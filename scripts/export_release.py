"""Maintainer-only export of frozen inference artifacts from an experiment workspace.
Never copies datasets, predictions, optimizer state, RNG state or private paths.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path, names):
    text = path.read_text()
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    return '\n\n'.join(''.join(lines[n.lineno-1:n.end_lineno]) for n in tree.body
                         if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sim-checkpoint', type=Path, required=True)
    parser.add_argument('--drive-checkpoint', type=Path, required=True)
    parser.add_argument('--sim-manifest', type=Path, required=True)
    parser.add_argument('--drive-audit', type=Path, required=True)
    args = parser.parse_args()
    import torch
    root = Path(__file__).resolve().parents[1]
    package = root / 'src/trailblazer_f'
    package.mkdir(parents=True, exist_ok=True)
    def emit(name, header, source, names, replacements=()):
        body = extract(args.source / source, names)
        for old, new in replacements:
            body = body.replace(old, new)
        (package / name).write_text(header + '\n\n' + body + '\n')
    emit('sim_core.py', '"""Frozen SimWAM inference primitives, extracted without numerical rewrites."""\nimport torch\nfrom torch import nn',
         'paper_simwam.py', {'PaperDraft','prefill','target','serial','feature_vector','speculative','denorm'},
         [('from paper_acceptance import acceptance_metrics','from .acceptance import acceptance_metrics'),("acceptance_mode='global'", "acceptance_mode='row_rms'")])
    base = extract(args.source/'test_speculative_drivewam.py', {'Runner'})
    # Replace only pilot draft initialization and pilot verifier: production methods below override feature/spec.
    tree = ast.parse(base).body[0]
    lines = base.splitlines(keepends=True)
    selected = {'setup','target','serial','rollout'}
    methods = '\n'.join(''.join(lines[n.lineno-1:n.end_lineno]) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in selected)
    methods = methods.replace('def rollout(self,eps=None,collect_seeds=None):','def rollout(self,eps):')
    begin=methods.index('            if collect_seeds is not None:')
    end=methods.index('            full[:,:3,0,:,0]=y',begin)
    methods=methods[:begin]+'            y,self.stats=self.spec(x,eps)\n'+methods[end:]
    formal = extract(args.source/'paper_drivewam_evaluate.py', {'FormalRunner'})
    ft = ast.parse(formal).body[0]; fl = formal.splitlines(keepends=True)
    feature = next(''.join(fl[n.lineno-1:n.end_lineno]) for n in ft.body if isinstance(n,ast.FunctionDef) and n.name=='feature')
    block = extract(args.source/'paper_drivewam_full.py', {'BlockRunner'})
    bt = ast.parse(block).body[0]; bl = block.splitlines(keepends=True)
    spec = next(''.join(bl[n.lineno-1:n.end_lineno]) for n in bt.body if isinstance(n,ast.FunctionDef) and n.name=='spec')
    spec = spec.replace('from paper_acceptance import acceptance_metrics','from .acceptance import acceptance_metrics')
    (package/'drive_core.py').write_text('"""Production DriveWAM block verifier; native CPU FP32 sigma deltas retained."""\nimport torch\n\nclass BlockRunner:\n    block=10\n    acceptance_mode="row_rms"\n    def __init__(self,e):\n        self.e=e\n        self.original=e._rollout_action_chunk\n        self.context=None\n\n'+feature+'\n'+methods+'\n'+spec+'\n')
    emit('drafts.py','"""Frozen DriveWAM structured MLP architecture."""\nimport torch\nfrom torch import nn','paper_drivewam_train.py',{'StructuredDraft'})
    emit('sim_loader.py', '"""Strict official supervised SimWAM loader."""\nimport torch\nfrom pathlib import Path\nfrom omegaconf import OmegaConf\nfrom simwam.models.wan22.simwam import SimWAM\nfrom simwam.models.wan22.wan_video_dit import WanVideoDiT\nfrom simwam.models.wan22.action_dit import ActionDiT\nfrom simwam.models.wan22.mot import MoT\nfrom simwam.models.wan22.helpers.loader import _load_registered_model\nfrom simwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer', 'test_simwam_baseline.py',{'load_model'}, [('def load_model():','def load_model(weights, upstream):\n    WEIGHTS=Path(weights)\n    ROOT=Path(upstream).parent'),("ROOT/'SimWAM/configs/model/simwam_navsim.yaml'","Path(upstream)/'configs/model/simwam_navsim.yaml'"),("weights_only=False","weights_only=True")])
    prompt = next(n.value for n in ast.parse((args.source/'test_simwam_baseline.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPT' for t in n.targets))
    with (package/'sim_loader.py').open('a') as f: f.write('\nPROMPT = '+repr(ast.literal_eval(prompt))+'\n')
    # Only inference-required configuration + immutable provenance identifiers survive export.
    metadata={}
    for model,path in [('simwam',args.sim_checkpoint),('drivewam',args.drive_checkpoint)]:
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if model=='simwam':
            cfg={k:v for k,v in saved['config'].items() if k!='train_scenes'}
            payload={k:saved[k] for k in ('schema','draft','projection','iteration','best_validation_mse','train_recordings','validation_recordings','frozen')}
            payload['config']=cfg
        else: payload=dict(saved)
        dest=root/'checkpoints'/f'{model}_draft.pt';dest.parent.mkdir(exist_ok=True)
        torch.save(payload,dest)
        metadata[model]={'source_sha256':sha(path),'export_sha256':sha(dest),'bytes':dest.stat().st_size,'keys':sorted(payload),'frozen':payload['frozen'],'projection_shape':list(payload['projection'].shape)}
        # All inference tensors must be bit-identical, including schedule buffers.
        statekey='draft' if model=='simwam' else 'model'
        reloaded=torch.load(dest,map_location='cpu',weights_only=True)
        for key,val in saved[statekey].items(): assert torch.equal(val,reloaded[statekey][key])
        assert torch.equal(saved['projection'],reloaded['projection'])
    m=json.loads(args.sim_manifest.read_text());a=json.loads(args.drive_audit.read_text());by={r['token']:r for r in m['full_rows']}
    rows=[{k:r[k] for k in ('token','log','recording','start')} for r in m['full_rows']]
    assert len(rows)==12146 and len({r['recording'] for r in rows})==44
    pilot=a['stratified88'];assert len(set(pilot))==88 and len({by[t]['recording'] for t in pilot})==44
    manifest={'split':'navtest','quality_scenes':12146,'recordings':44,'latency_tokens':pilot,'rows':rows,'source_manifest_sha256':sha(args.sim_manifest),'source_latency_audit_sha256':sha(args.drive_audit)}
    (root/'manifests').mkdir(exist_ok=True)
    (root/'manifests/navtest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'checkpoints/metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    sources=['paper_simwam.py','paper_drivewam_full.py','paper_drivewam_evaluate.py','paper_drivewam_train.py','test_speculative_drivewam.py','test_simwam_baseline.py']
    (root/'manifests/source_hashes.json').write_text(json.dumps({s:sha(args.source/s) for s in sources},indent=2)+'\n')

if __name__=='__main__': main()
