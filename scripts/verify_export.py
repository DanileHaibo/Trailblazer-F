"""Maintainer CPU equivalence audit against the untouched experiment workspace."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
import torch


def extract(path,name):
    text=path.read_text();lines=text.splitlines(keepends=True)
    n=next(n for n in ast.parse(text).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
    return ''.join(lines[n.lineno-1:n.end_lineno])


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--sim-checkpoint',type=Path,required=True)
    a=p.parse_args()
    from trailblazer_f.runtime import ROOT,load_draft,sha
    from trailblazer_f import sim_core
    from trailblazer_f.drive_core import BlockRunner
    env={'torch':torch,'nn':torch.nn}
    exec(extract(a.source/'paper_simwam.py','PaperDraft'),env)
    exec(extract(a.source/'paper_drivewam_train.py','StructuredDraft'),env)
    for model,name,source in [('simwam','PaperDraft',a.sim_checkpoint),('drivewam','StructuredDraft',a.source/'work_dirs/paper_drivewam_formal/paper_drivewam_flow_frozen.pt')]:
        draft,projection,saved=load_draft(model)
        original=torch.load(source,map_location='cpu',weights_only=True)
        reference=env[name]()
        reference.load_state_dict(original['draft' if model=='simwam' else 'model'],strict=True)
        torch.manual_seed(200)
        f=torch.randn(3,256 if model=='simwam' else 273)
        x=torch.randn((3,8,3) if model=='simwam' else (3,3,8))
        idx=torch.tensor([0,4,9])
        torch.testing.assert_close(draft(f,x,idx),reference(f,x,idx),rtol=0,atol=0)
        assert torch.equal(projection,original['projection'])
    # Compare production algorithms using deterministic fake teacher, real frozen draft and BF16 states.
    acceptance=types.ModuleType('paper_acceptance')
    exec((a.source/'paper_acceptance.py').read_text(),acceptance.__dict__)
    sys.modules['paper_acceptance']=acceptance
    env={'torch':torch}
    for name in ('target','serial','speculative'):exec(extract(a.source/'paper_simwam.py',name),env)
    class Scheduler:
        def step(self,v,d,x):
            d=d.to(x.device,dtype=x.dtype)
            return x+v*(d if d.ndim==0 else d.view(-1,1,1))
    model=types.SimpleNamespace(infer_action_scheduler=Scheduler(),_predict_action_noise_with_cache=lambda **kw:kw['latents_action']*.2)
    c={'context':torch.zeros(1,1,1),'context_mask':torch.ones(1,1),'video_kv_cache':[],'attention_mask':None,'video_seq_len':1}
    draft,projection,_=load_draft('simwam');torch.manual_seed(200)
    x=torch.randn(1,8,3).bfloat16();f=torch.randn(1,256);ts=torch.linspace(1,0,10).bfloat16();ds=torch.full((10,),-.1,dtype=torch.bfloat16)
    left,ls=sim_core.speculative(model,draft,c,f,x,ts,ds,.0375,'row_rms')
    right,rs=env['speculative'](model,draft,c,f,x,ts,ds,.0375,'row_rms')
    assert torch.equal(left,right) and ls==rs
    original=extract(a.source/'paper_drivewam_full.py','BlockRunner')
    denv={'torch':torch,'FormalRunner':object}
    exec(original,denv)
    r=BlockRunner(types.SimpleNamespace(_rollout_action_chunk=None,action_scheduler=types.SimpleNamespace(sigmas=torch.linspace(1,0,11))))
    r.draft,_,_=load_draft('drivewam');r.f=torch.randn(1,273)
    r.target=lambda x,i:x*.2
    orig=object.__new__(denv['BlockRunner']);orig.__dict__.update(r.__dict__);orig.acceptance_mode='row_rms'
    x=torch.randn(1,3,8).bfloat16()
    left,ls=r.spec(x,.1);right,rs=orig.spec(x,.1)
    assert torch.equal(left,right) and ls==rs and r.trace==orig.trace
    # Preserve train-only identifiers, not samples or original local paths.
    train=json.loads((a.source/'work_dirs/paper_drivewam_formal/paper_drivewam_manifest.json').read_text())
    recordings=sorted({r['recording'] for r in train['rows']});tokens=sorted({r['token'] for r in train['rows']})
    assert all(r['source_split']=='navtrain' for r in train['rows'])
    m=json.loads((ROOT/'manifests/navtest.json').read_text())
    assert not set(recordings)&{r['recording'] for r in m['rows']}
    assert not set(tokens)&{r['token'] for r in m['rows']}
    provenance={'drivewam':{'manifest_identity':train['identity'],'recordings':recordings,'tokens':tokens,'source_split':'navtrain','test_fitting':False}}
    (ROOT/'manifests/training_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    report={'cpu_only':True,'draft_forward_bit_exact':True,'projection_bit_exact':True,'simwam_production_verifier_bit_exact':True,'drivewam_production_verifier_bit_exact':True,'gpu_tested':False,'full_evaluation_tested':False}
    (ROOT/'manifests/verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
