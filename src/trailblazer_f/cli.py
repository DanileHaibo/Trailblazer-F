"""Compute fresh predictions, NAVSIM v1 scores, and coverage-gated CSV results."""
import argparse
import csv
import dataclasses
import json
import os
from pathlib import Path
import random
import subprocess


def read(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def append(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def manifest(args):
    from .runtime import ROOT
    m=json.loads((ROOT/'manifests/navtest.json').read_text())
    by={r['token']:r for r in m['rows']}
    tokens=m['latency_tokens'] if args.scope=='latency' else sorted(by)
    rows=[by[t] for t in tokens]
    assert len(set(tokens))==len(tokens)==(88 if args.scope=='latency' else 12146)
    assert len({r['recording'] for r in rows})==44
    return rows[:args.limit or None]


def validate_split(args,rows):
    import yaml
    from .runtime import load_draft
    upstream=Path(args.upstream)
    relative='navsim/planning/script/config/common/train_test_split/scene_filter'
    directory=upstream/('navsim/'+relative if args.model=='simwam' else relative)
    test=yaml.safe_load((directory/'navtest.yaml').read_text())
    train=yaml.safe_load((directory/'navtrain.yaml').read_text())
    tt=set(test['tokens']);tr=set(train['tokens'])
    assert len(tt)==12146 and not tt&tr
    assert not set(test['log_names'])&set(train['log_names'])
    assert {r['token'] for r in rows}<=tt
    assert {r['log'] for r in rows}<=set(test['log_names'])
    _,_,saved=load_draft(args.model)
    testrec={'_'.join(x.split('_')[:2]) for x in test['log_names']}
    if args.model=='simwam':
        assert not (set(saved['train_recordings'])|set(saved['validation_recordings']))&testrec
    else:
        from .runtime import ROOT
        provenance=json.loads((ROOT/'manifests/training_provenance.json').read_text())['drivewam']
        assert provenance['manifest_identity']==saved['manifest_identity']
        assert not set(provenance['recordings'])&testrec
        assert set(provenance['tokens'])<=tr and not set(provenance['tokens'])&tt


def gpu_audit():
    """Audit only the CUDA device selected for this process; never kill other jobs."""
    import torch
    props=torch.cuda.get_device_properties(torch.cuda.current_device())
    uuid=str(props.uuid)
    if not uuid.startswith('GPU-'):uuid='GPU-'+uuid
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    processes=[s for s in output.splitlines() if s.split(',')[0].strip()==uuid]
    foreign=[s for s in processes if int(s.split(',')[1])!=os.getpid()]
    return {'gpu_uuid':uuid,'shared':bool(foreign),'processes':processes}


def infer(args):
    from .runtime import bootstrap,sha,ROOT
    bootstrap(args)
    import torch
    from .runtime import SimAdapter,DriveAdapter
    rows=manifest(args);validate_split(args,rows)
    output=Path(args.output)/args.scope/args.model
    output.mkdir(parents=True,exist_ok=True)
    methods=[args.method] if args.method!='both' else ['serial10','sp']
    identity=dict(model=args.model,scope=args.scope,methods=methods,tokens=[r['token'] for r in rows],seed=200,
        dtype='bfloat16',epsilon=.0375 if args.model=='simwam' else .1,block=10,
        checkpoint_sha256=sha(ROOT/'checkpoints'/f'{args.model}_draft.pt'),
        weights=str(Path(args.weights).resolve()),base_weights=args.base_weights,
        upstream=str(Path(args.upstream).resolve()),warmups=3,repeats=3 if args.scope=='latency' else 1,
        smoke=bool(args.limit))
    config=output/'config.json'
    if config.exists() and json.loads(config.read_text())!=identity:raise RuntimeError('Resume identity differs; use a new output directory')
    config.write_text(json.dumps(identity,indent=2)+'\n')
    path=output/'predictions.jsonl';previous=read(path)
    done={(r['token'],r['method'],r['repeat']) for r in previous}
    assert len(done)==len(previous),'Duplicate prediction rows'
    adapter=(SimAdapter if args.model=='simwam' else DriveAdapter)(args,rows)
    runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        python=__import__('sys').version,seed=200,dtype='bfloat16')
    if args.model=='drivewam':
        runtime['attention_module']=adapter.e.transformer.blocks[0].attn1.attn_op.__module__
        adapter.e._prepare_timesteps(adapter.e.action_scheduler,10)
        runtime['scheduler_sigmas']=adapter.e.action_scheduler.sigmas.cpu().tolist()
    else:
        runtime['timesteps']=adapter.ts.cpu().tolist();runtime['deltas']=adapter.ds.cpu().tolist()
    (output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    repeats=3 if args.scope=='latency' else 1
    with torch.inference_mode():
        first=adapter.load(rows[0])
        for method in methods:
            for _ in range(3):adapter.execute(first,method)
        for index,row in enumerate(rows):
            data=adapter.load(row)
            for rep in range(repeats):
                order=list(methods);random.Random(f"200:{row['token']}:{rep}").shuffle(order)
                for method in order:
                    key=(row['token'],method,rep)
                    if key in done:continue
                    before=gpu_audit()
                    if args.scope=='latency' and before['shared']:raise RuntimeError('Latency requires an exclusive selected GPU')
                    result=adapter.execute(data,method);after=gpu_audit()
                    if args.scope=='latency' and after['shared']:raise RuntimeError('GPU contamination detected; measurement not saved')
                    append(path,dict(token=row['token'],recording=row['recording'],method=method,repeat=rep,
                        shared_gpu=before['shared'] or after['shared'],**result));done.add(key)
            print(f"{args.model} {args.scope}: {index+1}/{len(rows)}",flush=True)


def score(args):
    from .runtime import bootstrap
    bootstrap(args)
    import numpy as np
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.common.dataclasses import Trajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    output=Path(args.output)/args.scope/args.model
    pred=read(output/'predictions.jsonl');path=output/'scores.jsonl'
    done={(r['token'],r['method'],r['repeat']) for r in read(path)}
    cache=MetricCacheLoader(Path(args.metric_cache));sampling=TrajectorySampling(num_poses=40,interval_length=.1)
    simulator=PDMSimulator(sampling);scorer=PDMScorer(sampling)
    for r in pred:
        key=(r['token'],r['method'],r['repeat'])
        if key in done:continue
        value=pdm_score(cache.get_from_token(r['token']),Trajectory(np.asarray(r['trajectory'],dtype=np.float64)),sampling,simulator,scorer)
        append(path,dict(token=r['token'],method=r['method'],repeat=r['repeat'],pdm={k:float(v) for k,v in dataclasses.asdict(value).items()}));done.add(key)


def aggregate(args):
    """Refuse partial, duplicate, mismatched or contaminated benchmark results."""
    from .runtime import ROOT
    m=json.loads((ROOT/'manifests/navtest.json').read_text());quality={r['token'] for r in m['rows']};latency=set(m['latency_tokens'])
    results=[]
    for model in ('simwam','drivewam'):
        qdir=Path(args.output)/'quality'/model;ldir=Path(args.output)/'latency'/model
        configs=[]
        for directory in (qdir,ldir):
            cfg=json.loads((directory/'config.json').read_text())
            assert not cfg['smoke'] and cfg['seed']==200 and cfg['dtype']=='bfloat16'
            assert cfg['epsilon']==(.0375 if model=='simwam' else .1)
            assert cfg['warmups']==3 and cfg['block']==10
            configs.append(cfg)
        assert configs[0]['checkpoint_sha256']==configs[1]['checkpoint_sha256']
        assert configs[0]['weights']==configs[1]['weights']
        assert configs[0]['base_weights']==configs[1]['base_weights']
        assert configs[1]['repeats']==3
        predictions=read(qdir/'predictions.jsonl');scores=read(qdir/'scores.jsonl');times=read(ldir/'predictions.jsonl')
        for method in ('serial10','sp'):
            pp=[r for r in predictions if r['method']==method];ss=[r for r in scores if r['method']==method];tt=[r for r in times if r['method']==method]
            assert len(pp)==len(ss)==12146 and {r['token'] for r in pp}=={r['token'] for r in ss}==quality
            assert all(r['repeat']==0 for r in pp+ss)
            assert len(tt)==264 and {(r['token'],r['repeat']) for r in tt}=={(t,i) for t in latency for i in range(3)}
            assert not any(r['shared_gpu'] for r in tt)
            accepted=sum(r['accepted'] for r in pp);checked=sum(r['checked'] for r in pp)
            item=dict(model=model,method=method,quality_N=12146,latency_N=88,repeats=3,
                e2e_ms=sum(r['e2e_ms'] for r in tt)/264,flow_ms=sum(r['flow_ms'] for r in tt)/264,
                accepted=accepted,verified=checked,prefix_acceptance=accepted/checked if checked else '',
                target_calls=sum(r['calls']+r['target_final_commit_calls'] for r in pp)/12146)
            item.update({k:100*sum(r['pdm'][k] for r in ss)/12146 for k in ss[0]['pdm']})
            results.append(item)
    for item in results:
        base=next(r for r in results if r['model']==item['model'] and r['method']=='serial10')
        item['speedup']=base['e2e_ms']/item['e2e_ms']
    dest=Path(args.output)/'main.csv'
    with dest.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print(dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('infer','score'):
        q=sub.add_parser(name)
        q.add_argument('--model',choices=('simwam','drivewam'),required=True)
        q.add_argument('--scope',choices=('quality','latency'),default='quality')
        q.add_argument('--upstream',required=True,help='Pinned SimWAM or DriveWAM repository root')
        q.add_argument('--output',default='outputs')
        q.add_argument('--data',help='NAVSIM v1 dataset root')
        if name=='infer':
            q.add_argument('--weights',required=True)
            q.add_argument('--base-weights',help='DriveWAM LingBot-VA base directory')
            q.add_argument('--method',choices=('serial10','sp','both'),default='both')
            q.add_argument('--limit',type=int,default=0,help='Smoke only; aggregate refuses capped runs')
        else:q.add_argument('--metric-cache',required=True)
    q=sub.add_parser('aggregate');q.add_argument('--output',default='outputs')
    a=p.parse_args()
    if a.command=='infer':
        if not a.data:p.error('infer requires --data')
        if a.limit<0:p.error('--limit must be nonnegative')
        if a.model=='drivewam' and not a.base_weights:p.error('DriveWAM requires --base-weights')
    globals()[a.command](a)

if __name__=='__main__':main()
