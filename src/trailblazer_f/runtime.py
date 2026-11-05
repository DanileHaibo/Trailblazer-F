"""Lazy model adapters: no CUDA or third-party imports on CLI --help."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if not (ROOT/'checkpoints').exists():
    ROOT=Path(os.environ.get('TRAILBLAZER_ROOT',Path.cwd()))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def bootstrap(args):
    root=Path(args.upstream).resolve()
    if args.model=='simwam' or args.command=='score':
        sys.path.insert(0,str(root/'src'))
        sys.path.insert(0,str(root/'navsim'))
    else:
        # DriveWAM owns top-level `src`; do NOT pip install this root as Trailblazer src.
        sys.path.insert(0,str(root))
    if args.data:
        os.environ['OPENSCENE_DATA_ROOT']=str(Path(args.data).resolve())
        os.environ.setdefault('NUPLAN_MAPS_ROOT',str(Path(args.data).resolve()/'maps'))
    return root


def load_draft(model,device='cpu'):
    import torch
    path=ROOT/'checkpoints'/f'{model}_draft.pt'
    meta=json.loads((ROOT/'checkpoints/metadata.json').read_text())[model]
    if sha(path)!=meta['export_sha256']:raise RuntimeError('Frozen draft SHA256 mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['frozen'] and tuple(saved['projection'].shape)==(3072,256)
    if model=='simwam':
        from .sim_core import PaperDraft
        cfg=saved['config'];assert cfg['mode']=='flow' and cfg['features'] and cfg['block_size']==10
        assert cfg['source_split']=='navtrain' and cfg['test_fitting'] is False
        draft=PaperDraft(cfg['mode'],cfg['features'],cfg['block_size'])
        draft.load_state_dict(saved['draft'],strict=True)
    else:
        from .drafts import StructuredDraft
        assert saved['mode']=='flow' and saved['feature_dim']==273
        draft=StructuredDraft(saved['feature_dim'],'flow')
        draft.load_state_dict(saved['model'],strict=True)
    draft=draft.to(device).eval().requires_grad_(False)
    return draft,saved['projection'].to(device),saved


def sim_input(row,data):
    import pickle
    import numpy as np
    import torch
    from PIL import Image
    with (Path(data)/'navsim_logs/test'/f"{row['log']}.pkl").open('rb') as f:frames=pickle.load(f)
    f=frames[row['start']+3];assert f['token']==row['token']
    with Image.open(Path(data)/'sensor_blobs/test'/f['cams']['CAM_F0']['data_path']) as im:
        im=im.convert('RGB').resize((672,384),Image.Resampling.BILINEAR)
        image=torch.from_numpy(np.array(im).copy()).permute(2,0,1).float()[None]/127.5-1
    state=torch.tensor(np.concatenate([f['ego_dynamic_state'],f['driving_command']]),dtype=torch.float32)[None]
    return image,state


def drive_batch(frames,cfg,data):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import default_collate
    from navsim.common.dataclasses import Scene,convert_absolute_to_relative_se2_array
    from nuplan.common.actor_state.state_representation import StateSE2
    egos=[Scene._build_ego_status(f) for f in frames]
    origin=StateSE2(*egos[3].ego_pose)
    history=convert_absolute_to_relative_se2_array(origin,np.array([e.ego_pose for e in egos[:4]],dtype=np.float64))
    path=Path(data)/'sensor_blobs/test'/frames[3]['cams']['CAM_F0']['data_path']
    with Image.open(path) as image:array=np.array(image)
    h,w=array.shape[:2];array=cv2.resize(array,(w//2,h//2),interpolation=cv2.INTER_AREA)
    images=torch.zeros((5,3,*array.shape[:2]),dtype=torch.uint8)
    images[0]=torch.as_tensor(array.copy()).permute(2,0,1)
    command=int(np.argmax(egos[3].driving_command))
    actions=torch.zeros((30,2,8,1),dtype=torch.float32);mask=torch.zeros_like(actions,dtype=torch.bool);mask[:3,1]=True
    sample=dict(images=images,prompt=cfg.command_prompts[command],command_id=torch.tensor(command),
        ego_velocity_acc=torch.as_tensor(np.concatenate([egos[3].ego_velocity,egos[3].ego_acceleration])),
        history_poses=torch.as_tensor(history,dtype=torch.float32),actions=actions,actions_mask=mask,
        scene_token=str(frames[3]['token']))
    return default_collate([sample])


class SimAdapter:
    def __init__(self,args,rows):
        import torch
        from .sim_loader import load_model
        self.args=args
        self.model,self.prompt=load_model(args.weights,args.upstream)
        self.draft,self.projection,_=load_draft('simwam','cuda')
        assert self.model.torch_dtype==torch.bfloat16
        self.ts,self.ds=self.model.infer_action_scheduler.build_inference_schedule(10,device='cuda',dtype=self.model.torch_dtype)
    def load(self,row):return sim_input(row,self.args.data)
    def execute(self,data,method):
        import torch
        from . import sim_core as p
        image,state=data;torch.cuda.synchronize();start=time.perf_counter()
        c=p.prefill(self.model,self.prompt,image,state,200)
        torch.cuda.synchronize();flow=time.perf_counter()
        if method=='serial10':
            x=p.serial(self.model,c,c['latents_action'].clone(),self.ts,self.ds)[-1:]
            stats=dict(calls=10,accepted=0,checked=0,candidate_passed=0,trace=[])
        else:
            f=p.feature_vector(c,self.projection)
            x,stats=p.speculative(self.model,self.draft,c,f,c['latents_action'].clone(),self.ts,self.ds,.0375,acceptance_mode='row_rms')
        action=x[0].float().cpu();trajectory=p.denorm(action).numpy().tolist()
        torch.cuda.synchronize();end=time.perf_counter()
        return dict(trajectory=trajectory,e2e_ms=(end-start)*1000,flow_ms=(end-flow)*1000,target_final_commit_calls=0,**stats)


class DriveAdapter:
    def __init__(self,args,rows):
        import copy
        import torch
        from src.configs import VA_CONFIGS
        from src.navsim.eval import NavSimEvaluator
        from navsim.common.dataloader import SceneLoader
        from navsim.common.dataclasses import SceneFilter
        from .drive_core import BlockRunner
        self.args=args;self.cfg=copy.deepcopy(VA_CONFIGS['navsim_cfg'])
        self.cfg.wan22_pretrained_model_name_or_path=args.base_weights
        self.e=NavSimEvaluator(self.cfg,args.weights,attn_mode='flashattn')
        assert self.e.dtype==torch.bfloat16
        self.e._encode_prompt(self.cfg.command_prompts);self.e.text_encoder.to('cpu');torch.cuda.empty_cache()
        self.runner=BlockRunner(self.e)
        self.runner.draft,self.runner.projection,_=load_draft('drivewam',self.e.device)
        self.loader=SceneLoader(Path(args.data)/'navsim_logs/test',Path(args.data)/'sensor_blobs/test',
            SceneFilter(num_history_frames=4,num_future_frames=10,frame_interval=1,tokens=[r['token'] for r in rows],log_names=sorted({r['log'] for r in rows})))
        self.video_ms=0
        original=self.e._rollout_video_frame
        def timed_video(*a,**kw):
            torch.cuda.synchronize();start=time.perf_counter();value=original(*a,**kw);torch.cuda.synchronize()
            self.video_ms+=(time.perf_counter()-start)*1000
            return value
        self.e._rollout_video_frame=timed_video
    def load(self,row):return drive_batch(self.loader.scene_frames_dicts[row['token']],self.cfg,self.args.data)
    def execute(self,batch,method):
        import torch
        self.runner.trace=[];self.runner.stats={};self.video_ms=0
        hook=self.runner.original if method=='serial10' else self.runner.rollout(.1)
        timing={}
        def timed(**kw):
            torch.cuda.synchronize();t=time.perf_counter();v=hook(**kw);torch.cuda.synchronize()
            timing['action_ms']=(time.perf_counter()-t)*1000;return v
        self.e._rollout_action_chunk=timed;torch.cuda.synchronize();start=time.perf_counter()
        try:
            actions,_=self.e.predict(batch,num_video_steps=3,num_action_steps=10,seed=200,video_denoise_fraction=.6)
            poses=self.e.pred_traj_to_poses(self.e.actions_to_trajectory(actions));actions.float().cpu();torch.cuda.synchronize()
        finally:self.e._rollout_action_chunk=self.runner.original
        stats=dict(calls=10,accepted=0,checked=0,candidate_passed=0) if method=='serial10' else self.runner.stats
        return dict(trajectory=poses.tolist(),e2e_ms=(time.perf_counter()-start)*1000,
            flow_ms=timing['action_ms']+self.video_ms,action_ms=timing['action_ms'],video_ms=self.video_ms,
            trace=list(self.runner.trace),target_final_commit_calls=1,**stats)
