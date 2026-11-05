"""Frozen SimWAM inference primitives, extracted without numerical rewrites."""
import torch
from torch import nn

def prefill(model,prompt,image,state,seed):
    class Ready(Exception):
        pass
    condition = {}
    original = model._predict_action_noise_with_cache
    def hook(**kwargs):
        condition.update(kwargs)
        raise Ready()
    model._predict_action_noise_with_cache = hook
    try:
        model.infer_action(prompt=None,input_image=image,action_horizon=8,proprio=state,
                           context=prompt['context'],context_mask=prompt['mask'],num_inference_steps=10,seed=seed)
    except Ready:
        pass
    finally:
        model._predict_action_noise_with_cache = original
    if not condition:
        raise RuntimeError('Prefill did not produce conditioning')
    return condition


def target(model,condition,x,ts):
    n = len(x)
    return model._predict_action_noise_with_cache(latents_action=x,timestep_action=ts,
        context=condition['context'].expand(n,-1,-1),context_mask=condition['context_mask'].expand(n,-1),
        video_kv_cache=[{k:v.expand(n,*v.shape[1:]) for k,v in c.items()} for c in condition['video_kv_cache']],
        attention_mask=condition['attention_mask'],video_seq_len=condition['video_seq_len'])


def serial(model,condition,x,ts,ds):
    states = [x]
    for t,d in zip(ts,ds):
        x = model.infer_action_scheduler.step(target(model,condition,x,t[None]),d,x)
        states.append(x)
    return torch.cat(states)


def feature_vector(condition,projection):
    feature = torch.stack([condition['video_kv_cache'][i]['v'].float().mean(1) for i in (0,14,29)]).mean(0)@projection
    return feature/(feature.square().mean().sqrt()+1e-6)


class PaperDraft(nn.Module):
    def __init__(self,mode='flow',features=True,block_size=10):
        super().__init__()
        self.mode,self.features,self.block_size = mode,features,block_size
        u = torch.linspace(1,0,11)
        self.register_buffer('sigma',5*u/(1+4*u))
        # Feature-free ablation has a smaller input architecture, independently trained.
        self.net = nn.Sequential(nn.Linear((256 if features else 0)+24+10,512),nn.SiLU(),
                                 nn.Linear(512,512),nn.SiLU(),nn.Linear(512,block_size*24))

    def forward(self,feature,x,index):
        parts = [x.flatten(1).float(),torch.nn.functional.one_hot(index,10).float()]
        if self.features:
            parts.insert(0,feature.float())
        delta = self.net(torch.cat(parts,1)).reshape(-1,self.block_size,8,3)
        if self.mode == 'residual':
            return x[:,None].float()+delta
        nxt = (index[:,None]+torch.arange(self.block_size,device=x.device)[None]+1).clamp(max=10)
        ratio = self.sigma[nxt]/self.sigma[index,None]
        return ratio[:,:,None,None]*x[:,None].float()+(1-ratio[:,:,None,None])*delta


def speculative(model,draft,c,f,x,ts,ds,epsilon,acceptance_mode='row_rms'):
    from .acceptance import acceptance_metrics
    candidate_passed = 0
    trace = []
    k,calls,accepted,checked,fallback = 0,0,0,0,0
    while k<10:
        n = min(draft.block_size,10-k)
        proposals = draft(f,x,torch.tensor([k],device=x.device))[0,:n].to(x.dtype)
        inputs = torch.cat([x,proposals[:-1]])
        v = target(model,c,inputs,ts[k:k+n])
        corrected = model.infer_action_scheduler.step(v,ds[k:k+n],inputs)
        calls += 1; checked += n
        metric, passing = acceptance_metrics(corrected, proposals, epsilon, acceptance_mode)
        passes = passing.tolist()
        candidate_passed += sum(passes)
        a = 0
        if epsilon>0:
            while a<n and passes[a]:
                a+=1
        accepted += a
        trace.append(dict(start=k,lanes=n,accepted=a,candidate_passed=sum(passes),metric=metric.tolist()))
        if a==n:
            x = corrected[-1:]; k+=n
            continue
        if epsilon==0:
            current = inputs[a:a+1]
            v = target(model,c,current,ts[k+a:k+a+1])
            x = model.infer_action_scheduler.step(v,ds[k+a],current)
            calls+=1; fallback+=1
        else:
            x = corrected[a:a+1]
        k += a+1
    return x,dict(calls=calls,accepted=accepted,checked=checked,fallback=fallback,candidate_passed=candidate_passed,trace=trace)


def denorm(x):
    return (x.float()+1)/2*x.new_tensor([66.74,42,3.53]).float()-x.new_tensor([1.57,19.68,1.67]).float()

