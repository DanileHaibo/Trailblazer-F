"""Production DriveWAM block verifier; native CPU FP32 sigma deltas retained."""
import torch

class BlockRunner:
    block=10
    acceptance_mode="row_rms"
    def __init__(self,e):
        self.e=e
        self.original=e._rollout_action_chunk
        self.context=None

    def feature(self,ego):
        values=[]
        for i in (0,14,29):
            cache=self.e.transformer.blocks[i].attn1.attn_caches[self.e.cache_name]
            values.append(cache['v'][:,cache['mask']].float().mean(1).flatten(1))
        feature=torch.stack(values).mean(0)@self.projection
        feature=feature/(feature.square().mean(1,keepdim=True).sqrt()+1e-6)
        state=torch.cat([ego['velocity_acc'].flatten(1),ego['history_poses'].flatten(1),ego['command_ids'][:,None]],1).float()
        return torch.cat([feature,state],1),state

    def setup(self,chunk_index,action_shape,action_mask,prompt_emb,ego_cond,num_steps):
        self.context=(chunk_index,action_shape,action_mask,prompt_emb,ego_cond)
        self.e._prepare_timesteps(self.e.action_scheduler,num_steps)
        self.f,_=self.feature(ego_cond)

    def target(self,x,i,commit=False):
        chunk,shape,mask,prompt,ego=self.context
        n=x.shape[0]
        full=torch.zeros((n,*shape[1:]),device=x.device,dtype=x.dtype)
        full[:,:3,0,:,0]=x
        ts=self.e.action_scheduler.timesteps[i.cpu()].to(x.device).reshape(n,1)
        data=self.e._make_action_input(full,ts,prompt.expand(n,-1,-1),{k:v.expand(n,*v.shape[1:]) for k,v in ego.items()},chunk)
        v=self.e.transformer(data,update_cache=1 if commit else 0,cache_name=self.e.cache_name,action_mode=True)
        return v[:,:,:3].transpose(1,2)

    def serial(self,x):
        states=[x]
        for i in range(10):
            v=self.target(x,torch.tensor([i],device=x.device))
            # Match the scheduler's float32 CPU sigma difference exactly.
            s=self.e.action_scheduler.sigmas
            d=(s[i+1] if i<9 else 0)-s[i]
            x=x+v*d
            states.append(x)
        return torch.cat(states)

    def rollout(self,eps):
        def hook(**kw):
            self.setup(**kw)
            shape=kw['action_shape']
            full=torch.randn(shape,device=self.e.device,dtype=self.e.dtype)
            full=torch.where(kw['action_mask'],full,torch.zeros_like(full))
            x=full[:,:3,0,:,0]
            y,self.stats=self.spec(x,eps)
            full[:,:3,0,:,0]=y
            # Native zero-timestep pass commits action KV; included in timing.
            data=self.e._make_action_input(full,torch.zeros((1,1),device=self.e.device),kw['prompt_emb'],kw['ego_cond'],kw['chunk_index'])
            self.e.transformer(data,update_cache=1,cache_name=self.e.cache_name,action_mode=True)
            return full
        return hook

    def spec(self,x,eps):
        self.trace=[];k=calls=accepted=checked=drafts=0
        while k<10:
            n=min(self.block,10-k)
            proposals=self.draft(self.f,x,torch.tensor([k],device=x.device))[0,:n].to(x.dtype)
            drafts+=1;inputs=torch.cat([x,proposals[:-1]]);ids=torch.arange(k,k+n,device=x.device)
            v=self.target(inputs,ids);s=self.e.action_scheduler.sigmas
            ds=[(s[j+1] if j<9 else 0)-s[j] for j in range(k,k+n)]
            corrected=inputs+torch.stack([v[j]*ds[j] for j in range(n)])
            from .acceptance import acceptance_metrics
            residual,passing=acceptance_metrics(corrected,proposals,eps,self.acceptance_mode,coordinate_dim=1)
            a=0
            if eps>0:
                for ok in passing.tolist():
                    if not ok:break
                    a+=1
            calls+=1;checked+=n;accepted+=a
            self.trace.append(dict(start=k,lanes=n,accepted=a,candidate_passed=int(passing.sum()),residual_rms=residual.tolist()))
            if a==n:
                x=corrected[-1:];k+=n;continue
            if eps==0:
                current=inputs[a:a+1];vi=self.target(current,ids[a:a+1]);calls+=1;x=current+vi*ds[a]
            else:x=corrected[a:a+1]
            k+=a+1
        return x,dict(calls=calls,accepted=accepted,checked=checked,draft_calls=drafts,candidate_passed=sum(t['candidate_passed'] for t in self.trace))

