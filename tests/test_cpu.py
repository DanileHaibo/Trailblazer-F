import json
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
import torch
from trailblazer_f.acceptance import acceptance_metrics,prefix_length
from trailblazer_f.runtime import ROOT,load_draft
from trailblazer_f.sim_core import denorm
from trailblazer_f.drive_core import BlockRunner


class CPUContracts(unittest.TestCase):
    def test_row_rms_strict_prefix(self):
        x=torch.zeros(3,8,3);y=x.clone()
        y[0,0]=.25;y[1,0]=.5;y[2,0]=.125
        metric,passing=acceptance_metrics(x,y,.5)
        self.assertEqual(metric.tolist(),[.25,.5,.125])
        self.assertEqual(passing.tolist(),[True,False,True])
        self.assertEqual(prefix_length(passing),1)
        mt,pt=acceptance_metrics(x.transpose(1,2),y.transpose(1,2),.5,coordinate_dim=1)
        torch.testing.assert_close(mt,metric,rtol=0,atol=0)
        self.assertTrue(torch.equal(pt,passing))
        with self.assertRaises(ValueError):acceptance_metrics(x,y,.5,'global')

    def test_checkpoint_load_shape(self):
        for model in ('simwam','drivewam'):
            draft,projection,saved=load_draft(model)
            x=torch.zeros(1,8,3) if model=='simwam' else torch.zeros(1,3,8)
            f=torch.zeros(1,256 if model=='simwam' else 273)
            y=draft(f,x,torch.tensor([0]))
            self.assertEqual(y.shape,(1,10,*x.shape[1:]))
            self.assertTrue(torch.isfinite(y).all())
            self.assertFalse(any(p.requires_grad for p in draft.parameters()))
            self.assertEqual(projection.shape,(3072,256))
            self.assertNotIn('optimizer',saved)
            if model=='simwam':self.assertNotIn('train_scenes',saved['config'])

    def test_distinct_action_decoding(self):
        sim=denorm(torch.zeros(8,3))
        torch.testing.assert_close(sim[0],torch.tensor([31.8,1.32,.095]),atol=2e-6,rtol=0)
        # DriveWAM predicts deltas and cumulatively sums, SimWAM absolute poses do not.
        q01=torch.tensor([-.01582209,-1.76743735,-.18790549])
        q99=torch.tensor([6.15755262,2.46998216,.18174374])
        deltas=(torch.zeros(1,3,8)+1)/2*(q99-q01)[:,None]+q01[:,None]
        drive=torch.cumsum(deltas,dim=-1)[0].T
        torch.testing.assert_close(drive[-1],8*drive[0])
        self.assertTrue(torch.equal(sim[0],sim[-1]))
        self.assertFalse(torch.allclose(sim,drive))

    def test_scheduler_formula(self):
        sim,_,_=load_draft('simwam');drive,_,_=load_draft('drivewam')
        u=torch.linspace(1,0,11)
        torch.testing.assert_close(sim.sigma,5*u/(1+4*u),rtol=0,atol=0)
        torch.testing.assert_close(drive.sigmas,u,rtol=0,atol=0)
        # Production verifier retains CPU FP32 scalar deltas in BF16 state math.
        e=SimpleNamespace(action_scheduler=SimpleNamespace(sigmas=u),_rollout_action_chunk=None)
        runner=BlockRunner(e);runner.target=lambda x,i:torch.ones_like(x)
        x=torch.zeros(1,3,8,dtype=torch.bfloat16)
        actual=runner.serial(x)[-1:]
        expected=x
        for i in range(10):expected=expected+torch.ones_like(expected)*((u[i+1] if i<9 else 0)-u[i])
        self.assertTrue(torch.equal(actual,expected))

    def test_manifest(self):
        m=json.loads((ROOT/'manifests/navtest.json').read_text())
        self.assertEqual(len(m['rows']),12146)
        self.assertEqual(len({r['token'] for r in m['rows']}),12146)
        self.assertEqual(len(set(m['latency_tokens'])),88)
        selected=[r for r in m['rows'] if r['token'] in m['latency_tokens']]
        self.assertEqual(len({r['recording'] for r in selected}),44)
        for r in m['rows']:self.assertEqual(set(r),{'token','log','recording','start'})

    def test_help_without_upstream(self):
        for extra in ([],['infer'],['score'],['aggregate']):
            result=subprocess.run([sys.executable,'-m','trailblazer_f.cli',*extra,'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertNotIn('Traceback',result.stderr)

if __name__=='__main__':unittest.main()
