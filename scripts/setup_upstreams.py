"""Download pinned upstreams and apply the production read-only verifier cache fix.
Run explicitly; never invoked by importing the package.
"""
import argparse
import json
from pathlib import Path
import subprocess

PINS = {
    'SimWAM': ('https://github.com/H-EmbodVis/SimWAM.git', '1433ee7e06a62bd8b89da74455df0d82bda7b726'),
    'DriveWAM': ('https://github.com/chenshi3/DriveWAM.git', '14d16d2412dc4d70af59a921cc450299f5207f10'),
}


def patch_drive(root):
    p = root/'src/modules/model.py'
    text = p.read_text()
    old = """            slots = self.update_cache(cache_name,
                                      key,
                                      value,
                                      is_pred=(update_cache == 1))
            key_pool = self.attn_caches[cache_name]['k']
            value_pool = self.attn_caches[cache_name]['v']
            mask = self.attn_caches[cache_name]['mask']
            valid = mask.nonzero(as_tuple=False).squeeze(-1)
            key = key_pool[:, valid]
            value = value_pool[:, valid]"""
    new = """            if key.shape[0] != kv_cache['k'].shape[0]:
                if update_cache != 0 or kv_cache['k'].shape[0] != 1:
                    raise ValueError('Batched verification requires a read-only batch-one prefix')
                # Independent batch lanes share only the immutable visual/action prefix.
                valid = kv_cache['mask'].nonzero(as_tuple=False).squeeze(-1)
                prefix_k = kv_cache['k'][:, valid].expand(key.shape[0], -1, -1, -1)
                prefix_v = kv_cache['v'][:, valid].expand(value.shape[0], -1, -1, -1)
                key = torch.cat([prefix_k, key], dim=1)
                value = torch.cat([prefix_v, value], dim=1)
            else:
                slots = self.update_cache(cache_name,
                                          key,
                                          value,
                                          is_pred=(update_cache == 1))
                key_pool = self.attn_caches[cache_name]['k']
                value_pool = self.attn_caches[cache_name]['v']
                mask = self.attn_caches[cache_name]['mask']
                valid = mask.nonzero(as_tuple=False).squeeze(-1)
                key = key_pool[:, valid]
                value = value_pool[:, valid]"""
    if new not in text:
        assert text.count(old) == 1, 'Unsupported upstream cache implementation'
        text = text.replace(old,new)
        text = text.replace("            if kv_cache is not None and kv_cache['k'] is not None:\n                self.restore_cache(cache_name, slots)", "            if slots is not None:\n                self.restore_cache(cache_name, slots)")
        p.write_text(text)
    # Preserve the experiment's import fallback exactly; runtime provenance records the backend.
    text=p.read_text()
    old_import='except:\n    from flash_attn import flash_attn_func'
    fallback='''except ImportError:
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
            # PyTorch SDPA selects the supported CUDA attention implementation.
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=dropout_p, is_causal=causal, scale=softmax_scale,
            ).transpose(1, 2)'''
    if fallback not in text:
        assert text.count(old_import)==1
        p.write_text(text.replace(old_import,fallback))
    p=root/'src/navsim/eval.py';text=p.read_text()
    old='        transformer_path = os.path.join(checkpoint_path, "transformer")'
    extra='\n        if not os.path.isdir(transformer_path):\n            transformer_path = checkpoint_path  # Official HF transformer-only release.'
    if extra not in text:
        assert text.count(old)==1
        p.write_text(text.replace(old,old+extra))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,default=Path('third_party'))
    a=p.parse_args();a.destination.mkdir(parents=True,exist_ok=True)
    for name,(url,revision) in PINS.items():
        dest=a.destination/name
        if not dest.exists(): subprocess.run(['git','clone',url,str(dest)],check=True)
        current=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
        if current!=revision:
            subprocess.run(['git','-C',str(dest),'checkout','--detach',revision],check=True)
    patch_drive(a.destination/'DriveWAM')
    print(json.dumps(PINS,indent=2))

if __name__=='__main__':main()
