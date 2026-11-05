"""Strict official supervised SimWAM loader."""
import torch
from pathlib import Path
from omegaconf import OmegaConf
from simwam.models.wan22.simwam import SimWAM
from simwam.models.wan22.wan_video_dit import WanVideoDiT
from simwam.models.wan22.action_dit import ActionDiT
from simwam.models.wan22.mot import MoT
from simwam.models.wan22.helpers.loader import _load_registered_model
from simwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

def load_model(weights, upstream):
    WEIGHTS=Path(weights)
    ROOT=Path(upstream).parent
    torch.manual_seed(42)
    dtype = torch.bfloat16
    cache = WEIGHTS/'fixed_prompt.pt'
    if not cache.exists():
        enc = _load_registered_model(str(WEIGHTS/'Wan/models_t5_umt5-xxl-enc-bf16.pth'),
                                     'wan_video_text_encoder', torch_dtype=dtype, device='cuda')
        tok = HuggingfaceTokenizer(name=str(WEIGHTS/'tokenizer/google/umt5-xxl'), seq_len=128, clean='whitespace')
        ids, mask = tok(PROMPT, return_mask=True, add_special_tokens=True)
        with torch.inference_mode():
            ctx = enc(ids.cuda(), mask.cuda())
            for i, length in enumerate(mask.gt(0).sum(dim=1)):
                ctx[i, length:] = 0
            mask = torch.ones_like(mask)
        torch.save({'context': ctx.cpu(), 'mask': mask.cpu()}, cache)
        del enc, tok, ctx
        torch.cuda.empty_cache()
    prompt = torch.load(cache, weights_only=True)
    cfg = OmegaConf.load(Path(upstream)/'configs/model/simwam_navsim.yaml')
    vc = OmegaConf.to_container(cfg.video_dit_config)
    ac = OmegaConf.to_container(cfg.action_dit_config)
    vc['use_gradient_checkpointing'] = ac['use_gradient_checkpointing'] = False
    # Construct on meta, then strictly assign official trained weights: no random missing parameters.
    with torch.device('meta'):
        video = WanVideoDiT(**vc)
        action = ActionDiT(**ac)
        mot = MoT(mixtures={'video': video, 'action': action}, mot_checkpoint_mixed_attn=False)
    payload = torch.load(WEIGHTS/'weights/SimWAM.pt', mmap=True, map_location='cpu', weights_only=True)
    print('strict weight load', mot.load_state_dict(payload['mot'], strict=True, assign=True), flush=True)
    mot.to(device='cuda', dtype=dtype)
    from simwam.models.wan22.wan_video_dit import precompute_freqs_cis, precompute_freqs_cis_3d
    video.freqs = precompute_freqs_cis_3d(vc['attn_head_dim'])
    action.freqs = precompute_freqs_cis(ac['attn_head_dim'], end=1024)
    vae = _load_registered_model(str(WEIGHTS/'Wan/Wan2.2_VAE.pth'), 'wan_video_vae', torch_dtype=dtype, device='cuda')
    model = SimWAM(video, action, mot, vae, text_dim=4096, proprio_dim=8, device='cuda', torch_dtype=dtype)
    model.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
    model.eval()
    return model, prompt


PROMPT = 'A high-quality, photorealistic dashboard camera view of autonomous driving. Based on the past 2 seconds videos, predict and generate the next 4 seconds of realistic driving continuation, Maintain temporal consistency, stable camera perspective, natural motion flow without jitter or artifacts, clear details, and realistic physics. '
