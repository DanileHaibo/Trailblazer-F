"""Download official pinned baseline artifacts (license acceptance remains your responsibility)."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('simwam','drivewam'),required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--metric-cache',action='store_true',help='Also download the DriveWAM-hosted NAVSIM cache; inspect archive before extraction')
    a=p.parse_args()
    from huggingface_hub import snapshot_download
    dest=a.destination.resolve()
    if a.model=='simwam':
        snapshot_download('H-EmbodVis/SimWAM',revision='5d4492937e357e6ca64de128cdccbd25c2b34f30',local_dir=dest,allow_patterns=['weights/SimWAM.pt'])
        snapshot_download('Wan-AI/Wan2.2-TI2V-5B',revision='921dbaf3f1674a56f47e83fb80a34bac8a8f203e',local_dir=dest/'Wan',allow_patterns=['Wan2.2_VAE.pth','models_t5_umt5-xxl-enc-bf16.pth'])
        snapshot_download('Wan-AI/Wan2.1-T2V-1.3B',revision='37ec512624d61f7aa208f7ea8140a131f93afc9a',local_dir=dest/'tokenizer',allow_patterns=['google/umt5-xxl/*'])
    else:
        patterns=['navsim/*']
        if a.metric_cache:patterns+=['*metric*','**/*metric*']
        snapshot_download('chenchenshi/DriveWAM',revision='c4fb6d2c5d26ee3018a9ee1507f36fd02b85b1ce',local_dir=dest,allow_patterns=patterns)
        snapshot_download('robbyant/lingbot-va-base',revision='68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c',local_dir=dest/'base',allow_patterns=['vae/*','tokenizer/*','text_encoder/*'])

if __name__=='__main__':main()
