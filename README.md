# Trailblazer-F

**Frozen structured speculative flow inference for SimWAM and DriveWAM on NAVSIM v1.**

This repository contains the minimal inference and evaluation release for four configurations: SimWAM serial-10, SimWAM + SP, DriveWAM serial-10, and DriveWAM + SP. It does not distribute a manuscript, datasets, baseline model weights, historical results, or training samples.

## Protocol

- Quality and acceptance: all **12,146 official navtest scenarios**, 44 recordings.
- Latency: the fixed **88-scenario / 44-recording** manifest, **3 warmups per method and 3 repetitions per scenario**, batch size 1. Use an exclusive GPU. Timing is synchronized, includes conditioning, draft features, native decoding and CPU transfer, and excludes disk I/O and offline PDM scoring.
- Seed **200**, BF16 target states, ten action integration steps, frozen FP32 MLP drafts, block size 10.
- SP accepts a **continuous prefix** using the maximum per-row RMS over three coordinates, computed from FP32 residuals with **strict `< epsilon`**. SimWAM epsilon is **0.0375**; DriveWAM epsilon is **0.1**. The teacher-corrected first rejected state is committed. No global-RMS policy, transformer draft, or test-time threshold fitting is exposed.
- DriveWAM retains three video steps, video denoising fraction 0.6, and the native final action-cache commit. Its CPU FP32 sigma differences must not be pre-rounded to BF16.

`configs/main.json` records the frozen protocol. `manifests/navtest.json` contains only split identifiers and log-window indices, not sensor data, poses, or trajectories. The latency token order is preserved from the production manifest, not resampled.

## 1. Installation

Use separate environments for SimWAM, DriveWAM and CPU scoring when their dependencies conflict. All commands below run from this repository root. Set `TRAILBLAZER_ROOT` to this root when invoking an installed package from another directory.

```bash
python scripts/setup_upstreams.py --destination third_party
```

This downloads pinned official repositories:

- [SimWAM](https://github.com/H-EmbodVis/SimWAM), revision `1433ee7e06a62bd8b89da74455df0d82bda7b726`.
- [DriveWAM](https://github.com/chenshi3/DriveWAM), revision `14d16d2412dc4d70af59a921cc450299f5207f10`.
- NAVSIM v1.1 is the release subset inside the pinned SimWAM repository. Do **not** substitute NAVSIM v2.

The setup script applies the production DriveWAM read-only batched-prefix cache fix and Hugging Face checkpoint-layout fix. It also retains the production FlashAttention import fallback to PyTorch SDPA; different attention backends, PyTorch versions, and hardware can affect BF16 numerics. The upstream repositories are not vendored into this release.

SimWAM environment (Python 3.10, following its pinned requirements):

```bash
conda create -n tb-simwam python=3.10 -y
conda activate tb-simwam
python -m pip install -r third_party/SimWAM/requirements.txt
python -m pip install -e third_party/SimWAM/navsim --no-deps
python -m pip install -e third_party/SimWAM --no-deps
python -m pip install -e . --no-deps
```

DriveWAM environment (following its checked-in environment, not a nonexistent requirements file):

```bash
conda env create -f third_party/DriveWAM/environment.yml -n tb-drivewam
conda activate tb-drivewam
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -e third_party/SimWAM/navsim --no-deps
python -m pip install -e . --no-deps
```

**Import roots matter:** Trailblazer-F installs only `trailblazer_f`. DriveWAM imports `src.*` from its repository root; SimWAM imports `simwam` from its `src/` directory. The CLI configures these roots lazily. Never install the DriveWAM root as this package's `src`, and never run both model adapters in the same Python process.

The original local DriveWAM runtime used Python 3.12.14, PyTorch 2.10.0+cu128, NumPy 2.2.6, diffusers 0.36.0, transformers 4.55.2, accelerate 1.12.0 and nuplan-devkit 1.1.1, with the SDPA fallback available. The installation above follows upstream-supported environments; cross-environment bitwise identity is not claimed. CPU semantic and export equivalence checks are included; a clean-environment GPU validation is still required.

## 2. Official baseline checkpoints

Official links verified against the upstream READMEs:

- [SimWAM weights](https://huggingface.co/H-EmbodVis/SimWAM): use supervised `weights/SimWAM.pt`, **not** the RL or other-dataset checkpoints.
- [Wan2.2 TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B): VAE and UMT5 encoder used by SimWAM; its official loader uses the [Wan2.1 T2V-1.3B tokenizer](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B).
- [DriveWAM weights and NAVSIM metric cache](https://huggingface.co/chenchenshi/DriveWAM): use `navsim/`.
- [LingBot-VA base](https://huggingface.co/robbyant/lingbot-va-base): DriveWAM VAE, tokenizer and text encoder.

Pinned download commands (large; run only after accepting applicable upstream terms):

```bash
python -m pip install huggingface_hub
python scripts/download_weights.py --model simwam --destination weights/simwam
python scripts/download_weights.py --model drivewam --destination weights/drivewam
```

Download revisions are the revisions recorded by the original local Hugging Face artifacts. SimWAM creates `fixed_prompt.pt` in its configurable weights directory on first use; it is regenerated from the exact fixed prompt, not distributed as dataset content.

The two included small drafts are safe-loadable with `torch.load(..., weights_only=True)`. `checkpoints/metadata.json` records original and exported SHA256 values. Export preserves all model tensors, projection tensors, architecture settings, frozen flags and training-manifest identities. Optimizer/RNG state and SimWAM's list of training samples are removed. DriveWAM's identity is checked against a path-free train-only provenance manifest.

## 3. NAVSIM v1 data and metric cache

Follow the [official NAVSIM v1.1 data instructions](https://github.com/autonomousvision/navsim/tree/v1.1) and [OpenScene](https://github.com/OpenDriveLab/OpenScene). Obtain the **test metadata and front-camera sensor blobs** and nuPlan maps under your own licensed storage root. No raw data is downloaded by importing or running inference.

Set paths, for example:

```bash
export OPENSCENE_DATA_ROOT="$PWD/data/navsim"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export TRAILBLAZER_ROOT="$PWD"
```

Expected layout: `$OPENSCENE_DATA_ROOT/navsim_logs/test/*.pkl`, `$OPENSCENE_DATA_ROOT/sensor_blobs/test/<log>/CAM_F0/*`, and maps under `$NUPLAN_MAPS_ROOT`. Input pickle files and metric caches must come from trusted official sources.

Generate the official **navtest** PDM metric cache using the pinned NAVSIM v1 release (CPU scoring dependencies are provided by the SimWAM environment):

```bash
conda activate tb-simwam
python third_party/SimWAM/navsim/navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest cache.cache_path="$OPENSCENE_DATA_ROOT/navtest_metric_cache"
```

Alternatively download the precomputed metric-cache archive from the official DriveWAM Hugging Face page above and extract it into your chosen cache root. The release does not assume or fabricate an archive filename. Verify all 12,146 navtest tokens are cached before a full evaluation. No processed DriveWAM training samples are necessary: evaluation constructs only the official current-image/history input with zero future-action conditioning directly from the logs.

## 4. Reproduce four configurations

Run each model sequentially on the GPU you choose. `CUDA_VISIBLE_DEVICES` is never overridden. Change `0` below as appropriate. The commands compute fresh trajectories; they do not read historical scores.

```bash
conda activate tb-simwam
CUDA_VISIBLE_DEVICES=0 trailblazer-f infer --model simwam --scope quality \
  --upstream third_party/SimWAM --weights weights/simwam \
  --data "$OPENSCENE_DATA_ROOT" --output outputs
trailblazer-f score --model simwam --scope quality --upstream third_party/SimWAM \
  --metric-cache "$OPENSCENE_DATA_ROOT/navtest_metric_cache" --output outputs

conda activate tb-drivewam
CUDA_VISIBLE_DEVICES=0 trailblazer-f infer --model drivewam --scope quality \
  --upstream third_party/DriveWAM --weights weights/drivewam/navsim \
  --base-weights weights/drivewam/base --data "$OPENSCENE_DATA_ROOT" --output outputs

conda activate tb-simwam
trailblazer-f score --model drivewam --scope quality --upstream third_party/SimWAM \
  --metric-cache "$OPENSCENE_DATA_ROOT/navtest_metric_cache" --output outputs
```

`--method both` is the default, running serial-10 and SP independently with the same seed. For a GPU smoke test append `--limit 1 --output outputs-smoke`; capped runs are explicitly ineligible for final aggregation. Resume is supported only when the saved configuration matches; malformed/partial JSONL raises rather than silently discarding results.

Latency (88 scenes, 44 recordings, three warmups and repetitions) must run on an **exclusive** selected GPU:

```bash
conda activate tb-simwam
CUDA_VISIBLE_DEVICES=0 trailblazer-f infer --model simwam --scope latency \
  --upstream third_party/SimWAM --weights weights/simwam \
  --data "$OPENSCENE_DATA_ROOT" --output outputs

conda activate tb-drivewam
CUDA_VISIBLE_DEVICES=0 trailblazer-f infer --model drivewam --scope latency \
  --upstream third_party/DriveWAM --weights weights/drivewam/navsim \
  --base-weights weights/drivewam/base --data "$OPENSCENE_DATA_ROOT" --output outputs

trailblazer-f aggregate --output outputs
```

`outputs/main.csv` is calculated from fresh token-level quality scores and latency measurements. Aggregation refuses missing/duplicate scenarios, capped smoke outputs, wrong thresholds, incomplete repeats and GPU-contaminated timing. Acceptance is `sum(accepted prefix steps) / sum(verified candidate steps)` over the full quality set, not a mean of per-scene percentages. Quality columns are on a 0–100 scale; speedup is relative to each model's newly measured serial-10 E2E latency. Flow timing is action flow for SimWAM and video plus action flow for DriveWAM.

## 5. CPU checks and release provenance

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
trailblazer-f --help
trailblazer-f infer --help
```

Tests cover strict row-RMS prefix semantics, both action layouts/decoding conventions, frozen checkpoint loading and shapes, shifted-sigma draft schedules, native FP32-scalar DriveWAM integration, manifest coverage, and CLI help without importing GPU models. Maintainer scripts `export_release.py` and `verify_export.py` document extraction from the production sources; they require an explicit source workspace and are not needed for inference.

`manifests/verification.json` distinguishes verified CPU bit-exact draft/projection/verifier equivalence from **not yet tested** clean-environment GPU inference and full-dataset reruns. No benchmark numbers are embedded in the release. Absolute runtime and BF16 agreement depend on the GPU, attention implementation and software environment.

## Third-party terms

SimWAM, DriveWAM, NAVSIM, nuPlan, OpenScene, Wan2.2 and LingBot-VA retain their respective licenses and model/data terms. Download them from their official repositories and retain their notices. This repository does not relicense those projects. A new license for original Trailblazer-F contributions is intentionally not asserted pending confirmation by the repository owner.
