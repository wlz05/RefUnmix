# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

RefUnmix separates a generator's conditional innovation from its shared source-conditioned transport. A joint response rule retains useful shared changes and reallocates their energy toward the requested edit. The backbone stays frozen.

## Method

```text
Source image + source/target prompts
                  |
        Paired conditional field
                  |
     Three matched native endpoints
                  |
 Conditional innovation / shared transport
                  |
       Joint response allocation
                  |
           Edited RGB image
```

Let $h_{0,0}$, $h_{1,0}$, and $h_{1,1}$ denote the source-zero, source-field, and target-field native latent endpoints. The first index indicates whether the field is applied (0 or 1); the second selects the source (0) or target (1) condition. Their decoded differences give conditional innovation $d$ and shared transport $a$:

```math
d=V(h_{1,1})-V(h_{1,0}),\qquad
a=V(h_{1,0})-V(h_{0,0}),\qquad
Y=X+\eta d+qa.
```

The interaction between d and a determines how much shared transport to retain while increasing conditional innovation. A per-pixel source-error envelope constrains allocation through clipping and 8-bit export. Responsibility is measured from model responses. Generation uses no annotation mask or CLIP-based output selection.

The main profile uses the paired source/target contrast under shared sampled noise (`--singlemode contrast --recenter 0 --no-singlebias`). Source-State Calibration is disabled in full: both noise recentering and source-bias projection are omitted. The three matched endpoints and joint response allocation remain active. The profile fixes `conditiongain=1.25`, `singlegain=1.5`, and `innovationgain=2`; the field time is 0.75 and native continuation time is 0.30. One final image update uses five denoiser condition samples in two logical forward calls, one VAE encode, and three VAE decodes. The controller and candidate-search branches are inactive.

## Installation

Experiments use one NVIDIA RTX 4090, FP32 inference, and three concurrent images sharing one set of weights. `cuda:0` denotes the first visible GPU.

The server uses Python 3.12, PyTorch 2.12.1 with CUDA 13.0, torchvision 0.27.1, diffusers 0.35.1, transformers 4.57.1, NumPy 1.26.4, and torchmetrics 1.8.2.

Use a Linux shell for installation. If the checkout already exists, enter its root and skip the first two commands:

```bash
git clone https://github.com/wlz05/RefUnmix.git
cd RefUnmix
python -m venv env
source env/bin/activate
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install diffusers==0.35.1 transformers==4.57.1 numpy==1.26.4 torchmetrics==1.8.2 accelerate safetensors pillow scipy scikit-image huggingface-hub
```

Run subsequent commands from the repository root with this environment active.

Download [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) into `models/`:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/sd-turbo', revision='b261bac6fd2cf515557d5d0707481eafa0485ec2', local_dir='models', allow_patterns=['model_index.json','unet/*','vae/*','scheduler/*','tokenizer/*','text_encoder/*'], ignore_patterns=['*.onnx*','*.fp16.*','*.bin'])"
```

The model directory contains `unet/`, `vae/`, `scheduler/`, `tokenizer/`, and `text_encoder/`. RefUnmix needs no additional training checkpoint.

## Data and evaluation

Use the original [PIE-Bench](https://github.com/cure-lab/PnPInversion) release with all 700 requests:

```text
data/
  mapping_file.json
  annotation_images/
models/
protocol/
cache/
outputs/
```

PIE-Bench is the final benchmark, covering ten edit categories. RefUnmix is training-free. There is no training split or checkpoint selection on these requests. Development subsets used for diagnostics retain their own identities and are reported separately.

[ChordEdit](https://github.com/ChordEdit/ChordEdit) refers to the PnPInversion data and metric protocol. Download the released calculator:

```bash
python -c "from pathlib import Path; from urllib.request import urlretrieve; Path('protocol').mkdir(exist_ok=True); urlretrieve('https://raw.githubusercontent.com/cure-lab/PnPInversion/07f97f448150e2ca220bebd54c8f687c5c50c67a/evaluation/matrics_calculator.py','protocol/metrics.py')"
```

Official scoring uses CLIP ViT-L/14, squeeze LPIPS, and DINO ViT-B/8. The first scoring run with `--network` populates `cache/metrics/`. Subsequent runs load it offline. Move the code, model components, data, and cache together while retaining their relative layout.

## Main experiment

The runner generates and evaluates seeds **42, 52, 3047, 3407**, in that order:

```bash
python -B conduct.py --network
```

With populated metric caches:

```bash
python -B conduct.py
```

`--batch 3` is the default. Four concurrent images use the same GPU:

```bash
python -B conduct.py --batch 4 --output outputs/four
```

Each seed writes its own images, inference summary, and metric report under `outputs/main/SEED/`. The final four-seed result is `outputs/main/mean.json`. Completed seeds are retained. An unfinished output directory requires a fresh destination. Saved recipes bind every seed to the same code and generation settings.

For the server layout with source in `RefUnmix/` and data, models, and caches in the surrounding workspace, run from that workspace:

```bash
env/bin/python -B RefUnmix/conduct.py --protocol protocol/official/metrics.py
```

Run the stages separately:

```bash
python -B conduct.py --action generate
python -B conduct.py --action score --network
python -B conduct.py --action aggregate
```

Individual seeds can also be completed in sequence:

```bash
python -B conduct.py --seeds 42 --network
python -B conduct.py --seeds 52
python -B conduct.py --seeds 3047
python -B conduct.py --seeds 3407
python -B conduct.py --action aggregate
```

## Ablations

Each command runs all 700 requests with the same four seeds, evaluator, precision, and batch size.

| Profile | Change from the main profile |
|---|---|
| `calibration` | Enable full Source-State Calibration: noise recentering at gain 0.5 and local source-bias projection |
| `bias` | Enable only local source-bias projection; keep noise recentering disabled |
| `locality` | Enable only global source-bias projection; compare with `bias` to isolate geometry |
| `amplification` | Set the innovation cap to one |
| `condition` | Remove conditional extrapolation; retain field gain 1.5 |

```bash
python -B conduct.py --profile calibration
python -B conduct.py --profile bias
python -B conduct.py --profile locality
python -B conduct.py --profile amplification
python -B conduct.py --profile condition
```

The optional `calibration` control uses `--singlemode recenter --recenter .5 --singlebias`, restoring the complete Source-State Calibration module. It uses six denoiser condition samples in three logical forward calls. The `bias` and `locality` controls isolate source-bias projection without noise recentering. The `amplification` and `condition` profiles retain the main profile's disabled calibration module.

Each profile writes separate seed reports and its own `mean.json`. Evaluation follows image generation. The masks and metric models never choose the generated output.

## Metrics and aggregation

For each quality metric, first average all valid requests within each seed, then average the four seed results with equal weight:

```math
\bar m_s=\frac{1}{|\mathcal I_s|}\sum_{i\in\mathcal I_s}m_{i,s},\qquad
\bar m=\frac{1}{4}\sum_{s\in\{42,52,3047,3407\}}\bar m_s.
```

| Metric | Measurement | Table scale |
|---|---|---|
| PSNR | Source fidelity outside the edit support; larger is better | dB |
| MSE | Squared pixel error outside the edit support; smaller is better | x1000 |
| SSIM | Structural similarity outside the edit support; larger is better | x100 |
| LPIPS | Perceptual difference outside the edit support; smaller is better | x1000 |
| CLIP-Whole | Whole-image similarity to the target caption; larger is better | Original score |
| CLIP-Edited | Target-caption similarity with non-edit pixels zeroed; larger is better | Original score |
| Time | Complete editing latency on one RTX 4090; smaller is better | Seconds/image |

Background metrics include requests with nonempty unedited support; CLIP metrics cover the complete request set. Fully edited masks have undefined background scores. Exact matches can have infinite PSNR; these retain an explicit status. Average per-request PSNR directly, rather than deriving it from mean MSE.

`aggregate.py` recomputes the per-seed tables from the individual rows. It requires all requested seeds, 700 distinct request identities per seed, matching generation settings, and the pinned calculator. Missing values are never substituted with zero.

```bash
python -B aggregate.py \
  --input outputs/main/42/metrics.json outputs/main/52/metrics.json \
          outputs/main/3047/metrics.json outputs/main/3407/metrics.json \
  --seeds 42 52 3047 3407
```

Time is measured separately with batch 1 on an otherwise idle GPU. After five warm-up requests, measure the same 50 fixed requests, five per edit category, for each seed. Synchronize CUDA before and after each request. Timing includes input loading, the complete editing algorithm, internal selection when used, and conversion to the final RGB image. It excludes checkpoint loading, output writes, and report-only evaluation. Average the 50 request times within each seed, then average the four seed means with equal weight.

Avg. Rank averages ranks across PSNR, MSE, SSIM, LPIPS, CLIP-Whole, CLIP-Edited, and Time. Rank each metric in its stated direction after rounding to the reported precision; ties receive average ranks. Lower Avg. Rank is better. T-Free denotes training-free editing and I-Free denotes inversion-free editing. Comparisons separate multi-step, few-step, and single-update methods, and record condition evaluations separately from image-update counts.

The final comparison uses the four-seed means produced by the commands above. Baselines retain their official algorithm parameters. The three RefUnmix gains apply to its own field and response allocation.

## Related methods

[ChordEdit](https://github.com/ChordEdit/ChordEdit), [WhereEdit](https://github.com/Xi-Mu-Yu/WhereEdit), [RRLS](https://github.com/yhz5613813/RRLS), [SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit), [FlowEdit](https://github.com/fallenshock/FlowEdit), [InfEdit](https://github.com/sled-group/InfEdit), [InstantEdit](https://github.com/Supercomputing-System-AI-Lab/InstantEdit), [TurboEdit](https://github.com/GiilDe/turbo-edit), [MasaCtrl](https://github.com/TencentARC/MasaCtrl), and [PnP](https://github.com/MichalGeyer/plug-and-play).
