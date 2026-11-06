# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

A target-conditioned prediction mixes the requested change with displacement already induced by the editing field. RefUnmix separates these responses through matched predictions, grounds their shared component in the source image, and jointly allocates them under a per-pixel displacement budget. This permits stronger prompt response without increasing the accepted source-error budget.

## Method

**Paired field.** Let $X$ be the source image, $z_0=E(X)$ its latent, and $\epsilon$ one shared noise realization. The native predictor $P_t$ returns a clean-latent estimate. Source and target field queries use the same anchor and noise:

```math
c_f=c_s+\gamma_c(c_t-c_s),\qquad
S=P_{t_f}(z_0,c_s;\epsilon),\qquad T=P_{t_f}(z_0,c_f;\epsilon),
```

```math
F=\gamma_f(T-S)+\beta(T-z_0),\qquad z_1=z_0+F.
```

The paired difference supplies the editing direction; target attraction retains a controlled part of the target-conditioned displacement. The main profile uses sampled noise without source-state recentering or source-bias projection.

**Matched response measurement.** With $c_0=c_s$ and $c_1=c_t$, evaluate three endpoints at the same endpoint time and noise:

```math
h_{(i,j)}=P_{t_e}(z_i,c_j;\epsilon),\qquad
R_{(i,j)}=V(h_{(i,j)}),\qquad
(i,j)\in\{(0,0),(1,0),(1,1)\}.
```

Native cross-attention of the source-zero and target-field endpoints provides a soft support $m$ for changed prompt tokens. The reference interpolates between the zero-field prediction and the source:

```math
B=(1-m)R_{(0,0)}+mX,\qquad
d=R_{(1,1)}-R_{(1,0)},\qquad a=R_{(1,0)}-B.
```

The prompt response $d$ compares prompts at a fixed transported anchor. Writing the original shared transport as $a_0=R_{(1,0)}-R_{(0,0)}$ gives $a=a_0+m(R_{(0,0)}-X)$: the shared component also absorbs the shift from the generated reference toward the source image. It can carry useful geometry as well as reconstruction drift. The latent prompt response supplies a responsibility map $r$, combined with support as $w=r+m(1-r)$. Attention is read from existing forwards without modifying attention probabilities or adding model queries.

**Joint assignment.** For one RGB pixel, define $A=\|a\|^2$ and $C=\langle d,a\rangle$. Shared adoption depends on the interaction of both responses:

```math
J_\eta(s)=(1-w)\|\eta d+sa\|^2+\lambda A(s-w)^2,\qquad 0\le s\le1.
```

At unit prompt gain, the closed-form candidate is

```math
s_0=\Pi_{[0,1]}\left(
\frac{\lambda wA-(1-w)C}{(\lambda+1-w)A}
\right).
```

Use $w$ when $A$ degenerates. Compare the clipped, quantized candidate with $Q(X+d+wa)$ and accept $s_0$ only where it does not increase squared RGB source error. Let $q$ denote the accepted coefficient and $Y_0=Q(X+d+qa)$. Here $Q$ converts to normalized 8-bit RGB.

**Budgeted innovation.** Increase the prompt response while allowing shared adoption to adjust. Choose the largest feasible $\eta$, then minimize $J_\eta(s)$ over its feasible interval:

```math
1\le\eta\le\eta_{\max},\qquad 0\le s\le1,\qquad
\|\eta d+sa\|^2\le\|d+qa\|^2.
```

Quantize the proposal and round its integer residual inward wherever rounding exceeds the unamplified pixel budget. Finally, retain the fraction $\rho+(1-\rho)m$ of that residual, again rounding inward. Writing $\mathcal E$ for quantized per-pixel squared RGB source error gives

```math
\mathcal E(Y;X)\le\mathcal E(Y_0;X)
\le\mathcal E\!\left(Q(X+d+wa);X\right).
```

The budget uses the source and matched predictions, not benchmark masks or CLIP scores.

## Installation

Experiments use one NVIDIA RTX 4090, FP32 inference, and one image per request. `cuda:0` denotes the first visible GPU.

The server uses Python 3.12, PyTorch 2.12.1 with CUDA 13.0, torchvision 0.27.1, diffusers 0.35.1, transformers 4.57.1, NumPy 1.26.4, and torchmetrics 1.8.2.

Use a Linux shell for installation. If the checkout already exists, enter its root and skip the first two commands:

```bash
git clone https://github.com/wlz05/RefUnmix.git
python -m venv env
source env/bin/activate
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install diffusers==0.35.1 transformers==4.57.1 numpy==1.26.4 torchmetrics==1.8.2 accelerate safetensors pillow scipy scikit-image huggingface-hub
```

Run subsequent commands from the workspace containing the RefUnmix checkout, with this environment active. Keep data, models, protocol, caches and outputs alongside the code directory.

Download [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) into `models/`:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/sd-turbo', revision='b261bac6fd2cf515557d5d0707481eafa0485ec2', local_dir='models', allow_patterns=['model_index.json','unet/*','vae/*','scheduler/*','tokenizer/*','text_encoder/*'], ignore_patterns=['*.onnx*','*.fp16.*','*.bin'])"
```

The model directory contains `unet/`, `vae/`, `scheduler/`, `tokenizer/`, and `text_encoder/`. RefUnmix needs no additional training checkpoint.

## Data and evaluation

Use all 700 requests from the original [PIE-Bench](https://github.com/cure-lab/PnPInversion) release, covering ten editing categories:

```text
data/
  mapping_file.json
  annotation_images/
models/
protocol/
cache/
outputs/
```

Parameter development uses the complete 700-request benchmark with seed 42. The selected RefUnmix configuration is then evaluated with seeds 42, 52, 3047 and 3407, without further parameter changes. Ablations and backbone comparisons use paired seed 42. RefUnmix requires no weight training.

Use the released PIE-Bench calculator:

```bash
python -c "from pathlib import Path; from urllib.request import urlretrieve; Path('protocol').mkdir(exist_ok=True); urlretrieve('https://raw.githubusercontent.com/cure-lab/PnPInversion/07f97f448150e2ca220bebd54c8f687c5c50c67a/evaluation/matrics_calculator.py','protocol/metrics.py')"
```

Official scoring uses CLIP ViT-L/14, squeeze LPIPS, and DINO ViT-B/8. A first scoring run with `--network` populates `cache/metrics/`; subsequent runs load the cache offline. Masks are supplied to evaluation, never to the editing pipeline.

## Main experiment

`conduct.py` binds the fixed main configuration, complete request set, and source hashes.

| Stage | Implementation |
|---|---|
| Main preset and paired controls | `conduct.py::generation()` |
| Request loading and inference records | `evaluate.py::request()` and `run.py::run()` |
| Paired field and matched endpoints | `engine/single.py::edit()` |
| Native latent prediction and RGB decoding | `engine/session.py` and `engine/bridge.py` |
| Read-only prompt support | `engine/attention.py` and `engine/grounding.py` |
| Coupled shared adoption | `engine/responsibility.py::resolve()` |
| Feasible innovation and digital budget | `engine/innovation.py::amplify()` |
| Official scoring and verified aggregation | `assess.py` and `aggregate.py` |

Run from the workspace containing the checkout:

```bash
python -B RefUnmix/conduct.py --seeds 42 52 3047 3407 --network
```

With populated caches, omit `--network`. The default is seed 42, FP32, batch 1. Each profile writes images, request records, `summary.json` and `metrics.json` under `outputs/PROFILE/SEED/`. The four-seed aggregate is `outputs/PROFILE/mean.json`; a seed-42 run writes `mean42.json`. Recipes prevent outputs from different code or settings from being combined.

```bash
python -B RefUnmix/conduct.py --action generate
python -B RefUnmix/conduct.py --action score --network
python -B RefUnmix/conduct.py --action aggregate
```

## Ablations

All core controls use the same seed, requests, field settings and official calculator. They change the specified operation, not the chosen output for each image.

| Paper row | Profile | Intervention |
|---|---|---|
| RefUnmix | `main` | Joint adoption and budgeted innovation |
| w/o Joint Assignment | `assignment` | Fix shared adoption to $w$ in both stages; retain feasible innovation with that coefficient fixed |
| w/o Zero-Field Reference | `reference` | Replace $B$ by $X$ and omit the zero-field RGB decode; retain its prediction for matched attention |
| w/o Innovation Enhancement | `amplification` | Set $\eta_{\max}=1$ and retain joint adoption |
| Unified Gating | `gating` | Use $Q(X+w(d+a))$ before the same final support contraction |
| Target-Endpoint Only | `target` | Decode only $R_{(1,1)}$; omit response allocation and final support contraction |
| w/o Prompt Support | `grounding` | Remove support from reference interpolation, responsibility supply and final contraction |
| w/o Target Attraction | `attraction` | Set $\beta=0$ and retain the remaining pipeline |

Unified Gating is an alternative allocation rule. Target-Endpoint Only is a constructed truncation of the editing chain. The reference control removes one decode, not the attention-producing source-zero query. Source-State Calibration is already disabled in the complete method.

```bash
python -B RefUnmix/conduct.py --profile assignment
python -B RefUnmix/conduct.py --profile reference
python -B RefUnmix/conduct.py --profile amplification
python -B RefUnmix/conduct.py --profile gating
python -B RefUnmix/conduct.py --profile target
python -B RefUnmix/conduct.py --profile grounding
python -B RefUnmix/conduct.py --profile attraction
```

Legacy diagnostic profiles remain available for archived experiments; they are not rows of the current core ablation.

## T2I backbones

The same response allocation operates on SD-Turbo, SwiftBrush-v2 and InstaFlow. A model root contains `unet/`, `vae/`, `scheduler/`, `tokenizer/` and `text_encoder/`. SwiftBrush-v2 uses its generator UNet and matching SD2.1 components. InstaFlow uses the one-step `XCLiu/instaflow_0_9B_from_sd_1_5` checkpoint.

| Backbone | Option | Native prediction |
|---|---|---|
| [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) | `sdturbo` | Noise prediction with VP recovery |
| [SwiftBrush-v2](https://github.com/VinAIResearch/SwiftBrushV2) | `swiftbrush` | Noise prediction with VP recovery |
| [InstaFlow](https://github.com/gnobitab/InstaFlow) | `instaflow` | Noise-to-data velocity with linear-bridge recovery |

For VP predictors, let $N$ be the number of training timesteps, $\alpha_k=\sqrt{\bar\alpha_k}$ and $\sigma_k=\sqrt{1-\bar\alpha_k}$:

```math
x_k=\alpha_k z+\sigma_k\epsilon,\qquad
P_t(z,c;\epsilon)=
\frac{x_k-\sigma_k\widehat\epsilon_\theta(x_k,k,c)}{\alpha_k},
\qquad k=\mathrm{round}((N-1)t).
```

InstaFlow uses its velocity parameterization directly:

```math
x_\sigma=(1-\sigma)z+\sigma\epsilon,\qquad
P_\sigma(z,c;\epsilon)=
x_\sigma+\sigma v_\theta(x_\sigma,1000\sigma,c).
```

At $\sigma=1$, this is the official one-step update $\epsilon+v_\theta(\epsilon,1000,c)$. The generation ODE starts at zero while the UNet receives time 1000. The adapters preserve each predictor's clock, sign and text dimensions.

### Native control

**Native** in the backbone table applies one target-conditioned native prediction directly to the encoded source:

```math
Y_{\mathrm N}=Q\!\left(V\!\left(
P_{t_e}(E(X),c_t;\epsilon)
\right)\right).
```

The endpoint time is 0.30, matching RefUnmix, with the same seed and shared-noise draw. Native uses the original target embedding and the selected backbone's noise or velocity recovery, without constructing an editing field. Each request requires one conditional evaluation, one logical forward, one encode and one decode. The `native` profile is distinct from the retained historical `naive` profile.

```bash
python -B RefUnmix/conduct.py --backbone sdturbo --model models --profile native --output outputs/turbo
python -B RefUnmix/conduct.py --backbone sdturbo --model models --profile main --output outputs/turbo
python -B RefUnmix/conduct.py --backbone swiftbrush --model models/brush --profile native --output outputs/brush
python -B RefUnmix/conduct.py --backbone swiftbrush --model models/brush --profile main --output outputs/brush
python -B RefUnmix/conduct.py --backbone instaflow --model models/insta --profile native --output outputs/insta
python -B RefUnmix/conduct.py --backbone instaflow --model models/insta --profile main --output outputs/insta
```

The method-by-backbone interface prints a plan by default. `--run` executes compatible cells, retaining separate recipes and reports:

```bash
python -B RefUnmix/compare.py --methods native refunmix
python -B RefUnmix/compare.py --methods native refunmix --seeds 42 --run
python -B RefUnmix/compare.py --methods chord rrls where swiftedit --seeds 42 --run
```

Completed reports are reused after verification. Cells without required released backbone-specific components remain unavailable.

## Comparisons

RefUnmix is compared with the following methods:

| Group | Methods |
|---|---|
| Multi-step | [PnP](https://github.com/MichalGeyer/plug-and-play) ([SD 1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4))<br>[MasaCtrl](https://github.com/TencentARC/MasaCtrl) ([SD 1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4))<br>[FlowEdit](https://github.com/fallenshock/FlowEdit) ([SD3-Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)) |
| Few-step | [InfEdit](https://github.com/sled-group/InfEdit) ([LCM-DreamShaper v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7))<br>[TurboEdit](https://github.com/GiilDe/turbo-edit) ([SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo))<br>[InstantEdit](https://github.com/Supercomputing-System-AI-Lab/InstantEdit) ([PeRFlow-SD1.5 DreamShaper](https://huggingface.co/hansyan/perflow-sd15-dreamshaper)) |
| One-step | [SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit) ([SwiftBrush-v2](https://github.com/VinAIResearch/SwiftBrushV2))<br>[ChordEdit](https://github.com/ChordEdit/ChordEdit) ([SD-Turbo](https://huggingface.co/stabilityai/sd-turbo))<br>[RRLS](https://github.com/yhz5613813/RRLS) ([SD-Turbo](https://huggingface.co/stabilityai/sd-turbo))<br>[WhereEdit](https://github.com/Xi-Mu-Yu/WhereEdit) ([SD-Turbo](https://huggingface.co/stabilityai/sd-turbo)) |

## Experimental settings

The main preset uses `conditiongain=1.25`, `singlegain=1`, `innovationgain=3`, `targetpull=0.15` and `retainedcontext=0.10`. Field and endpoint times are 0.75 and 0.30. The responsibility exponent is 0.5 and the shared-adoption penalty is 1. Source-state calibration is off: `--singlemode contrast --recenter 0 --no-singlebias`.

Prompt support reads source-zero and target-field cross-attention groups at resolutions up to 1024 spatial queries. Token matching includes insertions, deletions and two following context tokens. Support uses 0.1/0.9 quantiles, a 0.25/0.75 transition, smoothstep and bilinear resizing. No attention weights are edited.

The main pipeline performs five conditional evaluations in two logical forwards, one VAE encode and three decodes per image. This is one image update, not one network evaluation. Generation uses FP32 and batch 1 on one RTX 4090. Main-method replication uses four seeds; paired ablations and backbone comparisons use seed 42.

## Metrics and aggregation

| Metric | Measurement | Table scale |
|---|---|---|
| PSNR | Background pixel fidelity; higher is better | dB |
| MSE | Background squared error; lower is better | x1000 |
| SSIM | Background structural similarity; higher is better | x100 |
| LPIPS | Background perceptual distance; lower is better | x1000 |
| CLIP-Whole | Whole-image target-caption similarity; higher is better | Original score |
| CLIP-Edited | Target-caption similarity with non-edit pixels zeroed; higher is better | Original score |
| Time | Complete editing latency; lower is better | Seconds/image |

Use the original calculator's valid support for each metric. Background metrics exclude requests without background pixels; CLIP covers all requests. The reported mean excludes nonfinite values separately for each metric and averages the remaining finite values. An infinite PSNR does not remove that request's valid MSE, SSIM, LPIPS or CLIP scores. Preserve raw calculator outputs and record finite and excluded counts. Average per-request PSNR directly.

```math
\bar m_s=\frac{1}{|\mathcal I_s|}
\sum_{i\in\mathcal I_s}m_{i,s}.
```

The four-seed main result is $\frac{1}{4}\sum_{s\in\{42,52,3047,3407\}}\bar m_s$. Paired seed-42 comparisons use $\bar m_{42}$. Each table identifies its seed scope; a four-seed RefUnmix result is not substituted into a seed-42 comparison. The aggregator verifies request identities, settings, calculator provenance and completeness.

```bash
python -B RefUnmix/aggregate.py --input outputs/main/42/metrics.json outputs/main/52/metrics.json outputs/main/3047/metrics.json outputs/main/3407/metrics.json --seeds 42 52 3047 3407
```

Time is measured separately on an otherwise idle RTX 4090 with batch 1. After five warm-up requests, measure 50 fixed requests, five per edit category, with CUDA synchronization around each request. Include input loading, complete editing and final RGB conversion; exclude checkpoint loading, output writes and report evaluation. Average the 50 durations. Dataset throughput is recorded separately.

```bash
python -B RefUnmix/timing.py --seed 42 --output outputs/timing
python -B RefUnmix/timing.py --seed 52 --output outputs/timing
python -B RefUnmix/timing.py --seed 3047 --output outputs/timing
python -B RefUnmix/timing.py --seed 3407 --output outputs/timing
```

Compute each seed's mean from its 50 timed requests, then average the four seed means equally.

Avg. Rank is the mean rank across six quality metrics and Time. Rank after rounding to the displayed precision, use average ranks for ties, and follow each metric's direction. Lower Avg. Rank is better. T-Free denotes training-free editing and I-Free denotes inversion-free editing.

## Related methods

Methods are grouped by editing steps and ordered by their first public paper release within each group.

### Multi-step

- **PnP** (2022-11-22): [Paper](https://arxiv.org/abs/2211.12572) | [Code](https://github.com/MichalGeyer/plug-and-play)
- **MasaCtrl** (2023-04-17): [Paper](https://arxiv.org/abs/2304.08465) | [Code](https://github.com/TencentARC/MasaCtrl)
- **FlowEdit** (2024-12-11): [Paper](https://arxiv.org/abs/2412.08629) | [Code](https://github.com/fallenshock/FlowEdit)

### Few-step

- **InfEdit** (2023-12-07): [Paper](https://arxiv.org/abs/2312.04965) | [Code](https://github.com/sled-group/InfEdit)
- **TurboEdit** (2024-08-01): [Paper](https://arxiv.org/abs/2408.00735) | [Code](https://github.com/GiilDe/turbo-edit)
- **InstantEdit** (2025-08-08): [Paper](https://arxiv.org/abs/2508.06033) | [Code](https://github.com/Supercomputing-System-AI-Lab/InstantEdit)

### One-step

- **SwiftEdit** (2024-12-05): [Paper](https://arxiv.org/abs/2412.04301) | [Code](https://github.com/Qualcomm-AI-research/SwiftEdit)
- **ChordEdit** (2026-02-22): [Paper](https://arxiv.org/abs/2602.19083) | [Code](https://github.com/ChordEdit/ChordEdit)
- **RRLS** (2026-06-23): [Paper](https://arxiv.org/abs/2606.24844) | [Code](https://github.com/yhz5613813/RRLS)
- **WhereEdit** (2026-07-23): [Paper](https://arxiv.org/abs/2607.20883) | [Code](https://github.com/Xi-Mu-Yu/WhereEdit)
