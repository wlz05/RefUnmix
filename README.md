# RefUnmix: Separating Prompt Response from Shared Transport in One-Step Image Editing

Even with the source prompt held fixed, applying an edit field can move the generator's output. Switching prompts at that transported anchor adds a second displacement. RefUnmix measures both responses, assigns them separate coefficients and writes the edit back to the source image in one update.

## Method

**Field construction.** Encode the source image $X$ as $z_0$. Evaluate the source prompt and the extrapolated target prompt at the same anchor, noise and field time. Scale the paired prediction difference by the field gain to obtain $F$ and the transported anchor $z_1=z_0+F$. With epsilon prediction, subtracting the clean-latent estimates cancels the explicit shared noisy input. The prompt difference still depends on where the denoiser is queried. The main profile uses the original sampled noise. The calibration control adds a source-only query, shifts the noise according to the source reconstruction residual and attenuates field components aligned with the local residual.

The field uses an extrapolated target embedding only during construction:

```math
c_f=c_s+\gamma_c(c_t-c_s),\qquad
F=\gamma_f\big[P_{t_f}(z_0,c_f;\epsilon)-P_{t_f}(z_0,c_s;\epsilon)\big],\qquad
z_1=z_0+F.
```

**Matched endpoint decomposition.** Write $c_0=c_s$ and $c_1=c_t$. The native clean-latent prediction $P$ uses the original prompts at endpoint time $t_e$. Decode each endpoint with $V$ before differencing. The decoded RGB endpoints remain unclipped during response measurement.

```math
h_{(i,j)}=P_{t_e}(z_i,c_j;\epsilon),\qquad
R_{(i,j)}=V(h_{(i,j)}),\qquad
(i,j)\in\{(0,0),(1,0),(1,1)\}.
```

```math
R_{(1,1)}-R_{(0,0)}
=\underbrace{R_{(1,1)}-R_{(1,0)}}_{d\ \text{(prompt response)}}
+\underbrace{R_{(1,0)}-R_{(0,0)}}_{a\ \text{(shared transport)}}.
```

The middle endpoint holds the transported anchor fixed when measuring $d$ and the source prompt fixed when measuring $a$. The latent prompt difference gives a spatial weight $w$. All three endpoints share one noise realization. The source image anchors the final edit. The generated zero-field endpoint anchors the transport measurement, so the offset between the source image and its zero-field prediction stays out of $a$.

**Joint assignment.** For each RGB pixel, the interaction $C=\langle d,a\rangle$ determines how retaining $a$ affects the source error. Aligned responses add to the squared displacement, while opposed responses can partially cancel. Set $A=\|a\|^2$ and minimize the following objective over $0\le s\le1$ with $\eta=1$.

```math
J_\eta(s)=(1-w)\|\eta d+s a\|^2+\lambda A(s-w)^2.
```

For $A>0$, interval projection gives the shared coefficient directly:

```math
s_0=\Pi_{[0,1]}\!\left(\frac{\lambda wA-(1-w)C}{(\lambda+1-w)A}\right).
```

Use $w$ when the shared response degenerates. Clip and quantize both the candidate and the fixed-weight proposal $X+d+wa$. Keep $s_0$ if its per-pixel RGB squared source error does not increase; otherwise keep $w$. Call the accepted coefficient $q$ and the unamplified output $Y_0=Q(X+d+qa)$, where $Q$ clips and quantizes to normalized 8-bit RGB.

**Budgeted innovation enhancement.** RefUnmix chooses the largest feasible gain up to its cap and minimizes $J_\eta(s)$ over the remaining feasible interval.

```math
1\le\eta\le\eta_{\max},\qquad 0\le s\le1,\qquad
\|\eta d+s a\|^2\le\|d+qa\|^2,\qquad
\widetilde Y=X+\eta d+s a.
```

The continuous proposal is clipped and quantized. If rounding exceeds the unamplified output's pixel budget, the integer RGB residual is rounded inward. Let $\mathcal E(U;X)$ denote a pixel's quantized RGB squared source error. The written image satisfies

```math
\mathcal E(Y;X)\le\mathcal E(Y_0;X)\le\mathcal E(X+d+wa;X).
```

This bound concerns squared source error for the same endpoints. The six quality metrics are measured independently after generation.

## Installation

Experiments use one NVIDIA RTX 4090, FP32 inference, and three concurrent images sharing one set of weights. `cuda:0` denotes the first visible GPU.

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

RefUnmix improves all six quality metrics over ChordEdit, RRLS and WhereEdit on the same single-update SD-Turbo backbone. ChordEdit and WhereEdit are faster per image. On PIE-Bench, RefUnmix ranks first on background SSIM and CLIP-Whole, second on CLIP-Edited, and first on the average rank over six quality metrics and runtime.

`conduct.py` fixes the paper's main configuration and generates and evaluates seeds **42, 52, 3047, 3407**, in that order. Its call chain is:

| Stage | Implementation |
|---|---|
| Fixed main settings and seed order | `conduct.py::generation()` |
| Request loading and output records | `evaluate.py::request()` and `run.py::run()` |
| Paired field and three matched endpoints | `engine/single.py::edit()` |
| Native clean prediction and unclipped RGB decoding | `engine/session.py::Editor.predict()` and `Session.decode()` |
| Shared coefficient and quantized acceptance | `engine/responsibility.py::resolve()` |
| Feasible prompt gain and integer pixel budget | `engine/innovation.py::amplify()` |
| Official quality scoring and seed means | `assess.py` and `aggregate.py` |

In these functions, `conditional`, `shared`, `responsibility`, and `penalty` denote $d$, $a$, $w$, and $\lambda$. `conditiongain`, `singlegain`, and `innovationgain` set $\gamma_c$, $\gamma_f$, and $\eta_{\max}$. Use `conduct.py` to reproduce the main profile; `run.py` also exposes the other experimental configurations.

```bash
python -B RefUnmix/conduct.py --network
```

With populated metric caches:

```bash
python -B RefUnmix/conduct.py
```

`--batch 3` is the default. Four concurrent images use the same GPU:

```bash
python -B RefUnmix/conduct.py --batch 4 --output outputs/four
```

Each seed writes its own images, inference summary, and metric report under `outputs/main/SEED/`. The four-seed quality result is `outputs/main/mean.json`. Completed seeds are retained. An unfinished output directory requires a fresh destination. Saved recipes bind every seed to the same code and generation settings.

For the server layout with source in `RefUnmix/` and data, models, and caches in the surrounding workspace, run from that workspace:

```bash
env/bin/python -B RefUnmix/conduct.py --protocol protocol/official/metrics.py
```

Run the stages separately:

```bash
python -B RefUnmix/conduct.py --action generate
python -B RefUnmix/conduct.py --action score --network
python -B RefUnmix/conduct.py --action aggregate
```

Individual seeds can also be completed in sequence:

```bash
python -B RefUnmix/conduct.py --seeds 42 --network
python -B RefUnmix/conduct.py --seeds 52
python -B RefUnmix/conduct.py --seeds 3047
python -B RefUnmix/conduct.py --seeds 3407
python -B RefUnmix/conduct.py --action aggregate
```

## Ablations

The core controls retain the main profile's uncalibrated paired field. Each profile changes the stated response measurement or output rule and writes separate predictions and metric reports.

| Paper row | Profile | Output rule |
|---|---|---|
| RefUnmix | `main` | Solve shared adoption, then enhance the prompt response within its pixel budget |
| w/o Joint Assignment | `assignment` | $Q(X+d+wa)$; retain fixed spatial adoption and unit prompt gain |
| w/o Zero-Field Reference | `reference` | Skip the source-zero prediction and decode, replace $R_{(0,0)}$ by $X$, and solve allocation with $a'=R_{(1,0)}-X$ |
| w/o Innovation Enhancement | `amplification` | $Q(X+d+qa)$; retain joint adoption with unit prompt gain |
| Unified Gating | `gating` | $Q(X+w(d+a))$; apply the same spatial gate to both responses |
| Target-Endpoint Only | `target` | $Q(R_{(1,1)})$; construct the field and decode only the target endpoint |

Target-Endpoint Only is a constructed baseline that removes the decomposition and downstream allocation together. The other rows retain their declared endpoint and allocation mechanisms. Source-State Calibration is disabled in the main profile; it is not an additional removal row.

```bash
python -B RefUnmix/conduct.py --profile assignment
python -B RefUnmix/conduct.py --profile reference
python -B RefUnmix/conduct.py --profile amplification
python -B RefUnmix/conduct.py --profile gating
python -B RefUnmix/conduct.py --profile target
```

Auxiliary field controls are `calibration` (restore noise recentering and local source-bias projection), `bias` (local projection only), `locality` (global projection only), and `condition` (remove conditional extrapolation). The calibration control adds the module back to the main configuration. Each profile uses the same seed order, requests and official evaluator. Masks and metric scores do not select generated outputs.

`ash
python -B RefUnmix/conduct.py --profile calibration
python -B RefUnmix/conduct.py --profile bias
python -B RefUnmix/conduct.py --profile locality
python -B RefUnmix/conduct.py --profile condition
`

## T2I backbones

RefUnmix uses SD-Turbo, SwiftBrush-v2 and InstaFlow through their native prediction interfaces. Model roots contain `unet/`, `vae/`, `scheduler/`, `tokenizer/` and `text_encoder/`. SwiftBrush-v2 uses its generator UNet and matching SD2.1 components. The InstaFlow root uses the one-step `XCLiu/instaflow_0_9B_from_sd_1_5` checkpoint.

| Backbone | Option | Native network output | Clean-latent recovery |
|---|---|---|---|
| [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) | `sdturbo` | Predicted noise | VP recovery |
| [SwiftBrush-v2](https://github.com/VinAIResearch/SwiftBrushV2) | `swiftbrush` | Predicted noise | VP recovery |
| [InstaFlow](https://github.com/gnobitab/InstaFlow) | `instaflow` | Noise-to-data velocity | Linear-bridge recovery |

For SD-Turbo and SwiftBrush-v2, write $alpha_k=sqrt{aralpha_k}$ and $sigma_k=sqrt{1-aralpha_k}$ from the checkpoint schedule:

```math
x_k=alpha_k z+sigma_kepsilon,qquad
P_t(z,c;epsilon)=rac{x_k-sigma_kwidehatepsilon_	heta(x_k,k,c)}{alpha_k},qquad
k=operatorname{round}((T-1)t).
```

For InstaFlow, the linear bridge and positive noise-to-data velocity give

```math
x_sigma=(1-sigma)z+sigmaepsilon,qquad
P_sigma(z,c;epsilon)=x_sigma+sigma v_	heta(x_sigma,1000sigma,c).
```

At $sigma=1$, the clean prediction is the native one-step update $epsilon+v_	heta(epsilon,1000,c)$. The ODE starts at time zero with UNet clock 1000. InstaFlow uses batch 1 with calibration disabled.

### Naive control

The table row is named **Naive**. Native denotes the backbone prediction interface $P$, not a separate editing method. Naive uses the original image and prompts, an instantaneous paired field, and one target refinement:

```math
F_{mathrm{naive}}=P_{t_f}(z_0,c_t;epsilon)-P_{t_f}(z_0,c_s;epsilon),qquad
z_1=z_0+F_{mathrm{naive}},qquad
Y_{mathrm{naive}}=Q!left(V!left(P_{t_e}(z_1,c_t;epsilon)
ight)
ight).
```

Its fixed settings are field time 0.90, time window $delta=0$, endpoint time 0.30, unit field scale, one shared noise sample, one image update and one target refinement without CFG. This follows the instantaneous-field control used by ChordEdit. RefUnmix's three gains apply to the main profile. Naive uses original prompt embeddings and no response allocation. It makes three condition evaluations in two logical forwards, one VAE encode and one decode.

```bash
python -B RefUnmix/conduct.py --backbone sdturbo --model models --profile naive --output outputs/turbo
python -B RefUnmix/conduct.py --backbone sdturbo --model models --profile main --output outputs/turbo
python -B RefUnmix/conduct.py --backbone swiftbrush --model models/brush --profile naive --output outputs/brush
python -B RefUnmix/conduct.py --backbone swiftbrush --model models/brush --profile main --output outputs/brush
python -B RefUnmix/conduct.py --backbone instaflow --model models/insta --batch 1 --profile naive --output outputs/insta
python -B RefUnmix/conduct.py --backbone instaflow --model models/insta --batch 1 --profile main --output outputs/insta
```

The default matrix command prints a plan without loading models. Add `--run` to execute the selected cells and official quality scoring:

```bash
python -B RefUnmix/compare.py --methods naive refunmix
python -B RefUnmix/compare.py --methods naive refunmix --run
python -B RefUnmix/compare.py --methods chord rrls where swiftedit --seeds 42 --run
```

Each cell has a distinct output directory, recipe and seed reports. Completed compatible reports are reused after integrity checks. A method requiring backbone-specific trained components remains unavailable when those components have not been released; the matrix records that state and leaves its metrics missing.

## Comparisons

RefUnmix is compared with the following methods:

| Group | Methods |
|---|---|
| Multi-step | [MasaCtrl](https://github.com/TencentARC/MasaCtrl), [PnP](https://github.com/MichalGeyer/plug-and-play), [FlowEdit](https://github.com/fallenshock/FlowEdit) |
| Few-step | [InfEdit](https://github.com/sled-group/InfEdit), [TurboEdit](https://github.com/GiilDe/turbo-edit), [InstantEdit](https://github.com/Supercomputing-System-AI-Lab/InstantEdit) |
| One-step | [SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit), [ChordEdit](https://github.com/ChordEdit/ChordEdit), [RRLS](https://github.com/yhz5613813/RRLS), [WhereEdit](https://github.com/Xi-Mu-Yu/WhereEdit) |

`baselines.py` provides inference, saved-output organization and official scoring entries for these ten methods and RefUnmix. `--action plan` is the default. Select the method with `--method`; source revisions and model identities are recorded in each generation recipe. The adapter implementation is kept in `engine/bench.py`, `engine/editors.py`, `engine/upstream.py` and `engine/flowedit.py`. `engine/comparison.py` retains the custom method registration API. The editing methods keep their own settings, while output geometry, request identities and scoring are checked by the common benchmark interface.

## Experimental settings

The main SD-Turbo profile uses `--singlemode contrast --recenter 0 --no-singlebias`. It sets `conditiongain=1.25`, `singlegain=1.5` and `innovationgain=2`. Field time is 0.75 and endpoint time is 0.30. The first logical forward groups two condition samples and the second groups three. One output update uses one VAE encode and three decodes. FP32 generation uses batch 3 on one RTX 4090. Annotation masks and CLIP scores are used only for evaluation.

The runs use the same 700 PIE-Bench requests and seeds 42, 52, 3047, 3407. Each baseline retains its official settings. All methods use the same evaluation and timing protocol.

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
python -B RefUnmix/aggregate.py \
  --input outputs/main/42/metrics.json outputs/main/52/metrics.json \
          outputs/main/3047/metrics.json outputs/main/3407/metrics.json \
  --seeds 42 52 3047 3407
```

Time is measured separately with batch 1 on an otherwise idle GPU. After five warm-up requests, measure the same 50 fixed requests, five per edit category, for each seed. Synchronize CUDA before and after each request. Timing includes input loading, the complete editing algorithm, internal selection when used, and conversion to the final RGB image. It excludes checkpoint loading, output writes, and report-only evaluation. Average the 50 request times within each seed, then average the four seed means with equal weight.

Avg. Rank averages ranks across PSNR, MSE, SSIM, LPIPS, CLIP-Whole, CLIP-Edited, and Time. Rank each metric in its stated direction after rounding to the reported precision; ties receive average ranks. Lower Avg. Rank is better. T-Free denotes training-free editing and I-Free denotes inversion-free editing. Comparisons separate multi-step, few-step, and single-update methods, and record condition evaluations separately from image-update counts.

The quality commands above produce per-seed reports and their equal-weight four-seed means. The comparison combines these quality means with the separately measured Time means to compute the seven-metric Avg. Rank. Inference summaries also record request durations and dataset throughput; the reported Time uses the dedicated batch-1 protocol above. Baselines retain their official algorithm parameters. The three RefUnmix gains apply to its own field and response allocation.

## Related methods

[ChordEdit](https://github.com/ChordEdit/ChordEdit), [WhereEdit](https://github.com/Xi-Mu-Yu/WhereEdit), [RRLS](https://github.com/yhz5613813/RRLS), [SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit), [FlowEdit](https://github.com/fallenshock/FlowEdit), [InfEdit](https://github.com/sled-group/InfEdit), [InstantEdit](https://github.com/Supercomputing-System-AI-Lab/InstantEdit), [TurboEdit](https://github.com/GiilDe/turbo-edit), [MasaCtrl](https://github.com/TencentARC/MasaCtrl), and [PnP](https://github.com/MichalGeyer/plug-and-play).
