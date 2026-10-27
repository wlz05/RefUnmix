# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

**RefUnmix** coordinates field and conditional-reference responses under a protected-output budget. It pairs reference motion with a field correction, compensates their protected-content response, and adopts the target change that remains.

RefUnmix uses one-step or few-step native backbones with finite-query control. Reports include all denoiser sample evaluations, forward calls, VAE decodes, latency, and peak memory.

[Quick start](#quick-start) | [Datasets](#datasets) | [Main experiments](#main-experiments-all-700-pie-bench-requests) | [Ablations](#benchmark-ablation-protocol) | [Parallel reproduction](#parallel-reproduction)

---

## Method

### Core Insight

A single editing channel changes target content and protected content together. Two channels can produce opposing protected responses while retaining useful target change. **RefUnmix measures and uses this compensation to edit within a fixed protection budget.**

The field and a matched source/target reference path provide the two interventions. Their decoded responses determine the paired correction and its common adoption step. Native endpoint calibration chooses the initial field within the supplied temporal-contrast family.

[ChordEdit](https://arxiv.org/abs/2602.19083) combines two temporal residuals before optional refinement. [ECT](https://arxiv.org/abs/2610.30827) analytically scales one VP conditional difference. [PreciseEdit](https://arxiv.org/abs/2610.23535) regulates directional and spatial release. RefUnmix fits joint output responses and generates compensated field-reference proposals. The temporal ablation compares initialization rules; the release ablation holds the candidate family fixed and compares query strategies.

### The Complete Method Chain

```text
Source image and source/target captions
                    |
       Shared-noise conditional contrasts
                    |
       Requested displacement and subspace
                    |
      Measure native endpoint transmission
                    |
       Invert response to construct F0
                    |
     Matched source/target zero-field endpoints
                    |
       Expose a controlled reference coordinate
                    |
    Fit joint semantic and protected RGB responses
                    |
     Compensate reference motion with field motion
                    |
     Select a common adoption step and query once
                    |
       Verify the actual output and reuse feedback
                    |
     Attainment-first selection -> Edited image
```

**1. Estimate the request.** Let $z_{\mathrm{src}}=E(X)$ and let $H(F,C)$ be the clean latent endpoint after applying field $F$ and the declared native continuation under condition $C$. Source and target predictions share the anchor and noise. Their contrasts at $t$ and $t-\delta$ define a request proxy and its resolved subspace:

$$
d_k=\widehat z_0(z_k,t_k,C_{\mathrm{tgt}})
-\widehat z_0(z_k,t_k,C_{\mathrm{src}}),\qquad
\bar d=\frac{d_1+d_2}{2},\qquad
Q_F=\operatorname{orth}[d_1,d_2],\quad c_0=Q_F^\top\bar d.
$$

The default times are 0.90 and 0.75. They provide paired conditional measurements; Haar frequency coordinates enter after field construction.

**2. Invert endpoint transmission.** Probe each resolved direction through the same continuation:

$$
r_k=\frac{H(\epsilon_F q_k,C_{\mathrm{tgt}})
-H(0,C_{\mathrm{tgt}})}{\epsilon_F},\qquad
R_F=[r_1,\ldots,r_r],\qquad r\le2.
$$

With request scale $\gamma_F$, solve the bounded inverse response problem:

$$
\begin{aligned}
b_F&=\gamma_F\bar d-R_Fc_0,\\
\delta_F^\star&=\arg\min_{\|\delta\|_2\le\Delta_F}
\left\{\|R_F\delta-b_F\|_2^2+\lambda_F\|\delta\|_2^2\right\},\\
F_0&=Q_F(c_0+\delta_F^\star).
\end{aligned}
$$

The full response Gram matrix preserves interactions between the conditional directions. The regularizer and correction radius control inverse amplification. The native output $Y_{\mathrm{base}}=V(H(F_0,C_{\mathrm{tgt}}))$ defines the per-field control reference.

$F_0$ remains in the span of the supplied temporal contrasts. With two independent contrasts it can also be written as $\beta_1d_1+\beta_2d_2$. Response measurements determine these coefficients. `field.temporal` records the signed weights, nonnegative-mixture fit, and input-span error; the temporal suite compares this rule with frozen mixture weights.

The measured response also separates strength from direction. Let $\widetilde R_F$ be its resolved SVD, $P_F$ its column-space projector, $v_F=\gamma_F\bar d$, and $w_F=\widetilde R_Fc_0$. For nonzero $v_F$, compare

$$
E_{\mathrm{line}}=\min_{\beta\in\mathbb R}\frac{\|\beta w_F-v_F\|_2^2}{\|v_F\|_2^2},\qquad
E_{\mathrm{span}}=\min_c\frac{\|\widetilde R_Fc-v_F\|_2^2}{\|v_F\|_2^2},\qquad
G_F=E_{\mathrm{line}}-E_{\mathrm{span}}\ge0.
$$

$G_F$ measures the error reduction available through direction changes beyond any signed scalar adjustment. $E_{\mathrm{span}}$ measures the request component outside the resolved response space. These quantities describe linear capacity before regularization, the correction ball, and nonlinear continuation; actual endpoint fit and final task completion are recorded separately.

**3. Recover the reference degree of freedom.** Query matched endpoints:

$$
h_{0,0}=H(0,C_{\mathrm{src}}),\qquad
h_{0,1}=H(0,C_{\mathrm{tgt}}),\qquad
h_{1,1}=H(F_0,C_{\mathrm{tgt}}).
$$

The target zero-field displacement contains two measured components:

$$
h_{0,1}-z_{\mathrm{src}}
=\underbrace{h_{0,0}-z_{\mathrm{src}}}_{b_{\mathrm{src}}}
+\underbrace{h_{0,1}-h_{0,0}}_{\Delta_{\mathrm{cond}}}.
$$

Subtracting $h_{0,1}$ also subtracts $\Delta_{\mathrm{cond}}$. A native reference path $\mathcal R(\eta)$ with $\mathcal R(0)=h_{0,1}$ and $\mathcal R(1)=h_{0,0}$ makes this condition contribution controllable:

$$
\begin{aligned}
F(\xi)&=\alpha_L\Pi_LF_0+\alpha_H\Pi_HF_0,\qquad
\xi=(\alpha_L,\alpha_H,\eta),\\
\ell(\xi)&=z_{\mathrm{src}}+H(F(\xi),C_{\mathrm{tgt}})-\mathcal R(\eta),\\
Y(\xi)&=V(\ell(\xi)).
\end{aligned}
$$

The two complementary Haar projections parameterize field corrections. Reference motion and field motion are evaluated jointly through the complete output. Embedding and multistep reference paths use their native endpoints.

**4. Compensate damage before choosing adoption.** For nonempty frozen protection weights $W$, define the decoded residual $\Phi_W(Y)=\operatorname{vec}(\sqrt W\odot(Y-X))/\sqrt{\sum W}$ and damage $D_W=\|\Phi_W\|_2^2$. Paid outputs fit local protected and semantic responses:

$$
\Phi_W(Y(\xi_j+u))\approx r_j+A_ju,\qquad
\widehat s(Y(\xi_j+u))\approx\widehat s_j+C_ju.
$$

Partition $A_j=[A_f,a_\eta]$ and $C_j=[C_f,c_\eta]$. For one unit of reference motion, the field compensation is

$$
q_\rho=\arg\min_q
\left\{\|A_fq+a_\eta\|_2^2+\rho\|q\|_{M_f}^2\right\},
\qquad
d_\rho=\begin{bmatrix}q_\rho\\1\end{bmatrix},
\qquad
e_\rho=A_jd_\rho,\quad k_\rho=C_jd_\rho.
$$

Here $e_\rho$ is the remaining protected response and $k_\rho$ is the retained semantic response. A single signed step $\omega$ adopts both changes. Its predicted protection cost is

$$
D_j(\omega)=\|r_j+\omega e_\rho\|_2^2
=D_j(0)+2\omega r_j^\top e_\rho+\omega^2\|e_\rho\|_2^2.
$$

Choose $\omega$ from the control box, trust region, semantic floors, and the native damage budget. The objective maximizes the least normalized editing-head gain with step regularization. When a release proposal is unavailable, the controller uses its joint QCQP, geometry repair, or measured fallback within the same budget.

**Why nonzero protected response can help.** Protection is measured against the source image. For $A_f=(2,0)^\top$, $a_\eta=(1,1)^\top$, $r=(0,-0.1)^\top$, $C_f=1$, and $c_\eta=2$, the correction $q=-1/2$ leaves $e=(0,1)^\top$ and $k=3/2$. Its damage change is $-0.2\omega+\omega^2$, so $0<\omega\le0.2$ respects the current budget. This permits gain 0.3, versus 0.2 from reference motion alone. The full protection matrix is invertible, so strict zero-response projection would allow no motion. Actual outputs verify each finite step.

Solver logs separate `damagecross`, `damagequadratic`, and `repaircredit`. Negative cross response provides repair credit against the current residual; it can offset the positive quadratic cost. `verification` compares this model prediction with the observed damage and target changes. Regional Jacobian and nullspace methods, including [LOCO Edit](https://arxiv.org/html/2409.02374v3), provide related geometric foundations.

**5. Close the loop on actual outputs.** Every visited candidate is decoded and scored under frozen control views. Completed candidates must satisfy the native protection budget and all semantic floors. Rejected outputs still inform response fitting. Initialization, field probes, reference queries, fallback, and enabled priors share the model allowance. Among the actual feasible outputs and $Y_{\mathrm{base}}$, select the highest attained minimum editing gain, then the lowest bounded preservation cost within that attainment band. Postselection PIE metrics and independent task labels score the frozen result.

### What Each Experiment Establishes

| Method claim | Matched comparison |
|--------------|--------------------|
| Response-calibrated initialization selects useful coefficients | Temporal mixtures, direct ECT, and scalar norm-matched alignment |
| Joint responses permit useful compensation or residual repair | Fixed-reference, release, and geometry controls in the same candidate family |
| The query policy improves final task quality per unit cost | Joint QCQP, polling, random search, and independent equal-budget evaluation |

The base field can run independently. The same finite-query controller can operate on instantaneous, average, blended, Chord, ECT, or pullback fields. These runs separate initialization from subsequent control.

---

## Project Structure

```text
RefUnmix/
|-- engine/             # Frozen backends, fields, references, control, and metrics
|   |-- field.py        # Coupled pullback field and matched scalar control
|   |-- session.py      # Native diffusion interface and cached predictions
|   |-- flow.py         # Native FLUX and SD3 velocity interfaces
|   |-- native.py       # One-step or few-step continuation
|   |-- path.py         # Matched endpoints and reference paths
|   |-- poll.py         # Finite-query controller
|   |-- release.py      # Response-aware protected compensation
|   |-- curvature.py    # Curvature monitoring and damping
|   |-- limit.py        # Forward-call and sample-evaluation accounting
|   |-- prior.py        # Optional generation-consistency priors
|   |-- judge.py        # Frozen control scorer
|   |-- corpus.py       # Benchmark conversion
|   |-- format.py       # Image, mask, and orientation normalization
|   `-- reproduce.py    # Seeds, environment snapshots, and hashes
|-- tests/              # Unit tests with synthetic backends
|-- main.py             # Single-image editing entry point
|-- run.py              # Shared CLI and editing dispatch
|-- evaluate.py         # Dataset inference and diagnostic summaries
|-- trial.py            # Benchmark and official-partition ablation matrix
|-- evidence.py         # Matched comparisons and evidence coverage
|-- prepare.py          # Local dataset conversion
|-- data.py             # Record and relative-path validation
|-- dataset.py          # Dataset and annotation interfaces
|-- split.py            # Grouped custom or official-tag partitions
|-- doctor.py           # Dependency, checkpoint, and dataset preflight
|-- assess.py           # External official PIE metric adapter interface
|-- report.py           # Independent task-annotation templates
|-- blind.py            # Anonymized comparison exports
|-- app.py              # Optional local Gradio interface
`-- README.md
```

---

## Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.11 or later |
| Core | [PyTorch](https://pytorch.org/get-started/locally/), NumPy, Pillow, SciPy |
| Frozen models | [diffusers](https://huggingface.co/docs/diffusers/), [transformers](https://huggingface.co/docs/transformers/), safetensors, accelerate |
| Flow text encoders | sentencepiece, protobuf |
| Saved Hugging Face datasets | datasets, pyarrow; optional |
| Local interface | gradio; optional |

### Installation

Install [PyTorch with CUDA support](https://pytorch.org/get-started/locally/), then install the dependencies:

```bash
python -m pip install numpy pillow scipy diffusers transformers accelerate safetensors sentencepiece protobuf
```

For saved Hugging Face datasets and the Gradio interface:

```bash
python -m pip install datasets pyarrow gradio
```

### Hardware and Random Seeds

Experiments use **one NVIDIA GeForce RTX 4090** and seeds **42, 52, 3047, and 3407**. Single-image inference defaults to 42. For dataset experiments, use `--seeds 42 52 3047 3407`.

`cuda:0` means the first GPU visible to the process. A scheduler's `CUDA_VISIBLE_DEVICES` setting remaps physical GPU indices within that process.

Dataset partitions remain fixed across seeds. Dataset inference defaults to `--batch 3`; `--batch 4` runs four independent image-seed requests when memory permits. Requests share one backbone and control model. Compatible VP predictions are combined into shared forwards; Flow velocity calls remain serialized while VAE calls can be combined. Jobs are ordered by source and then seed, so a concurrent group can include several seeds of the same source image.

Use `--batch 1` for standalone latency, strict `--deterministic` runs, attention-hook localization, or external adapters. The main 4090 configuration uses the VP backbone in fp32. Flow backends load their complete checkpoints into GPU memory; request batching does not reduce resident weight memory.

---

## Pretrained Models

Download the backbone and control scorer before running. All checkpoints are loaded from local directories using [Diffusers local loading](https://huggingface.co/docs/diffusers/using-diffusers/loading).

For `--backbone vp`, place a compatible [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) checkpoint in `models/`, including `unet/`, `vae/`, `scheduler/`, `tokenizer/`, and `text_encoder/`. The text encoder can also be stored in `text/`.

Place a [Transformers CLIP model and processor](https://huggingface.co/docs/transformers/model_doc/clip) in `clip/` for controlled editing. The `raw` and `fixed` modes run without CLIP. Evaluation with `--evalclip` requires a separate checkpoint from the control scorer.

For `--backbone flux` or `--backbone sd3`, provide the corresponding pipeline with its transformer, VAE, scheduler, tokenizers, and text encoders. The checkpoint determines the supported continuation schedule.

### Workspace Layout

Keep runtime assets outside the source checkout:

```text
workspace/
|-- RefUnmix/           # Source checkout
|-- models/             # Frozen backbone checkpoint
|-- clip/               # Frozen control checkpoint
|-- cache/metrics/      # Downloaded metric weights and repositories
|-- data/
|   |-- mapping_file.json
|   `-- annotation_images/
|-- protocol/           # External official metric calculator
|   `-- metrics.py
|-- input.png
`-- outputs/            # Generated results and reports
```

**Run inference from `workspace/`.** Paths are relative to this directory; absolute paths and `..` are rejected. Outputs must use a new file or directory name.

---

## Quick Start

### 1. Clone the Source

```bash
git clone https://github.com/wlz05/RefUnmix.git
```

Place `models/`, `clip/`, and `input.png` alongside the checkout.

### 2. Check Local Inputs

```bash
python -B RefUnmix/doctor.py --batch 3 --model models --clip clip --device cuda:0 --precision fp32
```

[doctor.py](doctor.py) checks dependencies and checkpoint structure without loading model weights.

### 3. Edit One Image

```bash
python -B RefUnmix/main.py --image input.png --source "A person wearing a blue shirt." --target "A person wearing a striped shirt." --model models --clip clip --device cuda:0 --precision fp32 --field pullback --method path --prior off --maxsamples 15 --maxcalls 15 --candidates 12 --seed 42 --no-control --no-heldout --output outputs/result.png --report outputs/result.json
```

Provide source and target image captions through `--source` and `--target`. Each run saves the edited image and a JSON report containing controls, inference costs, and checkpoint hashes.

### 4. Compare Field-Only and Fixed-Reference Modes

```bash
python -B RefUnmix/main.py --image input.png --source "A person wearing a blue shirt." --target "A person wearing a striped shirt." --model models --device cuda:0 --precision fp32 --field pullback --method raw --prior off --seed 42 --output outputs/raw.png --report outputs/raw.json
python -B RefUnmix/main.py --image input.png --source "A person wearing a blue shirt." --target "A person wearing a striped shirt." --model models --device cuda:0 --precision fp32 --field pullback --method fixed --axes none --eta 0 --prior off --seed 42 --output outputs/fixed.png --report outputs/fixed.json
```

These commands use `--prior off`. Optional consistency priors consume additional model queries.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--field` | `pullback` | Field: pullback, chord, instant, average, blend, ect, or zero |
| `--fieldweight` | `0.5` | Second-time weight for blend, in [0, 1] |
| `--fieldshrink` | `0.7` | ECT shrinkage, in [0, 1]; requires epsilon VP and delta zero |
| `--localizationtime` | start time | Shared protection probe time for native modes with baseline localization |
| `--fieldgeometry` | `coupled` | Coupled response fit or scalar norm-matched ablation |
| `--method` | `path` | Primary modes: path, raw, fixed |
| `--start`, `--delta` | `0.90`, `0.15` | Contrast times are start and start minus delta |
| `--end` | `0.30` | Native continuation noise time |
| `--poststeps` | `1` | Steps per native continuation, not total editor cost |
| `--backbone` | `vp` | Native backend: vp, flux, sd3 |
| `--precision` | `fp32` | fp32, fp16, or bf16; CPU requires fp32 |
| `--seed` | `42` | Single-image inference seed |
| `--batch` | `3` | Dataset inference only: 1-4 concurrent requests sharing model weights |
| `--deterministic` | off | Request deterministic PyTorch execution |
| `--maxsamples` | `64` | Maximum denoiser sample evaluations |
| `--maxcalls` | unset | Per-request logical denoiser forward-call cap |
| `--candidates` | `12` | Candidate-control budget; not an NFE count |
| `--prior` | `off` | Optional prior: ddpm, rec, flow, off |
| `--algorithm` | `trust` | Trust-region, poll, or random control search |
| `--geometry` | `output` | Compensation geometry: output, latent, attenuate |
| `--axes` | `all` | Active control coordinates |
| `--release` | on | Main protected-response reference-field compensation |
| `--compensation` | off | Additional rescue probes after unreliable proposals |

See `python -B RefUnmix/main.py --help` for all options. The report records the field used and the completed queries when a budget limit shortens execution.

---

## Datasets

PIE-Bench is the main benchmark. The data interfaces also load MagicBrush, Emu Edit, ImgEdit-Bench, and GEdit-Bench.

| Dataset | Task | Profile |
|---------|------|---------|
| [PIE-Bench](https://github.com/cure-lab/PnPInversion) | Prompt-guided editing; 700 samples across 10 edit categories | `pie` |
| [MagicBrush](https://github.com/OSU-NLP-Group/MagicBrush) | Single-turn and multi-turn instruction-guided editing | `magicbrush` |
| [Emu Edit](https://huggingface.co/datasets/facebook/emu_edit_test_set) | Local, global, background, and attribute editing | `emu` |
| [ImgEdit-Bench](https://github.com/PKU-YuanGroup/ImgEdit) | Instruction adherence, editing quality, and detail preservation | `imgedit` |
| [GEdit-Bench](https://github.com/stepfun-ai/Step1X-Edit/tree/main/GEdit-Bench) | Real-world image editing requests | `gedit` |

### Annotation Format

PIE-Bench commands below use the original release layout: `data/mapping_file.json` and `data/annotation_images/`. Keep its source/target captions, masks, and category folders. Custom datasets can instead use a JSON object or a list of records. Example `custom/mapping.json`:

```json
{
  "sample": {
    "image": "source.png",
    "source": "A person wearing a blue shirt.",
    "target": "A person wearing a striped shirt.",
    "instruction": "Add stripes to the shirt.",
    "group": "source001",
    "benchmark": "pie"
  }
}
```

With `--dataset custom --mapping mapping.json --images images`, `source.png` resolves to `custom/images/source.png`. Assign edits of the same source image to one group. Field names and paths are matched case-insensitively.

`prepare.py` converts local **PIE**, **Emu Edit**, **MagicBrush**, **ImgEdit**, and **GEdit** annotations. For instruction-only records, supply a JSON sidecar with source and target captions. Print an example with:

```bash
python -B RefUnmix/prepare.py --kind emu --template
```

### Reproduction Workflow

#### 1. Use the Released Partitions

The main PIE-Bench evaluation follows [ChordEdit Section 5.1](https://arxiv.org/html/2602.19083v2) and [WhereEdit](https://arxiv.org/html/2607.20883v1): edit all **700 requests across 10 categories at 512 x 512**, then score the generated images. PIE-Bench supplies the evaluation set, and pretrained model weights remain fixed throughout this protocol. The [main commands](#main-experiments-all-700-pie-bench-requests) and [predeclared ablations](#benchmark-ablation-protocol) use the complete released mapping with `--split all`.

| Dataset | Partition used for configuration selection | Final evaluation |
|---------|--------------------------------------------|------------------|
| [PIE-Bench](https://github.com/cure-lab/PnPInversion) | Fixed experiment configuration; no released development partition | All 700 released requests |
| [MagicBrush](https://github.com/OSU-NLP-Group/MagicBrush) | Official development partition | Official test partition; retain session and turn order for single-turn or multi-turn evaluation |
| [Emu Edit](https://huggingface.co/datasets/facebook/emu_edit_test_set) | Official validation partition | Official test partition |
| [ImgEdit-Bench](https://github.com/PKU-YuanGroup/ImgEdit) | Fixed experiment configuration | Released benchmark evaluation set |
| [GEdit-Bench](https://github.com/stepfun-ai/Step1X-Edit/tree/main/GEdit-Bench) | Fixed experiment configuration | Released benchmark evaluation set |

For converted datasets, `prepare.py` retains official partition tags and maps `dev`/`validation` to `val`. `split.py --protocol official` creates the corresponding manifest. Released training records remain `train`; RefUnmix keeps its pretrained weights fixed. Additional dataset interfaces supply inputs for their published evaluation workflows. MagicBrush session evaluation and the official ImgEdit/GEdit judges run through the respective dataset tools.

#### 2. Prepare Data and Caches

Download the [pretrained models](#pretrained-models) into `models/` and `clip/`. For PIE-Bench, place the original mapping and image tree in `data/` as shown in [Annotation Format](#annotation-format); no conversion or feature precomputation is required. Check the complete benchmark before generation:

```bash
python -B RefUnmix/doctor.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --report outputs/benchmarkcheck.json
```

For the complete PIE release, the preflight report has `dataset.total = 700`, `dataset.inspected = 700`, and `dataset.complete = true`.

A process loads one shared backbone and one control scorer. Conditional predictions and native field/reference endpoints are cached in memory within each image-seed request and reused by that request's candidates. Each ablation arm and each validation/test run starts fresh request caches. They are released when the request group finishes; `recipe.json` stores configuration metadata, not cached predictions. Generation writes the selected PNGs, per-request reports, and `summary.json`; official scoring reads these saved outputs. Each run uses a new output directory.

Downloaded metric weights use `cache/metrics/`, with Hugging Face assets under `huggingface/` and PyTorch Hub assets under `torch/`. The scoring process routes inherited cache settings to this relative directory before loading the official calculator. Reuse this cache across runs or copy it with the workspace when moving machines. The generator and control scorer continue to load their local `models/` and `clip/` checkpoints.

#### 3. Frozen Models and Sample Selection

RefUnmix optimizes editing controls at inference time. The UNet, VAE, text encoder, and CLIP weights stay frozen; there are no training epochs or trained-checkpoint selection. `dataset.training(...)` and `dataset.loader(...)` expose official training/development records for downstream experiments; the main method runs directly from the released pretrained checkpoints.

Every PIE main or benchmark ablation arm processes the same complete mapping and seeds **42, 52, 3047, 3407**. Omit `--limit` and `--categories` for the final PIE table. Candidate selection chooses one output per image-seed request under the declared budget; dataset membership remains fixed.

#### 4. Optional Development Validation and Configuration Selection

For development on a dataset with official validation/test partitions, first convert its local export and retain the released partition tags. This Emu Edit example requires `datasets` and reads a local DatasetDict saved with [save_to_disk](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.DatasetDict.save_to_disk), containing both official validation and test partitions:

```bash
python -B RefUnmix/prepare.py --kind emu --input raw/emu --dataset raw --output development
python -B RefUnmix/split.py --dataset development --mapping mapping.json --images images --protocol official --output outputs/development.json
```

The following pair evaluates the complete `core` design on the official validation partition and then replays that design on the official test partition. Run validation first. If selecting one global variant, record its name from the validation aggregate before running test; its final result is `outputs/developmenttest/<variant>/summary.json`. The pretrained checkpoint stays the same:

```bash
python -B RefUnmix/trial.py --batch 3 --dataset development --mapping mapping.json --images images --manifest outputs/development.json --split val --suite core --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/developmentval
python -B RefUnmix/trial.py --batch 3 --dataset development --mapping mapping.json --images images --manifest outputs/development.json --split test --recipe outputs/developmentval/recipe.json --suite core --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/developmenttest
```

The validation run saves `recipe.json` with every declared variant, code and checkpoint identities, seed set, and shared configuration. The test command replays the same matrix. Keep the model paths, mapping, official manifest, code revision, batch width, seeds, and base control options identical between these commands; only the split, output, and recipe argument change. To study different time, field, prior, or budget settings, run a new validation/test pair in a new output tree. The PIE-Bench commands below directly evaluate the fixed main configuration and predeclared variants on all 700 requests.

#### 5. Final Generation and Official Scoring

Run the [main experiments](#main-experiments-all-700-pie-bench-requests) and [benchmark ablations](#benchmark-ablation-protocol), then run [Official PIE Metrics](#official-pie-metrics) on each generated variant summary. This follows the generate-then-score workflow in the [ChordEdit runner](https://github.com/ChordEdit/ChordEdit/blob/main/run_pie_bench.py), the [PIE release](https://github.com/cure-lab/PnPInversion), and the [WhereEdit evaluation guide](https://github.com/Xi-Mu-Yu/WhereEdit/blob/main/eval/README.md). Use the official metric implementation and the same evaluation assets for every method. Annotation masks define scoring regions; the generation commands use automatic localization.

---

## Experiments

RefUnmix achieves state-of-the-art image editing performance on **PIE-Bench**, with improved protected-content preservation and target-text alignment. All experiments run on a **single NVIDIA GeForce RTX 4090** with seeds **42, 52, 3047, and 3407**.

### Baselines

We compare against [ChordEdit](https://github.com/ChordEdit/ChordEdit), [WhereEdit](https://github.com/Xi-Mu-Yu/WhereEdit), [PreciseEdit](https://arxiv.org/abs/2610.23535), [SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit), and [FlowEdit](https://github.com/fallenshock/FlowEdit), covering one-step diffusion editing and inversion-free flow editing.

### Evaluation Metrics

Evaluation follows the PIE-Bench protocol in [ChordEdit](https://arxiv.org/abs/2602.19083). Background preservation and editing semantics are measured separately:

| Metric | Region | Measurement | Better |
|--------|--------|-------------|--------|
| PSNR | Non-edited | Pixel reconstruction fidelity in dB | Higher |
| MSE x 1,000 | Non-edited | Mean squared pixel error | Lower |
| LPIPS x 1,000 | Non-edited | Perceptual distance from the source | Lower |
| CLIP-Whole | Whole image | Alignment with the target caption | Higher |
| CLIP-Edited | Edited region | Local alignment with the target caption | Higher |

Efficiency is reported as denoiser evaluations, forward calls, VAE decodes, runtime in seconds, and peak GPU memory. Field and controller comparisons use matched evaluation budgets.

### Choosing Three or Four Concurrent Requests

Run one dataset command at a time on the single RTX 4090. `--batch` schedules independent image-seed requests inside that process. Use the same width for each compared method and each arm in a suite.

| Configuration | Default | Four-request alternative |
|---------------|---------|--------------------------|
| Dataset preflight and generation | `--batch 3` | `--batch 4` |
| Result prefix | `outputs/` | `outputs/four/` |
| Development recipe | `outputs/developmentval/recipe.json` | `outputs/four/developmentval/recipe.json` |

For the four-request alternative, replace `--batch 3` with `--batch 4` throughout the selected workflow and replace the `outputs/` prefix in its output, recipe, and subsequent assessment input paths with `outputs/four/`. Keep data and model paths unchanged. Start a fresh validation run at that width before its test replay. Sample/call limits and the twelve-candidate cap remain per request; batch width does not change the budget or the four seeds.

### Main Experiments: All 700 PIE-Bench Requests

Run the following commands from `workspace/` in one process on one RTX 4090. The command profile uses FP32, `cuda:0`, and seeds **42 52 3047 3407**. Keep the same backbone, control checkpoint, precision, and input files across compared runs. Current paper experiments use `evaluate.py` and `trial.py`.

| Setting | Full-controller run | Equal-budget run |
|---------|---------------------|------------------|
| Model-sample cap | 64 | 15 |
| Forward-call cap | 64 | 15 |
| Candidate cap | 12 | 12 |
| Native continuation | 1 step | 1 step |
| Main compensation | `release` on | `release` on |
| Extra rescue / prior / held-out probes | off | off |
| Annotation-mask access | off | off |

**Full-controller run, 64-sample cap:**

```bash
python -B RefUnmix/evaluate.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/main
```

**Equal-budget run, 15-sample cap:**

```bash
python -B RefUnmix/evaluate.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 15 --maxcalls 15 --output outputs/matched
```

With the complete release, each command exports **2,800 image-seed outputs from 700 requests**. The inference summary contains 2,800 `rows`; each benchmark suite records `variants.<name>.rows = 2800` for every arm. `--split all` evaluates every mapped request; it does not use a split manifest. The 64- and 15-sample outputs belong to different budget settings. Report their actual sample evaluations, calls, and runtime alongside quality. The budget suite below also covers 10, 12, 20, and 30.

[evaluate.py](evaluate.py) saves images, per-image JSON reports, `summary.cost`, `summary.controller`, and diagnostic scores. Run [assess.py](assess.py) after generation for the official PIE metric table.

### Benchmark Ablation Protocol

Each predeclared suite runs on the same **700 PIE-Bench requests and four seeds** as the main experiment. Run its command once with `--split all`; the suite evaluates every named variant and writes one result directory per variant. The commands below need no split manifest or validation recipe.

Each suite inherits the supplied main configuration and applies its named interventions. `full` retains the main controls; its label is `pullbackfull` in `fields` and `controlled` in `alignment`. Explicit settings such as native steps, reference path, prior, and rescue are preserved in that reference arm.

Check the twelve-arm field plan before loading data or checkpoints:

```bash
python -B RefUnmix/trial.py --batch 3 --suite fields --split all --model models --clip clip --device cuda:0 --precision fp32 --seed 42 --seeds 42 52 3047 3407 --field pullback --method path --release --no-compensation --prior off --poststeps 1 --candidates 12 --maxsamples 64 --maxcalls 64 --dry
```

The printed `design.main` is `pullbackfull`, whose `changes` is empty. Each arm lists its effective settings and changes from the supplied main configuration. Main and variant summaries save `controlsignature`; suite summaries save the complete `design`, seeds, input identities, code hash, and checkpoint metadata. The mechanism suite uses a separate 256-sample allowance for every arm.

| Suite | Intervention | Sample/call cap |
|-------|--------------|----------------|
| `fields` | Four fields with raw, fixed-reference, and full control | 64 |
| `null` | Explicit zero-field continuation, centering, and condition recovery | 64 |
| `alignment` | Coupled and scalar norm-matched field construction | 64 |
| `temporal` | Fixed temporal mixtures and ECT direct/continued/controlled arms | 64 |
| `core` | Guarded field and reference coordinate controls | 64 |
| `release` | Main reference-field compensation versus joint QCQP | 64 |
| `geometry` | Output, latent, and reference-only compensation | 64 |
| `search` | Trust, poll, random search, and curvature monitoring | 64 |
| `attainment` | Attainment-first and protection-first selection | 64 |
| `reference` | Latent, RGB, mixture, random, and conditional representations | 64 |
| `diagnosis` | Unfiltered reference interventions | 64 |
| `goals` | Editing-head objective with unchanged feasibility checks | 64 |
| `leverage` | Additional rescue probes | 64 |
| `prior` | Explicit bounded, hard, and soft DDPM consistency | 64 |
| `guidance` | CFG and prediction-path guidance controls | 64 |
| `solver` | Solver iterations, certificates, and monitoring | 64 |
| `candidates` | Candidate caps of 8 and 24; full uses 12 | 64 |
| `cost` | Reference-only and shorter candidate configurations | 64 |
| `budget` | Sample/call caps of 10, 12, 15, 20, and 30 | 10/12/15/20/30; full reference 64 |
| `mechanism` | Decoder-only and five-step backend diagnostics | 256 |
| `frontier` | First-model local capacity diagnostic | 64 |

The 64-sample controller ablations share the full-controller run's allowance. The `mechanism` suite uses 256 for every arm so five-step continuation can be studied with its complete field construction; record its actual cost as a backend diagnostic. Equal-budget comparisons use the named rows in `budget`. Its `full` row remains the separate 64-sample reference. The `candidates` suite changes candidate limits while keeping the model allowance at 64.

`--release` controls the main compensation operator. `--compensation` controls additional rescue probes and is enabled only by the corresponding `leverage` arm. `--poststeps` counts native continuation steps; candidate count and sample allowance do not change it. `mechanism/multistep` changes only `poststeps` to 5. The `prior` suite explicitly enables its DDPM prior arms even though the shared command starts with `--prior off`.

`core/target` and `core/source` use guarded selection. `diagnosis/target`, `diagnosis/middle`, and `diagnosis/source` export unfiltered fixed-reference interventions. `frontier` measures the fitted local model's capacity; empirical quality-cost frontiers come from the budget runs. Some suites include aliases of `full`; these are reference configurations, not additional algorithms.

#### Fields

Four fields with raw, fixed-reference, and full control. Named outputs: `pullbackraw`, `pullbackfixed`, `pullbackfull`, `chordraw`, `chordfixed`, `chordfull`, `instantraw`, `instantfixed`, `instantfull`, `averageraw`, `averagefixed`, `averagefull`.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite fields --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/fieldstest
```

#### Null

`zero` is an explicit field intervention, independent of the input scale. Named outputs are `native` (target-conditioned zero-field endpoint), `centered` (VAE source reconstruction), `condition` (source anchor plus the target/source zero-field difference), `reference` (adaptive zero-field reference control), `raw` (the supplied main field without control), and `full` (the unchanged main method). Fixed interventions are exported directly. The shared localization contrast remains charged; zero-field arms do not pay pullback probe costs, so actual costs differ under the common cap.

~~~bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite null --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/nulltest
~~~

`centered` is generally a VAE reconstruction rather than the exact input pixels. `condition` retains a conditional effect despite its zero input field. These cases isolate different operations and should not be combined into one no-field label.

#### Alignment

Coupled and scalar norm-matched field construction. Named outputs: `aligned`, `scalar`, `controlled`, `matched`.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite alignment --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/alignmenttest
```

#### Temporal

This VP-only suite compares five fixed temporal mixtures with response-calibrated initialization and the ECT transport in Equation 22. Named mixture outputs are `firstraw/full`, `earlyraw/full`, `meanraw/full`, `lateraw/full`, and `lastraw/full`; their second-time weights are 0, 0.25, 0.5, 0.75, and 1. The benchmark suite reports the complete predefined curve; choosing one global weight belongs to development validation. Each arm is one editor configuration; the suite does not select a different winner for each test request.

`ectdirect` applies ECT transport without continuation or reference control at time 0.75 and shrinkage 0.7. `ectraw` passes that field through the common native continuation; `ectfull` adds RefUnmix control. `chorddirect` exports the internal Chord field without refinement. These labels separate direct transport from transplanted controls; external headline comparisons use the published complete pipelines and aligned preprocessing.

All arms inherit the supplied main budget, seeds, and shared protocol; `full` is the unchanged main configuration. Mixtures use the shared two-time contrast pool, including its localization contrast. Report actual conditioned samples and complete latency for every arm.

Controlled arms freeze the same protection-probe time. `ectfull` uses that time even though its editing field is measured at 0.75, and pays the extra probe when needed. Direct ECT keeps its single temporal support and does not prepare a control head.

~~~bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite temporal --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/temporaltest
~~~

#### Core

Guarded field and reference coordinate controls. Named outputs: `raw`, `target`, `source`, `eta`, `field`, `full`.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite core --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/coretest
```

#### Release

Main reference-field compensation versus joint QCQP. Named outputs: `released`, `joint`, `random`, `field`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite release --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/releasetest
```

#### Geometry

Output, latent, and reference-only compensation. Named outputs: `output`, `latent`, `attenuated`, `joint`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite geometry --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/geometrytest
```

#### Search

Trust, poll, random search, and curvature monitoring. Named outputs: `poll`, `randomsearch`, `unguarded`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite search --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/searchtest
```

#### Attainment

Attainment-first and protection-first selection. Named outputs: `protected`, `attained`, `field`, `random`, `poll`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite attainment --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/attainmenttest
```

#### Reference

Latent, RGB, mixture, random, and conditional representations. Named outputs: `rgb`, `mixture`, `random`, `conditional`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite reference --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/referencetest
```

#### Diagnosis

Unfiltered reference interventions. Named outputs: `raw`, `target`, `middle`, `source`, `mixture`, `rgb`, `random`, `full`.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite diagnosis --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/diagnosistest
```

#### Goals

Editing-head objective with unchanged feasibility checks. Named outputs: `targetonly`, `globalonly`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite goals --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/goalstest
```

#### Leverage

Additional rescue probes. Named outputs: `compensated`, `uncompensated`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite leverage --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/leveragetest
```

#### Prior

Explicit bounded, hard, and soft DDPM consistency. Named outputs: `noprior`, `boundedprior`, `hardprior`, `softprior`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite prior --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/priortest
```

#### Verify Optional Mechanisms

The main method uses `--release --no-compensation --prior off`. `release` implements protected-response reference-field compensation. The `compensation` switch enables an additional recovery probe only when the regular proposal needs a fallback and the fitted response is trusted. `prior` adds separately charged consistency probes. Their effects are measured by the `leverage` and `prior` suites; they are not required by the main configuration.

Compare those suites after generation:

```bash
python -B RefUnmix/audit.py --summary outputs/leveragetest/uncompensated/summary.json --compare outputs/leveragetest/compensated/summary.json --output outputs/rescueaudit.json
python -B RefUnmix/audit.py --summary outputs/priortest/noprior/summary.json --compare outputs/priortest/boundedprior/summary.json --output outputs/prioraudit.json
```

`configuration.changes` checks the intended intervention. `agreement` counts exact matching PNG hashes and matching visited coordinates; equal rounded averages do not establish equal outputs. The per-request `rescue` record separates considered, eligible, proposed, budget-blocked, queried, accepted, and finally selected probes, with skipped reasons. The per-request `prioractivity` records evaluations, positive excess, penalties, hard vetoes, and whether removing the prior changes selection on the same observed pool. Summary coverage retains missing activity records separately.

With attainment-first selection, a finite bounded prior cannot change a unique winner in the attainment band. A prior can therefore be evaluated and assign nonzero penalties without changing the selected image. The same-pool comparison isolates selection; separate on/off runs also measure changed query trajectories and probe cost. Exact matching outputs count as zero output effect for that intervention.

#### Guidance

CFG and prediction-path guidance controls. Named outputs: `cfgone`, `cfgtwo`, `cfgfour`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite guidance --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/guidancetest
```

#### Solver

Solver iterations, certificates, and monitoring. Named outputs: `nocertificate`, `shortsolve`, `unguarded`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite solver --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/solvertest
```

#### Candidates

Candidate caps of 8 and 24; full uses 12. Named outputs: `small`, `large`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite candidates --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/candidatestest
```

#### Cost

Reference-only and shorter candidate configurations. Named outputs: `light`, `short`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite cost --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/costtest
```

#### Budget

Sample/call caps of 10, 12, 15, 20, and 30. Named outputs: `ten`, `twelve`, `fifteen`, `twenty`, `thirty`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite budget --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/budgettest
```

#### Mechanism

Decoder-only and five-step backend diagnostics. Named outputs: `decoder`, `multistep`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite mechanism --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 256 --maxcalls 256 --output outputs/mechanismtest
```

#### Frontier

First-model local capacity diagnostic. Named outputs: `capacity`, `unchecked`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite frontier --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/frontiertest
```

### Reading and Comparing Outputs

Each variant writes `outputs/<suite>test/<variant>/summary.json`. For example, the field comparison reads:

```bash
python -B RefUnmix/evidence.py --reference outputs/fieldstest/chordfull/summary.json --comparison outputs/fieldstest/pullbackfull/summary.json --contrast field --seed 42 --repeats 2000 --output outputs/fieldcomparison.json
python -B RefUnmix/evidence.py --reference outputs/releasetest/joint/summary.json --comparison outputs/releasetest/released/summary.json --contrast outcome --seed 42 --repeats 2000 --output outputs/releasecomparison.json
```

Inspect `fields.actual`, `fields.budgetfallback`, and `summary.controller` together with image quality. Per-image reports record the backend step count, initialization state, actual release/rescue queries and acceptances, semantic bottlenecks, and budget termination. Named switches describe configuration; the query counters show execution. The evidence commands export paired diagnostics; independent task annotations can be attached with `--taskreference` and `--taskcomparison`.

### Endpoint Response Diagnostics

Run matched endpoint diagnostics for the coupled field, its norm-matched scalar control, and the uncorrected average field:

~~~bash
python -B RefUnmix/probe.py --image input.png --source "a blue shirt" --target "a striped shirt" --model models --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --poststeps 1 --prior off --field pullback --fieldgeometry coupled --withfield --etas 0 1 --maxsamples 64 --maxcalls 64 --output outputs/probecoupled
python -B RefUnmix/probe.py --image input.png --source "a blue shirt" --target "a striped shirt" --model models --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --poststeps 1 --prior off --field pullback --fieldgeometry scalar --withfield --etas 0 1 --maxsamples 64 --maxcalls 64 --output outputs/probescalar
python -B RefUnmix/probe.py --image input.png --source "a blue shirt" --target "a striped shirt" --model models --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --poststeps 1 --prior off --field average --fieldgeometry coupled --withfield --etas 0 1 --maxsamples 64 --maxcalls 64 --output outputs/probeaverage
~~~

`field.relativefit` is the response-error norm divided by the request norm; `responsecosine` measures their direction agreement. `relativegap` compares the true response with the measured linear prediction, normalized by the request norm. `alignment` records input angle, norm ratio, orthogonal correction, and the retained singular-value ratio. Undefined directions and zero denominators are null; numerical underflow and overflow have explicit status fields.

`field.capacity` contains `lineerror`, `spanerror`, and `directionroom`: the normalized squared errors $E_{\mathrm{line}}$, $E_{\mathrm{span}}$, and their difference $G_F$. Its `rank` is the resolved output-response rank; `field.rank` is the input-contrast rank. Both fits use the same SVD truncation and the same paid response probes. The scalar fit permits either sign. Zero requests have null normalized errors; zero responses have unit errors and zero direction room. The same diagnostic is saved for coupled and norm-matched scalar arms before their different fields are applied.

These diagnostics use the native field endpoint before reference control. Pullback runs already save them from paid endpoints; evaluation summaries aggregate available values by request before averaging, and `audit.py` includes them in paired comparisons. The probe command adds source-condition endpoints for mechanism analysis and reports its complete cost. Response fit is evaluated separately from final editing quality.

Controller history records `verification` for every queried non-geometry proposal, including rejected proposals: `predictedgain`, `actualgain`, `predicteddamagechange`, and `actualdamagechange`. Each change is measured from the current anchor. Gains use the frozen control heads and scales; independent task annotations evaluate the final image. `audit.py` averages release-query changes under `release:verified...`, then averages seeds within each request for paired comparisons. This connects predicted compensation to its measured finite-step effect.

Native probes use raw mode with priors and controller diagnostics disabled. Discrete reference paths require `--etas 0 1`. Raw inference rejects ignored prior or controller switches, and held-out consistency requires an enabled prior.

### Official PIE Metrics

Save the [PIE-Bench metric calculator](https://github.com/cure-lab/PnPInversion/blob/main/evaluation/matrics_calculator.py) as `protocol/metrics.py` outside the RefUnmix checkout. Keep its `MetricsCalculator` interface and use the same calculator version and metric weights for every method. Install a torchvision build matching PyTorch, then the metric dependencies:

```bash
python -m pip install torchmetrics lpips
```

The official calculator loads CLIP, LPIPS, and DINO assets. Use `--network` to populate or update `cache/metrics`, then use `--offline` to score with the completed cache. These modes are mutually exclusive. The relative cache uses the [Hugging Face cache settings](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) and PyTorch Hub directory, and is shared by the three- and four-request workflows. Generation remains separate from postselection scoring.

```bash
python -B RefUnmix/assess.py --input outputs/main/summary.json --dataset data --mapping mapping_file.json --images annotation_images --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/mainmetrics.json
python -B RefUnmix/assess.py --input outputs/matched/summary.json --dataset data --mapping mapping_file.json --images annotation_images --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/matchedmetrics.json
python -B RefUnmix/assess.py --input outputs/fieldstest/pullbackfull/summary.json --dataset data --mapping mapping_file.json --images annotation_images --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/fieldmetrics.json
```

Score other variants with the same command by changing only `--input` and `--output`. The input is a variant's inference summary, not the suite-level summary. Official scoring requires 512 x 512 source images, outputs, and annotation geometry. Omitting `--protocol` runs pixel-only scoring and leaves perceptual and CLIP metrics unavailable.

Assessments contain `raw`, scaled `table`, and metric-specific `coverage`. MSE, LPIPS, and DINO structure distance use x1,000; SSIM uses x100; PSNR is in dB. Empty protected regions remain undefined for preservation metrics; zero-error PSNR is recorded as positive infinity. The scorer averages seeds within each request key, then averages those request scores. Report each metric's valid denominator, alongside all 700 requests and four seeds.

### Reuse the Metric Cache After Moving

Copy the complete `cache/metrics/` tree with the workspace. Retain Hugging Face repository `refs/`, `snapshots/`, and `blobs/`, plus the PyTorch Hub repository directories and `checkpoints/`. Keep relative symlinks intact or copy their resolved file contents. Absolute snapshot links from the old machine are replaced with local blob copies when the matching blobs are present.

From the new workspace root, run the same Python command with `--offline` and a fresh assessment output:

```bash
python -B RefUnmix/assess.py --input outputs/main/summary.json --dataset data --mapping mapping_file.json --images annotation_images --protocol protocol/metrics.py --device cuda:0 --offline --cache cache/metrics --output outputs/mainoffline.json
```

Offline scoring reads cached assets and blocks network connections during calculator loading and scoring. Missing assets stop the run; populate the cache with `--network` before copying it. The saved assessment records its cache paths and offline mode. Pixel-only scoring needs neither flag and creates no metric cache.

### Independent Task Assessment

Create annotations only after the output images are frozen:

```bash
python -B RefUnmix/report.py --summary outputs/main/summary.json --template --output outputs/mainlabels.json
```

Complete every annotation with `success`, `preserved`, and `artifact` booleans plus a `naturalness` score from 1 to 5, retaining the image hashes. Then aggregate the completed labels:

```bash
python -B RefUnmix/report.py --summary outputs/main/summary.json --annotations outputs/mainlabels.json --seed 42 --repeats 2000 --output outputs/maintasks.json
```

For anonymized matched comparisons, export a pair from the same suite:

```bash
python -B RefUnmix/blind.py --left outputs/releasetest/joint/summary.json --right outputs/releasetest/released/summary.json --seed 42 --output outputs/releaseblind
```

### Tests

Run the regression suite from `workspace/`:

```bash
cd RefUnmix
python -B -m unittest discover -s tests -p '*.py' -q
```

The suite covers field construction, control constraints, query accounting, image formats, dataset conversion, README experiment plans, complete-benchmark ablations, and official validation-to-test replay at batch widths three and four. Temporal diagnostics distinguish numerical underflow from an exact fit.

For the focused parallel checks, run from the checkout root:

```bash
python -B -m unittest discover -s tests -p batch.py -q
```

CUDA availability enables allocator-retention and CUDA random-state tests with lightweight backends. Without CUDA, those two cases are reported as skipped.

---

## Implementation Details

Backbone parameters remain frozen throughout editing. Conditional contrasts at times 0.90 and 0.75 define the default field subspace; `--start` and `--delta` set these times. Endpoint probes fit a coupled inverse response within this subspace. The controller then adjusts the field and reference path using observed outputs.

`--poststeps` sets the steps in each native continuation. `--maxsamples` and `--maxcalls` limit denoiser sample evaluations and forward calls, including optional probes and priors. The candidate budget is set independently by `--candidates`.

The default sample cap is 64; the equal-budget experiments use explicit caps of 10, 12, 15, 20, and 30. Sample evaluations count conditioned branches. Per-request `calls` and `--maxcalls` count logical forwards; `execution.denoisercalls` records physical shared denoiser dispatches across concurrent requests. These counters remain separate from candidate counts and paper-reported NFE. At rank two, one noise draw, and one conditional step per endpoint, contrasts, field probes, the raw output, and the source reference require nine sample evaluations before initialization.

Each candidate reserves all uncached endpoints and its optional prior before generation. When a joint update exceeds the remaining allowance, the controller tries an affordable update that reuses a cached field endpoint or reference and verifies the actual output. Only completed candidates enter final selection. Global edits with an empty protected region continue semantic search with zero protection cost.

Inputs are converted to RGB with EXIF orientation applied consistently to images and masks. Deterministic execution is available through `--deterministic`; reports store the runtime settings and package versions.

Invalid budgets are rejected before checkpoints load. Nonfinite endpoints are rejected before caching, and diagnostic failures restore the previous accounting stage. Reports require finite numbers and string object keys; publication is atomic and never overwrites an existing result.

### Parallel Reproduction

Use the same batch width for the main method and its ablations. Every image-seed request keeps its own noise generator, reference cache, controller, CLIP counters, and sample budget. Four seeds generate four runs per source; batch width sets concurrent requests and does not multiply the per-request allowance.

Group barriers release inference-queue inputs and outputs before the next group starts. Failed groups clear completed worker frames. Prior probability tensors are copied before normalization; random search and random-reference directions use request-local generators.

For dataset inference and ablations, use `--batch 3` or `--batch 4`. The last group may contain fewer requests. Parallel request latency includes queue waits; `execution.throughput` records completed image-seed requests per second over the full dataset loop, including scoring and output writes. `execution.peakvram` is the shared-device peak; per-row VRAM is undefined during concurrent execution. Compare standalone latency with `--batch 1`, and compare throughput at the same batch width.

`execution.denoisersamples` and `execution.denoiserrequests` reconcile with summed per-request samples and logical calls. `execution.denoisercalls` counts shared physical forwards. A mismatch stops summary publication. `execution.accounting` records this check.

`execution.memorytrace` records live `allocated` and allocator `reserved` memory after each completed group, in MiB. `finalvram`, `reservedvram`, and `peakreservedvram` record final live, final reserved, and peak reserved memory. Cached allocator reservations can remain high after live tensors are released; inspect live allocations across groups when checking retention.

CUDA memory errors stop execution without changing precision or budgets. Completed row JSON files and their images remain available; a caught execution error writes `failure.json` and does not publish a complete `summary.json`. Start a new output directory after adjusting batch width. A checkpoint-loading memory error requires freeing GPU memory or choosing a compatible checkpoint or precision before inference.
