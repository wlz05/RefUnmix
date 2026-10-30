# RefUnmix: Protected-Response Compensation for Image Editing

**RefUnmix** constructs an editing field by protecting within a fixed target-response class. A coupled pullback field supplies the anchor; measured endpoint responses identify alternative input directions that retain the proposed target change while reducing protected distortion.

The primary `--method field` profile incorporates response allocation and verification into field construction, then exports its cached native endpoint. It runs no separate reference controller. Frozen one-step or few-step continuation is queried a finite number of times; all model samples, forwards, VAE decodes, control scores, latency, and peak memory remain charged.

[Quick start](#quick-start) | [Datasets](#datasets) | [Main experiments](#main-experiments-all-700-pie-bench-requests) | [Ablations](#benchmark-ablation-protocol) | [Parallel reproduction](#parallel-reproduction)

---

## Method

### Core Insight

**A protected update can be useful when it repairs an existing error.** The decisive quantity is the total output residual, including its cross term with the proposed change. RefUnmix uses this residual when choosing a tangent field direction and when deciding how much of it to adopt. Tangency removes first-order radial scaling; the generated endpoint determines semantic benefit and protection feasibility.

[ChordEdit](https://arxiv.org/abs/2602.19083) stabilizes coarse-step transport, [PreciseEdit](https://arxiv.org/abs/2610.23535) allocates strong transport by directional and spatial reliability, and [LOCO Edit](https://arxiv.org/html/2409.02374v3) uses regional Jacobian projections. RefUnmix allocates an unresolved request in the coupled field's tangent dictionary using complete generated-output responses. The optional reference-control extension is tested separately against the integrated field.

### The Complete Method Chain

```text
Source image and source/target captions
                    |
       Shared-noise conditional contrasts
                    |
       Coupled inverse endpoint response
                    |
       Baseline field and protected output
                    |
       Unresolved request / tangent dictionary
                    |
       Measure only lateral output responses
                    |
       Propose a target response from the residual
                    |
       Repair protection within its response class
                    |
       Choose feasible adoption / verify actual output
                    |
       Export cached endpoint or retain baseline
```

**1. Estimate the request.** Let $`z_{\mathrm{src}}=E(X)`$ and let $`H(F,C)`$ be the clean latent endpoint after applying field $`F`$ and the declared native continuation under condition $`C`$. Source and target predictions share the anchor and noise. Their contrasts at $`t`$ and $`t-\delta`$ define a request proxy and its resolved subspace:

```math
d_k=\widehat z_0(z_k,t_k,C_{\mathrm{tgt}})
-\widehat z_0(z_k,t_k,C_{\mathrm{src}}),\qquad
\bar d=\frac{d_1+d_2}{2},\qquad
Q_F=\mathrm{orth}[d_1,d_2],\quad c_0=Q_F^\top\bar d.
```

The default times are 0.90 and 0.75. They provide paired conditional measurements. The integrated field uses tangent coordinates; the optional path comparison uses Haar amplitude coordinates.

**2. Invert endpoint transmission.** Probe each resolved direction through the same continuation:

```math
r_k=\frac{H(\epsilon_F q_k,C_{\mathrm{tgt}})
-H(0,C_{\mathrm{tgt}})}{\epsilon_F},\qquad
R_F=[r_1,\ldots,r_r],\qquad r\le2.
```

With request scale $`\gamma_F`$, solve the bounded inverse response problem:

```math
\begin{aligned}
b_F&=\gamma_F\bar d-R_Fc_0,\\
\delta_F^\star&=\arg\min_{\lVert \delta\rVert_2\le\Delta_F}
\bigl(\lVert R_F\delta-b_F\rVert_2^2+\lambda_F\lVert \delta\rVert_2^2\bigr),\\
F_0&=Q_F(c_0+\delta_F^\star).
\end{aligned}
```

The full response Gram matrix preserves interactions between the conditional directions. The regularizer and correction radius control inverse amplification. The native output $`Y_{\mathrm{base}}=V(H(F_0,C_{\mathrm{tgt}}))`$ defines the per-field control reference.

$`F_0`$ remains in the span of the supplied temporal contrasts. With two independent contrasts it can also be written as $`\beta_1d_1+\beta_2d_2`$. Response measurements determine these coefficients. `field.initialtemporal` retains the initial fit when tangent allocation runs, while `field.temporal` records the final field's signed weights, nonnegative-mixture fit, and input-span error; the temporal suite compares this rule with frozen mixture weights.

The measured response also separates strength from direction. Let $`\widetilde R_F`$ be its resolved SVD, $`P_F`$ its column-space projector, $`v_F=\gamma_F\bar d`$, and $`w_F=\widetilde R_Fc_0`$. For nonzero $`v_F`$, compare

```math
E_{\mathrm{line}}=\min_{\beta\in\mathbb R}\frac{\lVert \beta w_F-v_F\rVert_2^2}{\lVert v_F\rVert_2^2},\qquad
E_{\mathrm{span}}=\min_c\frac{\lVert \widetilde R_Fc-v_F\rVert_2^2}{\lVert v_F\rVert_2^2},\qquad
G_F=E_{\mathrm{line}}-E_{\mathrm{span}}\ge0.
```

$`G_F`$ measures the error reduction available through direction changes beyond any signed scalar adjustment. $`E_{\mathrm{span}}`$ measures the request component outside the resolved response space. These quantities describe linear capacity before regularization, the correction ball, and nonlinear continuation; actual endpoint fit and final task completion are recorded separately.

**3. Allocate the residual inside the field.** Let $`B_b=\lVert F_b\rVert_2 Q_b`$, $`F_b=B_bc_b`$, and let $`Q_b`$ span the baseline, temporal contrasts, and unresolved latent request. With $`M=I`$, measure responses only along an orthonormal basis of $`c_b^\perp`$. For $`k`$ dictionary directions this needs $`k-1`$ probes, rather than $`k`$. Let $`S`$ denote target/local score changes divided by the frozen desired gain, and $`P`$ the protected RGB residual response. With $`r_b=\Phi_W(Y_b)`$ and $`d_b=\sqrt{D_b}`$ when $`D_b>0`$ (otherwise 1), solve

```math
v_0=\arg\min_{c_b^\top Mv=0,\ \lVert v\rVert_M\le\Delta_b}
\left(\lVert Sv-\mathbf1\rVert_2^2
+\gamma\lVert(r_b+Pv)/d_b\rVert_2^2
+\lambda_b\lVert v\rVert_M^2\right).
```

The conditional profile sets $`v^\star=\mathcal C_{Sv_0}(v_0)`$ by minimizing the regularized protected residual at fixed target response; the joint comparison sets $`v^\star=v_0`$. Details and the shared-response dictionary appear in [Protection at a Fixed Target Response](#protection-at-a-fixed-target-response). The protected term includes the current residual. Its cross term lets a direction repair existing damage; penalizing only $`\lVert Pv\rVert^2`$ misses this opportunity. With $`p=Pv^\star`$, adoption obeys

```math
F_\star=F_b+\alpha B_bv^\star,\qquad
0\le\alpha\le1,\qquad
2\alpha r_b^\top p+\alpha^2\lVert p\rVert_2^2\le0.
```

**4. Verify and export.** Up to `--tangentchecks 3` actual amplitudes are decoded and scored. Every accepted output improves the selected target heads, retains every frozen semantic floor, and satisfies the original baseline's total protection budget. Failure retains the baseline; budget exhaustion after acceptance retains the already verified result. The selected endpoint, pixels, and scores are cached and exported without another model query, decode, or control evaluation. `--method field` performs no reference subtraction or three-coordinate search. `--candidates` belongs to the optional path controller and does not set this profile's query count.

The optional `--tangentnorm` comparison checks the actual endpoint displacement relative to the same target zero-field endpoint within 1%. It intersects the full norm tolerance band with the protection interval, including both connected components when present. Tangency alone does not preserve finite-step input or output norms.

### Optional Reference-Control Extension

`--method path` selects the earlier reference-control formulation below. It remains a comparison, and `--method path --tangent` measures the additional value of reference search after the integrated field. These outputs use a different renderer; no equivalence with native field output is assumed.

**3. Recover the reference degree of freedom.** Query matched endpoints:

```math
h_{0,0}=H(0,C_{\mathrm{src}}),\qquad
h_{0,1}=H(0,C_{\mathrm{tgt}}),\qquad
h_{1,1}=H(F_0,C_{\mathrm{tgt}}).
```

The target zero-field displacement contains two measured components:

```math
h_{0,1}-z_{\mathrm{src}}
=\underbrace{h_{0,0}-z_{\mathrm{src}}}_{b_{\mathrm{src}}}
+\underbrace{h_{0,1}-h_{0,0}}_{\Delta_{\mathrm{cond}}}.
```

Subtracting $`h_{0,1}`$ also subtracts $`\Delta_{\mathrm{cond}}`$. A native reference path $`\mathcal R(\eta)`$ with $`\mathcal R(0)=h_{0,1}`$ and $`\mathcal R(1)=h_{0,0}`$ makes this condition contribution controllable:

```math
\begin{aligned}
F(\xi)&=\alpha_L\Pi_LF_0+\alpha_H\Pi_HF_0,\qquad
\xi=(\alpha_L,\alpha_H,\eta),\\
\ell(\xi)&=z_{\mathrm{src}}+H(F(\xi),C_{\mathrm{tgt}})-\mathcal R(\eta),\\
Y(\xi)&=V(\ell(\xi)).
\end{aligned}
```

The two complementary Haar projections parameterize field corrections. Reference motion and field motion are evaluated jointly through the complete output. Embedding and multistep reference paths use their native endpoints.

**4. Compensate damage before choosing adoption.** For nonempty frozen protection weights $`W`$, define the decoded residual $`\Phi_W(Y)=\mathrm{vec}(\sqrt W\odot(Y-X))/\sqrt{\sum W}`$ and damage $`D_W=\lVert \Phi_W\rVert_2^2`$. Paid outputs fit local protected and semantic responses:

```math
\Phi_W(Y(\xi_j+u))\approx r_j+A_ju,\qquad
\widehat s(Y(\xi_j+u))\approx\widehat s_j+C_ju.
```

Partition $`A_j=[A_f,a_\eta]`$ and $`C_j=[C_f,c_\eta]`$. For a fixed reference adoption $`\omega`$, the field correction minimizes the current protected residual together with the new conditional response:

```math
v_\rho(\omega)=\arg\min_v
\bigl(\lVert r_j+A_fv+\omega a_\eta\rVert_2^2+\rho\lVert v\rVert_{M_f}^2\bigr)
=p_\rho+\omega q_\rho.
```

Both terms use the same measured inverse:

```math
B_\rho=(A_f^\top A_f+\rho M_f)^{-1}A_f^\top,\qquad
p_\rho=-B_\rho r_j,\quad q_\rho=-B_\rho a_\eta.
```

Positive damping gives a unique solution. At zero damping, the implementation uses the resolved minimum-metric-norm solution. Define $`s_\rho=(p_\rho^\top,0)^\top`$ and $`d_\rho=(q_\rho^\top,1)^\top`$. The residual policy compares the affine path $`u(\omega)=s_\rho+\omega d_\rho`$ with the retained incremental path $`u(\omega)=\omega d_\rho`$. The offset consumes the original trust radius. Every path obeys the same control box, semantic floors, and protected-output budget.

For the full protected increment $`w=A_ju(\omega)`$,

```math
D_j(\omega)-D_j(0)=2r_j^\top w+\lVert w\rVert_2^2.
```

The cross term records whether the change repairs or amplifies existing distortion. The controller optimizes the least editing-head gain with step regularization on at most six paths, then generates one selected proposal. The additional paths reuse the fitted responses and require no additional model probes. A generic joint QCQP and measured fallback remain available.

**Why residual repair changes the attainable edit.** Consider $`A_f=(1,0)^\top`$, $`a_\eta=(0,1)^\top`$, $`r=(0.2,-0.1)^\top`$, $`C_f=0`$, and $`c_\eta=1`$. The current damage is 0.05. Incremental cancellation has $`q=0`$, allowing positive reference adoption only up to 0.2. Residual compensation gives $`p=-0.2`$, removes the first error component, and permits adoption up to $`0.1+\sqrt{0.05}\approx0.324`$ under the same damage budget. Both fit within a control radius of 0.5 at the test's interior anchor. This is a fixed linear example, not a measured PIE result.

`--releasepolicy residual` is the reference-control policy. `--releasepolicy increment` reproduces the earlier incremental rule. Logs record the repair `offset`, its target response, the two path-family objective values, and actual output verification. At the same fitted state and constraints, the residual family retains all incremental paths; its best regularized local objective is therefore no worse. Final generated-image improvements are evaluated independently.

**5. Close the loop on actual outputs.** Every visited candidate is decoded and scored under frozen control views. Completed candidates must satisfy the native protection budget and all semantic floors. Rejected outputs still inform response fitting. Initialization, field probes, reference queries, fallback, and enabled priors share the model allowance. Among the actual feasible outputs and $`Y_{\mathrm{base}}`$, select the highest attained minimum editing gain, then the lowest bounded preservation cost within that attainment band. Postselection PIE metrics and independent task labels score the frozen result.

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
|   |-- curve.py        # Fixed-field and matched operating-point comparisons
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
| Official PIE scoring | torchvision matched to PyTorch, torchmetrics, lpips; external calculator below |
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

Experiments use **one NVIDIA GeForce RTX 4090** and seeds **42, 52, 3047, and 3407**. For each seed, we average each metric over all test requests, then report the arithmetic mean of the four seed-level results. Single-image inference defaults to 42. For dataset experiments, use `--seeds 42 52 3047 3407`.

`cuda:0` means the first GPU visible to the process. A scheduler's `CUDA_VISIBLE_DEVICES` setting remaps physical GPU indices within that process.

Dataset partitions remain fixed across seeds. Dataset inference defaults to `--batch 3`; `--batch 4` runs four independent image-seed requests when memory permits. Requests share one backbone and control model. Compatible VP predictions are combined into shared forwards; Flow velocity calls remain serialized while VAE calls can be combined. Jobs are ordered by source and then seed, so a concurrent group can include several seeds of the same source image.

Use `--batch 1` for standalone latency, strict `--deterministic` runs, attention-hook localization, or external adapters. The main 4090 configuration uses the VP backbone in fp32. Flow backends load their complete checkpoints into GPU memory; request batching does not reduce resident weight memory.

---

## Pretrained Models

Download the backbone and control scorer before running. All checkpoints are loaded from local directories using [Diffusers local loading](https://huggingface.co/docs/diffusers/using-diffusers/loading).

For `--backbone vp`, place a compatible [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) checkpoint in `models/`, including `unet/`, `vae/`, `scheduler/`, `tokenizer/`, and `text_encoder/`. The text encoder can also be stored in `text/`.

Place a [Transformers CLIP model and processor](https://huggingface.co/docs/transformers/model_doc/clip) in `clip/` for controlled editing. Standard `raw` and `fixed` modes run without CLIP; `raw --tangent` requires the frozen control scorer. Evaluation with `--evalclip` requires a separate checkpoint from the control scorer.

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

## Move the Workspace to a Server

Copy the workspace tree with the relative directory names and capitalization shown above: `RefUnmix/`, `models/`, `clip/`, `data/`, `protocol/`, and `cache/metrics/`. For a validation-to-test transfer, also copy `development/`, `outputs/development.json`, and the matching `outputs/developmentval/` directory. Copy resolved checkpoint file contents into `models/` and `clip/` when those directories contain links to an external download cache. Retain the exact source commit and unchanged checkpoint, mapping, and manifest file contents. Match the Python package versions recorded in the source run's environment report.

Open a shell at the new workspace root. Its absolute parent directory can change; the commands continue to use the same relative paths. Run the server preflight into a new report:

```bash
python -B RefUnmix/doctor.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --report outputs/servercheck.json
```

For an existing validation recipe, keep its batch width, seed set, precision, model paths, and control settings. The same recipe then replays from the new root. Relative parameter spellings such as `models`, `./models`, and Windows-style separators are normalized. Use the forward-slash commands shown here on Linux. Run the existing test command only; retain its validation recipe and use a fresh test output directory. A new code revision or a change from three to four requests starts a new validation/test pair using the [four-request configuration](#choosing-three-or-four-concurrent-requests).

Use one visible RTX 4090 as `cuda:0` and run one dataset process at a time with `--batch 3` or `--batch 4`. Weights are shared within that process; each image-seed request retains its own generator, cache, controller, and sample/call allowance. Copied completed inference summaries can be scored directly using the [offline cache command](#reuse-the-metric-cache-after-moving). New generation runs use new output directories.

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
python -B RefUnmix/main.py --image input.png --source "A person wearing a blue shirt." --target "A person wearing a striped shirt." --model models --clip clip --device cuda:0 --precision fp32 --field pullback --method field --semantic region --tangentbasis shared --tangentpolicy conditional --prior off --maxsamples 15 --maxcalls 15 --tangentchecks 3 --seed 42 --no-control --no-heldout --output outputs/result.png --report outputs/result.json
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
| `--fieldrepair` | `0` | Verified endpoint residual repair: zero, one, or two rounds |
| `--semantic` | `patches` | Local control views: separate patches or one complete-region goal |
| `--tangent` | off | Enable tangent allocation in raw/path mode; field mode includes it directly |
| `--fieldfactor` | `1` | Multiply a fixed coupled raw field after construction for amplitude diagnostics |
| `--tangentfeedback` | off | Reallocate with prediction errors from paid checks, sharing the original check cap |
| `--tangentgain` | `0.02` | Desired increase of each frozen target control head |
| `--tangentpenalty` | `1` | Relative total protected-residual penalty |
| `--tangentchecks` | `3` | Maximum real-output checks within the shared model budget |
| `--tangentbasis` | `residual` | Temporal/request directions; `shared` adds a matched common-mode candidate |
| `--tangentpolicy` | `joint` | `conditional` repairs protection within the proposed target-response class |
| `--tangentnorm` | off | Accept only actual endpoint-response norm ratios within 1% of baseline |
| `--fieldweight` | `0.5` | Second-time weight for blend, in [0, 1] |
| `--fieldshrink` | `0.7` | ECT shrinkage, in [0, 1]; requires epsilon VP and delta zero |
| `--localizationtime` | start time | Shared protection probe time for native modes with baseline localization |
| `--fieldgeometry` | `coupled` | Coupled response fit or scalar norm-matched ablation |
| `--method` | `path` | Select `field` for the integrated profile; `path` is the optional reference controller |
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
| `--release` | on | Optional path profile: protected-response reference-field compensation |
| `--releasepolicy` | `residual` | Repair the current residual and compensate new reference motion; `increment` retains the earlier rule |
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
python -B RefUnmix/trial.py --batch 3 --dataset development --mapping mapping.json --images images --manifest outputs/development.json --split val --suite core --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/developmentval
python -B RefUnmix/trial.py --batch 3 --dataset development --mapping mapping.json --images images --manifest outputs/development.json --split test --recipe outputs/developmentval/recipe.json --suite core --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/developmenttest
```

The validation run saves `recipe.json` with every declared variant, code and checkpoint identities, seed set, and shared configuration. The test command replays the same matrix. Keep the model paths, mapping, official manifest, code revision, batch width, seeds, and base control options identical between these commands; only the split, output, and recipe argument change. To study different time, field, prior, or budget settings, run a new validation/test pair in a new output tree. The PIE-Bench commands below directly evaluate the fixed main configuration and predeclared variants on all 700 requests.

#### 5. Final Generation and Official Scoring

Run the [main experiments](#main-experiments-all-700-pie-bench-requests) and [benchmark ablations](#benchmark-ablation-protocol), then run [Official PIE Metrics](#official-pie-metrics) on each generated variant summary. This follows the generate-then-score workflow in the [ChordEdit runner](https://github.com/ChordEdit/ChordEdit/blob/main/run_pie_bench.py), the [PIE release](https://github.com/cure-lab/PnPInversion), and the [WhereEdit evaluation guide](https://github.com/Xi-Mu-Yu/WhereEdit/blob/main/eval/README.md). Use the official metric implementation and the same evaluation assets for every method. Annotation masks define scoring regions; the generation commands use automatic localization.

---

## Experiments

The **PIE-Bench** protocol measures protected-content preservation and target-text alignment. The experiment profile uses a **single NVIDIA GeForce RTX 4090** with seeds **42, 52, 3047, and 3407**. For both main and ablation experiments, we compute the mean across test requests separately for each seed, then average the four seed-level results.

### Baselines

We compare against [ChordEdit](https://github.com/ChordEdit/ChordEdit), [WhereEdit](https://github.com/Xi-Mu-Yu/WhereEdit), [PreciseEdit](https://arxiv.org/abs/2610.23535), [SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit), and [FlowEdit](https://github.com/fallenshock/FlowEdit), covering one-step diffusion editing and inversion-free flow editing.

### Evaluation Metrics

Evaluation follows the PIE-Bench protocol in [ChordEdit](https://arxiv.org/abs/2602.19083). Background preservation and editing semantics are measured separately:

| Metric | Region | Measurement | Better |
|--------|--------|-------------|--------|
| PSNR | Non-edited | Pixel reconstruction fidelity in dB | Higher |
| MSE x 1,000 | Non-edited | Mean squared pixel error | Lower |
| LPIPS x 1,000 | Non-edited | Perceptual distance from the source | Lower |
| SSIM x 100 | Non-edited | Structural similarity to the source | Higher |
| DINO Structure Distance x 1,000 | Whole image | Difference between DINO self-similarity structures | Lower |
| CLIP-Whole | Whole image | Alignment with the target caption | Higher |
| CLIP-Edited | Edited region | Local alignment with the target caption | Higher |

PIE applies the non-edited mask to both images before computing preservation metrics. MSE retains the full-canvas denominator, and PSNR uses that MSE. CLIP-Edited scores the full canvas with pixels outside the edit region set to black. The calculator uses CLIP ViT-L/14, SqueezeNet LPIPS, and DINO ViT-B/8. Use the postselection assessment's `raw` or `table` fields for benchmark results. The inference `summary.json` contains diagnostics, including region-normalized errors, which use a different denominator.

Efficiency is reported as denoiser evaluations, forward calls, VAE decodes, runtime in seconds, and peak GPU memory. Field and controller comparisons use matched evaluation budgets.

### Choosing Three or Four Concurrent Requests

Run one dataset command at a time on the single RTX 4090. `--batch` schedules independent image-seed requests inside that process. Use the same width for each compared method and each arm in a suite.

| Configuration | Default | Four-request alternative |
|---------------|---------|--------------------------|
| Dataset preflight and generation | `--batch 3` | `--batch 4` |
| Result prefix | `outputs/` | `outputs/four/` |
| Development recipe | `outputs/developmentval/recipe.json` | `outputs/four/developmentval/recipe.json` |

For the four-request alternative, replace `--batch 3` with `--batch 4` throughout the selected workflow and replace the `outputs/` prefix in its output, recipe, and subsequent assessment input paths with `outputs/four/`. Keep data and model paths unchanged. Start a fresh validation run at that width before its test replay. Sample/call limits, field amplitude checks, and the optional path candidate cap remain per request; batch width does not change the budget or the four seeds.

### Main Experiments: All 700 PIE-Bench Requests

Run the following commands from `workspace/` in one process on one RTX 4090. The command profile uses FP32, `cuda:0`, and seeds **42 52 3047 3407**. Keep the same backbone, control checkpoint, precision, and input files across compared runs. Current paper experiments use `evaluate.py` and `trial.py`.

The primary conditional field uses `--method field --semantic region --tangentbasis shared --tangentpolicy conditional`. Response allocation is enabled by this method; no additional `--tangent` flag is required. Its baseline, desired gain, protection support, and semantic floors stay fixed through verification. The parser retains the earlier `path` default for existing scripts, so select `field` explicitly.

**Conditional field, 64-sample cap:**

```bash
python -B RefUnmix/evaluate.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --fieldrepair 0 --method field --semantic region --tangentgain 0.02 --tangentpenalty 1 --tangentchecks 3 --tangentbasis shared --tangentpolicy conditional --no-tangentnorm --poststeps 1 --guidance 1 --no-cfg --referencepath auto --prior off --localization baseline --no-control --no-heldout --no-compensation --no-frontier --maxsamples 64 --maxcalls 64 --output outputs/main
```

**Conditional field, 15-sample cap:**

```bash
python -B RefUnmix/evaluate.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --fieldrepair 0 --method field --semantic region --tangentgain 0.02 --tangentpenalty 1 --tangentchecks 3 --tangentbasis shared --tangentpolicy conditional --no-tangentnorm --poststeps 1 --guidance 1 --no-cfg --referencepath auto --prior off --localization baseline --no-control --no-heldout --no-compensation --no-frontier --maxsamples 15 --maxcalls 15 --output outputs/matched
```

The main shared dictionary adds one matched source zero-field endpoint and at most one lateral direction. With two temporal directions, one residual, one shared candidate, and three checks, the single-condition one-step profile uses at most 15 sample evaluations. All costs remain per request.

**Static residual/joint field reference, 64-sample cap:**

```bash
python -B RefUnmix/evaluate.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --fieldrepair 0 --method field --semantic region --tangentgain 0.02 --tangentpenalty 1 --tangentchecks 3 --tangentbasis residual --tangentpolicy joint --no-tangentnorm --poststeps 1 --guidance 1 --no-cfg --referencepath auto --prior off --localization baseline --no-control --no-heldout --no-compensation --no-frontier --maxsamples 64 --maxcalls 64 --output outputs/static
```

This earlier profile uses at most 13 sample evaluations under the same one-step assumptions. Historical suites retain their explicit reference settings; `conditional/full` matches the current main configuration. Extra noise samples, CFG, native steps, or localization times change the cost. Start fresh output directories when changing the configuration.

The following commands retain the earlier reference-controller comparison. They do not define the integrated field profile.

| Setting | Full-controller run | Equal-budget run |
|---------|---------------------|------------------|
| Model-sample cap | 64 | 15 |
| Forward-call cap | 64 | 15 |
| Candidate cap | 12 | 12 |
| Native continuation | 1 step | 1 step |
| Main compensation | `release` on | `release` on |
| Extra rescue / prior / held-out probes | off | off |
| Annotation-mask access | off | off |

**Optional reference-controller comparison, 64-sample cap:**

```bash
python -B RefUnmix/evaluate.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/controller
```

**Optional reference-controller comparison, 15-sample cap:**

```bash
python -B RefUnmix/evaluate.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 15 --maxcalls 15 --output outputs/controllermatched
```

With the complete release, each command exports **2,800 image-seed outputs from 700 requests**. The inference summary contains 2,800 `rows`; each benchmark suite records `variants.<name>.rows = 2800` for every arm. `--split all` evaluates every mapped request; it does not use a split manifest. The 64- and 15-sample outputs belong to different budget settings. Report their actual sample evaluations, calls, and runtime alongside quality. The budget suite below also covers 10, 12, 20, and 30.

[evaluate.py](evaluate.py) saves images, per-image JSON reports, `summary.cost`, `summary.controller`, and diagnostic scores. Run [assess.py](assess.py) after generation for the official PIE metric table.

### Benchmark Ablation Protocol

Each predeclared suite runs on the same **700 PIE-Bench requests and four seeds** as the main experiment. Run its command once with `--split all`; the suite evaluates every named variant and writes one result directory per variant. The commands below need no split manifest or validation recipe.

For every variant and metric, first compute one mean over all test requests for each seed, then average the four seed means with equal weight. Each variant's own `summary.json` and the suite's `variants[arm]` entry retain the same `byseed`, `summary`, `seeds`, and `aggregation` fields. A complete four-seed benchmark contains 700 request scores per seed and 2,800 rows per variant. Missing or duplicate request-seed rows stop publication of a complete inference summary.

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
| `semantic` | Region-level goals and current-state endpoint repair | 64 |
| `residual` | Incremental versus residual compensation with the same responses and query caps | 64 |
| `tangent` | Coupled raw anchor, tangent allocation, norm matching, and temporal-only dictionary | 64 |
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
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite fields --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/fieldstest
```

#### Null

`zero` is an explicit field intervention, independent of the input scale. Named outputs are `native` (target-conditioned zero-field endpoint), `centered` (VAE source reconstruction), `condition` (source anchor plus the target/source zero-field difference), `reference` (adaptive zero-field reference control), `raw` (the supplied main field without control), and `full` (the unchanged main method). Fixed interventions are exported directly. The shared localization contrast remains charged; zero-field arms do not pay pullback probe costs, so actual costs differ under the common cap.

~~~bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite null --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/nulltest
~~~

`centered` is generally a VAE reconstruction rather than the exact input pixels. `condition` retains a conditional effect despite its zero input field. These cases isolate different operations and should not be combined into one no-field label.

#### Alignment

Coupled and scalar norm-matched field construction. Named outputs: `aligned`, `scalar`, `controlled`, `matched`.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite alignment --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/alignmenttest
```

#### Temporal

This VP-only suite compares five fixed temporal mixtures with response-calibrated initialization and the ECT transport in Equation 22. Named mixture outputs are `firstraw/full`, `earlyraw/full`, `meanraw/full`, `lateraw/full`, and `lastraw/full`; their second-time weights are 0, 0.25, 0.5, 0.75, and 1. The benchmark suite reports the complete predefined curve; choosing one global weight belongs to development validation. Each arm is one editor configuration; the suite does not select a different winner for each test request.

`ectdirect` applies ECT transport without continuation or reference control at time 0.75 and shrinkage 0.7. `ectraw` passes that field through the common native continuation; `ectfull` adds RefUnmix control. `chorddirect` exports the internal Chord field without refinement. These labels separate direct transport from transplanted controls; external headline comparisons use the published complete pipelines and aligned preprocessing.

All arms inherit the supplied main budget, seeds, and shared protocol; `full` is the unchanged main configuration. Mixtures use the shared two-time contrast pool, including its localization contrast. Report actual conditioned samples and complete latency for every arm.

Controlled arms freeze the same protection-probe time. `ectfull` uses that time even though its editing field is measured at 0.75, and pays the extra probe when needed. Direct ECT keeps its single temporal support and does not prepare a control head.

~~~bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite temporal --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/temporaltest
~~~

#### Core

Guarded field and reference coordinate controls. Named outputs: `raw`, `target`, `source`, `eta`, `field`, `full`.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite core --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/coretest
```

#### Release

Main reference-field compensation versus joint QCQP. Named outputs: `released`, `joint`, `random`, `field`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite release --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/releasetest
```

#### Geometry

Output, latent, and reference-only compensation. Named outputs: `output`, `latent`, `attenuated`, `joint`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite geometry --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/geometrytest
```

#### Search

Trust, poll, random search, and curvature monitoring. Named outputs: `poll`, `randomsearch`, `unguarded`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite search --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/searchtest
```

#### Attainment

Attainment-first and protection-first selection. Named outputs: `protected`, `attained`, `field`, `random`, `poll`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite attainment --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/attainmenttest
```

#### Residual Compensation

`--suite residual` adds `incremental`, `repaired`, and `joint` to the six core references. These arms respectively use incremental release, residual-aware affine release, and generic joint control. They freeze the initial field, semantic view policy, native step count, candidate cap, and model budgets. Keep `--fieldrepair 0 --semantic patches` to isolate the control operator. At each visited state, the log's `comparison` evaluates both path families from identical responses without generating either counterfactual image; final arms run and score their own outputs.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite residual --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/residualtest
```

Score `outputs/residualtest/incremental/summary.json`, `outputs/residualtest/repaired/summary.json`, and `outputs/residualtest/joint/summary.json` using the official assessment workflow below. Preserve all four seeds and compare target completion, preservation, and complete cost together. `release:modelimprovement` is the request-level mean of frozen-model objective improvements before seed-first aggregation; it is separate from CLIP-Edited. `release:residualqueries` and `release:residualaccepted` count actual affine-path proposals and accepted queries.

#### Reference

Latent, RGB, mixture, random, and conditional representations. Named outputs: `rgb`, `mixture`, `random`, `conditional`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite reference --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/referencetest
```

#### Diagnosis

Unfiltered reference interventions. Named outputs: `raw`, `target`, `middle`, `source`, `mixture`, `rgb`, `random`, `full`.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite diagnosis --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/diagnosistest
```

#### Goals

Editing-head objective with unchanged feasibility checks. Named outputs: `targetonly`, `globalonly`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite goals --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/goalstest
```

#### Leverage

Additional rescue probes. Named outputs: `compensated`, `uncompensated`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite leverage --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/leveragetest
```

#### Prior

Explicit bounded, hard, and soft DDPM consistency. Named outputs: `noprior`, `boundedprior`, `hardprior`, `softprior`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite prior --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/priortest
```

#### Verify Optional Mechanisms

The reference-control comparison uses `--method path --release --no-compensation --prior off`. `release` implements protected-response reference-field compensation. The `compensation` switch enables an additional recovery probe only when the regular proposal needs a fallback and the fitted response is trusted. `prior` adds separately charged consistency probes. Their effects are measured by the `leverage` and `prior` suites; they are not required by the main configuration.

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
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite guidance --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/guidancetest
```

#### Solver

Solver iterations, certificates, and monitoring. Named outputs: `nocertificate`, `shortsolve`, `unguarded`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite solver --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/solvertest
```

#### Candidates

Candidate caps of 8 and 24; full uses 12. Named outputs: `small`, `large`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite candidates --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/candidatestest
```

#### Cost

Reference-only and shorter candidate configurations. Named outputs: `light`, `short`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite cost --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/costtest
```

#### Budget

Sample/call caps of 10, 12, 15, 20, and 30. Named outputs: `ten`, `twelve`, `fifteen`, `twenty`, `thirty`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite budget --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/budgettest
```

#### Mechanism

Decoder-only and five-step backend diagnostics. Named outputs: `decoder`, `multistep`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite mechanism --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 256 --maxcalls 256 --output outputs/mechanismtest
```

#### Frontier

First-model local capacity diagnostic. Named outputs: `capacity`, `unchecked`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite frontier --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/frontiertest
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

`field.capacity` contains `lineerror`, `spanerror`, and `directionroom`: the normalized squared errors $`E_{\mathrm{line}}`$, $`E_{\mathrm{span}}`$, and their difference $`G_F`$. Its `rank` is the resolved output-response rank; `field.rank` is the input-contrast rank. Both fits use the same SVD truncation and the same paid response probes. The scalar fit permits either sign. Zero requests have null normalized errors; zero responses have unit errors and zero direction room. The same diagnostic is saved for coupled and norm-matched scalar arms before their different fields are applied.

These diagnostics use the native field endpoint before reference control. Pullback runs already save them from paid endpoints; evaluation summaries first average available request values within each seed, then average the seed results, and `audit.py` includes them in paired comparisons. The probe command adds source-condition endpoints for mechanism analysis and reports its complete cost. Response fit is evaluated separately from final editing quality.

Controller history records `verification` for every queried non-geometry proposal, including rejected proposals: `predictedgain`, `actualgain`, `predicteddamagechange`, and `actualdamagechange`. Each change is measured from the current anchor. Gains use the frozen control heads and scales; independent task annotations evaluate the final image. `audit.py` first averages release-query changes within each request, then applies the same request-within-seed and seed-level averaging used by the main tables. Paired confidence intervals resample source groups and recompute that seed-first statistic.

Native probes use raw mode with priors and controller diagnostics disabled. Discrete reference paths require `--etas 0 1`. Raw inference rejects ignored prior or controller switches, and held-out consistency requires an enabled prior.

### Protection at a Fixed Target Response

Similarity is ambiguous: source-like responses may represent preserved content or an edit that has not happened. The conditional branch instead asks whether protected damage can be reduced while retaining the target response of a proposed field direction.

Let $`S`$ and $`P`$ be measured target/local and protected-output responses in tangent coordinates, and let $`v_0`$ be the initial joint proposal. Define $`q=S v_0`$. The conditional repair is

```math
\mathcal C_q(v_0)=\arg\min_{v\in\mathcal T_b,\,Sv=q,\,\lVert v\rVert_M\le\Delta}
\left(\lVert r_b+Pv\rVert_2^2+\lambda_c\lVert v-v_0\rVert_M^2\right).
```

The original proposal is feasible. Therefore the repaired proposal preserves the resolved predicted target response and cannot increase its modeled protected residual. This is a change of priority from a weighted semantic/protection compromise to protection within a target-response equivalence class. The regularized projection has a unique solution; its shifted ball is solved in the semantic nullspace. Its algebraic solve uses no model queries. The actual native output is still generated and checked against the original protection budget and frozen semantic floors. Amplitude selection follows repair, so the equality concerns the direction at a fixed amplitude; actual image scores are reported independently.

The conditional policy uses its first real check for the unmodified joint proposal when feasible. Once an edit is accepted, subsequent candidates must retain every selected target/local control-head gain within the declared numerical tolerance. At tied target gain, lower actual damage can replace the retained edit. The same feasible amplitude is tested preferentially after projection. `tangentchecks` includes this seed and all repairs; budget exhaustion preserves the best verified image. Logs distinguish seed checks, repair checks, target-lock rejections, and whether the final image came from a repair. This is a measured control-head guarantee; independent official metrics score the frozen image afterward.

A two-dimensional tangent dictionary may already be spanned by two independent target heads. In that case no semantic-neutral direction exists. The `shared` dictionary adds one matched common-mode candidate:

```math
h_s=H(0,C_{\mathrm{src}}),\qquad h_t=H(0,C_{\mathrm{tgt}}),\qquad
b_c=(I-\Pi_{h_t-h_s})\left(\frac{h_s+h_t}{2}-z_{\mathrm{src}}\right).
```

$`\Pi_d`$ projects onto a nonzero conditional chord; it is zero when the chord vanishes. This candidate is orthogonalized with the existing field, temporal contrasts, and unresolved request. The paid source zero-field endpoint is cached and reused by the optional reference controller. Common-mode evidence supplies a possible correction direction; its semantic role is determined by the measured response, not inferred from similarity alone. No spatial region or feature is frozen merely because predictions agree.

Use `--method field --tangentbasis shared --tangentpolicy conditional`. The earlier static profile remains `--tangentbasis residual --tangentpolicy joint`. The shared configuration adds one source zero-field endpoint and at most one tangent probe; all are charged to the request. With two temporal directions, one residual, one common-mode candidate, and three real checks, the one-step single-condition profile uses at most 15 sample evaluations. A zero common mode or absent semantic nullspace retains the corresponding original proposal. Logs record nullity, predicted semantic defect, damage before/after projection, source endpoints, and actual acceptance.

[CEF](https://arxiv.org/abs/2609.33735) separates prompt- and trajectory-induced fields, [SynPS](https://arxiv.org/abs/2512.14423) modulates attention sharing with similarity evidence, and [ARRO](https://arxiv.org/abs/2610.06021) learns from requested and unintended changes. This experiment fixes a measured target response and tests whether the available input dictionary contains a lower-damage realization. Nullspace projection itself is established mathematics; the intervention and its equal-query comparisons determine the method's contribution.

**Seven-arm comparison:**

| Arm | Dictionary and rule |
|---|---|
| `anchor` | Coupled raw field |
| `plain` | Residual dictionary, joint allocation |
| `expanded` | Shared dictionary, joint allocation |
| `residual` | Residual dictionary, conditional protection |
| `full` | Shared dictionary, conditional protection |
| `feedback` | Full branch with paid-response feedback |
| `matched` | Full branch with actual endpoint norm within 1% of anchor |

`expanded` versus `full` uses the same dictionary, initial probes, and query limits and changes only the protection rule. `plain` versus `expanded` isolates added common-mode evidence. `residual` records whether the original dictionary already has a usable nullspace. Match actual cost as well as the common cap.

**64-sample matrix, one RTX 4090, three concurrent requests:**

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite conditional --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --fieldrepair 0 --fieldfactor 1 --method field --semantic region --tangentgain 0.02 --tangentpenalty 1 --tangentchecks 3 --tangentbasis shared --tangentpolicy conditional --no-tangentnorm --no-tangentfeedback --poststeps 1 --guidance 1 --no-cfg --referencepath auto --prior off --localization baseline --no-control --no-heldout --no-compensation --no-frontier --maxsamples 64 --maxcalls 64 --output outputs/conditional
```

**Official assessment and fixed-configuration comparison:**

```bash
python -B RefUnmix/assess.py --input outputs/conditional/full/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/conditionalmetrics/full.json
python -B RefUnmix/assess.py --input outputs/conditional/anchor/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/conditionalmetrics/anchor.json
python -B RefUnmix/assess.py --input outputs/conditional/plain/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/conditionalmetrics/plain.json
python -B RefUnmix/assess.py --input outputs/conditional/expanded/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/conditionalmetrics/expanded.json
python -B RefUnmix/assess.py --input outputs/conditional/residual/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/conditionalmetrics/residual.json
python -B RefUnmix/assess.py --input outputs/conditional/feedback/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/conditionalmetrics/feedback.json
python -B RefUnmix/assess.py --input outputs/conditional/matched/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/conditionalmetrics/matched.json
python -B RefUnmix/audit.py --summary outputs/conditional/summary.json --sweep outputs/conditionalmetrics --repeats 2000 --seed 42 --output outputs/conditionalcomparison.json
```

**15-sample matrix, one RTX 4090, three concurrent requests:**

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite conditional --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --fieldrepair 0 --fieldfactor 1 --method field --semantic region --tangentgain 0.02 --tangentpenalty 1 --tangentchecks 3 --tangentbasis shared --tangentpolicy conditional --no-tangentnorm --no-tangentfeedback --poststeps 1 --guidance 1 --no-cfg --referencepath auto --prior off --localization baseline --no-control --no-heldout --no-compensation --no-frontier --maxsamples 15 --maxcalls 15 --output outputs/conditionalfifteen
```

**Official assessment and fixed-configuration comparison:**

```bash
python -B RefUnmix/assess.py --input outputs/conditionalfifteen/full/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/conditionalmetricsfifteen/full.json
python -B RefUnmix/assess.py --input outputs/conditionalfifteen/anchor/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/conditionalmetricsfifteen/anchor.json
python -B RefUnmix/assess.py --input outputs/conditionalfifteen/plain/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/conditionalmetricsfifteen/plain.json
python -B RefUnmix/assess.py --input outputs/conditionalfifteen/expanded/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/conditionalmetricsfifteen/expanded.json
python -B RefUnmix/assess.py --input outputs/conditionalfifteen/residual/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/conditionalmetricsfifteen/residual.json
python -B RefUnmix/assess.py --input outputs/conditionalfifteen/feedback/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/conditionalmetricsfifteen/feedback.json
python -B RefUnmix/assess.py --input outputs/conditionalfifteen/matched/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/conditionalmetricsfifteen/matched.json
python -B RefUnmix/audit.py --summary outputs/conditionalfifteen/summary.json --sweep outputs/conditionalmetricsfifteen --repeats 2000 --seed 42 --output outputs/conditionalcomparisonfifteen.json
```

All seven configurations retain all 700 requests and seeds 42, 52, 3047, and 3407. Aggregate requests within each seed, then average the four seed results. The comparison validates hashes, populations, official metrics, and the common protection support. It reports dictionary, rule, nullspace-capacity, feedback, and norm-matching comparisons without selecting outputs using official scores. Use `--batch 4` consistently for the four-request alternative.

### Fixed-Field Amplitude and Matched Comparisons

This matrix isolates amplitude from direction. Every request first constructs the same coupled anchor $`F_b`$ with native `--scale 1`. Radial arms apply $`F_a=aF_b`$ after construction; changing `--scale` would change the inverse response problem and is not this experiment. `anchorfieldsha256` and `protectionsha256` must match across all arms. All paths below are relative to the workspace layout above.

| Arm | Field or verification |
|---|---|
| `zero`, `half`, `weak` | Fixed anchor multiplied by 0, 0.5, 0.75 |
| `anchor`, `strong`, `large`, `double` | Fixed anchor multiplied by 1, 1.25, 1.5, 2 |
| `full` | Static tangent response allocation |
| `matched` | Tangent field with actual endpoint-response norm within 1% of anchor |
| `feedback` | Residual feedback using already paid output observations |

**Generate all ten arms on one RTX 4090:**

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite amplitude --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --fieldrepair 0 --fieldfactor 1 --method field --semantic region --tangentgain 0.02 --tangentpenalty 1 --tangentchecks 3 --tangentbasis residual --no-tangentnorm --no-tangentfeedback --poststeps 1 --guidance 1 --no-cfg --referencepath auto --prior off --localization baseline --no-control --no-heldout --no-compensation --no-frontier --maxsamples 64 --maxcalls 64 --output outputs/amplitude
```

The matrix uses all 700 official PIE requests and seeds 42, 52, 3047, and 3407. Each seed averages all test requests first; the final result averages those four seed results. `--batch 3` shares weights across three requests; use 4 consistently for the four-request workflow. No annotation mask guides generation. `tangentchecks=3` includes every initial or feedback amplitude check; feedback does not add a second candidate allowance or calibration loop. The radial arms have no controller. Report actual samples, calls, decodes, scores, and runtime, since a common cap does not imply equal work.

For a configuration check, use the generation command with `--dry` and omit `--output`. The zero arm exports the target-conditioned zero-field endpoint.

**Generate the separate 15-sample matrix:**

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite amplitude --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --fieldrepair 0 --fieldfactor 1 --method field --semantic region --tangentgain 0.02 --tangentpenalty 1 --tangentchecks 3 --tangentbasis residual --no-tangentnorm --no-tangentfeedback --poststeps 1 --guidance 1 --no-cfg --referencepath auto --prior off --localization baseline --no-control --no-heldout --no-compensation --no-frontier --maxsamples 15 --maxcalls 15 --output outputs/amplitudefifteen
```

**Score every frozen arm with the official calculator:**

```bash
python -B RefUnmix/assess.py --input outputs/amplitude/full/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetrics/full.json
python -B RefUnmix/assess.py --input outputs/amplitude/anchor/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetrics/anchor.json
python -B RefUnmix/assess.py --input outputs/amplitude/zero/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetrics/zero.json
python -B RefUnmix/assess.py --input outputs/amplitude/half/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetrics/half.json
python -B RefUnmix/assess.py --input outputs/amplitude/weak/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetrics/weak.json
python -B RefUnmix/assess.py --input outputs/amplitude/strong/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetrics/strong.json
python -B RefUnmix/assess.py --input outputs/amplitude/large/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetrics/large.json
python -B RefUnmix/assess.py --input outputs/amplitude/double/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetrics/double.json
python -B RefUnmix/assess.py --input outputs/amplitude/matched/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetrics/matched.json
python -B RefUnmix/assess.py --input outputs/amplitude/feedback/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetrics/feedback.json
```

The calculator and cache setup are given in [Official PIE Metrics](#official-pie-metrics). These commands score stored images after selection. Keep the same calculator, mapping, metric weights, and PNG encoding for every arm.

**Produce all four diagnostics and the feedback comparison:**

```bash
python -B RefUnmix/audit.py --summary outputs/amplitude/summary.json --sweep outputs/amplitudemetrics --repeats 2000 --seed 42 --output outputs/amplitudecomparison.json
```

The report contains:

- `curve`: all seven fixed radial operating points, including seed-level results.
- `equaledit`: radial protection at the full field's mean official CLIP-Edited. It records every adjacent crossing, including nonmonotone curves, and the lowest-MSE crossing.
- `equalbudget`: radial CLIP-Edited at the full field's mean official protected-region MSE. PSNR, LPIPS, SSIM, and other metrics accompany the comparison; mean PSNR is never reconstructed from mean MSE.
- `normmatched`: paired anchor-versus-matched results, with every request's actual latent endpoint norm checked against the same target zero-field reference. Failed edits remain baseline outputs; zero response anchors remain in quality coverage with undefined norm ratios.
- `feedback`: paired static-versus-feedback outcomes under the same maximum number of real checks.

`equaledit` and `equalbudget` describe postselection benchmark curves. A crossing uses one fixed interpolation weight across every request and all four seeds. It is a curve estimate, not a generated image or a per-request oracle. The report also lists actually generated feasible grid points. It never extrapolates beyond measured points or assumes monotone CLIP response. Source-grouped paired intervals condition on the selected curve weights; they do not include operating-point selection uncertainty. Official test scores do not select a deployed method, prompt, or hyperparameter.

The comparison loader verifies inference hashes, complete request/seed populations, fixed metric coverage, the calculator and mapping identities, original field hashes, and protection support. Missing seeds, stale scores, changed anchors, or violated norm matching stop analysis. Undefined metric seeds remain undefined; infinite PSNR retains explicit status. The official MSE used for curve matching is distinct from the method's inferred-support protection constraint; both are reported under their declared support.

**Score and compare the separate 15-sample matrix:**

```bash
python -B RefUnmix/assess.py --input outputs/amplitudefifteen/full/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetricsfifteen/full.json
python -B RefUnmix/assess.py --input outputs/amplitudefifteen/anchor/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetricsfifteen/anchor.json
python -B RefUnmix/assess.py --input outputs/amplitudefifteen/zero/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetricsfifteen/zero.json
python -B RefUnmix/assess.py --input outputs/amplitudefifteen/half/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetricsfifteen/half.json
python -B RefUnmix/assess.py --input outputs/amplitudefifteen/weak/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetricsfifteen/weak.json
python -B RefUnmix/assess.py --input outputs/amplitudefifteen/strong/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetricsfifteen/strong.json
python -B RefUnmix/assess.py --input outputs/amplitudefifteen/large/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetricsfifteen/large.json
python -B RefUnmix/assess.py --input outputs/amplitudefifteen/double/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetricsfifteen/double.json
python -B RefUnmix/assess.py --input outputs/amplitudefifteen/matched/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetricsfifteen/matched.json
python -B RefUnmix/assess.py --input outputs/amplitudefifteen/feedback/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/amplitudemetricsfifteen/feedback.json
python -B RefUnmix/audit.py --summary outputs/amplitudefifteen/summary.json --sweep outputs/amplitudemetricsfifteen --repeats 2000 --seed 42 --output outputs/amplitudecomparisonfifteen.json
```

Both matrices use fresh output directories. Run one generation or assessment command at a time on the single GPU. A failed optional numerical feedback update retains the best verified endpoint and records `feedbackfailure`; it does not discard a completed edit.

### Response-Conditioned Residual Feedback

`--tangentfeedback` enables the optional feedback arm. The query is the unresolved semantic/protection residual; keys are measured output response modes and values are tangent input directions. For the stacked tangent response $`A=U\Sigma V^\top`$ and requested response $`b`$, the regularized allocation is

```math
y=V\,\mathrm{diag}\left(\frac{\sigma_i}{\sigma_i^2+\lambda+\mu}\right)U^\top b,
\qquad \mu\ge0.
```

The filter depends on both residual alignment and response strength. Signed coefficients can cancel errors; a positive softmax over raw residual magnitudes cannot express this allocation. Logged spectral gates and coefficient-energy shares describe the same inverse operator, not a newly trained attention network. Individual modes are not unique at repeated singular values.

A second solve with unchanged responses and objective repeats the same optimum. Feedback instead uses the actual prediction discrepancy of a paid candidate with tangent coordinates $`u`$:

```math
\epsilon^{(2)}=z_{\mathrm{observed}}-A u,\qquad
A^+=A+\frac{\epsilon^{(2)}u^\top}{\lVert u\rVert_2^2+\nu},
\qquad \nu=\zeta_F\lVert u\rVert_2^2>0.
```

Semantic, protected-pixel, and latent responses use their own observed discrepancies. The update is the unique minimizer of a regularized secant-fit objective. Reallocation uses the corrected responses while preserving the original anchor, tangent space, total protection budget, radius, semantic floors, and remaining check count. Both accepted and rejected outputs provide response evidence; only actually verified feasible images can be exported. This feedback corrects transmission within the measured dictionary and does not create unmeasured directions. [FASTER](https://arxiv.org/abs/2610.04538) reuses endpoint supervision to train a per-image network, [ARRO](https://arxiv.org/abs/2610.06021) learns from requested and unintended output changes, and [RBE](https://arxiv.org/abs/2412.12912) studies regional Jacobian components. This branch reuses finite observations to update a small analytic response operator without training the backbone or an auxiliary network.

### Integrated Field Ablation

This six-arm matrix compares `full` (integrated field), `anchor` (coupled raw baseline), `temporal` (temporal-only tangent dictionary), `matched` (actual endpoint norm within 1%), `legacy` (reference controller on the anchor), and `controlled` (reference controller after tangent allocation). All arms use the same region semantics, seeds, native steps, and model budget. Actual query counts may differ and are reported. The field arms ignore path-candidate limits; `legacy` and `controlled` retain the declared path cap of 12.

```bash
python -B RefUnmix/trial.py --suite integrated --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --fieldrepair 0 --method field --semantic region --tangentgain 0.02 --tangentpenalty 1 --tangentchecks 3 --tangentbasis residual --no-tangentnorm --poststeps 1 --guidance 1 --no-cfg --referencepath auto --prior off --localization baseline --no-control --no-heldout --no-compensation --no-frontier --maxsamples 64 --maxcalls 64 --output outputs/integrated
```

For an equal-cap comparison replace both 64 limits by 15 and use a fresh output directory. For a dry configuration check add `--dry` and omit `--output`. Independently score each arm's `summary.json` using [Official PIE Metrics](#official-pie-metrics); average all requests within each seed, then average the four seed means. Historical controller suites below retain their explicit `--method path` commands and references.

### Tangent Endpoint Response Allocation

`--tangent` refines the raw field around a **coupled pullback protection anchor**. It adds a lateral residual direction selected by measured native-output responses. It can run with `--method raw`, without subsequent reference search. `--method path --tangent` uses the refined field in the existing controller. Both modes charge all calibration and verification to the same request's model budget.

Let $`F_b=B_b c_b`$ be the frozen anchor field. The dictionary contains the baseline, its temporal contrasts, and the normalized unfulfilled latent endpoint request. A local solve allocates the measured target response while accounting for the total protected residual:

```math
\min_{c_b^\top Mv=0,\;\lVert v\rVert_M\le\Delta_b}
\left(\lVert R_{\mathrm{edit}}v-s_{\mathrm{target}}\rVert_2^2
+\gamma\lVert(r_b+R_{\mathrm{protect}}v)/d_b\rVert_2^2
+\lambda_b\lVert v\rVert_M^2\right).
```

The metric tangent is defined at the final coupled coefficients, not at the initial temporal mean. `s_target` is a frozen desired control-score increase: target-caption similarity and, for nonempty edit support, the aggregated local target gain. Edit responses are normalized by `--tangentgain`; the direction penalty uses the total residual divided by the baseline residual norm when nonzero (`d_b=1` otherwise). Adoption and actual verification use the original protection units. These are development settings, not values derived from the official test metric. The solve uses metric whitening and the exact tangent nullspace, then a bounded regularized inverse response.

The actual field is $`F_b+\alpha B_bv`$. Positive adoption must satisfy

```math
\lVert r_b+\alpha R_{\mathrm{protect}}v\rVert_2^2\le D_b,
\qquad 0\le\alpha\le1.
```

If a nonzero protected response is orthogonal to the existing residual, no positive amplitude preserves the identical budget in this model. Every visited amplitude is generated, decoded, and scored. Acceptance requires actual target gains, all frozen semantic floors, and actual protected damage no greater than the anchor's. A request with no accepted candidate retains the paid coupled baseline; interruption after acceptance retains the verified field and its matching endpoint. Protection weights define measurement support; they do not multiply the raw field. The standard run uses inferred support, and annotation masks remain reserved for postselection assessment unless oracle mode is explicitly requested.

Tangency removes first-order radial scaling but does not conserve finite-step field energy. `--tangentnorm` therefore provides a separate control: it intersects the full 1% latent-response norm band with the protection interval and accepts only actual response norms within that band. The band may have two connected intervals; no exact nonzero root is required. It retains baseline when no such verified improvement exists. `--tangentbasis temporal` excludes the extra residual direction, testing whether an improvement is explainable by temporal reweighting alone.

The branch requires `--field pullback --fieldgeometry coupled --cleanup --clip clip` and `--fieldrepair 0`. Standard `raw` editing remains available without CLIP; tangent `raw` explicitly loads the frozen control scorer. Its native continuation still uses `--poststeps 1`, while the complete editor performs the charged endpoint queries.

Standalone raw-field refinement on all PIE requests:

```bash
python -B RefUnmix/evaluate.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method raw --tangent --tangentbasis residual --tangentgain 0.02 --tangentpenalty 1 --tangentchecks 3 --fieldrepair 0 --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/tangentraw
```

Run the complete comparison at the same sample/call caps and four seeds:

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite tangent --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/tangenttest
```

| Arm | Output and intervention |
|-----|--------------------------|
| `full` | Supplied original main configuration |
| `anchorraw` | Coupled raw baseline |
| `tangentraw` | Verified tangent residual allocation, raw output |
| `matchedraw` | Tangent raw output with actual endpoint norm matching |
| `temporalraw` | Tangent raw output restricted to temporal directions |
| `controlled` | Tangent field followed by reference control |

Score the raw outputs with the same official calculator:

```bash
python -B RefUnmix/assess.py --input outputs/tangenttest/anchorraw/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/anchorrawmetrics.json
python -B RefUnmix/assess.py --input outputs/tangenttest/tangentraw/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/tangentrawmetrics.json
python -B RefUnmix/assess.py --input outputs/tangenttest/matchedraw/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/matchedrawmetrics.json
python -B RefUnmix/assess.py --input outputs/tangenttest/temporalraw/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/temporalrawmetrics.json
```

Every arm retains all 700 requests and seeds 42, 52, 3047, and 3407; failed refinements return baseline rather than dropping rows. `field.tangent` saves actual gains, damage, adoption, tangent error, input norm ratio, endpoint norm ratio, charged calls/samples, and each verification. `audit.py` exports these under `field:tangent...`, including adoption coverage. Preserve the request-within-seed then four-seed mean. The caps bound total work; report actual NFE, scorer work, decodes, and runtime as well.

### Local Semantic Attainment

`--candidates` limits controlled candidates per image-seed request, including initialization and fallback proposals; the retained native baseline is separate. Values such as 1, 2, 3, and 12 are valid. The default `--poststeps 1` sets one native continuation step per endpoint. Neither value is the total model-query count: field and reference queries also consume `--maxsamples` and `--maxcalls`, and early stopping can leave the candidate allowance partly unused.


PIE's [official CLIP calculator](https://github.com/cure-lab/PnPInversion/blob/f480116dda7576ea258aca9ac6346c9d9000d49d/evaluation/matrics_calculator.py) evaluates the target caption against the full image with pixels outside the annotation mask set to black. It does not average crop scores. Control views remain separate: they use frozen inferred support and neutral gray masking; official masks enter only postselection assessment unless an oracle run is explicitly requested.

Two interventions test local under-editing. `--semantic region` evaluates the complete masked support together with its bounding view, pads rectangular views to preserve their extent, and averages their target gains into one `local0` goal. The individual `view0` and `view1` values are diagnostics. The previous `patches` setting treats up to four crops as separate goals; an unchanged crop can keep their minimum gain at zero even when the requested object improves. Global, target, attribute, retention, and protected-output checks remain active.

`--fieldrepair 1` remeasures the native endpoint response at the constructed field. With the initial request held fixed, define

```math
e_j=\gamma_F\bar d-[H(F_j,C_{\mathrm{tgt}})-H(0,C_{\mathrm{tgt}})].
```

The probe basis includes the original temporal contrasts, the current field, and the normalized residual. Each direction is measured through the same native continuation at the current field. A bounded damped solve proposes a correction:

```math
u_j^\star=\arg\min_{\lVert u\rVert_2\le\Delta_{\mathrm{rep}}}
\left(\lVert R_j u-e_j\rVert_2^2+\lambda_j\lVert u\rVert_2^2\right),
\qquad F_{j+1}=F_j+Q_j u_j^\star.
```

The repair radius is fixed to $`\Delta_{\mathrm{rep}}=\rho_F\lVert F_0\rVert_2`$ in each round. The implementation accepts only a measured reduction in endpoint residual. A residual vector is a proposed input direction, not an assumed inverse Jacobian. Failed corrections retain the previous field. Path control retains the pre-repair native output as its protection budget, semantic reference, and fallback; field repair cannot loosen that budget. Raw and fixed diagnostic modes expose the repaired field directly. All probes share the request's sample/call caps, and the accepted endpoint is reused by the controller. Two repair rounds are available; zero preserves the initial construction. The generator's native step count stays unchanged. This is additional field calibration, so its query cost is included in the complete editor cost.

[FlowEdit](https://openaccess.thecvf.com/content/ICCV2025/html/Kulikov_FlowEdit_Inversion-Free_Text-Based_Editing_Using_Pre-Trained_Flow_Models_ICCV_2025_paper.html) updates the target velocity at the evolving target state. [DecFlowEdit](https://arxiv.org/abs/2609.34237) separates localization from editing guidance, and [PreciseEdit](https://arxiv.org/abs/2610.23535) separates strong transport from spatial and directional reliability. These mechanisms motivate testing state-dependent response and regional attainment separately. SD3/SD3.5-versus-SD-Turbo comparisons also include backbone differences. SD-Turbo retains its [documented configuration without extra CFG](https://huggingface.co/stabilityai/sd-turbo); FlowEdit's source/target guidance scales are not transplanted into it.

The `semantic` suite runs the following controlled comparison. Defaults remain `--fieldrepair 0 --semantic patches` for the established main profile. Select a revised profile on the development partition and freeze it before final testing. Endpoint residual reduction is a calibration measurement; official CLIP-Edited and independent task completion measure transfer to image editing.

| Arm | Field repair | Semantic views |
|-----|--------------|----------------|
| `full` | Supplied main setting | Supplied main setting |
| `anchored` | 0 | patches |
| `regional` | 0 | region |
| `repaired` | 1 | patches |
| `combined` | 1 | region |

Use the same 64-sample/64-call cap, twelve control candidates, one native continuation step, and four seeds in every arm. At a 15-sample cap, repair can leave fewer controller queries; compare actual costs and accepted repair counts rather than assuming equal work from equal caps. The existing validation-to-test recipe workflow accepts `--suite semantic`. After freezing the profile, the complete PIE comparison is:

```bash
python -B RefUnmix/trial.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --suite semantic --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldrepair 0 --semantic patches --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --releasepolicy residual --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/semantic
```

Score all four interventions with the unchanged official protocol:

```bash
python -B RefUnmix/assess.py --input outputs/semantic/anchored/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/anchoredmetrics.json
python -B RefUnmix/assess.py --input outputs/semantic/regional/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/regionalmetrics.json
python -B RefUnmix/assess.py --input outputs/semantic/repaired/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/repairedmetrics.json
python -B RefUnmix/assess.py --input outputs/semantic/combined/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/combinedmetrics.json
```

Read `field.repair` for accepted rounds, per-round residuals, and charged model calls; `audit.py` aggregates its scalar diagnostics under `field:repair...`; `field.temporal` describes the final field while `initialtemporal` describes its initializer. `semanticviews` records frozen crop geometry, and `attainment.bottlenecks` identifies the active control goals. Compare CLIP-Edited, CLIP-Whole, PSNR, LPIPS, DINO structure distance, independent task labels, and runtime on the same outputs. Each metric is averaged over requests within each seed and then over the four seed means, with the corresponding `byseed` entries retained.

### Official PIE Metrics

Download the [PIE-Bench metric calculator](https://raw.githubusercontent.com/cure-lab/PnPInversion/f480116dda7576ea258aca9ac6346c9d9000d49d/evaluation/matrics_calculator.py) at revision `f480116` and save its Python source as `protocol/metrics.py` outside the RefUnmix checkout. Keep its `MetricsCalculator` interface and use the same calculator version, package versions, and metric weights for every method. Install a torchvision build matching PyTorch, then the metric dependencies:

```bash
python -m pip install torchmetrics lpips
```

The official calculator loads CLIP, LPIPS, and DINO assets. Use `--network` to populate or update `cache/metrics`, then use `--offline` to score with the completed cache. These modes are mutually exclusive. The relative cache uses the [Hugging Face cache settings](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) and PyTorch Hub directory, and is shared by the three- and four-request workflows. Generation remains separate from postselection scoring.

```bash
python -B RefUnmix/assess.py --input outputs/main/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/mainmetrics.json
python -B RefUnmix/assess.py --input outputs/matched/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/matchedmetrics.json
python -B RefUnmix/assess.py --input outputs/fieldstest/pullbackfull/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --network --cache cache/metrics --output outputs/fieldmetrics.json
```

Score other variants with the same command by changing only `--input` and `--output`. The input is a variant's inference summary, not the suite-level summary. `--scope benchmark` checks all 700 request keys, categories 0 through 9, and exactly one result per declared request-seed pair: 2,800 rows for the four-seed experiment. Source images, outputs, and annotation geometry must be 512 x 512. Prediction paths must be distinct. Saved image hashes, source hashes, captions, and the mapping hash are checked when present in the inference summary, before metric weights load. Results retain the calculator hash, package versions, population counts, and per-metric coverage.

`--protocol` and `--pixels` are explicit, mutually exclusive modes. Missing calculator files stop external scoring. Pixel diagnostics require `--pixels`; they report PSNR and MSE without downloading metric weights. For a short smoke run using the main defaults:

```bash
python -B RefUnmix/evaluate.py --batch 3 --dataset data --mapping mapping_file.json --images annotation_images --split all --limit 3 --model models --clip clip --device cuda:0 --precision fp32 --seeds 42 --maxsamples 64 --maxcalls 64 --output outputs/smoke
python -B RefUnmix/assess.py --input outputs/smoke/summary.json --dataset data --mapping mapping_file.json --images annotation_images --pixels --scope subset --seeds 42 --output outputs/smokepixels.json
```

`--scope subset` assesses the request keys present in a diagnostic summary and requires every declared seed for each included request. Such reports record `population.benchmarkcomplete = false`. Use `--protocol protocol/metrics.py --network` instead of `--pixels` to measure the full metric set on that same subset. Pixel mode accepts neither `--network` nor `--offline`.

The [ChordEdit export script](https://github.com/ChordEdit/ChordEdit/blob/main/run_pie_bench.py) defaults to seed 42. For a single-seed reproduction, generate both methods with seed 42 and assess with `--seeds 42`; for the four-seed table, rerun both methods with all four seeds. RefUnmix saves lossless PNGs. Use the same image encoding for newly generated comparisons: ChordEdit's exporter retains the input extension, so a JPEG extension adds compression. Score the saved files for every method with the same calculator.

Assessments contain `raw`, scaled `table`, per-seed results in `byseed`, and metric-specific `coverage`. MSE, LPIPS, and DINO structure distance use x1,000; SSIM uses x100; PSNR is in dB. Empty protected regions remain undefined for preservation metrics; zero-error PSNR is recorded as positive infinity. For each metric, first average its valid request scores separately for seeds 42, 52, 3047, and 3407, then take the arithmetic mean of those four results. Each seed has equal weight. A metric with no valid requests for any one seed has an undefined overall mean. Report the per-seed results and valid denominators alongside the final four-seed mean.

Final means are computed directly from the saved seed-level means. The result locations are:

| Report | Per-seed metric mean | Final four-seed mean |
|--------|----------------------|----------------------|
| Main or individual variant inference | `byseed[seed][metric].mean` | `summary[metric].mean` |
| Ablation suite | `variants[arm].byseed[seed][metric].mean` | `variants[arm].summary[metric].mean` |
| Official unscaled assessment | `byseed[seed].raw[metric].mean` | `raw[metric].mean` |
| Official table assessment | `byseed[seed].table[metric].mean` | `table[metric].mean` |

For example, the final MSE is `(MSE42 + MSE52 + MSE3047 + MSE3407) / 4`, where each term is that seed's dataset mean. Inference reports also retain cost means in `cost.byseed`; `cost.metrics[metric].mean` follows the same seed-first order. Cost medians and p95 describe the distribution of per-request seed averages. Postselection assessment supplies the official table metrics.

### Auxiliary Summaries and Undefined Metrics

Audits, paired comparisons, task annotations, blind ratings, operator scans, and time-search score tables use the same seed-first reduction. Result readers validate the declared seed list and the complete request-seed grid. Historical summaries without a seed declaration use their observed seed set; retain the declared `seeds` field when transferring results. A complete declaration detects an omitted entire seed as well as a missing individual request.

| Condition | Reported result |
|-----------|-----------------|
| Missing declared seed, missing request-seed row, or duplicate row | Stop before publishing a complete summary |
| A request has an undefined metric | Exclude that value from its seed's metric mean and retain the valid count |
| Any seed has no valid value for a metric | Final mean is `null`, with `status: "undefined seed"` and `undefinedseeds` |
| Defined PSNR contains positive infinity | Mean is `null` with `status: "positive infinity"` |
| Paired PSNR has one finite and one infinite value | Preserve the signed infinity in the difference report |
| Paired PSNR compares two positive infinities | Difference is undefined and contributes no finite observation |
| Positive and negative infinities meet in one mean | Mean is `null` with an explicit undefined or mixed-infinity status |

An undefined seed takes precedence over an infinite value from another seed. Reports remain valid JSON and retain the individual seed states. Finite confidence intervals require a finite point estimate, at least two independent source groups, and valid seed means in every bootstrap replicate. Blind ratings average raters within each request-seed case before averaging requests and seeds; every frozen case must receive a rating, and `unsure` remains undefined.

| Auxiliary report | Per-seed values | Final point estimate |
|------------------|-----------------|----------------------|
| Audit | `summary.byseed[seed][metric].mean` | `summary.metrics[metric].mean` |
| Paired audit or evidence | `paired.byseed` or the corresponding comparison's `byseed` | Metric `difference` |
| Independent task report | `byseed[seed][metric].mean` | `metrics[metric].mean` |
| Blind comparison | `byseed[seed][metric].mean` | `metrics[metric].rightpreference` |
| Operator scan | `byseed[configuration][seed][metric].mean` | `summary[configuration][metric].mean` |
| Time search | Each table row's `byseed[seed][metric].mean` | That row's metric value |
| Frequency prediction diagnostic | Each time entry's `byseed[seed][band:metric].mean` | That entry's `low[metric]` or `high[metric]` |

Frozen same-pool replay exports discard the parent's `byseed` and all output-dependent aggregates. Score the exported images again with `assess.py` and regenerate independent labels for those outputs. Time-search control scores require finite inputs; postselection assessments and audits retain explicit infinite PSNR states. Time-search protection changes and violation rates follow the same seed-first reduction. Undefined protection prevents a feasibility claim and falls back to the native candidate when no verified feasible candidate remains; an undefined MSE is excluded from the two-axis Pareto set.

### Reuse the Metric Cache After Moving

Copy the complete `cache/metrics/` tree with the workspace. Retain Hugging Face repository `refs/`, `snapshots/`, and `blobs/`, plus the PyTorch Hub repository directories and `checkpoints/`. Keep relative symlinks intact or copy their resolved file contents. Absolute snapshot links from the old machine are replaced with local blob copies when the matching blobs are present.

From the new workspace root, run the same Python command with `--offline` and a fresh assessment output:

```bash
python -B RefUnmix/assess.py --input outputs/main/summary.json --dataset data --mapping mapping_file.json --images annotation_images --scope benchmark --seeds 42 52 3047 3407 --protocol protocol/metrics.py --device cuda:0 --offline --cache cache/metrics --output outputs/mainoffline.json
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

The suite covers field construction, control constraints, query accounting, image formats, dataset conversion, README experiment plans, complete-benchmark ablations, cache relocation, offline scoring, and official validation-to-test replay at batch widths three and four. Transfer checks copy the checkout and validation files into a new root and run test replay in a fresh Python process with synthetic backends. Serial/parallel checks compare all four seeds, output tensors, and per-request counters. Temporal diagnostics distinguish numerical underflow from an exact fit.

For the focused parallel checks, run from the checkout root:

```bash
python -B -m unittest discover -s tests -p batch.py -q
```

CUDA availability enables allocator-retention and CUDA random-state tests with lightweight backends. Without CUDA, those two cases are reported as skipped.

---

## Implementation Details

Backbone parameters remain frozen throughout editing. Conditional contrasts at times 0.90 and 0.75 define the default field subspace; `--start` and `--delta` set these times. Endpoint probes fit a coupled inverse response within this subspace. The integrated profile allocates a tangent residual using complete output responses and exports its verified native endpoint. The optional path comparison adjusts field and reference coordinates.

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
