# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

**RefUnmix** formulates training-free image editing as inverse control of a frozen generator's measured response. It reconstructs an input field for the requested endpoint displacement, then couples reference motion with a field correction that compensates protected-content damage.

RefUnmix uses one-step or few-step native backbones with finite-query control. Reports include all denoiser sample evaluations, forward calls, VAE decodes, latency, and peak memory.

---

## Method

### Core Insight

A conditional prediction estimates a desired change. The frozen continuation and decoder determine the change that reaches the final image. Direction-dependent amplification and mixing separate these two quantities. Reference calibration creates a second coupling: subtracting the target zero-field endpoint removes reconstruction drift together with a finite target-condition response.

**RefUnmix controls the semantic response that remains after protected damage is compensated.** The same inverse-response principle determines both the base field and the subsequent reference-field intervention.

[ChordEdit](https://arxiv.org/abs/2602.19083) stabilizes conditional transport through temporal smoothing. [WhereEdit](https://arxiv.org/abs/2607.20883) strengthens target attraction and localizes transport. [PreciseEdit](https://arxiv.org/abs/2610.23535) coordinates transport strength with cross-scale agreement and spatial gating. RefUnmix measures the response of the chosen intervention through native continuation and decoding, then solves for its adoption under output constraints.

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

**Why compensation can preserve editing capacity.** In a local linear example with one field coordinate, let $A_f=2$, $a_\eta=1$, $C_f=1$, and $c_\eta=2$. The undamped correction is $q=-1/2$. The joint motion has zero incremental protected response, $2q+1=0$, while retaining semantic response $q+2=3/2$. This is the quantity the controller seeks: a reference change whose side effect can be counteracted without cancelling its useful target response. Finite steps are checked on the generated image.

**5. Close the loop on actual outputs.** Every visited candidate is decoded and scored under frozen control views. Completed candidates must satisfy the native protection budget and all semantic floors. Rejected outputs still inform response fitting. Initialization, field probes, reference queries, fallback, and enabled priors share the model allowance. Among the actual feasible outputs and $Y_{\mathrm{base}}$, select the highest attained minimum editing gain, then the lowest bounded preservation cost within that attainment band. Postselection PIE metrics and independent task labels score the frozen result.

### What Each Experiment Establishes

| Method claim | Matched comparison |
|--------------|--------------------|
| Endpoint transmission changes the required field direction | Four-field raw outputs and scalar norm-matched alignment |
| The reference supplies a controllable condition response | Unfiltered diagnosis and same-dimensional reference controls |
| Protected-output response determines a useful joint intervention | Release and geometry with the same field, allowance, and output selection |
| Queries improve the editing-preservation trade-off | Search, budget, and independent postselection metrics |
| Final selection retains generated editing gains | Attainment, goals, and same-pool diagnostics |

The base field can run independently. The same finite-query controller can operate on the instantaneous, average, Chord, or pullback field. These runs separate field construction from the value of subsequent control.

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
|-- trial.py            # Frozen validation/test ablation matrix
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
| Core | PyTorch, NumPy, Pillow, SciPy |
| Frozen models | diffusers, transformers, safetensors, accelerate |
| Flow text encoders | sentencepiece, protobuf |
| Saved Hugging Face datasets | datasets, pyarrow; optional |
| Local interface | gradio; optional |

### Installation

Install PyTorch with CUDA support, then install the dependencies:

```bash
python -m pip install numpy pillow scipy diffusers transformers accelerate safetensors sentencepiece protobuf
```

For saved Hugging Face datasets and the Gradio interface:

```bash
python -m pip install datasets pyarrow gradio
```

### Hardware and Random Seeds

Experiments use **one NVIDIA GeForce RTX 4090** and seeds **42, 52, 3047, and 3407**. Single-image inference defaults to 42. For dataset experiments, use `--seeds 42 52 3047 3407`.

Dataset partitions remain fixed across seeds. Flow backends load their complete checkpoints into GPU memory.

---

## Pretrained Models

Download the backbone and control scorer before running. All checkpoints are loaded from local directories.

For `--backbone vp`, place a compatible SD-Turbo-style checkpoint in `models/`, including `unet/`, `vae/`, `scheduler/`, `tokenizer/`, and `text_encoder/`. The text encoder can also be stored in `text/`.

Place a Transformers CLIP model and processor in `clip/` for controlled editing. The `raw` and `fixed` modes run without CLIP. Evaluation with `--evalclip` requires a separate checkpoint from the control scorer.

For `--backbone flux` or `--backbone sd3`, provide the corresponding pipeline with its transformer, VAE, scheduler, tokenizers, and text encoders. The checkpoint determines the supported continuation schedule.

### Workspace Layout

Keep runtime assets outside the source checkout:

```text
workspace/
|-- RefUnmix/           # Source checkout
|-- models/             # Frozen backbone checkpoint
|-- clip/               # Frozen control checkpoint
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
python -B RefUnmix/doctor.py --model models --clip clip --device cuda:0 --precision fp32
```

This checks dependencies and checkpoint structure without loading model weights.

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
| `--field` | `pullback` | Field: pullback, chord, instant, or average |
| `--fieldgeometry` | `coupled` | Coupled response fit or scalar norm-matched ablation |
| `--method` | `path` | Primary modes: path, raw, fixed |
| `--start`, `--delta` | `0.90`, `0.15` | Contrast times are start and start minus delta |
| `--end` | `0.30` | Native continuation noise time |
| `--poststeps` | `1` | Steps per native continuation, not total editor cost |
| `--backbone` | `vp` | Native backend: vp, flux, sd3 |
| `--precision` | `fp32` | fp32, fp16, or bf16; CPU requires fp32 |
| `--seed` | `42` | Single-image inference seed |
| `--deterministic` | off | Request deterministic PyTorch execution |
| `--maxsamples` | `64` | Maximum denoiser sample evaluations |
| `--maxcalls` | unset | Optional limit on denoiser forward calls |
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

### Dataset Protocols

The complete PIE-Bench benchmark and grouped ablations use separate commands in [Experiments](#experiments). Full-release evaluation uses all 700 requests; custom validation/test splits retain their actual subset sizes. For datasets with official partition tags, `split.py --protocol official` preserves those tags. Generation never receives annotation masks in the commands below; oracle experiments add `--oracle` and are reported separately.

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
python -B RefUnmix/evaluate.py --dataset data --mapping mapping_file.json --images annotation_images --split all --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/main
```

**Equal-budget run, 15-sample cap:**

```bash
python -B RefUnmix/evaluate.py --dataset data --mapping mapping_file.json --images annotation_images --split all --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 15 --maxcalls 15 --output outputs/matched
```

With the complete release, each command exports **2,800 image-seed outputs from 700 requests**. `--split all` evaluates every mapped request; it does not use a split manifest. The 64- and 15-sample outputs belong to different budget settings. Report their actual sample evaluations, calls, and runtime alongside quality. The budget suite below also covers 10, 12, 20, and 30.

`evaluate.py` saves images, per-image JSON reports, `summary.cost`, `summary.controller`, and diagnostic scores. Run `assess.py` after generation for the official PIE metric table.

### Frozen Ablation Protocol

Use a grouped validation/test manifest for ablations. The custom split below uses partition seed 42 once; generation uses all four seeds on that fixed partition. Validation and test sources remain disjoint. The manifest stores train/val/test keys; these commands use val/test and keep all model weights frozen. Test-subset results report their actual request count separately from the full 700-request benchmark. Freeze full-benchmark hyperparameters using a separate development source.

```bash
python -B RefUnmix/split.py --dataset data --mapping mapping_file.json --images annotation_images --protocol custom --seed 42 --ratios 0.70 0.15 0.15 --output outputs/split.json
python -B RefUnmix/doctor.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --report outputs/valcheck.json
python -B RefUnmix/doctor.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --report outputs/testcheck.json
```

Run each validation command first. Its test command reads that suite's `recipe.json` with the same seed set, code, checkpoints, precision, and base options. Use a new output directory for every run; outputs are not overwritten. If an existing official manifest is available, use that manifest consistently instead of creating a custom split.

Each suite inherits the supplied main configuration and applies its named interventions. Keep the shared command flags unchanged between its validation and test runs. `full` uses the same effective controls as the main run. In the default `fields` suite it is named `pullbackfull`; in `alignment` it is named `controlled`. Explicit settings such as native steps, reference path, prior, and rescue are preserved in the full arm.

Check the complete twelve-arm field plan before loading data or checkpoints:

```bash
python -B RefUnmix/trial.py --suite fields --manifest outputs/split.json --split val --model models --clip clip --device cuda:0 --precision fp32 --seed 42 --seeds 42 52 3047 3407 --field pullback --method path --release --no-compensation --prior off --poststeps 1 --candidates 12 --maxsamples 64 --maxcalls 64 --dry
```

The printed `design.main` is `pullbackfull`, whose `changes` must be empty. Each arm lists its effective settings, configuration hash, and changes relative to the supplied main configuration. The run stops before model loading if the full arm differs. Dry runs describe the current plan; actual test runs still verify the frozen recipe, checkpoints, and disjoint sources. Regenerate validation recipes after updating the code.

Main and variant summaries save `controlsignature`; suite summaries save the complete `design`. Matching hashes identify matching control settings. Seeds, input identities, code hashes, checkpoint hashes, and split membership are recorded separately and must also match for a repeated run. The 256-cap mechanism diagnostic therefore has a different budget signature from the 64-cap main run.

| Suite | Intervention | Sample/call cap |
|-------|--------------|----------------|
| `fields` | Four fields with raw, fixed-reference, and full control | 64 |
| `alignment` | Coupled and scalar norm-matched field construction | 64 |
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
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite fields --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/fieldsval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite fields --recipe outputs/fieldsval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/fieldstest
```

#### Alignment

Coupled and scalar norm-matched field construction. Named outputs: `aligned`, `scalar`, `controlled`, `matched`.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite alignment --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/alignmentval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite alignment --recipe outputs/alignmentval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/alignmenttest
```

#### Core

Guarded field and reference coordinate controls. Named outputs: `raw`, `target`, `source`, `eta`, `field`, `full`.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite core --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/coreval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite core --recipe outputs/coreval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/coretest
```

#### Release

Main reference-field compensation versus joint QCQP. Named outputs: `released`, `joint`, `random`, `field`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite release --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/releaseval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite release --recipe outputs/releaseval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/releasetest
```

#### Geometry

Output, latent, and reference-only compensation. Named outputs: `output`, `latent`, `attenuated`, `joint`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite geometry --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/geometryval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite geometry --recipe outputs/geometryval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/geometrytest
```

#### Search

Trust, poll, random search, and curvature monitoring. Named outputs: `poll`, `randomsearch`, `unguarded`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite search --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/searchval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite search --recipe outputs/searchval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/searchtest
```

#### Attainment

Attainment-first and protection-first selection. Named outputs: `protected`, `attained`, `field`, `random`, `poll`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite attainment --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/attainmentval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite attainment --recipe outputs/attainmentval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/attainmenttest
```

#### Reference

Latent, RGB, mixture, random, and conditional representations. Named outputs: `rgb`, `mixture`, `random`, `conditional`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite reference --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/referenceval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite reference --recipe outputs/referenceval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/referencetest
```

#### Diagnosis

Unfiltered reference interventions. Named outputs: `raw`, `target`, `middle`, `source`, `mixture`, `rgb`, `random`, `full`.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite diagnosis --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/diagnosisval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite diagnosis --recipe outputs/diagnosisval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/diagnosistest
```

#### Goals

Editing-head objective with unchanged feasibility checks. Named outputs: `targetonly`, `globalonly`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite goals --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/goalsval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite goals --recipe outputs/goalsval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/goalstest
```

#### Leverage

Additional rescue probes. Named outputs: `compensated`, `uncompensated`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite leverage --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/leverageval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite leverage --recipe outputs/leverageval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/leveragetest
```

#### Prior

Explicit bounded, hard, and soft DDPM consistency. Named outputs: `noprior`, `boundedprior`, `hardprior`, `softprior`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite prior --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/priorval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite prior --recipe outputs/priorval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/priortest
```

#### Verify Optional Mechanisms

The main method uses `--release --no-compensation --prior off`. `release` implements protected-response reference-field compensation. The `compensation` switch enables an additional recovery probe only when the regular proposal needs a fallback and the fitted response is trusted. `prior` adds separately charged consistency probes. Their effects are measured by the `leverage` and `prior` suites; they are not required by the main configuration.

Compare those suites after the test runs:

```bash
python -B RefUnmix/audit.py --summary outputs/leveragetest/uncompensated/summary.json --compare outputs/leveragetest/compensated/summary.json --output outputs/rescueaudit.json
python -B RefUnmix/audit.py --summary outputs/priortest/noprior/summary.json --compare outputs/priortest/boundedprior/summary.json --output outputs/prioraudit.json
```

`configuration.changes` checks the intended intervention. `agreement` counts exact matching PNG hashes and matching visited coordinates; equal rounded averages do not establish equal outputs. The per-request `rescue` record separates considered, eligible, proposed, budget-blocked, queried, accepted, and finally selected probes, with skipped reasons. The per-request `prioractivity` records evaluations, positive excess, penalties, hard vetoes, and whether removing the prior changes selection on the same observed pool. Summary coverage retains missing activity records separately.

With attainment-first selection, a finite bounded prior cannot change a unique winner in the attainment band. A prior can therefore be evaluated and assign nonzero penalties without changing the selected image. The same-pool comparison isolates selection; separate on/off runs also measure changed query trajectories and probe cost. Exact matching outputs count as zero output effect for that intervention.

#### Guidance

CFG and prediction-path guidance controls. Named outputs: `cfgone`, `cfgtwo`, `cfgfour`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite guidance --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/guidanceval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite guidance --recipe outputs/guidanceval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/guidancetest
```

#### Solver

Solver iterations, certificates, and monitoring. Named outputs: `nocertificate`, `shortsolve`, `unguarded`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite solver --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/solverval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite solver --recipe outputs/solverval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/solvertest
```

#### Candidates

Candidate caps of 8 and 24; full uses 12. Named outputs: `small`, `large`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite candidates --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/candidatesval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite candidates --recipe outputs/candidatesval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/candidatestest
```

#### Cost

Reference-only and shorter candidate configurations. Named outputs: `light`, `short`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite cost --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/costval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite cost --recipe outputs/costval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/costtest
```

#### Budget

Sample/call caps of 10, 12, 15, 20, and 30. Named outputs: `ten`, `twelve`, `fifteen`, `twenty`, `thirty`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite budget --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/budgetval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite budget --recipe outputs/budgetval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/budgettest
```

#### Mechanism

Decoder-only and five-step backend diagnostics. Named outputs: `decoder`, `multistep`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite mechanism --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 256 --maxcalls 256 --output outputs/mechanismval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite mechanism --recipe outputs/mechanismval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 256 --maxcalls 256 --output outputs/mechanismtest
```

#### Frontier

First-model local capacity diagnostic. Named outputs: `capacity`, `unchecked`. The six core references (`raw`, `target`, `source`, `eta`, `field`, `full`) are included.

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split val --suite frontier --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/frontierval
python -B RefUnmix/trial.py --dataset data --mapping mapping_file.json --images annotation_images --manifest outputs/split.json --split test --suite frontier --recipe outputs/frontierval/recipe.json --model models --clip clip --backbone vp --device cuda:0 --precision fp32 --size 512 --seed 42 --seeds 42 52 3047 3407 --noise 1 --start 0.90 --delta 0.15 --end 0.30 --scale 1 --cleanup --field pullback --fieldgeometry coupled --fieldridge 0.05 --fieldprobe 0.25 --fieldradius 1 --fieldtolerance 0.000001 --method path --axes all --eta 0 --representation latent --algorithm trust --selection attain --gainloss 0 --poststeps 1 --guidance 1 --no-cfg --referencepath auto --candidates 12 --prior off --priorprobes 2 --release --geometry output --no-compensation --monitor --certificate --localization baseline --no-control --no-heldout --no-frontier --no-archive --maxsamples 64 --maxcalls 64 --output outputs/frontiertest
```

### Reading and Comparing Outputs

Each variant writes `outputs/<suite>test/<variant>/summary.json`. For example, the field comparison reads:

```bash
python -B RefUnmix/evidence.py --reference outputs/fieldstest/chordfull/summary.json --comparison outputs/fieldstest/pullbackfull/summary.json --contrast field --seed 42 --repeats 2000 --output outputs/fieldcomparison.json
python -B RefUnmix/evidence.py --reference outputs/releasetest/joint/summary.json --comparison outputs/releasetest/released/summary.json --contrast outcome --seed 42 --repeats 2000 --output outputs/releasecomparison.json
```

Inspect `fields.actual`, `fields.budgetfallback`, and `summary.controller` together with image quality. Per-image reports record the backend step count, initialization state, actual release/rescue queries and acceptances, semantic bottlenecks, and budget termination. Named switches describe configuration; the query counters show execution. The evidence commands export paired diagnostics; independent task annotations can be attached with `--taskreference` and `--taskcomparison`.

### Official PIE Metrics

Save the [PIE-Bench metric calculator](https://github.com/cure-lab/PnPInversion/blob/main/evaluation/matrics_calculator.py) as `protocol/metrics.py` outside the RefUnmix checkout. Keep its `MetricsCalculator` interface and use the same calculator version and metric weights for every method. Install a torchvision build matching PyTorch, then the metric dependencies:

```bash
python -m pip install torchmetrics lpips
```

The official calculator loads CLIP, LPIPS, and DINO assets. `--network` enables that external backend, including its first-use downloads. Generation remains separate from postselection scoring.

```bash
python -B RefUnmix/assess.py --input outputs/main/summary.json --dataset data --mapping mapping_file.json --images annotation_images --protocol protocol/metrics.py --device cuda:0 --network --output outputs/mainmetrics.json
python -B RefUnmix/assess.py --input outputs/matched/summary.json --dataset data --mapping mapping_file.json --images annotation_images --protocol protocol/metrics.py --device cuda:0 --network --output outputs/matchedmetrics.json
python -B RefUnmix/assess.py --input outputs/fieldstest/pullbackfull/summary.json --dataset data --mapping mapping_file.json --images annotation_images --protocol protocol/metrics.py --device cuda:0 --network --output outputs/fieldmetrics.json
```

Score other variants with the same command by changing only `--input` and `--output`. The input is a variant's inference summary, not the suite-level summary. Official scoring requires 512 x 512 source images, outputs, and annotation geometry. Omitting `--protocol` runs pixel-only scoring and leaves perceptual and CLIP metrics unavailable.

Assessments contain `raw`, scaled `table`, and metric-specific `coverage`. MSE, LPIPS, and DINO structure distance use x1,000; SSIM uses x100; PSNR is in dB. Empty protected regions remain undefined for preservation metrics; zero-error PSNR is recorded as positive infinity. Report each metric's valid denominator, alongside all 700 requests and four seeds.

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

Run the synthetic-backend regression suite from the repository root:

```bash
cd RefUnmix
python -B -m unittest discover -s tests -p '*.py' -q
```

The suite covers field construction, control constraints, query accounting, image formats, and dataset conversion.

---

## Implementation Details

Backbone parameters remain frozen throughout editing. Conditional contrasts at times 0.90 and 0.75 define the default field subspace; `--start` and `--delta` set these times. Endpoint probes fit a coupled inverse response within this subspace. The controller then adjusts the field and reference path using observed outputs.

`--poststeps` sets the steps in each native continuation. `--maxsamples` and `--maxcalls` limit denoiser sample evaluations and forward calls, including optional probes and priors. The candidate budget is set independently by `--candidates`.

The default sample cap is 64; the equal-budget experiments use explicit caps of 10, 12, 15, 20, and 30. Sample evaluations count conditioned branches, while forward calls count batches. Paper-reported batched NFE and sample evaluations are reported separately. At rank two, one noise draw, and one conditional step per endpoint, contrasts, field probes, the raw output, and the source reference require nine sample evaluations before initialization.

Each candidate reserves all uncached endpoints and its optional prior before generation. When a joint update exceeds the remaining allowance, the controller tries an affordable update that reuses a cached field endpoint or reference and verifies the actual output. Only completed candidates enter final selection. Global edits with an empty protected region continue semantic search with zero protection cost.

Inputs are converted to RGB with EXIF orientation applied consistently to images and masks. Deterministic execution is available through `--deterministic`; reports store the runtime settings and package versions.
