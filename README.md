# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

**RefUnmix** is a training-free image editing framework that aligns an editing field with the response of a frozen generator and controls the resulting edit under content-preservation constraints. It constructs a coupled pullback field from matched conditional contrasts, separates reference-dependent changes using matched endpoints, and selects an edit through finite-query response control.

The implementation provides native diffusion and rectified-flow interfaces, field and controller ablations, grouped dataset partitions, and independent evaluation interfaces. A one-step or few-step generator is the backbone; the complete controlled editor uses multiple queries. Candidate count, denoiser sample evaluations, forward calls, decoding, latency, and memory are recorded separately.

---

## Architecture

```text
Source image + source caption + target caption
                         |
           Frozen image and text encoders
                         |
        Matched conditional contrasts at two times
                         |
         Endpoint-aligned pullback editing field
                         |
       Matched endpoints and reference compensation
                         |
     Finite-query protected-response editing control
                         |
                Frozen decoder -> Edited image
```

### Key Contributions

- **Endpoint-Aligned Editing Field**: Fits a coupled inverse response in the conditional-contrast subspace so that an input displacement better matches the requested output displacement.
- **Reference Compensation**: Uses matched native endpoints to distinguish source reconstruction drift from the direct target-conditioned response.
- **Protected-Response Control**: Selects field strength and reference compensation from observed generator responses, with trust-region proposals, curvature monitoring, and explicit query budgets.
- **Reproducible Evaluation**: Separates controller scores from final evaluation, freezes validation configurations before testing, and records checkpoint, source, and output hashes.

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

All Python filenames contain lowercase letters only. Source folders are Python namespace packages without `__init__.py`. Every Python file has a UTF-8 declaration, a project title, and an author footer. Single blank lines separate logical code blocks.

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

Install a CUDA-compatible PyTorch build for your system, then install the remaining dependencies in a dedicated environment:

```bash
python -m pip install numpy pillow scipy diffusers transformers accelerate safetensors sentencepiece protobuf
```

Optional dataset and interface dependencies:

```bash
python -m pip install datasets pyarrow gradio
```

PyTorch must be installed separately if it is not already available. The runtime report records the actual package versions used for each experiment.

### Hardware and Random Seeds

The reference experiment setup uses **one NVIDIA GeForce RTX 4090** and random seeds **42, 52, 3047, and 3407**. Single-image inference defaults to seed 42. Dataset evaluation and trials accept all four through `--seeds 42 52 3047 3407`.

Use the same source groups, seeds, model weights, precision, and budgets across compared methods. Keep the dataset partition seed separate from inference seeds. Large flow checkpoints may exceed single-card memory; the interface does not automatically offload or quantize them.

---

## Pretrained Models

No RefUnmix training stage is required. Download the frozen backbone and control scorer separately, following their respective access and usage terms. Native loading is local-only and does not silently download missing weights.

For `--backbone vp`, place a compatible SD-Turbo-style diffusion checkpoint in `models/` with its native `unet/`, `vae/`, `scheduler/`, `tokenizer/`, and `text_encoder/` components. A `text/` folder is also supported. A bare UNet weight file is insufficient.

For controlled editing, place a compatible Transformers CLIP model and processor in `clip/`. Raw-field and fixed-reference modes do not require a CLIP control model. A report-only CLIP scorer, if used, must have different checkpoint weights from the controller.

`--backbone flux` and `--backbone sd3` load their corresponding complete native pipelines. They require the actual transformer, VAE, scheduler, tokenizers, and text encoders for that architecture. Changing this flag does not convert a diffusion checkpoint into a flow checkpoint or distill a multistep model into a one-step model.

### Workspace Layout

Keep runtime assets outside the source checkout:

```text
workspace/
|-- RefUnmix/           # Source checkout
|-- models/             # Frozen backbone checkpoint
|-- clip/               # Frozen control checkpoint
|-- data/
|   |-- mapping.json
|   `-- images/
|-- input.png
`-- outputs/            # Generated results and reports
```

All configured paths are relative to the current working directory. Absolute paths and parent-directory escapes are rejected. **Run the following inference commands from `workspace/`, not from inside `RefUnmix/`.** Use a new output name for every run; existing outputs are not overwritten.

---

## Quick Start

### 1. Clone the Source

```bash
git clone https://github.com/wlz05/RefUnmix.git
```

Stay in the parent workspace and prepare `models/`, `clip/`, and `input.png` as described above.

### 2. Check Local Inputs

```bash
python -B RefUnmix/doctor.py --model models --clip clip --device cuda --precision fp16
```

The preflight checks local dependencies and checkpoint structure without performing model inference.

### 3. Edit One Image

```bash
python -B RefUnmix/main.py --image input.png --source "A person wearing a blue shirt." --target "A person wearing a striped shirt." --model models --clip clip --device cuda --precision fp16 --field pullback --method path --prior off --maxsamples 15 --maxcalls 15 --candidates 12 --seed 42 --no-control --no-heldout --output outputs/result.png --report outputs/result.json
```

`--source` and `--target` are captions of the source and desired image, not interchangeable with a free-form edit instruction. The JSON report includes the selected controls, actual cost, checkpoint metadata, and reproducibility information.

### 4. Compare Field-Only and Fixed-Reference Modes

```bash
python -B RefUnmix/main.py --image input.png --source "A person wearing a blue shirt." --target "A person wearing a striped shirt." --model models --device cuda --precision fp16 --field pullback --method raw --prior off --seed 42 --output outputs/raw.png --report outputs/raw.json
python -B RefUnmix/main.py --image input.png --source "A person wearing a blue shirt." --target "A person wearing a striped shirt." --model models --device cuda --precision fp16 --field pullback --method fixed --axes none --eta 0 --prior off --seed 42 --output outputs/fixed.png --report outputs/fixed.json
```

The examples explicitly disable optional consistency priors. Enable them only for the corresponding supported backend and include their queries in the total budget.

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
| `--prior` | `ddpm` | Optional prior: ddpm, rec, flow, off |
| `--algorithm` | `trust` | Trust-region, poll, or random control search |
| `--geometry` | `output` | Compensation geometry: output, latent, attenuate |
| `--axes` | `all` | Active control coordinates |

Run `python -B RefUnmix/main.py --help` for the complete interface. A low hard budget can trigger the documented field fallback or shorten search; inspect the recorded actual field and cost rather than equating a candidate budget with completed queries.

---

## Dataset Format and Split Provenance

The native mapping accepts an object or a list of records. For example, `data/mapping.json` can contain:

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

With `--dataset data --images images`, the image resolves to `data/images/source.png`. Records derived from the same source must share a group. Field names and path matching are case-insensitive; ambiguous case collisions are rejected. Caption content is preserved.

`prepare.py` provides local conversion interfaces for **PIE**, **Emu Edit**, **MagicBrush**, **ImgEdit**, and **GEdit**. It does not download datasets. Instruction-only records require a source/target caption sidecar; inspect its format with:

```bash
python -B RefUnmix/prepare.py --kind emu --template
```

### Grouped Partitions

For a custom development split with at least three independent source groups:

```bash
python -B RefUnmix/split.py --dataset data --mapping mapping.json --images images --protocol custom --seed 17 --output outputs/split.json
python -B RefUnmix/doctor.py --dataset data --mapping mapping.json --images images --manifest outputs/split.json --split val --model models --clip clip
```

The custom default ratios are 0.70/0.15/0.15. They are a development protocol, not an official PIE-Bench split. Use `--protocol official` only when the input supplies official partition tags; the implementation preserves those tags and checks source-group leakage. A train partition is a dataset interface, not a requirement to train RefUnmix.

### Four-Seed Dataset Evaluation

```bash
python -B RefUnmix/evaluate.py --dataset data --mapping mapping.json --images images --manifest outputs/split.json --split test --model models --clip clip --device cuda --precision fp16 --prior off --maxsamples 15 --maxcalls 15 --candidates 12 --seeds 42 52 3047 3407 --output outputs/evaluation
```

Use `--split all` without a manifest when evaluating a complete benchmark release. Official edit-region annotations are evaluation inputs; exposing them to the controller requires the explicit `--oracle` option and should be reported separately.

---

## Experiments

### Field and Controller Matrix

The `fields` suite compares pullback, chord, instant, and average fields, each with raw, fixed-reference, and full-controller configurations. The `alignment` suite compares coupled and scalar norm-matched field construction with matched probe cost.

Freeze a validation configuration, then use its recipe on disjoint test sources:

```bash
python -B RefUnmix/trial.py --dataset data --mapping mapping.json --images images --manifest outputs/split.json --split val --suite fields --model models --clip clip --device cuda --precision fp16 --prior off --maxsamples 15 --maxcalls 15 --candidates 12 --seeds 42 52 3047 3407 --no-control --no-heldout --output outputs/fieldval
python -B RefUnmix/trial.py --dataset data --mapping mapping.json --images images --manifest outputs/split.json --split test --suite fields --recipe outputs/fieldval/recipe.json --model models --clip clip --device cuda --precision fp16 --prior off --maxsamples 15 --maxcalls 15 --candidates 12 --seeds 42 52 3047 3407 --no-control --no-heldout --output outputs/fieldtest
```

Compare matched full-controller outputs:

```bash
python -B RefUnmix/evidence.py --reference outputs/fieldtest/chordfull/summary.json --comparison outputs/fieldtest/pullbackfull/summary.json --contrast field --output outputs/comparison.json
```

Report the equal-budget frontier across sample-evaluation limits, together with actual forward calls, VAE decodes, wall-clock latency, and peak memory. The built-in `chord` field is a matched field ablation, not a replacement for reproducing the original ChordEdit release under its own protocol.

### Independent Evaluation

`evaluate.py` exports per-image outputs and diagnostic summaries. Its pixel, frequency, and controller scores are not official PIE metric implementations.

`assess.py` loads a separately supplied, verified official PIE metric adapter through `--protocol`. `report.py` generates independent task-annotation templates, and `blind.py` exports anonymized comparisons. Use these interfaces to report edit completion, protected-content preservation, and image quality after output selection. Do not reuse controller scores as independent evidence of task completion.

### Tests

Run the synthetic-backend regression suite from the repository root:

```bash
cd RefUnmix
python -B -m unittest discover -s tests -p '*.py' -q
```

Unit tests check algebra, constraints, budgets, image formats, dataset conversion, and fallback behavior. Real-image experiments additionally require the local checkpoints and benchmark data described above.

---

## Implementation Details

- All backbone parameters remain frozen. Field construction uses paired conditional predictions and native endpoint probes, not additional model training.
- The main path uses a coupled low-dimensional inverse fit, matched reference endpoints, and response-aware control. Legacy frequency-only experiments remain available in separate CLI modes and are not the default method.
- The default contrast times are 0.90 and 0.75. They are configurable noise coordinates, not universal optimal times or image-frequency labels. Flow backends use their declared sigma/velocity convention.
- Candidate evaluations, batched denoiser samples, forward calls, and continuation steps are different quantities. Optional priors and probes add work and are included in the recorded accounting.
- Image inputs use explicit RGB conversion and aligned EXIF handling. Masks require declared geometry; ambiguous labels and malformed RLE are rejected.
- Deterministic mode records its runtime controls. Cross-device or cross-version bitwise equality is not implied by using the same seed.
