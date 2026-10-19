# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

**RefUnmix** performs training-free image editing with a frozen diffusion or rectified-flow model. It estimates how the generator transforms an editing displacement, fits an endpoint-aligned field, and controls the edit while preserving protected content.

RefUnmix uses one-step or few-step backbones with a finite-query controller. Reports include denoiser evaluations, forward calls, VAE decodes, latency, and peak memory.

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

- **Endpoint-Aligned Field**: Fits the editing displacement to the generator's measured endpoint response in a coupled conditional-contrast subspace.
- **Reference Compensation**: Separates source reconstruction drift from the direct target-conditioned response using matched endpoints.
- **Protected-Response Control**: Adjusts field strength and reference compensation under preservation constraints and a fixed query budget.

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

Dataset partitions remain fixed across seeds. Large flow checkpoints may require more memory than a single RTX 4090; automatic offloading and quantization are not implemented.

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
|   |-- mapping.json
|   `-- images/
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
python -B RefUnmix/doctor.py --model models --clip clip --device cuda --precision fp16
```

This checks dependencies and checkpoint structure without loading model weights.

### 3. Edit One Image

```bash
python -B RefUnmix/main.py --image input.png --source "A person wearing a blue shirt." --target "A person wearing a striped shirt." --model models --clip clip --device cuda --precision fp16 --field pullback --method path --prior off --maxsamples 15 --maxcalls 15 --candidates 12 --seed 42 --no-control --no-heldout --output outputs/result.png --report outputs/result.json
```

Provide source and target image captions through `--source` and `--target`. Each run saves the edited image and a JSON report containing controls, inference costs, and checkpoint hashes.

### 4. Compare Field-Only and Fixed-Reference Modes

```bash
python -B RefUnmix/main.py --image input.png --source "A person wearing a blue shirt." --target "A person wearing a striped shirt." --model models --device cuda --precision fp16 --field pullback --method raw --prior off --seed 42 --output outputs/raw.png --report outputs/raw.json
python -B RefUnmix/main.py --image input.png --source "A person wearing a blue shirt." --target "A person wearing a striped shirt." --model models --device cuda --precision fp16 --field pullback --method fixed --axes none --eta 0 --prior off --seed 42 --output outputs/fixed.png --report outputs/fixed.json
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
| `--prior` | `ddpm` | Optional prior: ddpm, rec, flow, off |
| `--algorithm` | `trust` | Trust-region, poll, or random control search |
| `--geometry` | `output` | Compensation geometry: output, latent, attenuate |
| `--axes` | `all` | Active control coordinates |

See `python -B RefUnmix/main.py --help` for all options. The report records the field used and the completed queries when a budget limit shortens execution.

---

## Dataset Format and Split Provenance

Store annotations as a JSON object or a list of records. Example `data/mapping.json`:

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

With `--dataset data --images images`, `source.png` resolves to `data/images/source.png`. Assign edits of the same source image to one group. Field names and paths are matched case-insensitively.

`prepare.py` converts local **PIE**, **Emu Edit**, **MagicBrush**, **ImgEdit**, and **GEdit** annotations. For instruction-only records, supply a JSON sidecar with source and target captions. Print an example with:

```bash
python -B RefUnmix/prepare.py --kind emu --template
```

### Grouped Partitions

For a custom development split with at least three independent source groups:

```bash
python -B RefUnmix/split.py --dataset data --mapping mapping.json --images images --protocol custom --seed 17 --output outputs/split.json
python -B RefUnmix/doctor.py --dataset data --mapping mapping.json --images images --manifest outputs/split.json --split val --model models --clip clip
```

Custom splits use train/validation/test ratios of 0.70/0.15/0.15. For datasets with official partition tags, use `--protocol official` to preserve them. Source groups are kept disjoint. RefUnmix requires no training on the train partition.

### Four-Seed Dataset Evaluation

```bash
python -B RefUnmix/evaluate.py --dataset data --mapping mapping.json --images images --manifest outputs/split.json --split test --model models --clip clip --device cuda --precision fp16 --prior off --maxsamples 15 --maxcalls 15 --candidates 12 --seeds 42 52 3047 3407 --output outputs/evaluation
```

Use `--split all` without a manifest to evaluate a complete release. Edit-region annotations are used for evaluation; `--oracle` also exposes them to the controller.

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

For equal-budget comparisons, vary `--maxsamples` and report quality alongside forward calls, VAE decodes, latency, and peak memory. The built-in `chord` field is the field-level ablation; full ChordEdit comparisons use its original implementation.

### Independent Evaluation

`evaluate.py` exports edited images and diagnostic scores. Official PIE scoring is run separately through `assess.py` with a metric adapter supplied via `--protocol`.

For independent task assessment, `report.py` creates annotation templates and `blind.py` exports anonymized comparisons. These assessments measure edit completion, preservation, and image quality after output selection, separately from the control scorer.

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

Inputs are converted to RGB with EXIF orientation applied consistently to images and masks. Deterministic execution is available through `--deterministic`; reports store the runtime settings and package versions.
