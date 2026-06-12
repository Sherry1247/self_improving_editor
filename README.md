# Self-Improving Image Editor

A modular **closed-loop image editing research framework**. The goal is to change the **background** of an image (e.g. river ↔ mountain) while **preserving the main subject and pose** (adult person / dog sitting or standing).

The system combines:

- **InstructPix2Pix** (Diffusers) for text-guided image editing
- **YOLOv8** for subject detection and object-consistency scoring
- **CLIP** for semantic similarity and instruction-alignment scoring
- A **closed-loop controller** that evaluates each edit, refines prompts, and re-generates until a score threshold is reached

---

## 1. Project structure

```
self_improving_editor/
├── main.py                          # Primary entry point (full dataset)
├── README.md
├── data/
│   ├── labels.csv                   # 15 labeled images (filename, object, action, background)
│   ├── metrics.csv                  # Legacy CSV export (optional)
│   ├── images/
│   │   ├── original/                # Source images
│   │   └── edited/                  # Best edited outputs
│   └── experiments/                 # JSON experiment results (auto-created)
├── src/
│   ├── closed_loop_editor.py        # Legacy wrapper (backward-compatible)
│   ├── configs/
│   │   ├── default.yaml             # Hyperparameters, model paths, critic weights
│   │   └── __init__.py              # load_config(), build_pipeline()
│   ├── detectors/
│   │   ├── base.py                  # Detector ABC
│   │   └── yolo_detector.py         # YOLOv8 person/dog detection
│   ├── editors/
│   │   ├── base.py                  # Editor ABC
│   │   └── pix2pix_editor.py        # InstructPix2Pix wrapper
│   ├── critics/
│   │   ├── base.py                  # Critic ABC
│   │   ├── object_consistency.py    # IoU-based structural scoring
│   │   ├── clip_similarity.py       # CLIP image-image similarity
│   │   ├── instruction_alignment.py # CLIP text-image alignment
│   │   ├── composite.py             # Weighted score combination
│   │   └── clip_utils.py            # Shared CLIP model backend
│   ├── prompts/
│   │   └── refinement.py            # Rule-based prompt refinement
│   ├── pipelines/
│   │   └── closed_loop_pipeline.py  # Generate → Evaluate → Refine loop
│   ├── utils/
│   │   ├── io_utils.py              # JSON/CSV I/O, image loading
│   │   ├── visualization.py         # Bounding-box overlays, comparisons
│   │   ├── logging_config.py        # Structured logging
│   │   └── device_utils.py          # Device detection, seeding
│   └── requirements.txt
├── experiments/
│   └── run_single.py                # Single-job runner (CHTC / batch)
├── jobs/
│   └── jobs.csv                     # Job definitions for batch execution
├── chtc/
│   ├── run.sh                       # HTCondor execute script
│   └── submit.sub                   # HTCondor submit file
└── logs/                            # Condor and job logs
```

### Data labels

`data/labels.csv` contains 15 images with columns: `filename, object, action, background`.

- `object ∈ {adult_person, dog}`
- `action ∈ {sit, stand}`
- `background ∈ {river, mountain}`

---

## 2. Architecture

### Closed-loop pipeline

```
Original Image → Editor (Pix2Pix) → Edited Image → Composite Critic
                                                        ↓
                                              score ≥ threshold?
                                                   ↓         ↓
                                                  Yes        No
                                                   ↓         ↓
                                            Save results   Refine prompt → loop
```

### Composite critic

The final score is a weighted combination of three critics (configurable in `src/configs/default.yaml`):

```
final_score = 0.4 × object_consistency
            + 0.3 × clip_similarity
            + 0.3 × instruction_alignment
```

| Critic | Method |
|--------|--------|
| `object_consistency` | YOLO bbox IoU between original and edited images |
| `clip_similarity` | CLIP image-image cosine similarity (semantic preservation) |
| `instruction_alignment` | CLIP text-image similarity (prompt adherence) |

### Abstract interfaces

All components implement ABCs for easy swapping:

- `Detector` — object localization (`src/detectors/base.py`)
- `Editor` — text-guided editing (`src/editors/base.py`)
- `Critic` — image quality scoring (`src/critics/base.py`)

---

## 3. Environment setup

### 3.1 Clone repo

```bash
git clone https://github.com/Sherry1247/self_improving_editor.git
cd self_improving_editor
```

### 3.2 Python & dependencies

Tested in **Python 3.12** with CUDA 12.1 (Colab / local GPU).

```bash
# GPU PyTorch (CUDA 12.1)
pip uninstall -y torch torchvision torchaudio
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core stack
pip install -r src/requirements.txt
```

Verify GPU:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
```

### 3.3 Configuration

Edit `src/configs/default.yaml` to adjust:

- `pipeline.max_iterations` / `pipeline.score_threshold`
- `critics.weights` (must sum to 1.0)
- `editor` settings (steps, guidance scales, image size)
- `output.base_dir` for experiment results

---

## 4. Running experiments

### Full dataset (recommended)

```bash
python main.py --limit 15 --export-csv
```

Options:

| Flag | Description |
|------|-------------|
| `--config PATH` | Custom YAML config |
| `--labels PATH` | Custom labels CSV |
| `--limit N` | Process first N images |
| `--export-csv` | Also write `data/metrics.csv` |
| `--no-backup` | Skip copying originals |

Results are saved to `data/experiments/<experiment_id>/`:

- `metadata.json` — full experiment record
- `iterations/NNN.json` — per-iteration scores and prompts
- `images/final_edit.jpg` — best edited image
- `images/compare_NNN.jpg` — side-by-side bbox visualizations

### Legacy script (backward-compatible)

```bash
python src/closed_loop_editor.py
```

Preserves the original behavior: processes 15 images, writes `data/metrics.csv` and `data/images/edited/`.

### Single job (local or batch)

```bash
python experiments/run_single.py \
  --job-id job_0001 \
  --filename adult_sit_river_01.jpg \
  --object adult_person \
  --action sit \
  --background river
```

---

## 5. CHTC / HTCondor batch execution

For large-scale evaluation across thousands of image-background combinations:

1. Expand `jobs/jobs.csv` with one row per job (`job_id, filename, object, action, background`).
2. Submit to HTCondor:

```bash
mkdir -p logs
condor_submit chtc/submit.sub
```

Each job runs independently via `chtc/run.sh` → `experiments/run_single.py`, writing results to `data/experiments/chtc/<job_id>/`.

To scale beyond the starter 15 jobs, add rows to `jobs/jobs.csv` (e.g. cartesian product of images × backgrounds) and resubmit.

---

## 6. Extending the framework

Swap components without modifying the pipeline:

```python
from src.configs import build_pipeline

pipeline = build_pipeline()  # wires detector, editor, critics from YAML
result = pipeline.run(
    original_image,          # BGR numpy array
    base_prompt="replace the background...",
    metadata={"filename": "example.jpg"},
)
print(result["best_score"], result["experiment_id"])
```

Add a new critic by subclassing `Critic`, registering it in `build_pipeline()`, and updating weights in `default.yaml`.

---

## 7. Prompt refinement

When scores are below threshold, the rule-based refiner (`src/prompts/refinement.py`) appends constraints:

- Low `object_consistency` → `", keep the main subject exactly the same"`
- Low `instruction_alignment` → `", follow the instruction precisely"`
- Low `clip_similarity` → `", preserve the original appearance of the subject"`

This can be replaced with an LLM-based refiner in the future.
