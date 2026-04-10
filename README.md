# Self-Improving Image Editor

A small prototype for a **closed-loop, self-improving image editing system**.  
The goal is to change the **background** of an image (e.g. river ↔ mountain) while **preserving the main subject and pose** (adult person / dog sitting or standing).

The system combines:

- **InstructPix2Pix** (Diffusers) for text-guided image editing  
- **YOLOv8** for subject detection and structure scoring  
- A simple **closed-loop controller** that evaluates each edit and can refine prompts

---

## 1. Project structure

Key files:

- `src/closed_loop_editor.py`  
  - Main closed-loop script: loads labels, runs InstructPix2Pix, evaluates edits with YOLO, and exports metrics to CSV.
- `data/labels.csv`  
  - 15 labeled images with columns: `filename, object, action, background`.
  - `object ∈ {adult_person, dog}`, `action ∈ {sit, stand}`, `background ∈ {river, mountain}`.
- `data/images/`  
  - Original images.
  - `edited/` subfolder: edited results.
  - `debug_boxes/` subfolder: original & edited images with YOLO bounding boxes overlaid.
- `data/metrics.csv`  
  - Per-image scores exported by the closed-loop script.

---

## 2. Environment setup (Colab / local GPU)

### 2.1 Clone repo

```bash
git clone https://github.com/Sherry1247/self_improving_editor.git
cd self_improving_editor
```

### 2.2 Python & dependencies

Tested in **Python 3.12** on Google Colab with CUDA 12.1.

Install PyTorch (CUDA) and core libraries:

```bash
# GPU PyTorch (Colab CUDA 12.1)
pip uninstall -y torch torchvision torchaudio
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Diffusion + detection stack
pip install -q "diffusers==0.30.3" "transformers==4.45.2" "huggingface_hub==0.25.2" accelerate
pip install -q ultralytics opencv-python pillow
```

> Note: the versions above are chosen to avoid common compatibility issues between `diffusers`, `transformers`, and `huggingface_hub`.[web:247]

On Colab, after installing, restart the runtime once, then verify GPU:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
```

---

## 3. Running the closed-loop editor

From the repo root:

```bash
cd self_improving_editor
python src/closed_loop_editor.py
```

What this script does:

1. Loads all rows from `data/labels.csv` (currently 15 images).
2. For each image:
   - Builds a text prompt:
     - `"change the background {background} to something different, keep the {object} and its {action} pose unchanged"`  
     - e.g. `"change the background river to something different, keep the adult_person and its sit pose unchanged"`.
   - Runs InstructPix2Pix for 30 denoising steps on a 384×384 resized version of the image.
3. Uses **YOLOv8l** (`yolov8l.pt`) to detect the main subject (person or dog) on:
   - original image (resized to 384×384),
   - edited image.
4. Computes a **structural score**:
   - Finds the best person/dog bounding box in both images.
   - Computes IoU between the original and edited bounding boxes.
   - Uses IoU as the structural score.
5. Saves:
   - Edited image to `data/images/edited/<filename>_edited.jpg`.
   - Debug visualizations with bounding boxes to `data/images/debug_boxes/`.
   - Metrics for each image to `data/metrics.csv`:
     - `filename, object, action, background, score, iou, orig_cls, edit_cls`.

The script overwrites `metrics.csv` on each run so that it always reflects the latest experiment.

---

## 4. Closed-loop design (current prototype)

The current loop is:

1. **Edit**:  
   \( y_t = \text{InstructPix2Pix}(x, p_t) \)

2. **Evaluate** structural consistency with YOLO:  
   - Detect subject bbox on original & edited.  
   - Compute IoU as structural score \( S_t \).

3. **(Optional) Refine prompt** (currently simple rule):  
   If \( S_t < 0.5 \), append  
   > ", keep the main subject exactly the same"  
   to the prompt and re-edit.

At the moment `max_iter=1`, so we effectively run a **single-loop pass**. The code is written so we can easily increase `max_iter` and plug in a more sophisticated LLM-based prompt refiner in the future.

---

## 5. Next steps (planned)

- Add **background-change scoring** using subject/background masks and CLIP embeddings.
- Extend the closed-loop controller to:
  - combine structural and background scores,  
  - ask an LLM (or rules) to refine prompts when scores are low.
- Explore **segmentation-based masks** for better subject protection and controlled background editing.
