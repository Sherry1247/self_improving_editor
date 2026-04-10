import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image
from ultralytics import YOLO


# Resolve data directory relative to this script file so the script works
# no matter which directory the user runs it from.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMG_DIR = DATA_DIR / "images"
LABELS_PATH = DATA_DIR / "labels.csv"


# ---------- 1. Load labels ----------

def load_labels(labels_path=LABELS_PATH):
    # Helpful error if labels file cannot be found. Print the resolved path
    # to make debugging easier for users running from a different CWD.
    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels.csv not found at {labels_path!s}.\n"
            f"Expected labels.csv under the repository 'data' directory.\n"
            f"If you intended to run from the src directory, run: cd src && python closed_loop_editor.py\n"
        )

    rows = []
    with labels_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------- 2. YOLO detection and structural score ----------

det_model = YOLO("yolov8n.pt")  # small, COCO-pretrained[web:107]


def detect_subject(img_path_or_array):
    """Return (cls_id, bbox_xyxy) for the strongest person/dog detection, or (None, None)."""
    results = det_model(img_path_or_array)[0]
    best = None
    best_conf = -1.0
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        # COCO: 0=person, 16=dog
        if cls_id in [0, 16]:
            conf = float(box.conf[0].item())
            if conf > best_conf:
                best_conf = conf
                best = box
    if best is None:
        return None, None
    x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
    return int(best.cls[0].item()), (x1, y1, x2, y2)


def bbox_iou(boxA, boxB):
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def structural_score(orig_path, edited_img):
    """0–1 score: subject class preserved and bbox IoU."""
    # Detect on original
    orig_cls, orig_box = detect_subject(str(orig_path))

    # Detect on edited (convert PIL to OpenCV BGR array)
    edited_bgr = cv2.cvtColor(np.array(edited_img), cv2.COLOR_RGB2BGR)
    edit_cls, edit_box = detect_subject(edited_bgr)

    if orig_cls is None or edit_cls is None:
        return 0.0

    # Require same subject type (person vs dog)
    if orig_cls != edit_cls:
        return 0.0

    iou = bbox_iou(orig_box, edit_box)
    # Map IoU to [0,1] with a soft threshold
    return max(0.0, min(1.0, iou / 0.7))


# ---------- 3. InstructPix2Pix editor ----------

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)  # pretrained editor[web:99][web:100]


def edit_once(img_path, prompt):
    image = load_image(str(img_path))
    result = pipe(
        prompt,
        image=image,
        num_inference_steps=20,
        image_guidance_scale=1.5,
        guidance_scale=7.5,
    )
    return result.images[0]


# ---------- 4. Simple closed loop (2 iterations max) ----------

def refine_prompt(base_prompt, scores):
    """Very simple heuristic 'LLM': add constraints based on low structural score."""
    p = base_prompt
    if scores["struct"] < 0.5:
        if "keep the subject" not in p:
            p = base_prompt + ", keep the main subject exactly the same"
    return p


def iterative_edit(row, max_iter=2, threshold=0.7):
    filename = row["filename"]
    obj = row["object"]
    action = row["action"]
    bg = row["background"]

    img_path = IMG_DIR / filename

    # Base prompt: change background, keep subject and pose
    base_prompt = f"change the background {bg} to something different, keep the {obj} and its {action} pose unchanged"
    p_t = base_prompt

    best_img = None
    best_score = -1.0

    for t in range(max_iter):
        print(f"\nIteration {t} for {filename}")
        y_t = edit_once(img_path, p_t)

        s_struct = structural_score(img_path, y_t)
        # TODO: add semantic / realism scores later
        S_t = s_struct
        print(f"  structural score: {s_struct:.3f}")

        if S_t > best_score:
            best_score = S_t
            best_img = y_t

        if S_t >= threshold:
            break

        p_t = refine_prompt(base_prompt, {"struct": s_struct})

    # Save best image
    out_dir = IMG_DIR / "edited"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / filename.replace(".jpg", "_edited.jpg")
    best_img.save(out_path)
    print(f"Saved best edit to {out_path}, score={best_score:.3f}")


# ---------- 5. Main ----------

def main():
    rows = load_labels()
    print(f"Loaded {len(rows)} labeled images")

    for row in rows:
        iterative_edit(row, max_iter=2, threshold=0.6)


if __name__ == "__main__":
    main()
