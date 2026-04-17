import csv
import shutil
from pathlib import Path
from shutil import copy2

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
from ultralytics import YOLO

# ================================================================
# 0. Path configuration
# ================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IMG_DIR = DATA_DIR / "images"

ORIG_IMG_DIR = IMG_DIR / "original"
EDITED_DIR = IMG_DIR / "edited"
LABELED_IOU_DIR = IMG_DIR / "boxes(labeled+iou)"

LABELS_PATH = DATA_DIR / "labels.csv"   # 如果你用 labels-2.csv，请改这里
METRICS_CSV = DATA_DIR / "metrics.csv"

DET_MAX_SIZE = 768
EDIT_MAX_SIZE = 768


# ================================================================
# 1. Label loading and original backup
# ================================================================

def load_labels(labels_path: Path = LABELS_PATH):
    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels csv not found at {labels_path!s}.\n"
            f"Expected labels.csv under the repository 'data' directory."
        )

    rows = []
    with labels_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def backup_originals():
    ORIG_IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_labels()

    for row in rows:
        fn = row["filename"]
        dst = ORIG_IMG_DIR / fn
        if dst.exists():
            continue

        src = IMG_DIR / fn
        if src.exists():
            copy2(src, dst)
            print(f"[backup] copied {src} -> {dst}")
        else:
            print(f"[warning] original image not found for {fn}")


# ================================================================
# 2. Resize helpers
# ================================================================

def resize_keep_ratio(pil_img: Image.Image, max_size: int = 768):
    """
    等比缩放到最长边 <= max_size
    返回:
      resized PIL image, scale_x, scale_y
    其中 new = old * scale
    """
    w, h = pil_img.size
    long_edge = max(w, h)

    if long_edge <= max_size:
        return pil_img, 1.0, 1.0

    scale = max_size / long_edge
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_rs = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return img_rs, scale, scale


def resize_for_edit(pil_img: Image.Image, max_size: int = EDIT_MAX_SIZE) -> Image.Image:
    img_rs, _, _ = resize_keep_ratio(pil_img, max_size=max_size)
    return img_rs


# ================================================================
# 3. YOLO detection
# ================================================================

# COCO: 0=person, 16=dog
det_model = YOLO("yolov8l.pt")


def detect_subject_on_resized(pil_img: Image.Image, max_size: int = DET_MAX_SIZE):
    """
    在统一的 resized 坐标系中做检测。
    返回:
      cls_id, bbox_xyxy(resized坐标系), conf, sx, sy, resized_pil
    """
    img_rs, sx, sy = resize_keep_ratio(pil_img, max_size=max_size)
    img_bgr = cv2.cvtColor(np.array(img_rs), cv2.COLOR_RGB2BGR)

    results = det_model(img_bgr)[0]

    best_box = None
    best_area = -1.0
    best_cls = None
    best_conf = None

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id not in [0, 16]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area <= 0:
            continue

        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)
            best_cls = cls_id
            best_conf = float(box.conf[0].item())

    if best_box is None:
        return None, None, None, sx, sy, img_rs

    return best_cls, best_box, best_conf, sx, sy, img_rs


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

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0:
        return 0.0

    return interArea / denom


def map_box_resized_to_original(box, sx, sy):
    """
    把 resized 坐标系的框映射回原图坐标系。
    new = old * scale  => old = new / scale
    """
    if box is None:
        return None

    if sx == 0 or sy == 0:
        return None

    x1, y1, x2, y2 = box
    return (
        int(x1 / sx),
        int(y1 / sy),
        int(x2 / sx),
        int(y2 / sy),
    )


# ================================================================
# 4. Visualization
# ================================================================

def draw_box_with_label(img, box, cls_id, conf):
    if box is None:
        return img

    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if cls_id is not None:
        label = det_model.names.get(int(cls_id), str(cls_id))
        text = label if conf is None else f"{label} {conf*100:.1f}%"

        font_scale = 0.45
        font_thickness = 1
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        y_text = max(y1 - th - 4, 0)

        cv2.rectangle(
            img,
            (x1, y_text),
            (x1 + tw + 4, y_text + th + baseline + 2),
            (0, 255, 0),
            thickness=-1,
        )

        cv2.putText(
            img,
            text,
            (x1 + 2, y_text + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    return img


def draw_iou_text(img, iou):
    if iou is None:
        return img

    text = f"IoU: {iou*100:.1f}%"
    font_scale = 0.6
    font_thickness = 1

    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )

    cv2.rectangle(
        img,
        (0, 0),
        (tw + 8, th + baseline + 6),
        (0, 255, 0),
        thickness=-1,
    )

    cv2.putText(
        img,
        text,
        (4, th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        font_thickness,
        cv2.LINE_AA,
    )

    return img


def visualize_boxes(
    orig_path: Path,
    edited_img: Image.Image,
    orig_box_vis,
    edit_box_vis,
    orig_cls=None,
    edit_cls=None,
    orig_conf=None,
    edit_conf=None,
    iou=None,
    out_prefix="debug",
):
    """
    保存一张横向拼接图:
      [original with bbox+label+iou | edited with bbox+label+iou]
    """
    LABELED_IOU_DIR.mkdir(parents=True, exist_ok=True)

    orig_pil = Image.open(str(orig_path)).convert("RGB")
    orig = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR)

    edit = cv2.cvtColor(np.array(edited_img), cv2.COLOR_RGB2BGR)

    orig = draw_box_with_label(orig, orig_box_vis, orig_cls, orig_conf)
    edit = draw_box_with_label(edit, edit_box_vis, edit_cls, edit_conf)

    orig = draw_iou_text(orig, iou)
    edit = draw_iou_text(edit, iou)

    h0, w0 = orig.shape[:2]
    edit_vis = cv2.resize(edit, (w0, h0), interpolation=cv2.INTER_LINEAR)

    combined = np.hstack([orig, edit_vis])

    out_path = LABELED_IOU_DIR / f"{out_prefix}.jpg"
    cv2.imwrite(str(out_path), combined)


def structural_score(orig_path: Path, edited_img: Image.Image):
    """
    关键修正：
    IoU 在“统一 resized 坐标系”里计算，而不是直接拿原图坐标和编辑图坐标硬比。
    """
    orig_pil = Image.open(str(orig_path)).convert("RGB")
    edit_pil = edited_img

    orig_cls, orig_box_rs, orig_conf, sx_o, sy_o, _ = detect_subject_on_resized(orig_pil)
    edit_cls, edit_box_rs, edit_conf, sx_e, sy_e, _ = detect_subject_on_resized(edit_pil)

    print(f"[DEBUG] orig_box_rs={orig_box_rs}, edit_box_rs={edit_box_rs}")

    if orig_cls is None or edit_cls is None:
        iou = 0.0
    elif orig_cls != edit_cls:
        iou = 0.0
    else:
        iou = bbox_iou(orig_box_rs, edit_box_rs)

    print(f"[DEBUG] orig_cls={orig_cls}, edit_cls={edit_cls}, IoU={iou:.4f}")

    # 原图可视化：把 resized 坐标映射回原图
    orig_box_vis = map_box_resized_to_original(orig_box_rs, sx_o, sy_o)

    # 编辑图可视化：直接画 resized 坐标即可（edited_img 本身就是当前编辑尺寸）
    edit_box_vis = edit_box_rs

    visualize_boxes(
        orig_path=orig_path,
        edited_img=edited_img,
        orig_box_vis=orig_box_vis,
        edit_box_vis=edit_box_vis,
        orig_cls=orig_cls,
        edit_cls=edit_cls,
        orig_conf=orig_conf,
        edit_conf=edit_conf,
        iou=iou,
        out_prefix=orig_path.stem,
    )

    return iou, iou, orig_cls, edit_cls


# ================================================================
# 5. InstructPix2Pix pipeline
# ================================================================

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=dtype,
)
pipe.to(device)


def dummy_safety_checker(images, clip_input):
    return images, [False] * len(images)


pipe.safety_checker = dummy_safety_checker
pipe.enable_attention_slicing()


def load_image_original_size(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def edit_once(img_path: Path, prompt: str) -> Image.Image:
    """
    编辑阶段等比缩小，避免 OOM；检测阶段会在统一 resized 坐标系下做。
    """
    image = load_image_original_size(img_path)
    image = resize_for_edit(image, max_size=EDIT_MAX_SIZE)

    result = pipe(
        prompt,
        image=image,
        num_inference_steps=30,
        image_guidance_scale=1.5,
        guidance_scale=7.5,
    )
    return result.images[0]


# ================================================================
# 6. Closed-loop prompt refinement
# ================================================================

def refine_prompt(base_prompt: str, scores: dict[str, float]) -> str:
    p = base_prompt
    if scores.get("struct", 0.0) < 0.5 and "keep the main subject exactly the same" not in p:
        p = base_prompt + ", keep the main subject exactly the same"
    return p


# ================================================================
# 7. One image closed-loop editing
# ================================================================

def iterative_edit(row: dict, max_iter: int = 1, threshold: float = 0.5):
    filename = row["filename"]
    obj = row["object"]
    action = row["action"]
    bg = row["background"]

    img_path = ORIG_IMG_DIR / filename

    base_prompt = (
        f"replace the current {bg} background with a different scene, "
        f"while keeping the {obj} and its {action} pose unchanged, "
        f"natural lighting, realistic photo"
    )
    p_t = base_prompt

    best_img = None
    best_score = -1.0
    best_iou = 0.0
    best_orig_cls = None
    best_edit_cls = None

    for t in range(max_iter):
        print(f"\nIteration {t} for {filename}")

        y_t = edit_once(img_path, p_t)

        s_struct, iou, o_cls, e_cls = structural_score(img_path, y_t)
        S_t = s_struct

        print(f"  structural score: {s_struct:.3f}")

        if S_t > best_score:
            best_score = S_t
            best_iou = iou
            best_orig_cls = o_cls
            best_edit_cls = e_cls
            best_img = y_t

        if S_t >= threshold:
            break

        p_t = refine_prompt(base_prompt, {"struct": s_struct})

    EDITED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EDITED_DIR / filename.replace(".jpg", "_edited.jpg")
    best_img.save(out_path)
    print(f"Saved best edit to {out_path}, score={best_score:.3f}")

    write_header = not METRICS_CSV.exists()
    with METRICS_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "filename",
                    "object",
                    "action",
                    "background",
                    "score",
                    "iou",
                    "orig_cls",
                    "edit_cls",
                ]
            )
        writer.writerow(
            [
                filename,
                obj,
                action,
                bg,
                f"{best_score:.4f}",
                f"{best_iou:.4f}",
                best_orig_cls,
                best_edit_cls,
            ]
        )


# ================================================================
# 8. Main
# ================================================================

def main():
    backup_originals()

    # 删除旧 metrics
    if METRICS_CSV.exists():
        METRICS_CSV.unlink()
        print(f"[info] removed old metrics: {METRICS_CSV}")

    # 删除旧 labeled+iou 文件夹，确保这次结果全新
    shutil.rmtree(LABELED_IOU_DIR, ignore_errors=True)
    print(f"[info] removed old folder: {LABELED_IOU_DIR}")

    rows = load_labels()
    print(f"Loaded {len(rows)} labeled images")

    for row in rows:
        iterative_edit(row, max_iter=1, threshold=0.5)


if __name__ == "__main__":
    main()