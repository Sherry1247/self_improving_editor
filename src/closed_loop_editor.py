import csv
import os
from pathlib import Path
from shutil import copy2

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
from ultralytics import YOLO

# ---------- 0. 路径配置 ----------

# 本文件: self_improving_editor/src/closed_loop_editor.py
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMG_DIR = DATA_DIR / "images"             # 顶层 images 目录
ORIG_IMG_DIR = IMG_DIR / "original"       # 原始图片统一放这里
EDITED_DIR = IMG_DIR / "edited"           # 编辑后的图片
BOXES_ROOT = IMG_DIR / "boxes"            # 带框可视化
BOXES_ORIG_DIR = BOXES_ROOT / "orig"
BOXES_EDIT_DIR = BOXES_ROOT / "edit"

LABELS_PATH = DATA_DIR / "labels.csv"     # 如果你用 labels-2.csv，改成那个名字
METRICS_CSV = DATA_DIR / "metrics.csv"    # 每次运行会覆盖


# ---------- 1. 读取 labels.csv ----------

def load_labels(labels_path=LABELS_PATH):
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


# ---------- 2. 备份原始图片到 data/images/original ----------

def backup_originals():
    """
    确保所有在 labels 里的文件名，在 data/images/original/ 下都有一份。
    如果 original 里还没有，但 images/ 下有，就 copy 一份过去。
    不会删除 / 覆盖任何东西。
    """
    ORIG_IMG_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_labels()
    for row in rows:
        fn = row["filename"]
        dst = ORIG_IMG_DIR / fn
        if dst.exists():
            continue

        # 原来散装的路径：data/images/filename
        src = IMG_DIR / fn
        if src.exists():
            copy2(src, dst)
            print(f"[backup] copied {src} -> {dst}")
        else:
            print(f"[warning] cannot find original image for {fn}")


# ---------- 3. YOLOv8 检测 + 结构评分 ----------

# 0=person, 16=dog (COCO); 用 l 模型做检测
det_model = YOLO("yolov8l.pt")


def detect_subject(img_path_or_array, resize_to=None):
    """
    返回 (cls_id, bbox_xyxy)，只取置信度最高的 person/dog。
    如果传的是路径，可以先 resize 到固定尺寸再给 YOLO，
    这样原图和编辑图坐标系一致，IoU 才有意义。
    """
    if isinstance(img_path_or_array, (str, Path)):
        img = Image.open(str(img_path_or_array)).convert("RGB")
        if resize_to is not None:
            img = img.resize((resize_to, resize_to), Image.LANCZOS)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        source = img
    else:
        source = img_path_or_array

    results = det_model(source)[0]
    best = None
    best_conf = -1.0
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id in [0, 16]:  # person 或 dog
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


def visualize_boxes(orig_path, edited_img, orig_box, edit_box,
                    orig_cls=None, edit_cls=None, out_prefix="debug"):
    """
    画绿框 + 类别文字，并分别保存到:
      data/images/boxes/orig/<name>.jpg
      data/images/boxes/edit/<name>.jpg
    """
    # 确保目录存在
    BOXES_ORIG_DIR.mkdir(parents=True, exist_ok=True)
    BOXES_EDIT_DIR.mkdir(parents=True, exist_ok=True)

    # 原图（统一 resize 到 384x384 再画框）
    orig = Image.open(str(orig_path)).convert("RGB")
    orig = orig.resize((384, 384), Image.LANCZOS)
    orig = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)

    # 编辑图（已经是 384x384）
    edit = cv2.cvtColor(np.array(edited_img), cv2.COLOR_RGB2BGR)

    # 画原图框 + 文本
    if orig_box is not None:
        x1, y1, x2, y2 = orig_box
        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 4)
        if orig_cls is not None:
            label = det_model.names.get(int(orig_cls), str(orig_cls))
            cv2.putText(orig, label,
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2, cv2.LINE_AA)

    # 画编辑图框 + 文本
    if edit_box is not None:
        x1, y1, x2, y2 = edit_box
        cv2.rectangle(edit, (x1, y1), (x2, y2), (0, 255, 0), 4)
        if edit_cls is not None:
            label = det_model.names.get(int(edit_cls), str(edit_cls))
            cv2.putText(edit, label,
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(str(BOXES_ORIG_DIR / f"{out_prefix}.jpg"), orig)
    cv2.imwrite(str(BOXES_EDIT_DIR / f"{out_prefix}.jpg"), edit)


def structural_score(orig_path, edited_img):
    """
    返回 (score, iou, orig_cls, edit_cls)，其中 score = IoU。
    """
    orig_cls, orig_box = detect_subject(str(orig_path), resize_to=384)
    edited_bgr = cv2.cvtColor(np.array(edited_img), cv2.COLOR_RGB2BGR)
    edit_cls, edit_box = detect_subject(edited_bgr)

    print(f"  orig_cls={orig_cls}, orig_box={orig_box}")
    print(f"  edit_cls={edit_cls}, edit_box={edit_box}")

    visualize_boxes(orig_path, edited_img, orig_box, edit_box,
                    orig_cls=orig_cls, edit_cls=edit_cls,
                    out_prefix=Path(orig_path).stem)

    if orig_cls is None or edit_cls is None:
        print("  structural score: 0.0 (missing detection)")
        return 0.0, 0.0, orig_cls, edit_cls

    if orig_cls != edit_cls:
        print("  structural score: 0.0 (class mismatch)")
        return 0.0, 0.0, orig_cls, edit_cls

    iou = bbox_iou(orig_box, edit_box)
    print(f"  IoU between boxes: {iou:.3f}")
    score = iou
    return score, iou, orig_cls, edit_cls


# ---------- 4. InstructPix2Pix 设置 ----------

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Using device:", device)


def load_and_resize(path, size=384):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return img


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


def edit_once(img_path, prompt):
    image = load_and_resize(str(img_path), size=384)
    result = pipe(
        prompt,
        image=image,
        num_inference_steps=30,
        image_guidance_scale=1.5,
        guidance_scale=7.5,
    )
    return result.images[0]


# ---------- 5. prompt refinement ----------

def refine_prompt(base_prompt, scores):
    """
    简单 rule-based：结构分太低时，更强调主体不变。
    后面可以换成 LLM。
    """
    p = base_prompt
    if scores["struct"] < 0.5 and "keep the main subject" not in p:
        p = base_prompt + ", keep the main subject exactly the same"
    return p


# ---------- 6. 单张图片 closed-loop ----------

def iterative_edit(row, max_iter=1, threshold=0.5):
    filename = row["filename"]
    obj = row["object"]
    action = row["action"]
    bg = row["background"]

    # 所有编辑都从 original_images 里的原图开始
    img_path = ORIG_IMG_DIR / filename

    base_prompt = (
        f"replace the current {bg} background with a different scene, "
        f"while keeping the {obj} and its {action} pose unchanged, natural lighting, realistic photo"
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

    # 保存编辑结果
    EDITED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EDITED_DIR / filename.replace(".jpg", "_edited.jpg")
    best_img.save(out_path)
    print(f"Saved best edit to {out_path}, score={best_score:.3f}")

    # 写入 metrics.csv（第一次写表头）
    write_header = not METRICS_CSV.exists()
    with METRICS_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "filename", "object", "action", "background",
                "score", "iou", "orig_cls", "edit_cls",
            ])
        writer.writerow([
            filename,
            obj,
            action,
            bg,
            f"{best_score:.4f}",
            f"{best_iou:.4f}",
            best_orig_cls,
            best_edit_cls,
        ])


# ---------- 7. main：跑前 15 张 ----------

def main():
    # 先备份所有原图到 data/images/original/
    backup_originals()

    # 每次运行前清空旧 metrics.csv
    if METRICS_CSV.exists():
        METRICS_CSV.unlink()

    rows = load_labels()
    print(f"Loaded {len(rows)} labeled images")

    rows = rows[:15]

    for row in rows:
        iterative_edit(row, max_iter=1, threshold=0.5)


if __name__ == "__main__":
    main()