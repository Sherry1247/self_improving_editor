import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
from ultralytics import YOLO


# ---------- 0. 路径配置：相对本文件定位 data 目录 ----------

# 本文件: self_improving_editor/src/closed_loop_editor.py
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMG_DIR = DATA_DIR / "images"
LABELS_PATH = DATA_DIR / "labels.csv"


# ---------- 1. 读取 labels.csv ----------

def load_labels(labels_path=LABELS_PATH):
    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels.csv not found at {labels_path!s}.\n"
            f"Expected labels.csv under the repository 'data' directory."
        )

    rows = []
    with labels_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------- 2. YOLO 检测 + 结构评分 ----------

det_model = YOLO("yolov8n.pt")  # 小模型，COCO 预训练


def detect_subject(img_path_or_array):
    """
    返回 (cls_id, bbox_xyxy)，只取置信度最高的 person/dog。
    cls_id: 0=person, 16=dog（COCO 定义）
    """
    results = det_model(img_path_or_array)[0]
    best = None
    best_conf = -1.0
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
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
    yB = max(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def structural_score(orig_path, edited_img):
    """
    0–1 结构分：主体类别是否一致 + bbox IoU。
    """
    # 原图检测
    orig_cls, orig_box = detect_subject(str(orig_path))

    # 编辑后检测（PIL -> OpenCV BGR）
    edited_bgr = cv2.cvtColor(np.array(edited_img), cv2.COLOR_RGB2BGR)
    edit_cls, edit_box = detect_subject(edited_bgr)

    if orig_cls is None or edit_cls is None:
        return 0.0
    if orig_cls != edit_cls:
        return 0.0

    iou = bbox_iou(orig_box, edit_box)
    return max(0.0, min(1.0, iou / 0.7))


# ---------- 3. Stable Diffusion (InstructPix2Pix) 设置 ----------

# 3.1 选择设备：M1 优先用 mps
device = "cpu"
print("Using device:", device)


# 3.2 预处理：统一 resize 到 384x384，降低内存压力
def load_and_resize(path, size=384):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return img


# 3.3 加载编辑模型（预训练）
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float32,
).to(device)

# 关闭 NSFW 安全过滤，不再把图变成黑色
def dummy_safety_checker(images, clip_input):
    return images, [False] * len(images)

pipe.safety_checker = dummy_safety_checker

# 减少显存/内存占用
pipe.enable_attention_slicing()

# 减少显存/内存占用
pipe.enable_attention_slicing()


def edit_once(img_path, prompt):
    image = load_and_resize(img_path, size=384)
    result = pipe(
        prompt,
        image=image,
        num_inference_steps=4,   # 先用 4 步，尽量不卡
        image_guidance_scale=1.0,
        guidance_scale=4.0,
    )
    return result.images[0]


# ---------- 4. 简单的 prompt refinement ----------

def refine_prompt(base_prompt, scores):
    """
    简单 heuristics：如果结构分太低，就在提示里强调“keep the main subject unchanged”。
    之后可以换成真 LLM。
    """
    p = base_prompt
    if scores["struct"] < 0.5 and "keep the main subject" not in p:
        p = base_prompt + ", keep the main subject exactly the same"
    return p


# ---------- 5. 单张图片的 closed-loop ----------

def iterative_edit(row, max_iter=1, threshold=0.5):
    filename = row["filename"]
    obj = row["object"]
    action = row["action"]
    bg = row["background"]

    img_path = IMG_DIR / filename

    base_prompt = (
        f"change the background {bg} to something different, "
        f"keep the {obj} and its {action} pose unchanged"
    )
    p_t = base_prompt

    best_img = None
    best_score = -1.0

    for t in range(max_iter):
        print(f"\nIteration {t} for {filename}")
        y_t = edit_once(img_path, p_t)

        s_struct = structural_score(img_path, y_t)
        S_t = s_struct
        print(f"  structural score: {s_struct:.3f}")

        if S_t > best_score:
            best_score = S_t
            best_img = y_t

        if S_t >= threshold:
            break

        p_t = refine_prompt(base_prompt, {"struct": s_struct})

    out_dir = IMG_DIR / "edited"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / filename.replace(".jpg", "_edited.jpg")
    best_img.save(out_path)
    print(f"Saved best edit to {out_path}, score={best_score:.3f}")


# ---------- 6. main：只跑一张图 ----------

def main():
    rows = load_labels()
    print(f"Loaded {len(rows)} labeled images")

    # 每次只跑第一张，防止 8GB M1 卡死
    rows = rows[:1]

    for row in rows:
        iterative_edit(row, max_iter=1, threshold=0.5)


if __name__ == "__main__":
    main()
