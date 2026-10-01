"""Environment + memory diagnostic. Loads each model, runs one forward pass, reports GPU / RAM usage.

    python experiments/check_env.py               # perception models + ip2p
    python experiments/check_env.py --skip-editor

Paste the whole output back when something runs out of memory.
"""

from __future__ import annotations

import platform
import time
import traceback

import numpy as np
from common import base_parser, setup

GB = 2**30


def ram() -> str:
    try:
        import psutil

        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        return (f"RAM used {vm.used / GB:.1f}/{vm.total / GB:.1f} GB (avail {vm.available / GB:.1f}) | "
                f"swap/pagefile used {sw.used / GB:.1f}/{sw.total / GB:.1f} GB")
    except ImportError:
        return "RAM: pip install psutil for host-memory numbers"


def gpu() -> str:
    import torch

    if not torch.cuda.is_available():
        return "GPU: none"
    free, total = torch.cuda.mem_get_info()
    return (f"GPU used {(total - free) / GB:.2f}/{total / GB:.2f} GB | torch allocated "
            f"{torch.cuda.memory_allocated() / GB:.2f} GB, peak {torch.cuda.max_memory_allocated() / GB:.2f} GB")


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--skip-editor", action="store_true")
    args = ap.parse_args()
    cfg = setup(args)

    import torch
    import transformers

    print(f"python {platform.python_version()} | torch {torch.__version__} (CUDA {torch.version.cuda}) | "
          f"transformers {transformers.__version__} | {platform.platform()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU0 {p.name} | sm_{p.major}{p.minor} | {p.total_memory / GB:.1f} GB | arch list {torch.cuda.get_arch_list()}")
    print(ram())
    print(gpu())
    print(f"memory_policy={cfg['memory_policy']} image.long_side={cfg['image']['long_side']}\n")

    from src.models import ModelRegistry
    from src.perception import Perceiver
    from src.spec import build_spec

    reg = ModelRegistry(cfg)
    per = Perceiver(reg, cfg)
    spec = build_spec("check", "dog", "sit", "river", "snow")
    ls = cfg["image"]["long_side"]
    img = (np.random.default_rng(0).random((ls, ls * 2 // 3, 3)) * 255).astype(np.uint8)
    mask = np.zeros(img.shape[:2], bool)
    mask[ls // 4: 3 * ls // 4, ls // 6: ls // 2] = True

    steps = [
        ("grounding_dino", lambda: per.detect(img, "dog .", 0.35, 0.25)),
        ("sam2", lambda: per.segment(img, [(50, 50, 300, 500)])),
        ("dinov2", lambda: per.dino(img, mask)),
        ("siglip", lambda: per.background_probs(img)),
        ("depth", lambda: per.depth(img)),
    ]
    if cfg["perception"].get("use_vlm"):
        def vlm():
            ans = per.vlm_answers(img, mask, spec)
            print("                answers:", {k: round(v, 3) for k, v in ans.items()})
            return ans

        steps.append(("vlm", vlm))
    if not args.skip_editor:
        name = cfg["loop"]["editor"]

        def editor(name=name):
            from src.editors import build_editor

            ed = build_editor(name, cfg, reg)
            p = ed.default_params() | {"num_inference_steps": 5}
            return ed.edit(img, spec.instruction(), p, 0, spec, mask)

        steps.append((name, editor))

    ok = True
    for name, fn in steps:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            fn()
            status = f"OK   {time.time() - t:5.1f}s"
        except Exception as e:  # report and keep going
            ok = False
            status = f"FAIL {type(e).__name__}: {str(e).splitlines()[0][:120]}"
            traceback.print_exc()
        print(f"{name:15s} {status}\n                {gpu()}\n                {ram()}")
    print("\nALL OK" if ok else "\nSOME STEPS FAILED — paste this whole output back")


if __name__ == "__main__":
    main()
