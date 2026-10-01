"""Download every model in the config to the Hugging Face cache (run once, on a machine with internet).

    python experiments/download_models.py                 # MVP set (~13 GB)
    python experiments/download_models.py --only ip2p sam2
"""

from __future__ import annotations

from common import base_parser, setup

# Only fetch the files the loaders need (skips duplicate .bin / .ckpt / onnx copies).
PATTERNS = {
    "sdxl_inpaint": ["*.json", "*.txt", "*/*fp16*.safetensors", "tokenizer*/*", "scheduler/*"],
}
IGNORE = ["*.bin", "*.ckpt", "*.onnx", "*.msgpack", "*.h5", "*.pt", "*.pth"]
EXTRA_IGNORE = {"ip2p": ["*fp16*"]}  # loader reads the default safetensors and casts


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--only", nargs="*", help="subset of model kinds")
    args = ap.parse_args()
    cfg = setup(args)
    from huggingface_hub import snapshot_download

    for kind, mcfg in cfg["models"].items():
        if args.only and kind not in args.only:
            continue
        print(f"==> {kind}: {mcfg['id']}")
        allow = PATTERNS.get(kind)
        path = snapshot_download(mcfg["id"], allow_patterns=allow,
                                 ignore_patterns=None if allow else IGNORE + EXTRA_IGNORE.get(kind, []))
        print(f"    -> {path}")


if __name__ == "__main__":
    main()
