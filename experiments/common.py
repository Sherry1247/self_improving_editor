"""Shared CLI plumbing: config, logging, component construction, run directories."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", action="append", default=[], help="extra YAML merged over configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="a.b=v", help="config override, repeatable")
    ap.add_argument("--log-level", default="INFO")
    return ap


def setup(args) -> dict:
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "huggingface_hub", "diffusers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(args.config, args.set)
    for k in ("labels", "images", "cache", "runs"):
        p = Path(cfg["paths"][k])
        cfg["paths"][k] = str(p if p.is_absolute() else ROOT / p)
    return cfg


def build_evaluation_stack(cfg: dict):
    from src.critics import build_critics
    from src.models import ModelRegistry
    from src.perception import Perceiver
    from src.pipelines import Evaluator
    from src.scoring import build_aggregator

    registry = ModelRegistry(cfg)
    perceiver = Perceiver(registry, cfg)
    evaluator = Evaluator(perceiver, build_critics(cfg), build_aggregator(cfg))
    return registry, perceiver, evaluator


def new_run_dir(cfg: dict, name: str | None) -> Path:
    name = name or time.strftime("%Y%m%d-%H%M%S")
    d = Path(cfg["paths"]["runs"]) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
