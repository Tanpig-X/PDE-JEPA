"""Training entry point for the four stages of PDE-JEPA."""

import argparse
import importlib
import os
from pathlib import Path

import torch
import yaml

from pde_jepa.config import load_config


STAGE_CONFIGS = {
    "pretrain": "stage1_pretrain.yaml",
    "cooldown": "stage1_cooldown.yaml",
    "pag": "stage2_pag.yaml",
    "psp": "stage3_psp.yaml",
    "decoder": "stage4_decoder.yaml",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=list(STAGE_CONFIGS))
    parser.add_argument("--task", default="vorticity", help="Configuration directory under configs (default: vorticity)")
    parser.add_argument("--config", help="Explicit YAML path; otherwise use configs/<task>/<stage YAML>")
    parser.add_argument("--device", default=None, help="cpu or cuda device; torchrun uses LOCAL_RANK")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--prepare-cache-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true", help="Run the decoder's validation loop")
    parser.add_argument("--checkpoint", help="Decoder checkpoint for --evaluate-only")
    parser.add_argument("--split", choices=["val", "ood"], default="ood")
    parser.add_argument("--limit", type=int, help="Optional evaluation trajectory limit")
    args = parser.parse_args()
    config = load_config(args.config or Path("configs") / args.task / STAGE_CONFIGS[args.stage], args.overrides)
    if args.device:
        config["device"] = args.device
    config.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    module = importlib.import_module(f"pde_jepa.training.{'pretrain' if args.stage == 'cooldown' else args.stage}")
    if args.evaluate_only:
        if args.stage != "decoder" or not args.checkpoint:
            parser.error("--evaluate-only requires --stage decoder and --checkpoint")
        module.evaluate_checkpoint(config, args.checkpoint, args.split, args.limit)
        return
    if args.prepare_cache_only:
        if args.stage == "decoder":
            module.prepare_caches(config)
        elif args.stage == "psp":
            for split in ("id_train", "id_val"):
                module.prepare_cache(config, split)
        else:
            parser.error("Cache preparation applies to PSP and decoder training")
        return
    folder = Path(config["folder"])
    if int(os.environ.get("RANK", 0)) == 0:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "params-pretrain.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    module.train(config)


if __name__ == "__main__":
    main()
