#!/usr/bin/env python
"""Package local checkpoints (SFT/GRPO runs + pretrained encoders) and push them
to the Hugging Face Hub as a plain weights dump — safetensors + config.json +
tokenizer.json + a generated model card. No `transformers.AutoModel` support;
loading a pushed checkpoint back means using this repo's own model classes
(see the "Loading it back" section the script writes into each README.md).

    # 1. build the export bundle locally first — no network, no token needed —
    #    so you can look at what would be uploaded:
    python scripts/export_to_hub.py

    # 2. once it looks right, push everything found under runs/ into one repo:
    export HF_TOKEN=hf_...                      # https://huggingface.co/settings/tokens
    python scripts/export_to_hub.py --push --repo-id YOUR_USERNAME/coffee-transformer

    # a separate repo per checkpoint instead of one repo with subfolders:
    python scripts/export_to_hub.py --push --repo-id YOUR_USERNAME/coffee-transformer \
        --one-repo-per-checkpoint

    # only export/push specific runs:
    python scripts/export_to_hub.py --push --repo-id YOUR_USERNAME/coffee-transformer \
        --only sft75_rl25 pretrain_swiglu_s0

Discovery: every directory under --runs-dir (default `runs/`) containing a
`model.pt` (an SFT/GRPO YieldModel from run_sft.py/run_grpo.py) or an
`encoder.pt` (a pretrained MLM encoder from pretrain.py) is treated as one
checkpoint. `encoder_latest.pt` (the crash-recovery rolling checkpoint) is
ignored — only the canonical best-by-val `encoder.pt` is exported.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import torch

from coffee_transformer.models.config import ModelConfig
from coffee_transformer.models.recurrent_depth import count_parameters
from coffee_transformer.utils.config import load_run_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs", help="root to scan for checkpoints (default: runs/)")
    p.add_argument("--out-dir", default="hf_export", help="local export staging dir (default: hf_export/)")
    p.add_argument("--only", nargs="+", default=None, help="only export these checkpoint dir names")
    p.add_argument("--config", nargs="+", default=[], metavar="NAME=PATH",
                   help="explicit configs/*.yaml to use for a run whose architecture can't be "
                        "auto-resolved (old model.pt saved before this script existed, no "
                        "config.json next to it, and no matching configs/*.yaml out_dir)")
    p.add_argument("--push", action="store_true", help="actually upload to the Hub (default: local export only)")
    p.add_argument("--repo-id", default=None, help="e.g. your-username/coffee-transformer (required with --push)")
    p.add_argument("--one-repo-per-checkpoint", action="store_true",
                   help="push each checkpoint to its own repo (repo-id-<name>) instead of one "
                        "repo with a subfolder per checkpoint")
    p.add_argument("--private", action="store_true", help="create Hub repo(s) as private")
    p.add_argument("--token", default=None, help="HF token; defaults to $HF_TOKEN or the cached `huggingface-cli login`")
    return p.parse_args()


def find_config_for_run(run_dir: pathlib.Path, root: pathlib.Path, overrides: dict) -> ModelConfig:
    if run_dir.name in overrides:
        return load_run_config(overrides[run_dir.name]).model

    local = run_dir / "config.json"
    if local.exists():
        return ModelConfig(**json.loads(local.read_text()))

    configs_dir = root / "configs"
    if configs_dir.is_dir():
        for yaml_path in sorted(configs_dir.glob("*.yaml")):
            try:
                cfg = load_run_config(yaml_path)
            except Exception:
                continue
            if pathlib.Path(cfg.train.out_dir).name == run_dir.name:
                return cfg.model

    raise SystemExit(
        f"can't resolve the architecture for '{run_dir.name}': no {local} and no "
        f"configs/*.yaml with train.out_dir matching it. Re-run this run_sft.py so it writes "
        f"config.json, or pass --config {run_dir.name}=configs/whatever.yaml"
    )


def clone_state_dict(state_dict: dict) -> dict:
    # safetensors refuses tensors that share underlying storage (e.g. tied
    # weights) — clone breaks the aliasing so save_file never trips on it.
    return {k: v.clone().contiguous() for k, v in state_dict.items()}


def save_safetensors(state_dict: dict, path: pathlib.Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise SystemExit(
            "safetensors is required: pip install -e '.[hub]'  (or: pip install safetensors huggingface_hub)"
        )
    save_file(clone_state_dict(state_dict), str(path))


def write_model_card(
    path: pathlib.Path, name: str, kind: str, cfg: ModelConfig, num_params: int, metrics: dict | None,
) -> None:
    lines = [
        f"# {name}",
        "",
        f"CoffeeTransformer checkpoint (`{kind}`) — a recurrent-depth transformer for chemical "
        "reaction-yield prediction. Weights-only export: `model.safetensors` + `config.json` + "
        "`tokenizer.json`, no `transformers` integration. See "
        "[the source repo](https://github.com/) for the model code "
        "(`src/coffee_transformer/`).",
        "",
        f"- kind: **{kind}** ({'YieldModel: encoder + histogram head' if kind == 'sft' else 'pretrained encoder only (MLM head not included)'})",
        f"- parameters: **{num_params / 1e6:.2f}M**",
        f"- d_model={cfg.d_model}, core_layers={cfg.core_layers}, activation={cfg.activation}, "
        f"recurrent={cfg.recurrent}, train_r={cfg.train_r}",
    ]
    if metrics:
        lines += ["", "## Metrics", "", "```json", json.dumps(metrics, indent=2), "```"]
    lines += [
        "",
        "## Loading it back",
        "",
        "```python",
        "import json",
        "from safetensors.torch import load_file",
        "from coffee_transformer.models.config import ModelConfig",
    ]
    if kind == "sft":
        lines += [
            "from coffee_transformer.models.heads import YieldModel",
            "",
            'cfg = ModelConfig(**json.load(open("config.json")))',
            "model = YieldModel(cfg)",
            'model.load_state_dict(load_file("model.safetensors"))',
        ]
    else:
        lines += [
            "from coffee_transformer.models.recurrent_depth import RecurrentDepthEncoder",
            "",
            'cfg = ModelConfig(**json.load(open("config.json")))',
            "encoder = RecurrentDepthEncoder(cfg)",
            'encoder.load_state_dict(load_file("model.safetensors"))',
        ]
    lines += [
        "from coffee_transformer.data.tokenizer import SmilesTokenizer",
        'tokenizer = SmilesTokenizer.load("tokenizer.json")',
        "```",
        "",
    ]
    path.write_text("\n".join(lines))


def discover(runs_dir: pathlib.Path, only: list[str] | None) -> list[tuple[str, str, pathlib.Path]]:
    found = []
    if not runs_dir.is_dir():
        raise SystemExit(f"--runs-dir {runs_dir} doesn't exist")
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if only and d.name not in only:
            continue
        if (d / "model.pt").exists():
            found.append((d.name, "sft", d))
        elif (d / "encoder.pt").exists():
            found.append((d.name, "pretrain", d))
    return found


def export_one(name: str, kind: str, run_dir: pathlib.Path, root: pathlib.Path, out_root: pathlib.Path, overrides: dict) -> pathlib.Path:
    dest = out_root / name
    dest.mkdir(parents=True, exist_ok=True)

    if kind == "sft":
        cfg = find_config_for_run(run_dir, root, overrides)
        state_dict = torch.load(run_dir / "model.pt", map_location="cpu")
        metrics_src = run_dir / "metrics.json"
    else:
        ckpt = torch.load(run_dir / "encoder.pt", map_location="cpu")
        cfg = ModelConfig(**ckpt["model_config"])
        state_dict = ckpt["encoder_state"]
        metrics_src = run_dir / "pretrain_metrics.json"

    save_safetensors(state_dict, dest / "model.safetensors")
    (dest / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    tok_src = run_dir / "tokenizer.json"
    if tok_src.exists():
        shutil.copy(tok_src, dest / "tokenizer.json")
    else:
        print(f"  [warn] {name}: no tokenizer.json found next to the checkpoint")

    metrics = None
    if metrics_src.exists():
        shutil.copy(metrics_src, dest / metrics_src.name)
        metrics = json.loads(metrics_src.read_text())

    num_params = sum(v.numel() for v in state_dict.values())
    write_model_card(dest / "README.md", name, kind, cfg, num_params, metrics)
    print(f"  exported {name} ({kind}, {num_params / 1e6:.2f}M params) -> {dest}")
    return dest


def write_index_readme(out_root: pathlib.Path, exported: list[tuple[str, str, pathlib.Path]]) -> None:
    lines = ["# CoffeeTransformer checkpoints", "", "Weights-only export — see each subfolder's README.md "
             "for how to load it back with `coffee_transformer`'s model classes.", "", "| checkpoint | kind |", "|---|---|"]
    for name, kind, _ in exported:
        lines.append(f"| `{name}/` | {kind} |")
    (out_root / "README.md").write_text("\n".join(lines) + "\n")


def push(out_root: pathlib.Path, exported: list[tuple[str, str, pathlib.Path]], args) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required for --push: pip install -e '.[hub]'  "
            "(or: pip install huggingface_hub safetensors)"
        )
    api = HfApi(token=args.token)

    if args.one_repo_per_checkpoint:
        for name, _, dest in exported:
            repo_id = f"{args.repo_id}-{name}"
            api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
            api.upload_folder(repo_id=repo_id, folder_path=str(dest),
                               commit_message=f"Add {name} checkpoint")
            print(f"  pushed -> https://huggingface.co/{repo_id}")
    else:
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, folder_path=str(out_root),
                           commit_message="Add CoffeeTransformer checkpoint(s)")
        print(f"  pushed -> https://huggingface.co/{args.repo_id}")


def main():
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parent.parent
    runs_dir = pathlib.Path(args.runs_dir)
    out_root = pathlib.Path(args.out_dir)

    overrides = {}
    for item in args.config:
        if "=" not in item:
            raise SystemExit(f"--config expects NAME=PATH, got: {item}")
        n, p = item.split("=", 1)
        overrides[n] = p

    if args.push and not args.repo_id:
        raise SystemExit("--push needs --repo-id YOUR_USERNAME/coffee-transformer")

    found = discover(runs_dir, args.only)
    if not found:
        raise SystemExit(f"no checkpoints found under {runs_dir} (looked for */model.pt or */encoder.pt)")

    print(f"found {len(found)} checkpoint(s) under {runs_dir}: {', '.join(n for n, _, _ in found)}")
    out_root.mkdir(parents=True, exist_ok=True)
    exported = [(name, kind, export_one(name, kind, d, root, out_root, overrides)) for name, kind, d in found]
    write_index_readme(out_root, exported)

    if args.push:
        print(f"\npushing to the Hub ({'one repo per checkpoint' if args.one_repo_per_checkpoint else args.repo_id})...")
        push(out_root, exported, args)
    else:
        print(f"\nlocal export only (pass --push --repo-id YOUR_USERNAME/coffee-transformer to upload) -> {out_root}")


if __name__ == "__main__":
    main()
