#!/usr/bin/env python
"""One-command data gathering.

    python scripts/fetch_data.py --all                 # everything reachable
    python scripts/fetch_data.py --hte                 # just the HTE sets (small)
    python scripts/fetch_data.py --pubchem --limit 10000000
    python scripts/fetch_data.py --uspto --ord

What lands where:
  data/hte/        Buchwald-Hartwig (+ Suzuki) HTE xlsx  -> TRACKED in git (small)
  data/raw/        PubChem / USPTO / ORD corpora         -> gitignored (large)

Network note (as observed from the Claude sandbox; your training box likely
differs): GitHub-hosted files (BH, ORD) download fine here; PubChem (NCBI) and
figshare (USPTO/Lowe) are blocked by this environment's proxy. Run the PubChem
and USPTO fetches on the H100 box where outbound network is open. Each step
degrades gracefully and prints the canonical URL if a host is unreachable, so
you can always download by hand and drop the file in the right place.
"""

from __future__ import annotations

import argparse
import gzip
import pathlib
import shutil
import sys
import urllib.request

HTE_DIR = pathlib.Path("data/hte")
RAW_DIR = pathlib.Path("data/raw")

# --- canonical sources -----------------------------------------------------
GH_RAW = "https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/data"
BH_FILE = "Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx"

# PubChem full CID->SMILES map (~2 GB gz, ~100M molecules); take the first
# `--limit` for Stage-1 grammar MLM.
PUBCHEM_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"

# USPTO reactions text-mined by Lowe (figshare, 1976-Sep2016).
USPTO_FIGSHARE = "https://ndownloader.figshare.com/files/8664379"  # 1976_Sep2016_USPTOgrants_smiles.rsmi.gz

# Open Reaction Database (schema-validated reactions with typed roles).
ORD_REPO = "https://raw.githubusercontent.com/open-reaction-database/ord-data/main"


def _download(url: str, dest: pathlib.Path, desc: str) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[{desc}] {url}")
        # stream to disk in chunks — these files are GBs, never buffer in RAM
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:  # noqa: S310
            shutil.copyfileobj(r, f, length=1 << 20)
        print(f"[{desc}] saved {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    except Exception as e:  # network blocked / host down
        print(f"[{desc}] FAILED ({e}). Download by hand from:\n    {url}\n  -> {dest}")
        return False


def fetch_hte() -> None:
    _download(f"{GH_RAW}/{BH_FILE}", HTE_DIR / BH_FILE, "BH")
    # Suzuki-Miyaura (Perera et al.) is not at a stable raw path in the repo;
    # drop the processed xlsx here to enable the multi-task transfer ablation.
    print("[Suzuki] optional (transfer ablation). Source: Perera et al., Science 2018 "
          "(nanomole-scale flow screening) / rxn4chemistry processed splits.\n"
          f"  place as {HTE_DIR / 'Suzuki-Miyaura' / 'Suzuki_Miyaura_input_data.xlsx'}")


def fetch_pubchem(limit: int | None) -> None:
    """Stream-decompress CID-SMILES.gz and write the first `limit` SMILES straight
    to data/raw/pubchem.smi (what prepare_corpus.py needs) — never stores the
    full ~2 GB / ~100M-line file."""
    out = RAW_DIR / "pubchem.smi"
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = limit or 10_000_000
    n = 0
    try:
        print(f"[PubChem] streaming {PUBCHEM_URL}")
        with urllib.request.urlopen(PUBCHEM_URL, timeout=120) as resp:  # noqa: S310
            with gzip.GzipFile(fileobj=resp) as gz, open(out, "w") as w:
                for raw in gz:
                    parts = raw.decode("ascii", "ignore").rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    w.write(parts[1] + "\n")  # column 1 is CID, column 2 is SMILES
                    n += 1
                    if n >= cap:
                        break
        print(f"[PubChem] wrote {n} SMILES -> {out}")
    except Exception as e:
        print(f"[PubChem] FAILED ({e}). Manual:\n"
              f"    curl -L {PUBCHEM_URL} -o data/raw/CID-SMILES.gz\n"
              f"    zcat data/raw/CID-SMILES.gz | cut -f2 | head -n {cap} > {out}")


def fetch_uspto() -> None:
    dest = RAW_DIR / "uspto_1976_Sep2016.7z"   # figshare ships a 7-Zip archive
    if _download(USPTO_FIGSHARE, dest, "USPTO"):
        print(f"[USPTO] 7-Zip archive — extract before use:\n"
              f"    sudo apt-get install -y p7zip-full\n"
              f"    7z x {dest} -odata/raw/\n"
              f"  then: python scripts/build_reactions_jsonl.py --uspto data/raw/<extracted>.rsmi ...")


def fetch_ord() -> None:
    # ORD is many sharded protobufs; grab the index and let the user pull shards.
    ok = _download(f"{ORD_REPO}/README.md", RAW_DIR / "ord" / "README.md", "ORD")
    if ok:
        print("[ORD] reachable. Full dataset is sharded protobufs under data/ in "
              "open-reaction-database/ord-data; use the ord-schema python API to "
              "iterate reactions and emit the slot-tagged reactions.jsonl.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--hte", action="store_true")
    p.add_argument("--pubchem", action="store_true")
    p.add_argument("--uspto", action="store_true")
    p.add_argument("--ord", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="PubChem molecule cap")
    return p.parse_args()


def main():
    args = parse_args()
    if not any([args.all, args.hte, args.pubchem, args.uspto, args.ord]):
        print("nothing selected; try --all or --hte (see --help)")
        sys.exit(1)
    if args.all or args.hte:
        fetch_hte()
    if args.all or args.pubchem:
        fetch_pubchem(args.limit)
    if args.all or args.uspto:
        fetch_uspto()
    if args.all or args.ord:
        fetch_ord()


if __name__ == "__main__":
    main()
