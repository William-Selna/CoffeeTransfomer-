# CoffeeTransformer pipeline orchestration.
# `make help` for targets. Paths are overridable: `make prepare PUBCHEM=... USPTO=...`.
.RECIPEPREFIX = >
.PHONY: help setup preflight smoke test fetch reactions prepare pretrain mps select uspto all clean tarball

PY ?= python
PUBCHEM ?= data/raw/pubchem.smi
# figshare ships USPTO as a .7z — `make fetch` downloads it, then extract:
#   7z x data/raw/uspto_1976_Sep2016.7z -odata/raw/   (needs p7zip-full)
USPTO ?= data/raw/1976_Sep2016_USPTOgrants_smiles.rsmi
PUBCHEM_LIMIT ?= 10000000
ENCODERS = runs/pretrain_gelu_s0 runs/pretrain_gelu_s1 runs/pretrain_swiglu_s0 runs/pretrain_swiglu_s1

help:
> @echo "Setup & checks:"
> @echo "  make setup       install the package (+ chem, dev extras)"
> @echo "  make preflight   env + plumbing check (torch/CUDA, data, 2-step smoke)"
> @echo "  make smoke       full pipeline on synthetic data, CPU (debug-first gate)"
> @echo "  make test        run the unit tests"
> @echo "Real run (on the H100 box, open network):"
> @echo "  make fetch       download corpora (BH vendored; PubChem/USPTO here)"
> @echo "  make reactions   USPTO -> slot-tagged data/raw/reactions.jsonl"
> @echo "  make prepare     clean + pre-tokenize corpus -> data/processed/"
> @echo "  make mps         start CUDA MPS (better concurrency for the 4 runs)"
> @echo "  make pretrain    the 2x2 grid, four runs in parallel"
> @echo "  make select      probe all four, fork the winner into the 4 SFT/RL runs"
> @echo "  make uspto       optional crude USPTO SFT (coarse 4-bin)"
> @echo "  make all         reactions -> prepare -> pretrain -> select"
> @echo "  make tarball     bundle the repo (+ data) into a release tarball"

setup:
> $(PY) -m pip install -e ".[chem,dev]"

preflight:
> $(PY) scripts/preflight.py

smoke:
> $(PY) scripts/prepare_corpus.py --synthetic --out data/processed
> for c in gelu_s0 swiglu_s0; do \
>   $(PY) scripts/pretrain.py --config configs/pretrain_$$c.yaml --synthetic --device cpu \
>     --stage1-epochs 1 --stage2-epochs 1 --num-workers 0 --batch-size 32 --no-compile || exit 1; \
> done
> $(PY) scripts/select_and_sweep.py --encoders runs/pretrain_gelu_s0 runs/pretrain_swiglu_s0 \
>   --device cpu --epochs 1 --probe-epochs 1 --batch-size 32
> $(PY) scripts/run_sft.py --config configs/run_uspto_crude.yaml --device cpu --epochs 1 --batch-size 16
> @echo "SMOKE OK"

test:
> $(PY) -m pytest -q

fetch:
> $(PY) scripts/fetch_data.py --pubchem --limit $(PUBCHEM_LIMIT) --uspto --ord

reactions:
> @test -f $(USPTO) || { echo "USPTO file $(USPTO) not found — did you extract the .7z? (7z x data/raw/uspto_1976_Sep2016.7z -odata/raw/)"; exit 1; }
> $(PY) scripts/build_reactions_jsonl.py --uspto $(USPTO) --out data/raw/reactions.jsonl

prepare:
> $(PY) scripts/prepare_corpus.py --pubchem $(PUBCHEM) --limit $(PUBCHEM_LIMIT) \
>   --reactions data/raw/reactions.jsonl \
>   --hte data/hte/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx --sheet FullCV_01 \
>   --out data/processed

mps:
> @echo "starting CUDA MPS daemon (co-schedules the 4 tiny runs on one H100)"
> nvidia-cuda-mps-control -d || true

pretrain:
> OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash -c '\
>   for c in gelu_s0 gelu_s1 swiglu_s0 swiglu_s1; do \
>     $(PY) scripts/pretrain.py --config configs/pretrain_$$c.yaml & \
>   done; wait'
> @echo "all four pretrainings finished"

select:
> $(PY) scripts/select_and_sweep.py --encoders $(ENCODERS)

uspto:
> $(PY) scripts/run_sft.py --config configs/run_uspto_crude.yaml

all: reactions prepare pretrain select

tarball:
> bash scripts/make_tarball.sh

clean:
> rm -rf runs/ data/processed/
