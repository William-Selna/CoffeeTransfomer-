import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from coffee_transformer.data.tagging import tag_agent, tag_reaction, write_reactions_jsonl


def test_tag_agent_known_and_fallback():
    assert tag_agent("O=C([O-])[O-]") == "BASE"       # carbonate
    assert tag_agent("CN(C)C=O") == "SOLVENT"          # DMF
    assert tag_agent("[Pd]") == "AGENT"                # unmatched -> fallback


def test_tag_reaction_structure():
    comps = tag_reaction("CC(=O)O.c1ccccc1N>O=C([O-])[O-].CN(C)C=O>CC(=O)Nc1ccccc1")
    slots = [s for s, _ in comps]
    assert slots.count("REACTANT") == 2
    assert slots.count("PRODUCT") == 1
    assert "BASE" in slots and "SOLVENT" in slots


def test_tag_reaction_rejects_malformed():
    assert tag_reaction("not a reaction") == []


def test_write_reactions_jsonl_roundtrip(tmp_path):
    rxns = ["CC(=O)O>CN(C)C=O>CC(=O)Nc1ccccc1", "CCBr.NCCO>[Pd]>CCOC(C)=O"]
    out = tmp_path / "reactions.jsonl"
    n = write_reactions_jsonl(rxns, out)
    assert n == 2
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert all(len(pair) == 2 for pair in first)   # [slot, smiles] pairs
