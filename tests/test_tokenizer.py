import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from coffee_transformer.data.slots import DEFAULT_SCHEMA
from coffee_transformer.data.synthetic import all_synthetic_smiles
from coffee_transformer.data.tokenizer import SmilesTokenizer, split_smiles


def _tok():
    return SmilesTokenizer.build(all_synthetic_smiles(), schema=DEFAULT_SCHEMA)


def test_split_keeps_multichar_atoms():
    toks = split_smiles("Brc1ccc(C)cc1")
    assert "Br" in toks
    assert "Cl" not in toks


def test_no_unk_on_known_molecules():
    tok = _tok()
    for smi in all_synthetic_smiles():
        assert tok.unk_id not in tok.encode_smiles(smi), smi


def test_encode_reaction_alignment_and_slots():
    tok = _tok()
    comps = [("ARYL_HALIDE", "Clc1ccccc1"), ("LIG", "CP(C)C"), ("ADD", "c1ccno1")]
    ids, slot_ids = tok.encode_reaction(comps)
    assert len(ids) == len(slot_ids)
    assert ids[0] == tok.cls_id
    assert slot_ids[0] == DEFAULT_SCHEMA.slot_id("NONE")
    # the token right after CLS is the first slot's opening token
    assert ids[1] == tok.slot_token_id("ARYL_HALIDE")


def test_roundtrip_save_load(tmp_path):
    tok = _tok()
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = SmilesTokenizer.load(p)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.encode_smiles("CP(C)C") == tok.encode_smiles("CP(C)C")
