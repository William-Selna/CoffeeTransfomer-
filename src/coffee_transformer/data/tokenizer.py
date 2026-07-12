"""Regex-based SMILES tokenizer (Molecular Transformer / Yield-BERT lineage).

Keeps multi-character atoms and bracket atoms intact (Br, Cl, [nH], [C@@H], ...)
so the vocabulary stays ~300 and every HTE molecule tokenizes without [UNK]
(a hygiene check the design calls for explicitly, Section 6).

Special tokens:
  [PAD] [UNK] [CLS] [SEP] [MASK]  plus one slot token per schema slot ([LIG], ...).

The vocabulary is built from a corpus (or a provided token list) and can be
saved/loaded as JSON so pretraining and fine-tuning share exactly one vocab.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .slots import DEFAULT_SCHEMA, SlotSchema

# Canonical SMILES atom/bond regex (Schwaller et al.). Order matters: the
# bracket-atom and two-letter-halogen alternatives must precede single chars.
SMILES_TOKEN_PATTERN = (
    r"(\[[^\]]+\]|Br|Cl|B|C|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%\d{2}|\d)"
)
_SMILES_RE = re.compile(SMILES_TOKEN_PATTERN)

PAD, UNK, CLS, SEP, MASK = "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
CORE_SPECIALS: tuple[str, ...] = (PAD, UNK, CLS, SEP, MASK)


def split_smiles(smiles: str) -> list[str]:
    """Tokenize a raw SMILES string into atoms/bonds/branch symbols."""
    return _SMILES_RE.findall(smiles)


@dataclass
class SmilesTokenizer:
    """Vocabulary + encode/decode for slot-structured reaction strings.

    Encoding a component span emits the slot token followed by the SMILES
    tokens; `encode_reaction` assembles [CLS] + per-slot spans and returns
    parallel `input_ids` and `slot_type_ids`.
    """

    token_to_id: dict[str, int]
    schema: SlotSchema

    # ---- construction -----------------------------------------------------
    @classmethod
    def build(
        cls,
        corpus: Iterable[str],
        schema: SlotSchema = DEFAULT_SCHEMA,
        max_vocab: int | None = None,
        min_freq: int = 1,
    ) -> "SmilesTokenizer":
        counts: Counter[str] = Counter()
        for smiles in corpus:
            counts.update(split_smiles(smiles))

        specials = list(CORE_SPECIALS) + schema.slot_tokens()
        vocab: list[str] = list(specials)
        budget = None if max_vocab is None else max(0, max_vocab - len(specials))
        for tok, freq in counts.most_common():
            if freq < min_freq or tok in specials:
                continue
            vocab.append(tok)
            if budget is not None and len(vocab) - len(specials) >= budget:
                break

        token_to_id = {tok: i for i, tok in enumerate(vocab)}
        return cls(token_to_id=token_to_id, schema=schema)

    # ---- special-token ids ------------------------------------------------
    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    @property
    def cls_id(self) -> int:
        return self.token_to_id[CLS]

    @property
    def sep_id(self) -> int:
        return self.token_to_id[SEP]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[MASK]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def slot_token_id(self, slot_name: str) -> int:
        return self.token_to_id[self.schema.slot_token(slot_name)]

    # ---- encode / decode --------------------------------------------------
    def encode_smiles(self, smiles: str) -> list[int]:
        unk = self.unk_id
        return [self.token_to_id.get(t, unk) for t in split_smiles(smiles)]

    def encode_reaction(
        self,
        components: Sequence[tuple[str, str]],
        add_cls: bool = True,
    ) -> tuple[list[int], list[int]]:
        """Encode a reaction given ordered (slot_name, smiles) pairs.

        Returns (input_ids, slot_type_ids) of equal length. The slot token and
        every SMILES token in a component span carry that component's slot id;
        [CLS] carries the NO_SLOT id.
        """
        input_ids: list[int] = []
        slot_type_ids: list[int] = []
        no_slot = self.schema.slot_id("NONE")

        if add_cls:
            input_ids.append(self.cls_id)
            slot_type_ids.append(no_slot)

        for slot_name, smiles in components:
            slot_tok_id = self.slot_token_id(slot_name)
            slot_id = self.schema.slot_id(slot_name)
            input_ids.append(slot_tok_id)
            slot_type_ids.append(slot_id)
            body = self.encode_smiles(smiles)
            input_ids.extend(body)
            slot_type_ids.extend([slot_id] * len(body))

        return input_ids, slot_type_ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        id_to_token = {i: t for t, i in self.token_to_id.items()}
        specials = set(CORE_SPECIALS) | set(self.schema.slot_tokens())
        out = []
        for i in ids:
            tok = id_to_token.get(int(i), UNK)
            if skip_special and tok in specials:
                continue
            out.append(tok)
        return "".join(out)

    # ---- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {
            "token_to_id": self.token_to_id,
            "slots": list(self.schema.slots),
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SmilesTokenizer":
        payload = json.loads(Path(path).read_text())
        schema = SlotSchema(slots=tuple(payload["slots"]))
        return cls(token_to_id=payload["token_to_id"], schema=schema)
