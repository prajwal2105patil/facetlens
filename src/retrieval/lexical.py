"""BM25 lexical retrieval, fused with dense retrieval.

WHY
---
Dense retrieval over MiniLM underperforms badly on this catalogue: recall@25 of
reference-labelled facets was 36.4%, and only 63% on facets that should be
scored. The reason is visible in the data - facet names are short, abstract
single words ("Naivety", "Brevity", "Orderliness") with almost no context for a
sentence embedding to work with, while the conversations are concrete and
narrative. The two live in different regions of the embedding space.

Lexical matching has the opposite failure profile: it is useless for paraphrase
but excellent when the conversation happens to use the facet's own vocabulary.
Fusing the two covers more than either alone.

No new dependency: BM25 is ~40 lines of numpy over a token count matrix.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")

# Standard BM25 parameters. k1 controls term-frequency saturation, b controls
# length normalisation. These are the conventional defaults and are not tuned
# on the benchmark - tuning them on 55 labelled pairs would overfit.
K1 = 1.5
B = 0.75

#: Reciprocal-rank-fusion constant. 60 is the value from the original RRF paper.
#: RRF is used instead of score normalisation because dense cosine and BM25 are
#: on incomparable scales, and rank fusion sidesteps that entirely.
RRF_K = 60

_STOP = frozenset("""a an the of and or to in on for with is are was were be been
being it its this that these those as at by from how what when where which who
whom you your i me my we our they them their he she his her not no do does did
have has had can could would should will shall may might must if then than so
such very more most other some any each own same s t just don now""".split())


# Conservative suffix normalisation. Plain stripping is not enough here:
# conversations use verbs ("assigned", "collaborating") while facets use nouns
# ("Delegation skills", "Collaboration"), and stripping alone maps those to
# different stems ("collabor" vs "collaborat"). Mapping suffixes to a shared
# replacement makes the noun, verb and adjective forms converge:
#
#     collaboration / collaborating / collaborate  -> collaborat
#     irritability  / irritable                    -> irritabl
#     talkativeness / talkative                    -> talkat
#
# Ordered longest-first; the first match wins. A minimum stem length stops it
# mangling short words. This is not a linguistically correct stemmer and is not
# trying to be - it only has to make these two vocabularies meet.
_SUFFIX_RULES: tuple[tuple[str, str], ...] = (
    ("ativeness", "at"), ("iveness", "iv"),
    ("ibility", "ibl"), ("ability", "abl"),
    ("ationally", "at"), ("ations", "at"), ("ation", "at"),
    ("ating", "at"), ("ative", "at"), ("ated", "at"), ("ates", "at"),
    ("liness", ""), ("fulness", "ful"), ("ousness", "ous"),
    ("iness", "y"), ("ness", ""),
    ("ements", "em"), ("ement", "em"),
    ("ities", "it"), ("ility", "il"), ("ivity", "iv"), ("ity", "it"),
    ("ible", "ibl"), ("able", "abl"),
    ("ingly", ""), ("ing", ""), ("edly", ""), ("ed", ""),
    ("ers", ""), ("est", ""), ("ly", ""),
    ("ive", "iv"), ("ate", "at"),
    ("ies", "y"), ("es", ""), ("s", ""),
)

MIN_STEM = 3


def stem(token: str) -> str:
    for suffix, replacement in _SUFFIX_RULES:
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM:
            return token[: -len(suffix)] + replacement
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower())
            if t not in _STOP and len(t) > 1]


class BM25Index:
    """Sparse BM25 over the same facet texts the dense index embeds."""

    def __init__(self, documents: list[str]) -> None:
        tokenized = [tokenize(doc) for doc in documents]
        self.n_docs = len(tokenized)
        self.doc_len = np.array([len(d) for d in tokenized], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if self.n_docs else 0.0

        self.vocab: dict[str, int] = {}
        for tokens in tokenized:
            for token in tokens:
                self.vocab.setdefault(token, len(self.vocab))

        # Term-frequency matrix, documents x vocabulary. At 399 x ~1500 this is
        # 2.4 MB dense, so there is no reason to reach for a sparse structure.
        self.tf = np.zeros((self.n_docs, len(self.vocab)), dtype=np.float32)
        for row, tokens in enumerate(tokenized):
            for token, count in Counter(tokens).items():
                self.tf[row, self.vocab[token]] = count

        doc_freq = (self.tf > 0).sum(axis=0)
        # BM25+ style idf floor keeps very common terms from going negative.
        self.idf = np.log(
            1.0 + (self.n_docs - doc_freq + 0.5) / (doc_freq + 0.5)
        ).astype(np.float32)

        self._denominator_base = K1 * (1 - B + B * self.doc_len / max(self.avg_len, 1e-6))

    def score(self, query: str) -> np.ndarray:
        """BM25 score of every document against the query."""
        columns = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not columns:
            return np.zeros(self.n_docs, dtype=np.float32)
        tf = self.tf[:, columns]
        numerator = tf * (K1 + 1)
        denominator = tf + self._denominator_base[:, None]
        return (self.idf[columns] * (numerator / denominator)).sum(axis=1)


def _ranks(scores: np.ndarray) -> np.ndarray:
    """0-based rank of each item, best first."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return ranks


def fuse(dense: np.ndarray, lexical: np.ndarray) -> np.ndarray:
    """Reciprocal rank fusion of two score vectors.

    RRF only looks at ordering, so a facet needs to rank well under *either*
    signal to survive. That is the property worth having here: dense retrieval
    finds paraphrase, BM25 finds shared vocabulary, and neither is trusted to
    be on a meaningful absolute scale.
    """
    return (1.0 / (RRF_K + 1 + _ranks(dense))
            + 1.0 / (RRF_K + 1 + _ranks(lexical))).astype(np.float32)
