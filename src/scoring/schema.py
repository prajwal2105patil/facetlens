"""Strict output contract for facet verdicts.

Validation rules enforced here (not merely documented):
  * status must come from the allowed set
  * score must be an integer 1-5 when status == "scored"
  * score must be null for every other status
  * confidence must be within [0, 1]
  * reason must be non-empty
  * unknown fields from the model are rejected, not silently absorbed

A model response that violates any of these is an error to be captured, not a
result to be trusted. `parser.py` turns violations into per-facet ERROR verdicts
so one malformed batch cannot take down a run.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Status(StrEnum):
    SCORED = "scored"
    NOT_OBSERVABLE = "not_observable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    #: Could plausibly be inferred, but we refuse to. See sensitivity.py.
    POLICY_BLOCKED = "policy_blocked"
    ERROR = "error"


#: Which gate produced a verdict. Makes the failure analysis attributable.
Origin = Literal["observability_gate", "policy_gate", "llm",
                 "evidence_verifier", "parser"]


class ModelVerdict(BaseModel):
    """Exactly what the LLM is asked to return, per facet. Nothing more."""

    model_config = ConfigDict(extra="forbid")

    # NOTE: every field here is deliberately REQUIRED (no defaults).
    # This model's JSON schema is handed to the backend as a constrained-decoding
    # grammar. Pydantic omits defaulted fields from `required`, and the model
    # then legitimately skips them - which made it emit status="scored" with no
    # score at all, turning correctly-scored facets into parse errors.
    # See DEBUGGING.md #4. `score` stays nullable, but must be PRESENT.
    facet_id: str
    status: Literal["scored", "insufficient_evidence"]
    score: int | None = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_quote: str = Field(
        description="Verbatim span copied from the conversation, or empty when abstaining.",
    )
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _score_matches_status(self) -> "ModelVerdict":
        if self.status == "scored" and self.score is None:
            raise ValueError("status 'scored' requires an integer score 1-5")
        if self.status != "scored" and self.score is not None:
            raise ValueError(f"status {self.status!r} must have score=null")
        return self


class FacetVerdict(BaseModel):
    """The final, enriched result the pipeline emits for one facet."""

    model_config = ConfigDict(extra="forbid")

    facet_id: str
    facet: str
    facet_type: str
    status: Status
    score: int | None = Field(default=None, ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    evidence_quote: str = ""
    origin: Origin
    retrieval_score: float | None = None
    #: Set when the evidence verifier rejected a quote the model invented.
    evidence_verified: bool | None = None
    #: True when a contradictory model verdict was repaired toward abstention.
    schema_repaired: bool = False

    @model_validator(mode="after")
    def _score_matches_status(self) -> "FacetVerdict":
        if self.status == Status.SCORED and self.score is None:
            raise ValueError("status 'scored' requires an integer score 1-5")
        if self.status != Status.SCORED and self.score is not None:
            raise ValueError(f"status {self.status!r} must have score=null")
        return self


class BatchResponse(BaseModel):
    """Envelope the model must return for a scoring batch."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[ModelVerdict]


class ConversationResult(BaseModel):
    """Everything produced for one conversation."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    conversation: str
    model: str
    top_k: int
    batch_size: int
    verdicts: list[FacetVerdict]

    @property
    def scored(self) -> list[FacetVerdict]:
        return [v for v in self.verdicts if v.status == Status.SCORED]

    @property
    def abstained(self) -> list[FacetVerdict]:
        return [
            v for v in self.verdicts
            if v.status in (Status.NOT_OBSERVABLE, Status.INSUFFICIENT_EVIDENCE,
                            Status.POLICY_BLOCKED)
        ]

    @property
    def errors(self) -> list[FacetVerdict]:
        return [v for v in self.verdicts if v.status == Status.ERROR]
