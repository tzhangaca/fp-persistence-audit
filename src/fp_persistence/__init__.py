"""Public API for the generic false-positive persistence audit."""

from .public_api import audit, flags_from_scores
from .public_types import (
    AuditResult,
    AuditStatus,
    AuditValidationError,
    ExposureSpec,
    FlagSelection,
    PermutationSpec,
)

__version__ = "0.1.1"

__all__ = [
    "AuditResult",
    "AuditStatus",
    "AuditValidationError",
    "ExposureSpec",
    "FlagSelection",
    "PermutationSpec",
    "audit",
    "flags_from_scores",
]
