"""Public API for the generic false-positive persistence audit."""

from ._version import PACKAGE_VERSION as __version__
from .public_api import audit, flags_from_scores
from .public_types import (
    AuditResult,
    AuditStatus,
    AuditValidationError,
    ExposureSpec,
    FlagSelection,
    PermutationSpec,
)

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
