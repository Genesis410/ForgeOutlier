"""Domain-specific exception hierarchy with sanitized messaging."""


class ForgeOutlierException(Exception):
    """Base exception for all ForgeOutlier errors."""

    def __init__(self, message: str, code: str = "GENERIC_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class SecurityIntegrityError(ForgeOutlierException):
    """Raised when artifact hash mismatch, untrusted model, or path traversal detected."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="SECURITY_INTEGRITY_VIOLATION")


class CorruptedMediaError(ForgeOutlierException):
    """Raised when image or video is truncated, invalid, or exceeds resource limits."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="CORRUPTED_MEDIA_ERROR")


class ResourceLimitExceededError(CorruptedMediaError):
    """Raised when media exceeds bounds (dimensions, file size, or frame count)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = "RESOURCE_LIMIT_EXCEEDED"


class NumericalInstabilityError(ForgeOutlierException):
    """Raised when covariance matrix singular, Cholesky fails, or NaN/Inf encountered."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="NUMERICAL_INSTABILITY_ERROR")


class InsufficientDataError(ForgeOutlierException):
    """Raised when sample count insufficient for statistical fitting or calibration."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="INSUFFICIENT_DATA_ERROR")


class ExecutionFailureError(ForgeOutlierException):
    """Raised when an internal processing step fails (e.g. face detection missed)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="EXECUTION_FAILURE_ERROR")
