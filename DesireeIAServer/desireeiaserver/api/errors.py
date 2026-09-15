"""OpenAI-compatible error envelope and the app-wide exception handler."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class OpenAIError(Exception):
    error_type = "server_error"
    default_status = 500

    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        code: Optional[str] = None,
        param: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status or self.default_status
        self.code = code or self.error_type
        self.param = param

    def payload(self) -> dict:
        error = {"message": self.message, "type": self.error_type, "code": self.code}
        if self.param is not None:
            error["param"] = self.param
        return {"error": error}


class InvalidRequestError(OpenAIError):
    error_type = "invalid_request_error"
    default_status = 400


class AuthenticationError(OpenAIError):
    error_type = "authentication_error"
    default_status = 401


class NotFoundError(OpenAIError):
    error_type = "not_found_error"
    default_status = 404


class NotSupportedError(OpenAIError):
    error_type = "not_supported_error"
    default_status = 400


class NotImplementedEndpointError(OpenAIError):
    error_type = "not_implemented_error"
    default_status = 501


class UnavailableError(OpenAIError):
    error_type = "unavailable_error"
    default_status = 503


def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, StarletteHTTPException):
        if exc.status_code == 404:
            error = NotFoundError("not found")
        else:
            error = InvalidRequestError(str(exc.detail), status=exc.status_code)
        return JSONResponse(error.payload(), status_code=error.status)
    if not isinstance(exc, OpenAIError):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        exc = OpenAIError("internal server error", status=500, code="server_error")
    return JSONResponse(exc.payload(), status_code=exc.status)