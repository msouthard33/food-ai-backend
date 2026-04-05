"""Audit logging middleware for future HIPAA compliance."""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("audit")


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Log every data-access request for compliance auditing.

    Captures: request ID, timestamp, method, path, user identity (from JWT sub),
    response status, and latency.  Sensitive body content is never logged.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        # Extract user identity from Authorization header (sub claim)
        user_id = "anonymous"
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            # In dev mode the token IS the user UUID
            try:
                user_id = str(uuid.UUID(token))
            except ValueError:
                user_id = "jwt-user"  # Will be resolved by auth dependency

        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "audit_event",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "user_id": user_id,
                "status_code": response.status_code,
                "latency_ms": round(elapsed_ms, 2),
                "ip": request.client.host if request.client else "unknown",
            },
        )

        response.headers["X-Request-ID"] = request_id
        return response
