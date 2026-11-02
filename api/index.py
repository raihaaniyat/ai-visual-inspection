"""
Vercel Serverless Function Entrypoint.
Exposes the FastAPI application as the ASGI handler for Vercel,
with automatic path normalization for Vercel rewrites and edge routing.
"""
from urllib.parse import parse_qs, urlencode
from backend.app import app as fastapi_app


class VercelASGIAdapter:
    """
    Transparent ASGI wrapper for Vercel Serverless functions.
    Extracts the true requested URL path from Vercel rewrite parameter (__path),
    or edge headers (x-forwarded-uri, x-matched-path), and normalizes query strings
    so FastAPI routes (/profiles, /inspections, /, /docs) match consistently.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            path = scope.get("path", "")
            raw_qs = scope.get("query_string", b"").decode("latin1")

            # 1. Check for rewrite query parameter (__path)
            if "__path=" in raw_qs:
                parsed = parse_qs(raw_qs, keep_blank_values=True)
                extracted = parsed.pop("__path", [None])[0]
                if extracted:
                    path = extracted
                    new_qs = urlencode(parsed, doseq=True)
                    scope["query_string"] = new_qs.encode("latin1")

            # 2. Check edge headers if path is still the raw entrypoint
            if path in ("/api/index.py", "/api/index"):
                headers = dict(scope.get("headers", []))
                forwarded = headers.get(b"x-forwarded-uri")
                if forwarded:
                    try:
                        orig = forwarded.decode("latin1").split("?")[0]
                        if orig and not orig.startswith("/api/index.py") and not orig.startswith("/api/index"):
                            path = orig
                    except Exception:
                        pass

            # 3. Strip any leftover function prefix
            for prefix in ("/api/index.py", "/api/index"):
                if path == prefix:
                    path = "/"
                    break
                elif path.startswith(prefix + "/"):
                    path = path[len(prefix):]
                    break

            scope["path"] = path
            scope["raw_path"] = path.encode("latin1")

        await self.app(scope, receive, send)


app = VercelASGIAdapter(fastapi_app)


