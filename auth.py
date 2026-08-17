"""JWT authentication for the FastAPI backend.

Validates Bearer tokens on protected endpoints in api.py. This module does
NOT issue tokens (no /token or /login route here) -- the spec's ACs test
against a token "generated locally" for verification, and no user-management
system (accounts, passwords, a tenant registry) exists anywhere in this
codebase to issue real ones against. Token issuance is a separate,
considerably larger piece of work (a login endpoint needs a persisted user
store, password hashing/verification, and a decision about who is allowed
to mint a token for which tenant_id) that this task does not ask for and
this module deliberately does not invent.

    from auth import get_current_user, CurrentUser

    @app.post("/api/research")
    def research(req: ResearchRequest, user: CurrentUser = Depends(get_current_user)):
        ...

SCOPE NOTE -- "multi-tenant isolation":
    get_current_user() extracts and returns tenant_id, so callers CAN attach
    it to logs/state (see api.py's use of it and research_agent.py's
    AgentState.tenant_id). That is authentication plus tenant ATTRIBUTION,
    not tenant ISOLATION -- this module has no opinion on, and does not
    enforce, which tenant's data a query is allowed to touch. Actual
    isolation would mean every Qdrant/Neo4j read is filtered by tenant_id,
    which requires a tenant_id field to exist on every stored record in the
    first place (it does not, anywhere in this schema today) -- a
    considerably larger, separate change than adding auth. Do not treat a
    validated tenant_id claim as a security boundary until that filtering
    actually exists.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import ExpiredSignatureError, JWTError, jwt
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# tokenUrl is a documentation-only hint -- it points Swagger UI's
# "Authorize" button at a path, and is never itself called by this
# dependency. This API has no such route (see module docstring); the value
# is kept descriptive rather than a dead literal "token" so /docs doesn't
# imply a working login flow that isn't there.
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/token",
    auto_error=True,  # a request with NO Authorization header is rejected
                      # by this dependency itself, before get_current_user
                      # ever runs -- one less case for that function to handle
    description="No /api/token route exists in this service; tokens are issued out of band.",
)


class CurrentUser(BaseModel):
    tenant_id: str
    user_email: str


def _unauthorized(detail: str) -> HTTPException:
    # WWW-Authenticate: Bearer is what tells a spec-compliant client THIS
    # 401 means "send a Bearer token", not "log in with a form" or some
    # other auth scheme -- part of the OAuth2 Bearer spec, not decoration.
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(token: str = Depends(oauth2_scheme)) -> CurrentUser:
    """Decode and validate the Bearer token, returning the caller's identity.

    Every failure mode raises 401, not a distinguishable status per failure
    reason -- telling an unauthenticated caller WHY a token is invalid
    (expired vs. wrong signature vs. malformed) is an oracle that helps an
    attacker iterate; the detail message differs for legitimate debugging,
    but the status code and the fact that access is denied does not.

    require_exp=True is NOT python-jose's default -- verified directly
    before writing this: jwt.decode() on a token with no `exp` claim at all
    silently succeeds and is treated as never-expiring unless this option
    is set. The spec's "validates the expiration (exp)" only actually holds
    with this passed explicitly.

    algorithms=[JWT_ALGORITHM] is passed as an explicit allow-list, not
    inferred from the token's own header -- also verified directly: without
    a restrictive algorithms= argument, a forged alg="none" token is a
    classic algorithm-confusion bypass. python-jose rejects "none" outright
    when a concrete algorithms list is supplied, confirmed live before
    relying on it here.
    """
    if not JWT_SECRET_KEY:
        # Fail closed: a missing secret must never be silently treated as
        # "accept everything" or "accept nothing verifiable" is picked by
        # jose's own defaults -- surface it as a server misconfiguration.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT_SECRET_KEY is not configured on the server.",
        )

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"require_exp": True},
        )
    except ExpiredSignatureError:
        raise _unauthorized("Token has expired.")
    except JWTError:
        raise _unauthorized("Could not validate credentials.")

    tenant_id = payload.get("tenant_id")
    user_email = payload.get("user_email") or payload.get("sub")
    if not tenant_id or not user_email:
        raise _unauthorized("Token is missing required claims (tenant_id, user_email).")

    return CurrentUser(tenant_id=tenant_id, user_email=user_email)
