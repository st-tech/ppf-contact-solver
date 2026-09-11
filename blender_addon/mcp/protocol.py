# File: mcp/protocol.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Wire-level vocabulary for the MCP server: protocol versions, error codes,
# result envelopes, and the header and `_meta` validation the Streamable HTTP
# transport requires.
#
# Everything here is pure: no bpy, no sockets, no add-on state. That is what
# lets the conformance gate in ``addon_host_tests`` drive it on a plain
# interpreter.
#
# TWO ERAS ARE SERVED FROM ONE ENDPOINT. Protocol version 2026-07-28 is
# stateless: a request carries its own version, client capabilities and client
# identity in ``params._meta``, and there is no handshake and no session.
# Version 2025-06-18 opens with an ``initialize`` request and carries an
# ``Mcp-Session-Id`` on every later request. A request selects its era by its
# own shape (see ``era_of_request``), so one server answers both.

from __future__ import annotations

import base64
import binascii

from typing import Any

# ---------------------------------------------------------------- versions

LATEST_PROTOCOL_VERSION = "2026-07-28"
LEGACY_PROTOCOL_VERSION = "2025-06-18"

# Ordered newest first. Every version this server answers in some era, which is
# what `server/discover` reports: a dual-era client may pick either, and reaches
# the legacy one through the `initialize` handshake rather than through `_meta`.
SUPPORTED_PROTOCOL_VERSIONS = (LATEST_PROTOCOL_VERSION, LEGACY_PROTOCOL_VERSION)

# The versions a STATELESS request may declare in its `_meta`. A legacy version
# is deliberately absent: `_meta` does not exist in that revision, so a request
# declaring it is incoherent, and serving it statelessly would hand a client
# validating against the legacy schema the fields that schema does not define.
# This is also what the `data.supported` of an UnsupportedProtocolVersion error
# on the stateless path must list, or the retry it invites fails the same way.
MODERN_PROTOCOL_VERSIONS = (LATEST_PROTOCOL_VERSION,)

# Versions whose requests are self-describing. Date-stamped version strings
# order correctly under plain string comparison, which is what makes this a
# range test rather than a membership test: a future revision is treated as
# modern and then rejected by name, not silently handled as legacy.
MODERN_ERA_FLOOR = LATEST_PROTOCOL_VERSION

ERA_MODERN = "modern"
ERA_LEGACY = "legacy"

SERVER_NAME = "zozo_contact_solver"
SERVER_VERSION = "0.1.0"


# ------------------------------------------------------------ `_meta` keys

META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
META_LOG_LEVEL = "io.modelcontextprotocol/logLevel"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"


# ----------------------------------------------------------- error codes

# JSON-RPC 2.0 base codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Codes the MCP specification allocates from the reserved -32020..-32099 band.
HEADER_MISMATCH = -32020
MISSING_REQUIRED_CLIENT_CAPABILITY = -32021
UNSUPPORTED_PROTOCOL_VERSION = -32022

# The HTTP status that must accompany each JSON-RPC error class. A client
# distinguishes an unimplemented method on a live MCP endpoint (404 with a
# parseable JSON-RPC body) from an endpoint that is not there at all (404 with
# no body), so the status and the body are chosen together, never separately.
_STATUS_FOR_CODE = {
    PARSE_ERROR: 400,
    INVALID_REQUEST: 400,
    METHOD_NOT_FOUND: 404,
    INVALID_PARAMS: 400,
    INTERNAL_ERROR: 500,
    HEADER_MISMATCH: 400,
    MISSING_REQUIRED_CLIENT_CAPABILITY: 400,
    UNSUPPORTED_PROTOCOL_VERSION: 400,
}


def http_status_for_error(code: int) -> int:
    """HTTP status that carries JSON-RPC error *code* on this transport."""
    return _STATUS_FOR_CODE.get(code, 400)


class ProtocolError(Exception):
    """A JSON-RPC error raised from anywhere in request handling.

    Carries the code, message and optional structured `data` member all the
    way out to the transport, which maps the code to an HTTP status. Raising
    this is the only way to fail a request: a handler never returns a
    half-formed envelope, and no failure reaches the client as a successful
    result.
    """

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


# ------------------------------------------------------- result envelopes


def server_info() -> dict[str, str]:
    return {"name": SERVER_NAME, "version": SERVER_VERSION}


def rpc_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def rpc_error(
    request_id: Any, code: int, message: str, data: Any = None
) -> dict[str, Any]:
    """A JSON-RPC error response.

    ``id`` is omitted entirely when *request_id* is None, which is the case
    when the id could not be read (an unparseable body). The 2026-07-28 schema
    types the error-response id as a string or number, so a null id is not a
    value the envelope admits.
    """
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    response: dict[str, Any] = {"jsonrpc": "2.0", "error": error}
    if request_id is not None:
        response["id"] = request_id
    return response


def complete(result: dict[str, Any], *, era: str) -> dict[str, Any]:
    """Stamp *result* as a finished result for the client's era.

    In the modern era a result carries ``resultType`` and identifies the
    server under ``_meta``. In the legacy era it carries neither, because the
    2025-06-18 schema has no place for them and a client validating against
    that schema would reject the extra fields.
    """
    if era != ERA_MODERN:
        return result
    stamped = dict(result)
    stamped["resultType"] = "complete"
    meta = dict(stamped.get("_meta") or {})
    meta.setdefault(META_SERVER_INFO, server_info())
    stamped["_meta"] = meta
    return stamped


def cacheable(
    result: dict[str, Any], *, era: str, ttl_ms: int, scope: str
) -> dict[str, Any]:
    """Attach the caching hints a cacheable result is required to carry.

    The set of operations that MUST carry them is closed: ``server/discover``,
    ``tools/list``, ``prompts/list``, ``resources/list``,
    ``resources/templates/list`` and ``resources/read``. Do not attach them
    elsewhere; ``tools/call`` in particular is not cacheable, because its
    result depends on Blender state the cache key does not name.

    ``scope`` is "private" for anything that reflects the state of this
    Blender session, and "public" only for content identical for every caller,
    which here means the documentation bundle shipped inside the add-on.
    """
    if era != ERA_MODERN:
        return result
    if ttl_ms < 0:
        raise ValueError(f"ttlMs must be >= 0, got {ttl_ms}")
    if scope not in ("public", "private"):
        raise ValueError(f"cacheScope must be 'public' or 'private', got {scope!r}")
    hinted = dict(result)
    hinted["ttlMs"] = ttl_ms
    hinted["cacheScope"] = scope
    return hinted


# --------------------------------------------------------- header values

_SENTINEL_PREFIX = "=?base64?"
_SENTINEL_SUFFIX = "?="


def decode_header_value(raw: str) -> str:
    """Decode a standard MCP header value, undoing the Base64 sentinel.

    A value that cannot travel as plain ASCII arrives as
    ``=?base64?<base64 of UTF-8>?=``. The markers are case-sensitive and
    lowercase. A value that is not in that form is returned unchanged.
    """
    if raw.startswith(_SENTINEL_PREFIX) and raw.endswith(_SENTINEL_SUFFIX):
        payload = raw[len(_SENTINEL_PREFIX) : -len(_SENTINEL_SUFFIX)]
        try:
            return base64.b64decode(payload, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
            raise ProtocolError(
                HEADER_MISMATCH,
                f"Malformed Base64 header value: {exc}",
            ) from exc
    return raw


def _header_value_is_transmittable(value: str) -> bool:
    """True when *value* is legal in an HTTP field value without encoding.

    RFC 9110 admits visible ASCII, space and horizontal tab, and forbids
    leading or trailing whitespace.
    """
    if value != value.strip(" \t"):
        return False
    return all(ch == "\t" or 0x20 <= ord(ch) <= 0x7E for ch in value)


# The methods whose `Mcp-Name` header is required, and the params member each
# one sources it from.
NAME_HEADER_SOURCE = {
    "tools/call": "name",
    "resources/read": "uri",
    "prompts/get": "name",
}


def validate_request_headers(
    headers: dict[str, str], message: dict[str, Any], meta: dict[str, Any]
) -> None:
    """Check the standard request headers against the request body.

    An intermediary may route on a header while the server executes on the
    body, so the two must agree or the request is refused. Raises
    ``ProtocolError`` with HEADER_MISMATCH on any disagreement, missing
    required header, or untransmittable value.
    """

    def header(name: str) -> str | None:
        # RFC 9110 field names are case-insensitive.
        lowered = name.lower()
        for key, value in headers.items():
            if key.lower() == lowered:
                return value
        return None

    method = message.get("method")

    version_header = header("MCP-Protocol-Version")
    if version_header is None:
        raise ProtocolError(
            HEADER_MISMATCH,
            "Missing required header: MCP-Protocol-Version",
        )
    body_version = meta.get(META_PROTOCOL_VERSION)
    if version_header != body_version:
        raise ProtocolError(
            HEADER_MISMATCH,
            f"Header mismatch: MCP-Protocol-Version header value "
            f"{version_header!r} does not match body value {body_version!r}",
        )

    method_header = header("Mcp-Method")
    if method_header is None:
        raise ProtocolError(HEADER_MISMATCH, "Missing required header: Mcp-Method")
    if method_header != method:
        raise ProtocolError(
            HEADER_MISMATCH,
            f"Header mismatch: Mcp-Method header value {method_header!r} "
            f"does not match body value {method!r}",
        )

    source = NAME_HEADER_SOURCE.get(method or "")
    if source is None:
        return

    name_header = header("Mcp-Name")
    if name_header is None:
        raise ProtocolError(
            HEADER_MISMATCH,
            f"Missing required header: Mcp-Name (required for {method})",
        )
    if not _header_value_is_transmittable(name_header):
        raise ProtocolError(
            HEADER_MISMATCH,
            "Mcp-Name contains characters that require the Base64 sentinel encoding",
        )
    params = message.get("params")
    body_name = params.get(source) if isinstance(params, dict) else None
    if decode_header_value(name_header) != body_name:
        raise ProtocolError(
            HEADER_MISMATCH,
            f"Header mismatch: Mcp-Name header value {name_header!r} does "
            f"not match body value {body_name!r}",
        )


# --------------------------------------------------- envelope and `_meta`


def validate_envelope(message: Any) -> None:
    """Check *message* is a single JSON-RPC request or notification.

    One POST carries exactly one of those. An array (the batch form), a
    response frame sent by a client, and a request whose id is null are all
    rejected here rather than being partially served.
    """
    if isinstance(message, list):
        raise ProtocolError(
            INVALID_REQUEST,
            "JSON-RPC batching is not supported: send one request or "
            "notification per POST",
        )
    if not isinstance(message, dict):
        raise ProtocolError(INVALID_REQUEST, "Request body must be a JSON-RPC object")
    if message.get("jsonrpc") != "2.0":
        raise ProtocolError(INVALID_REQUEST, 'Request body must carry "jsonrpc": "2.0"')
    if "result" in message or "error" in message:
        raise ProtocolError(
            INVALID_REQUEST,
            "Clients must not send JSON-RPC responses to this endpoint",
        )
    method = message.get("method")
    if not isinstance(method, str) or not method:
        raise ProtocolError(
            INVALID_REQUEST, 'Request body must carry a non-empty "method"'
        )
    if "id" in message and message["id"] is None:
        raise ProtocolError(
            INVALID_REQUEST,
            "A request id must be a string or a number; use a notification "
            '(omit "id") when no response is wanted',
        )
    if "id" in message and (
        isinstance(message["id"], bool)
        or not isinstance(message["id"], (str, int, float))
    ):
        # bool is a subclass of int, so it passes an isinstance check written
        # for numbers unless it is excluded first.
        raise ProtocolError(
            INVALID_REQUEST, "A request id must be a string or a number"
        )
    params = message.get("params")
    if params is not None and not isinstance(params, dict):
        raise ProtocolError(INVALID_REQUEST, '"params" must be an object')


def request_meta(message: dict[str, Any]) -> dict[str, Any]:
    """The `_meta` object of a request's params, or an empty dict."""
    params = message.get("params")
    if not isinstance(params, dict):
        return {}
    meta = params.get("_meta")
    return meta if isinstance(meta, dict) else {}


def era_of_request(message: dict[str, Any], headers: dict[str, str]) -> str:
    """Which protocol era *message* is written in.

    A request is modern when it describes itself: either its `_meta` carries
    a protocol version, or its ``MCP-Protocol-Version`` header names a version
    from the modern era. Anything else is legacy, which covers the
    ``initialize`` handshake and every request that follows it.

    The header is consulted as well as the body so that a modern client which
    omits `_meta` is answered with the specific error naming what is missing,
    rather than being served silently under legacy rules.
    """
    if request_meta(message).get(META_PROTOCOL_VERSION) is not None:
        return ERA_MODERN
    for key, value in headers.items():
        if key.lower() == "mcp-protocol-version" and value >= MODERN_ERA_FLOOR:
            return ERA_MODERN
    return ERA_LEGACY


def validate_modern_meta(meta: dict[str, Any]) -> None:
    """Check the `_meta` fields every modern request is required to carry.

    ``protocolVersion`` and ``clientCapabilities`` are both required. A
    version the server does not serve is refused by name, with the list of
    versions it does serve, so the client can pick one and retry.
    """
    version = meta.get(META_PROTOCOL_VERSION)
    if not isinstance(version, str) or not version:
        raise ProtocolError(
            INVALID_PARAMS,
            f"Missing required request metadata: params._meta[{META_PROTOCOL_VERSION!r}]",
        )
    if version not in MODERN_PROTOCOL_VERSIONS:
        raise ProtocolError(
            UNSUPPORTED_PROTOCOL_VERSION,
            "Unsupported protocol version",
            {
                "supported": list(MODERN_PROTOCOL_VERSIONS),
                "requested": version,
            },
        )
    capabilities = meta.get(META_CLIENT_CAPABILITIES)
    if not isinstance(capabilities, dict):
        raise ProtocolError(
            INVALID_PARAMS,
            f"Missing required request metadata: params._meta[{META_CLIENT_CAPABILITIES!r}]",
        )


def reject_unknown_cursor(params: dict[str, Any]) -> None:
    """Refuse a pagination cursor, because this server mints none.

    Every list this server serves fits in one page, so it never returns a
    ``nextCursor``. A cursor arriving here was therefore not issued by this
    server, and serving page one in answer to it would silently repeat
    results the client believes it has already read.
    """
    if "cursor" in params and params.get("cursor") is not None:
        raise ProtocolError(
            INVALID_PARAMS,
            "Invalid cursor: this server returns every list in a single page "
            "and issues no pagination cursors",
        )
