from __future__ import annotations

import os
import ssl
from dataclasses import dataclass, field
from importlib.metadata import distribution

READONLY_TOKEN_ENV = "TINVEST_READONLY_TOKEN"
SANDBOX_TOKEN_ENV = "TINVEST_SANDBOX_TOKEN"
TBANK_TLS_VERIFY_ENV = "SSL_TBANK_VERIFY"


class MissingTokenError(RuntimeError):
    def __init__(self, env_name: str) -> None:
        self.env_name = env_name
        super().__init__(env_name)


@dataclass(frozen=True, slots=True)
class TInvestTokens:
    readonly: str = field(repr=False)
    sandbox: str = field(repr=False)


def load_readonly_token() -> str:
    return _required_secret(READONLY_TOKEN_ENV)


def load_sandbox_token() -> str:
    return _required_secret(SANDBOX_TOKEN_ENV)


def load_tokens() -> TInvestTokens:
    return TInvestTokens(readonly=load_readonly_token(), sandbox=load_sandbox_token())


def token_presence() -> dict[str, bool]:
    return {
        "readonly_token_detected": bool(os.getenv(READONLY_TOKEN_ENV, "").strip()),
        "sandbox_token_detected": bool(os.getenv(SANDBOX_TOKEN_ENV, "").strip()),
    }


def tbank_tls_context() -> ssl.SSLContext:
    if os.getenv(TBANK_TLS_VERIFY_ENV, "").strip().lower() != "true":
        raise RuntimeError(TBANK_TLS_VERIFY_ENV)
    certificate = distribution("t-tech-investments").locate_file(
        "t_tech/invest/certs/RussianTrustedRootCA.pem"
    )
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=str(certificate))
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise RuntimeError("TINVEST_TLS_VERIFICATION_REQUIRED")
    return context


def _required_secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise MissingTokenError(name)
    return value
