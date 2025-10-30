"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Simple Fernet-based encryption helpers.
"""

from __future__ import annotations

import base64
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken


def get_fernet(encryption_key: Optional[str]) -> Fernet:
    if not encryption_key:
        raise RuntimeError(
            "ENCRYPTION_KEY is required for token storage. Generate one with: "
            "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
        )
    # Accept already-base64 key
    key_bytes = encryption_key.encode()
    return Fernet(key_bytes)


def encrypt_text(fernet: Fernet, text: str) -> str:
    return fernet.encrypt(text.encode()).decode()


def decrypt_text(fernet: Fernet, token: str) -> str:
    try:
        return fernet.decrypt(token.encode()).decode()
    except InvalidToken as e:
        raise RuntimeError("Invalid encryption token; check ENCRYPTION_KEY") from e
