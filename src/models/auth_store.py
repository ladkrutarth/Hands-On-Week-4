"""
Simple in-memory user store for the demo login flow.

This is NOT production-ready auth. It exists to:
- Demonstrate username/password verification
- Provide roles for different demo personas
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Optional


_PBKDF_SALT = b"veriscan-demo-salt"  # static salt for demo; do NOT copy to prod
_PBKDF_ITER = 100_000


def _hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC (demo only)."""
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), _PBKDF_SALT, _PBKDF_ITER)
    return dk.hex()


@dataclass
class UserRecord:
    username: str
    password_hash: str
    role: str = "viewer"


class DemoUserStore:
    """
    Extremely small in-memory user store.

    Backed by a dict of username -> UserRecord. No persistence; data resets
    when the API process restarts.
    """

    def __init__(self) -> None:
        self._users: Dict[str, UserRecord] = {}
        self._seed()

    def _seed(self) -> None:
        # Seed a few demo users. Passwords are simple on purpose for the demo.
        demo_users = {
            "admin": ("admin123!", "admin"),
            "analyst": ("analyst123", "analyst"),
            "viewer": ("viewer123", "viewer"),
        }
        for username, (pw, role) in demo_users.items():
            self._users[username] = UserRecord(
                username=username,
                password_hash=_hash_password(pw),
                role=role,
            )

    def verify_user(self, username: str, password: str) -> bool:
        rec = self._users.get(username)
        if not rec:
            return False
        return _hash_password(password) == rec.password_hash

    def get_user_role(self, username: str) -> Optional[str]:
        rec = self._users.get(username)
        return rec.role if rec else None


# Singleton-style helper for the API
_GLOBAL_STORE: Optional[DemoUserStore] = None


def get_user_store() -> DemoUserStore:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = DemoUserStore()
    return _GLOBAL_STORE

