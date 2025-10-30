"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

UI helper functions for Streamlit pages.

Consolidates common UI utilities like dialog factories and rerun triggers.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import streamlit as st


def get_dialog_factory() -> Optional[Callable]:
    """Get Streamlit dialog factory function if available.

    Checks for both st.dialog (newer API) and st.experimental_dialog (older API).

    Returns:
        Optional[Callable]: Dialog factory function or None if not available
    """
    if hasattr(st, "dialog"):
        return getattr(st, "dialog")
    if hasattr(st, "experimental_dialog"):
        return getattr(st, "experimental_dialog")
    return None


def trigger_rerun() -> None:
    """Trigger a Streamlit rerun.

    Convenience wrapper for st.rerun().
    """
    st.rerun()

