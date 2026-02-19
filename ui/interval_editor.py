"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Shared Streamlit widgets for editing interval loop structures.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from services.interval_utils import (
    clone_steps,
    new_loop,
    new_recovery_action,
    new_run_action,
    normalize_steps,
    serialize_steps,
)
from utils.constants import INTERVAL_TARGET_TYPES

Steps = Dict[str, Any]
Action = Dict[str, Any]

def _state_key(prefix: str) -> str:
    return f"{prefix}-interval-editor-state"


def _source_key(prefix: str) -> str:
    return f"{prefix}-interval-editor-source"


def _parse_steps_value(steps_value: Any) -> Dict[str, Any]:
    if isinstance(steps_value, dict):
        return steps_value
    if isinstance(steps_value, str):
        try:
            parsed = json.loads(steps_value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _ensure_state(prefix: str, steps_value: Any) -> Steps:
    state_key = _state_key(prefix)
    source_key = _source_key(prefix)
    raw = _parse_steps_value(steps_value)
    try:
        payload_hash = json.dumps(raw, sort_keys=True)
    except Exception:
        payload_hash = ""
    if st.session_state.get(source_key) != payload_hash or state_key not in st.session_state:
        st.session_state[state_key] = normalize_steps(raw)
        st.session_state[source_key] = payload_hash
    return st.session_state[state_key]


def _set_state(prefix: str, steps: Steps) -> None:
    st.session_state[_state_key(prefix)] = steps


def _ensure_minimum_blocks(blocks: List[Action], *, defaults: List[Action]) -> None:
    if not blocks:
        blocks.extend(defaults)


def _render_kind_selector(action: Action, key: str) -> str:
    kind_options = ["run", "recovery"]
    current_kind = action.get("kind") if action.get("kind") in kind_options else "run"
    idx = kind_options.index(current_kind)
    new_kind = st.selectbox("Type", kind_options, index=idx, key=f"{key}-kind")
    if new_kind != current_kind:
        action["kind"] = new_kind
        if new_kind == "recovery":
            action["targetType"] = None
            action["targetLabel"] = None
            action["ascendM"] = 0
            action["descendM"] = 0
        else:
            action.setdefault("targetType", "pace")
            action.setdefault("targetLabel", None)
    return new_kind


def _target_inputs(
    action: Action,
    *,
    key: str,
    threshold_names: List[str],
) -> None:
    target_type = action.get("targetType") or "pace"
    if target_type not in INTERVAL_TARGET_TYPES:
        target_type = "pace"
    idx = INTERVAL_TARGET_TYPES.index(target_type)
    new_target = st.selectbox(
        "Cible",
        INTERVAL_TARGET_TYPES,
        index=idx,
        key=f"{key}-target-type",
    )
    if new_target == "none":
        action["targetType"] = None
        action["targetLabel"] = None
        return
    action["targetType"] = new_target

    if new_target in {"pace", "hr", "threshold"}:
        choices = threshold_names or ["Fundamental", "Threshold 30", "Threshold 60"]
        if action.get("targetLabel") not in choices:
            action["targetLabel"] = choices[0]
        label_idx = choices.index(action.get("targetLabel"))
        action["targetLabel"] = st.selectbox(
            "Seuil",
            choices,
            index=label_idx,
            key=f"{key}-target-label",
        )
    else:
        default_label = action.get("targetLabel") or ""
        action["targetLabel"] = (
            st.text_input(
                "Valeur",
                value=str(default_label),
                key=f"{key}-target-value",
            ).strip()
            or None
        )


def _render_action_card(
    action: Action,
    *,
    key: str,
    threshold_names: List[str],
    allow_delete: bool,
    delete_callback: Optional[Callable[[], None]],
) -> None:
    st.markdown("<div class='rm-interval-action'>", unsafe_allow_html=True)
    cols = st.columns([1, 1, 2, 1, 1])
    with cols[0]:
        kind = _render_kind_selector(action, key)
    with cols[1]:
        sec = st.number_input(
            "Temps (sec)",
            min_value=5,
            value=int(action.get("sec") or 60),
            step=5,
            key=f"{key}-sec",
        )
        action["sec"] = int(sec)
    if kind == "run":
        with cols[2]:
            _target_inputs(action, key=key, threshold_names=threshold_names)
        with cols[3]:
            asc = st.number_input(
                "D+ (m)",
                min_value=0,
                value=int(action.get("ascendM") or 0),
                step=5,
                key=f"{key}-asc",
            )
            action["ascendM"] = int(asc)
        with cols[4]:
            desc = st.number_input(
                "D- (m)",
                min_value=0,
                value=int(action.get("descendM") or 0),
                step=5,
                key=f"{key}-desc",
            )
            action["descendM"] = int(desc)
    else:
        action["targetType"] = None
        action["targetLabel"] = None
        action["ascendM"] = 0
        action["descendM"] = 0
        cols[2].markdown("—")
        cols[3].markdown("—")
        cols[4].markdown("—")

    if allow_delete and delete_callback:
        if st.button("🗑️", key=f"{key}-delete", help="Supprimer l'action"):
            delete_callback()
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_interval_editor(
    prefix: str,
    steps_value: Any,
    threshold_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Render the interval loops editor and return a serialized dictionary."""
    threshold_names = threshold_names or []
    state = _ensure_state(prefix, steps_value)

    st.markdown("<div class='rm-interval-editor'>", unsafe_allow_html=True)
    st.markdown("#### Avant les boucles")
    pre_blocks = state.get("preBlocks") or []
    _ensure_minimum_blocks(
        pre_blocks,
        defaults=[new_run_action(sec=600, target_type="sensation", target_label="Fundamental")],
    )

    for idx, block in enumerate(pre_blocks):
        block_key = f"{prefix}-pre-{idx}"

        def _delete_block() -> None:
            new_state = clone_steps(state)
            new_state["preBlocks"].pop(idx)
            _ensure_minimum_blocks(
                new_state["preBlocks"],
                defaults=[
                    new_run_action(sec=600, target_type="sensation", target_label="Fundamental")
                ],
            )
            _set_state(prefix, new_state)

        _render_action_card(
            block,
            key=block_key,
            threshold_names=threshold_names,
            allow_delete=len(pre_blocks) > 1,
            delete_callback=_delete_block if len(pre_blocks) > 1 else None,
        )

    if st.button("Ajouter un bloc avant", key=f"{prefix}-add-pre"):
        new_state = clone_steps(state)
        new_state["preBlocks"].append(
            new_run_action(sec=300, target_type="sensation", target_label="Fundamental")
        )
        _set_state(prefix, new_state)
        st.rerun()

    st.divider()
    st.markdown("#### Boucles")
    loops = state.get("loops") or []
    _ensure_minimum_blocks(loops, defaults=[new_loop()])

    for loop_idx, loop in enumerate(loops):
        loop_key = f"{prefix}-loop-{loop_idx}"
        st.markdown("<div class='rm-loop-card'>", unsafe_allow_html=True)
        header_cols = st.columns([2, 1, 0.2])
        with header_cols[0]:
            st.markdown(f"**Boucle {loop_idx + 1}**")
        with header_cols[1]:
            repeats = st.number_input(
                "Répétitions",
                min_value=1,
                value=int(loop.get("repeats") or 1),
                step=1,
                key=f"{loop_key}-repeats",
            )
            loop["repeats"] = int(repeats)
        with header_cols[2]:
            if len(loops) > 1 and st.button(
                "🗑️", key=f"{loop_key}-delete", help="Supprimer la boucle"
            ):
                new_state = clone_steps(state)
                new_state["loops"].pop(loop_idx)
                _ensure_minimum_blocks(new_state["loops"], defaults=[new_loop()])
                _set_state(prefix, new_state)
                st.rerun()

        actions = loop.get("actions") or []
        _ensure_minimum_blocks(actions, defaults=[new_run_action(), new_recovery_action()])
        loop["actions"] = actions

        for action_idx, action in enumerate(actions):
            action_key = f"{loop_key}-action-{action_idx}"

            def _delete_action(loop_index: int = loop_idx, act_index: int = action_idx) -> None:
                new_state = clone_steps(state)
                new_actions = new_state["loops"][loop_index]["actions"]
                new_actions.pop(act_index)
                _ensure_minimum_blocks(
                    new_actions, defaults=[new_run_action(), new_recovery_action()]
                )
                _set_state(prefix, new_state)

            _render_action_card(
                action,
                key=action_key,
                threshold_names=threshold_names,
                allow_delete=len(actions) > 1,
                delete_callback=_delete_action if len(actions) > 1 else None,
            )
        if st.button("Ajouter une action", key=f"{loop_key}-add-action"):
            new_state = clone_steps(state)
            new_state["loops"][loop_idx]["actions"].append(new_run_action())
            _set_state(prefix, new_state)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Ajouter une boucle", key=f"{prefix}-add-loop"):
        new_state = clone_steps(state)
        new_state["loops"].append(new_loop())
        _set_state(prefix, new_state)
        st.rerun()

    st.divider()
    st.markdown("#### Entre les boucles")
    between = state.get("betweenBlock")
    if between and isinstance(between, dict):
        between_key = f"{prefix}-between"
        _render_action_card(
            between,
            key=between_key,
            threshold_names=threshold_names,
            allow_delete=True,
            delete_callback=lambda: _set_state(prefix, {**state, "betweenBlock": None}),
        )
        if st.button("Supprimer le bloc entre boucles", key=f"{prefix}-remove-between"):
            new_state = clone_steps(state)
            new_state["betweenBlock"] = None
            _set_state(prefix, new_state)
            st.rerun()
    else:
        if st.button("Ajouter un bloc entre boucles", key=f"{prefix}-add-between"):
            new_state = clone_steps(state)
            new_state["betweenBlock"] = new_recovery_action(sec=45)
            _set_state(prefix, new_state)
            st.rerun()

    st.divider()
    st.markdown("#### Après les boucles")
    post_blocks = state.get("postBlocks") or []
    _ensure_minimum_blocks(post_blocks, defaults=[new_recovery_action(sec=300)])

    for idx, block in enumerate(post_blocks):
        block_key = f"{prefix}-post-{idx}"

        def _delete_post() -> None:
            new_state = clone_steps(state)
            new_state["postBlocks"].pop(idx)
            _ensure_minimum_blocks(new_state["postBlocks"], defaults=[new_recovery_action(sec=300)])
            _set_state(prefix, new_state)

        _render_action_card(
            block,
            key=block_key,
            threshold_names=threshold_names,
            allow_delete=len(post_blocks) > 1,
            delete_callback=_delete_post if len(post_blocks) > 1 else None,
        )

    if st.button("Ajouter un bloc après", key=f"{prefix}-add-post"):
        new_state = clone_steps(state)
        new_state["postBlocks"].append(new_recovery_action(sec=300))
        _set_state(prefix, new_state)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    return serialize_steps(state)
