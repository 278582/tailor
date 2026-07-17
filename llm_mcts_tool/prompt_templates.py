from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def _prompt_pack_dir(context: dict[str, Any]) -> Path:
    return Path(context.get("_prompt_pack_dir") or "prompt_pack")


def _render_pack_template(context: dict[str, Any], template_name: str, **values: Any) -> str | None:
    template_dir = _prompt_pack_dir(context) / "templates"
    template_path = template_dir / template_name
    if not template_path.exists():
        return None
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    env.filters["tojson"] = lambda value, indent=None: json.dumps(value, ensure_ascii=False, indent=indent)
    payload = {key: value for key, value in context.items() if not key.startswith("_")}
    payload.update(values)
    return env.get_template(template_name).render(**payload)


def init_prompt(context: dict[str, Any], n_init: int) -> str:
    rendered = _render_pack_template(context, "init_prompt.j2", n_init=int(n_init))
    if rendered is not None:
        return rendered
    return (
        "You propose initial theta strategies for theta-guided Pareto postprocessing.\n"
        "Return JSON only with key proposals.\n"
        f"Generate exactly {int(n_init)} proposals. Each proposal must contain theta, prior_score, reason.\n"
        "theta fields: col_1ds, col_2ds, col_ps, col_u. Use only non-target feature columns.\n\n"
        f"Context:\n{context}\n"
        "JSON schema shape:\n"
        '{"proposals":[{"theta":{"col_1ds":[],"col_2ds":[],"col_ps":[],"col_u":""},'
        '"prior_score":0.5,"reason":"short"}]}'
    )


def refine_prompt(context: dict[str, Any], n_expand: int) -> str:
    rendered = _render_pack_template(context, "refine_prompt.j2", n_expand=int(n_expand))
    if rendered is not None:
        return rendered
    return (
        "You refine one parent theta for theta-guided Pareto postprocessing.\n"
        "Return JSON only with key proposals.\n"
        f"Generate exactly {int(n_expand)} child proposals. Each proposal must contain actions, theta, prior_score, reason.\n"
        "Allowed actions: add_col_1d, replace_col_1d, add_col_2d, replace_col_2d, "
        "add_col_p, replace_col_p, replace_col_u.\n\n"
        f"Context:\n{context}\n"
        "JSON schema shape:\n"
        '{"proposals":[{"actions":[{"type":"replace_col_u","old":"","new":""}],'
        '"theta":{"col_1ds":[],"col_2ds":[],"col_ps":[],"col_u":""},'
        '"prior_score":0.5,"reason":"short"}]}'
    )


def prior_only_scoring_prompt(context: dict[str, Any]) -> str:
    return (
        "Score the potential of this theta before rollout. Return JSON only.\n"
        "The score is a prior for MCTS exploration, not an observed reward.\n"
        "Use a number in [0, 1].\n\n"
        f"Context:\n{context}\n"
        'JSON schema shape: {"prior_score":0.5,"reason":"short"}'
    )
