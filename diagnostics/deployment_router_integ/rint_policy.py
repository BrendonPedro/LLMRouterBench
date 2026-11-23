"""ROUTER-INTEG: the placement-policy adapter and the data-policy feasibility of every execution mode.

Two data policies are evaluated and reported separately; neither is switched after a result:
  A  no raw query transfer: the query text and its token ids stay on the edge.
  B  raw query transfer allowed.
What an execution mode sends off the edge decides its feasibility. REMOTE sends the token ids of the query, which are
the raw query, so it violates A. EDGE sends nothing. A layer split would send hidden states and no token ids; it is
not run in this protocol (no correctness gate exists for a layer split of either model of the pair).

Placement is one fixed rule per data policy (rint_spec.PLACEMENT), the same for both models, fixed before any execution
of this pair, and never fitted to the evaluation records. This module decides nothing else.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rint_spec as S  # noqa: E402

SENDS = {S.EDGE: "nothing", S.REMOTE: "token ids of the query (raw query)", S.SPLIT: "hidden states (no token ids)"}


def feasible(mode, policy):
    """Data-policy feasibility of an execution mode: True, or False with the reason."""
    if policy not in S.MODE_FEASIBLE:
        raise ValueError(policy)
    if mode == S.SPLIT:
        return False, "not run: no correctness gate has been passed for a layer split of this model pair"
    if mode in S.MODE_FEASIBLE[policy]:
        return True, None
    return False, f"{mode} sends {SENDS[mode]}, which data policy {policy} prohibits"


def config_of(model, mode):
    for name, (m, x) in S.CONFIGS.items():
        if (m, x) == (model, mode):
            return name
    raise ValueError(f"no configuration for {model} on {mode}")


def place(model, policy, server_available=True):
    """The execution decision for `model` under data policy `policy`.
    Returns {"config", "mode", "fallback", "infeasible"}. A mode that the data policy prohibits is never returned."""
    if model not in S.MODELS:
        raise ValueError(f"{model} is not a model of the router's pair")
    mode = S.PLACEMENT[policy]
    ok, why = feasible(mode, policy)
    if not ok:
        raise AssertionError(f"the frozen placement violates its own data policy: {why}")
    if mode == S.REMOTE and not server_available:
        fb = S.PLACEMENT_FALLBACK[policy]
        if fb is None or not feasible(fb, policy)[0]:
            return {"config": None, "mode": None, "fallback": False, "infeasible": "the server does not answer and the data policy admits no other mode"}
        return {"config": config_of(model, fb), "mode": fb, "fallback": True, "infeasible": None}
    return {"config": config_of(model, mode), "mode": mode, "fallback": False, "infeasible": None}


def matrix():
    """Model x execution mode x data policy: feasible or not, and why."""
    out = []
    for model in S.MODEL_ORDER:
        for mode in (S.EDGE, S.REMOTE, S.SPLIT):
            row = {"model": S.MODELS[model]["name"], "mode": mode, "sends_off_edge": SENDS[mode]}
            for pol in (S.POLICY_A, S.POLICY_B):
                ok, why = feasible(mode, pol)
                row[pol] = "feasible" if ok else f"infeasible: {why}"
            row["placed_under"] = [pol for pol in (S.POLICY_A, S.POLICY_B) if S.PLACEMENT[pol] == mode]
            out.append(row)
    return out
