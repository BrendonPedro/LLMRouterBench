"""E1-OPT command hub. Every subcommand is implemented in one module; this file only dispatches.

Stage 0   inventory collect | inventory roles               (e1_inventory)   no GPU code, no model
Stage 1a  freeze | check-freeze                             (e1_freeze)      written once; verified by every later command
Stage 1b  equivalence reference|local|joint|decide          (e1_equivalence) correctness only; no timing fields
Stage 2   profile-server-segment                            (e1_profile)     server GPU, synthetic activations
Stage 3   fit-split | fit-competitor | profile-edge-segment | summarize-fit   (e1_profile)  edge machine, capped, fresh
          gate-ub                                           (e1_gate)        G-UB-E1 with provenance check + rounded rule
Stage 4   serve | run-cell | run-remote-cell | linkcheck    (e1_joint)       the lab pair; frozen cells only
          gate-r                                            (e1_gate)        G-R with provenance check + rounded rule

Every GPU-executing command requires --freeze <frozen/runtime_freeze.json>; no freeze record, no run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path

ROUTES = {
    "inventory": ("e1_inventory", None),
    "freeze": ("e1_freeze", "freeze"), "check-freeze": ("e1_freeze", "check-freeze"),
    "equivalence": ("e1_equivalence", None),
    "fit-split": ("e1_profile", "fit-split"), "fit-competitor": ("e1_profile", "fit-competitor"),
    "profile-edge-segment": ("e1_profile", "profile-edge-segment"), "profile-server-segment": ("e1_profile", "profile-server-segment"),
    "summarize-fit": ("e1_profile", "summarize-fit"),
    "serve": ("e1_joint", "serve"), "run-cell": ("e1_joint", "run-cell"), "run-remote-cell": ("e1_joint", "run-remote-cell"),
    "linkcheck": ("e1_joint", "linkcheck"),
    "gate-ub": ("e1_gate", "gate-ub"), "gate-r": ("e1_gate", "gate-r"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in ROUTES:
        print(__doc__); print("commands:", ", ".join(ROUTES)); sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    mod_name, sub = ROUTES[argv[0]]
    mod = __import__(mod_name)
    mod.main(([sub] if sub else []) + argv[1:])


if __name__ == "__main__":
    main()
