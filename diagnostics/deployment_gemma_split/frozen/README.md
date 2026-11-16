# frozen/ (empty until Stage 1)

`freeze_record.json` is written once by `gs_bench.py freeze` from commit A (capture/ + adapter/) and committed alone
in commit B. A record is never edited, renamed, deleted or overwritten (git history is checked), and no other
record-shaped file may exist here. A new record `freeze_record.rN.json` (`--supersede-reason` required; at most three
records) is accepted only when no run of the chain is final and the previous run produced nothing, ended INVALID for a
cause that needs a new record, or crashed at the next stage because of an orchestration defect, shown by both hosts'
committed exit-status logs (PREREG §6.4). A new record counts as a repeat of its causes; repeat caps carry over. The superseded record can no longer drive a command; its run stays in `runs/` and in the archive
and is listed as history with the outcome.

`HARNESS_FIXES.md` (only if needed): one line per orchestration, test or adapter file that differs from the tag,
`<path> | <fix commit> | <regression test> | <reason>`. Decision files (`gs_spec.DECISION_FILES`, including the
measurement drivers) can never differ from the tag, documented or not.
