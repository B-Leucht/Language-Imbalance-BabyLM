# Agent Experiment Log

Maintained by the Claude Code agent running on Lambda Labs.
Append timestamped entries throughout the session.
Format: `[YYYY-MM-DD HH:MM:SS] message`

See `CLAUDE.md` for project context and task description.

---

## Pre-session: Existing Results Summary

**Monolingual scaling laws** (tiny config, 10 epochs, 3 languages):
- eng: ppl at 10M tokens ≈ 100 (see scaling_results.csv)
- dut: similar curve
- ind: similar curve

**Bilingual eng-dut** (tiny config, 10M tokens, 5 ratios):
- Dutch MLTE peaks ~1.4–1.5 at 90/10 ratio
- At 99/1, Dutch MLTE ≈ 1.0 (no benefit)
- English mildly hurt but recovers at high eng ratio

**Open questions entering this session:**
- Does eng-ind show weaker transfer than eng-dut (language distance hypothesis)?
- Does trilingual training amplify or dilute minority-language transfer?
- What are the optimal hyperparameters for the small config on GPU?

---

<!-- Agent: append new entries below this line -->
