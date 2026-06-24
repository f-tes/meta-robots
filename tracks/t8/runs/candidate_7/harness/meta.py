"""
meta.py — Machine-readable hypothesis metadata for candidate_7.

Read by run_analyzer to correlate SR results with mechanism descriptions.
Do NOT add executable code here.
"""

TARGET_FAILURE_CLASSES = ["floor_exploration_budget_window"]

TARGET_SCENES = ["p53SfW6mjZe", "mL8ThkuaVTM", "XB4GS9ShBRE"]

HYPOTHESIS = (
    "The stair-mode transition fires at the wrong floor_step in three scenes: "
    "p53SfW6mjZe exhausts 380 of 500 steps on floor-0 (TV is on floor-1), leaving only "
    "13 upper-floor steps after a confirmed physical floor change; mL8ThkuaVTM's "
    "look_for_downstair trigger fires at floor_step=47 on floor-2 before the couch room "
    "frontier is reached, causing premature oscillation; XB4GS9ShBRE commits to upstairs "
    "after only 22-47 floor_steps, abandoning the couch present on the starting floor. "
    "All three share a missing floor_step budget gate: no minimum exploration requirement "
    "before stair entry and no maximum cap before forced transition."
)

MECHANISM = (
    "floor.py Fix 7 (T8) — Floor-step budget window [MIN=80, MAX=200]: "
    "(1) should_force_floor_switch_by_coverage returns True when steps_on_floor > "
    "_MAX_FLOOR_STEPS_BEFORE_FORCED_STAIR=200 AND frontiers remain (MAX gate); "
    "(2) patch.py Fix 5 wires the MAX gate: patches _explore after Fix 1; before LLM "
    "frontier selection, checks get_harness().should_force_floor_switch_by_coverage() "
    "and forces _navigate_stair_if_unexplored_floor (up then down) when it returns True. "
    "Log tag: [T8_FLOOR_BUDGET_MAX]. "
    "(3) patch.py Fix 6 wires the MIN gate: patches _navigate_stair_if_unexplored_floor "
    "to return None when _floor_num_steps < _MIN_FLOOR_STEPS_ON_FLOOR=80, blocking "
    "early upstair/downstair navigation from the frontier-exhaustion path. "
    "Log tag: [T8_FLOOR_BUDGET_MIN_NAV]. "
    "(4) patch.py Fix 7 wires MIN gate into _look_for_downstair: when floor_steps < 80, "
    "clears _look_for_downstair_flag and falls back to _explore. "
    "Log tag: [T8_FLOOR_BUDGET_MIN_LFD]. "
    "Only floor.py and patch.py are changed."
)

PREDICTED_CHANGE = (
    "p53SfW6mjZe: [T8_FLOOR_BUDGET_MAX] must appear in logs at floor_step<=200 (vs 380 "
    "baseline); subsequent mode must show 'get_close_to_stair'; upper-floor floor_step "
    "count must exceed 100 (vs 13 baseline). "
    "mL8ThkuaVTM: [T8_FLOOR_BUDGET_MIN_LFD] must appear at floor_step=47 suppressing "
    "look_for_downstair; 'look_for_downstair' mode must not appear before floor_step=80 "
    "on floor-2; couch room frontier vicinity must be visited before first downstair. "
    "XB4GS9ShBRE: [T8_FLOOR_BUDGET_MIN_NAV] must appear suppressing upstair navigation "
    "before floor_step=80; couch must generate a STOP candidate on starting floor."
)

PREDICTED_SR_DELTA = 0.1

WHY_THIS_WILL_WORK = (
    "p53SfW6mjZe: capping floor-0 at 200 steps provides 200+ upper-floor steps vs "
    "current 13 (14x budget increase); TV is confirmed physically reachable post-stair-"
    "climb (T6 SUCCESS at step ~393), so additional budget directly translates to "
    "exploration coverage. "
    "mL8ThkuaVTM: untested analysis explicitly states 'suppressing look_for_downstair "
    "trigger until floor_step>=80 could allow the couch room frontier to be reached "
    "before downstairs oscillation begins'; stair traversal already succeeds in "
    "candidates 1, 4, 5, and 6 so the sole residual failure is the premature downstair "
    "trigger at floor_step=47. "
    "XB4GS9ShBRE: the untested list identifies 'minimum starting-floor coverage gate "
    "before stair entry' as 'the sole untested lever that addresses the causal bottleneck "
    "at step 64'; couch is on the starting floor and would be found during the extended "
    "exploration window [22-80 additional floor steps]."
)

WHY_ALTERNATIVES_REJECTED = (
    "DP12 as sole gate is ruled out for p53SfW6mjZe; DP12 is ruled out entirely for "
    "XB4GS9ShBRE — but the proposed floor_step pre-condition is a NEW gate evaluated "
    "before DP12, consistent with the untested item description. "
    "(hooks.py, STOP_gate_calibration) and (hooks.py, per_category_stop_gate) both "
    "scored 0.4 with no improvement. (patch.py, false_positive_stop) scored 0.1 SR "
    "(severe regression). stair.py T6_EARLY_STAIR_DISABLE streak threshold fixes only "
    "mL8ThkuaVTM and provides no lever for p53SfW6mjZe or XB4GS9ShBRE. floor.py has "
    "not been targeted in any prior T8 candidate, making (floor.py, "
    "floor_exploration_budget_window) a guaranteed new (file, class) pair."
)

FALSIFIABILITY_CHECK = (
    "p53SfW6mjZe: logs must show get_close_to_stair transition at floor_step<=200 "
    "(vs 380 baseline); upper-floor floor_step count must exceed 100 (vs 13 baseline). "
    "mL8ThkuaVTM: look_for_downstair must NOT appear in logs before floor_step=80 on "
    "floor-2; couch room frontier [-2.53, 1.30] vicinity must be visited before any "
    "downstair transition. "
    "XB4GS9ShBRE: first stair-entry log must appear no earlier than floor_step=80 on "
    "starting floor (vs 22-47 baseline); couch on starting floor must generate a STOP "
    "candidate before stair entry."
)
