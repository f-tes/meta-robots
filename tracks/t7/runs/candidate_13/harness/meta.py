"""
meta.py — Candidate 13 metadata for Track7Harness.

TARGET_FAILURE_CLASSES = ["post_floor_switch_goal_inaccessibility"]
TARGET_SCENES = ["XB4GS9ShBRE"]

HYPOTHESIS:
  After the first successful stair climb in XB4GS9ShBRE (floor switch at ~step
  80-100), the passive stair detection hysteresis guard in Fix 10 (patch.py)
  uses threshold=350 floor-steps. In single-switch episodes the spurious passive
  detection fires at _floor_num_steps ~380-400, which EXCEEDS the 350 threshold
  and is therefore NOT blocked by Fix 10. The agent re-enters stair climbing mode,
  consuming the remaining episode budget on an already-explored floor instead of
  stopping near the target object (DTG_min=0.74m).

  Fix 10 only covers multi-switch (mapping_floor_confusion) episodes where the
  most recent floor switch is recent enough that _floor_num_steps < 350. Single-
  switch episodes where the agent explores floor 2 for ~380-400 steps before the
  spurious trigger remain unprotected.

MECHANISM:
  Two-file change: stair.py (primary) + patch.py (wiring).

  stair.py adds:
    _T13_POST_SWITCH_HYS = 500   (module constant, larger than Fix 10's 350)
    StairMixin.get_post_switch_passive_hysteresis(self) -> int
      Returns _T13_POST_SWITCH_HYS. Acts as the authoritative threshold for
      passive-detection suppression on non-ground floors.

  patch.py Fix 10 updated:
    _PASSIVE_STAIR_HYSTERESIS_13 = 500  (from stair.py SDP method at apply() time)
    In _patched_detect_passive:
      floor_idx = mc_self._cur_floor_index[env]
      threshold = _PASSIVE_STAIR_HYSTERESIS_13 if floor_idx > 0 else _PASSIVE_STAIR_HYSTERESIS
      if floor_step < threshold: suppress + log [T7_PASSIVE_HYS_13]

  The floor_idx > 0 guard ensures the 500-step window ONLY applies after at
  least one successful stair climb. Floor 0 retains the original 350-step
  threshold so passive detection on the ground floor is minimally disrupted.

  Falsifiability: [T7_PASSIVE_HYS_13] log line must appear at step ~482 in
  XB4GS9ShBRE episodes (floor_idx=1, floor_step ~380-400 < 500 → blocked).
  No second floor-switch event should appear after step ~120 in those episodes.

PREDICTED_CHANGE:
  XB4GS9ShBRE single-switch episodes: spurious passive detection at step ~482
  now blocked (floor_idx=1, floor_step ~392 < 500). Agent continues exploring
  floor 2 through end of episode; target already at DTG_min=0.74m suggests
  SUCCESS is achievable if goal-finding continues.

PREDICTED_SR_DELTA = 0.067

WHY_THIS_WILL_WORK:
  candidate_10 (Fix 10, SR=0.595) improved over baseline by suppressing passive
  detection for floor_step < 350 on ALL floors. Telemetry from XB4GS9ShBRE
  shows spurious trigger at step ~482 with _floor_num_steps ~392 > 350.
  Raising the threshold to 500 on post-switch floors (floor_idx > 0) covers
  the 350-500 gap where Fix 10 fails. The 500-step window is conservative:
  at floor_num_steps=500 the agent is ~8+ minutes into floor 2 exploration,
  well past any legitimate need for passive stair re-entry detection.

WHY_ALTERNATIVES_REJECTED:
  stair.py threshold increase without floor_idx guard: would also raise the
  ground-floor threshold from 350 to 500, potentially suppressing legitimate
  passive detection on floor 0 at step 350-500 in other scenes.

  patch.py-only change (no stair.py): possible but bypasses the SDP interface.
  Moving the threshold into stair.py makes it the canonical source for passive
  hysteresis configuration, consistent with other per-SDP tunable constants.

  floor.py / hooks.py: These fire at the floor-switch dispatcher level, after
  detection state is committed. Candidate_8 (floor.py) and candidate_9 (hooks.py)
  showed no improvement — dispatching too late.

  frontier.py / dps.py: No DP lever controls passive detection frequency.
  DP9/DP12 forbidden per instructions. LLM DPs (DP2/3/5/6/7) forbidden.

FALSIFIABILITY_CHECK:
  [T7_PASSIVE_HYS_13] log line must appear at floor_step ~380-400 in
  XB4GS9ShBRE episodes (confirming suppression in the 350-500 gap).
  No mode transition to look_for_upstair or look_for_downstair should appear
  after step ~120 in XB4GS9ShBRE. Episode SR for XB4GS9ShBRE should
  increase from ~0.5 (c10 estimate for this scene) to ~0.7+.
"""

TARGET_FAILURE_CLASSES = ["post_floor_switch_goal_inaccessibility"]
TARGET_SCENES = ["XB4GS9ShBRE"]
HYPOTHESIS = (
    "Fix 10 (patch.py) suppresses passive stair detection for floor_step < 350. "
    "In single-switch XB4GS9ShBRE episodes the spurious passive detection fires at "
    "floor_step ~392 > 350, which Fix 10 does NOT block. Raising the threshold to "
    "500 on post-switch floors (cur_floor_index > 0) closes this gap."
)
MECHANISM = (
    "stair.py: add _T13_POST_SWITCH_HYS=500 constant + "
    "StairMixin.get_post_switch_passive_hysteresis() SDP. "
    "patch.py Fix 10: select threshold=500 when floor_idx>0, 350 otherwise. "
    "Log tag: [T7_PASSIVE_HYS_13]."
)
PREDICTED_CHANGE = (
    "XB4GS9ShBRE spurious passive detection at step ~482 "
    "(floor_step ~392) now blocked by 500-step post-switch window."
)
PREDICTED_SR_DELTA = 0.067
WHY_ALTERNATIVES_REJECTED = (
    "Flat threshold raise to 500 on all floors risks suppressing ground-floor "
    "passive detection at steps 350-500 in other scenes. "
    "floor.py/hooks.py candidates (c8, c9) failed — fire after state commit. "
    "No valid DP levers for passive detection frequency."
)
WHY_THIS_WILL_WORK = (
    "candidate_10 improved to SR=0.595 but floor_step ~392 > 350 escapes Fix 10. "
    "Threshold 500 covers the 350-500 gap on non-ground floors only, "
    "minimising regression risk on floor 0."
)
FALSIFIABILITY_CHECK = (
    "[T7_PASSIVE_HYS_13] must appear at floor_step ~380-400 in XB4GS9ShBRE. "
    "No second floor-switch after step ~120. XB4GS9ShBRE SR expected >=0.7."
)
