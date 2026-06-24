"""
meta.py — machine-readable hypothesis metadata for candidate_18.
Read by run_analyzer.py and classify_failures.py. No executable code.
"""

TARGET_FAILURE_CLASSES = ["post_floor_switch_goal_inaccessibility"]

TARGET_SCENES = ["XB4GS9ShBRE"]

HYPOTHESIS = """
Targeting: post_floor_switch_goal_inaccessibility (XB4GS9ShBRE).

After a successful upstair climb in XB4GS9ShBRE, the agent reaches floor 2 and
gets within 0.74m of the bed (dtg_min achieved) but never stops. At step ~482
the agent drifts back into the stair pixel footprint and accumulates
PASSIVE_STAIR_DETECTION_THRESHOLD=3 consecutive in-stair frames, causing
Map_Controller._detect_passive_stair_entry to re-trigger _trigger_stair_climbing,
which yanks the agent off floor 2 before a STOP can fire. The root method
(_detect_passive_stair_entry in map_controller.py:629) has no cooldown after a
completed floor switch.

patch.py Fix 10 (incumbent c10, floor_step < 350) blocks most spurious triggers
but fails for XB4GS9ShBRE where the trigger fires at floor_step ~392 > 350.
This candidate adds a wall-clock layer (N=400) implemented in llm.py that catches
the [350, 400) gap.
"""

MECHANISM = """
Fix 18 in llm.py: wall-clock passive stair detection hysteresis, installed as a
module-level monkey-patch at import time.

Installation order:
  1. Module import: _install_llm_passive_hys_patch() wraps the ORIGINAL
     Map_Controller._detect_passive_stair_entry (_c18_inner).
  2. PatchMixin.apply() runs later: Fix 10 wraps _c18_inner with _c10_outer.
  Net chain: _c10_outer (floor_step<350) → _c18_inner (wall-clock N=400) → original.

_c18_inner logic (per-call):
  - Compare mc_self._cur_floor_index[env] to _env_state_hys[env]["last_floor"].
  - If first call (state absent): initialize {"last_floor": cur_floor, "until": 0},
    pass through to original (no blocking on initial floor).
  - If floor index changed: set "until" = floor_step + 400, log
    [T7_PASSIVE_HYS_18 env=... floor X→Y floor_step=... until=...].
  - If floor_step < "until": reset passive counters, log
    [T7_PASSIVE_HYS_BLOCKED env=... step=... until=...], return (suppress).
  - Else: call original.

LLMMixin.on_episode_start resets _env_state_hys[env] to prevent cross-episode
floor-index confusion. Chains to super().on_episode_start() to preserve
HooksMixin telemetry.

XB4GS9ShBRE trace (predicted):
  - Floor switch at episode_step ~80 → _cur_floor_index changes to 1.
  - First passive-detection call on floor 2 (floor_step ~1): floor change detected,
    until = 1 + 400 = 401.
  - At episode_step ~472 (floor_step ~392): c10 gate: 392 >= 350 → passes.
    c18 gate: 392 < 401 → BLOCKED. _trigger_stair_climbing not called.
  - Agent continues exploring floor 2, achieves SUCCESS (dtg_min=0.74m already).
"""

PREDICTED_CHANGE = "SR delta +0.033 (1 episode fixed in XB4GS9ShBRE)"

PREDICTED_SR_DELTA = 0.033

WHY_ALTERNATIVES_REJECTED = """
patch.py tried T7 c10 (SR=0.595, Fix 10 floor_step<350): threshold insufficient
for XB4GS9ShBRE trigger at floor_step~392. c13 raised Fix 10 threshold to 500
(no_improvement, SR=0.3667) by blocking legitimate passive detections in other
scenes. The wall-clock approach in llm.py targets only the [350,400) gap left by
c10, not floor_step<500, avoiding the c13 regression.

floor.py tried T7 c8 (no_improvement): wrapped _detect_passive_stair_entry via
FloorMixin.on_episode_start; same hysteresis mechanism (N=400) but different file.
c8 targeted the floor-switch decision path rather than layering on top of c10.

hooks.py tried T7 c9 (no_improvement): dtg-gated stop; passive detection at
step 482 fires before should_stop, so the gate is bypassed.

stair.py tried T7 c13 (no_improvement): combined stair.py + patch.py with N=500.

frontier.py tried T7 c15 (no_improvement): scored frontiers on floor 2 but did
not suppress the passive detection re-trigger.

dps.py tried T7 c17 (no_improvement): parameter tuning.

llm.py is the sole untried (file, class) pair for post_floor_switch_goal_inaccessibility.
The mechanism (wall-clock N=400 as inner layer under Fix 10) is new: no prior
candidate combined Fix 10 with a secondary wall-clock gate.
"""

WHY_THIS_WILL_WORK = """
c10 (Fix 10, SR=0.595) confirmed that suppressing passive stair detection after
floor switches is the right lever for XB4GS9ShBRE. Fix 10 blocked detections at
floor_step<350, improving from 0.4667 → 0.595. The remaining XB4GS9ShBRE failure
mode has the spurious trigger at floor_step~392 which is above c10's threshold.

Fix 18 adds a secondary gate (floor_step < until, where until = floor_step_at_switch
+ 400). When the first passive-detection call after a floor switch arrives
(floor_step~1), until is set to ~401. This blocks floor_step in [1, 401). Combined
with c10's block of [0, 350), the effective blocked range extends to [0, 401),
covering the ~392 trigger.

N=400 matches T6_PASSIVE_HYS=400 which Track 6 c8 validated as sufficient to cover
the XB4GS9ShBRE spurious trigger window (~392 steps after switch).

Risk: legitimate passive detections at floor_step ∈ (350, 400) in other scenes
would be blocked. However, since c10 already blocks [0, 350) in ALL scenes without
regression, those scenes' legitimate passive triggers must occur at floor_step >= 350.
The additional [350, 400) block is a narrow window (50 steps), and T6 c8 with the
same N=400 did not show regression, suggesting legitimate triggers in other scenes
fire at floor_step >= 400.
"""

FALSIFIABILITY_CHECK = """
Log line '[T7_PASSIVE_HYS_BLOCKED env=0 step=392 until=401]' (or similar floor_step
value) must appear in XB4GS9ShBRE episode output.
Log line '[T7_PASSIVE_HYS_18 env=0 floor 0→1 ...]' must appear shortly after the
initial floor switch (~step 80).
_trigger_stair_climbing must NOT fire a second time after the first successful climb.
floor_step must not reset to 0 again after the initial floor switch.
Episode must achieve SUCCESS or dtg < 0.74m (already-achieved dtg_min threshold).
"""
