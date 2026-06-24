"""
meta.py — machine-readable hypothesis metadata for candidate_16.
Read by run_analyzer.py and classify_failures.py; no executable code.
"""

TARGET_FAILURE_CLASSES = ["post_floor_switch_goal_inaccessibility"]

TARGET_SCENES = ["XB4GS9ShBRE"]

HYPOTHESIS = """
In XB4GS9ShBRE, after the agent successfully climbs the stairs and begins second-floor
exploration (where dtg_min=0.74m of the bed was achieved), a spurious passive stair
detection fires at step ~482. _detect_passive_stair_entry triggers when the robot passes
near an out-of-map region on the second floor (common near stair exits), initiating a
second floor-switch cycle that pulls the agent away before BLIP-2 can cross the detection
threshold.

candidate_10 Fix 10 (SR=0.595) already patches _detect_passive_stair_entry using
floor_step = _obstacle_map[env]._floor_num_steps < 350. However, telemetry from
XB4GS9ShBRE shows the spurious trigger at step ~482 with _floor_num_steps ~392 > 350,
so candidate_10 misses it.

candidate_13 attempted to raise the threshold to 500 via a stair.py SDP but caused
SR regression to 0.3667, likely due to stair.py bugs unrelated to the threshold value.

candidate_16 Fix 11 raises the effective suppression window to 400 steps post-floor-switch
using the same patch.py code path (no stair.py involvement) with an absolute episode-step
clock: sum of all floors' _floor_num_steps across the obstacle_map_list gives total
episode steps. The switch step is recorded in _patched_new_floor_init; in
_detect_passive_stair_entry the delta (cur_ep - switch_ep = floor1._floor_num_steps)
is compared against T7_PASSIVE_HYS=400. This covers _floor_num_steps=392 (< 400).
"""

MECHANISM = """
Single-file change: patch.py only.

Fix 11 replaces Fix 10 (_PASSIVE_STAIR_HYSTERESIS=350 floor-local) with a wall-clock
absolute step approach (T7_PASSIVE_HYS=400):

1. Add _t7_passive_hys = {} dict (env → switch_ep_step, default -9999) to apply() closure.
2. Reset _t7_passive_hys[env] = -9999 in _reset_ep_state().
3. In _patched_new_floor_init, after _orig_new_floor_init or early-return path completes,
   record:
     _t7_passive_hys[env] = sum(m._floor_num_steps
                                for m in mc_self._obstacle_map_list[env])
   This records the absolute episode step at floor switch time.
4. In _patched_detect_passive:
   - Compute cur_ep = sum(m._floor_num_steps for m in mc_self._obstacle_map_list[env])
   - If switch_ep >= 0 and (cur_ep - switch_ep) < T7_PASSIVE_HYS=400: suppress and log
     [T7_PASSIVE_HYS] with env/cur_ep/switch_ep/delta
   - Otherwise: call _orig_detect_passive normally

The effective check reduces to: floor1._floor_num_steps < 400 (since floor0 is frozen
at switch time and floor1 starts at 0). This is equivalent to c10's mechanism but with
threshold 400 instead of 350, covering the observed _floor_num_steps=392 spurious trigger.
Log tag: [T7_PASSIVE_HYS] (replaces [T7_PASSIVE_HYS_10] from c10).
"""

PREDICTED_CHANGE = "XB4GS9ShBRE spurious passive detection at step ~482 suppressed; agent remains on floor 2 to find bed (dtg_min=0.74m already achieved). No regression for qyAac8rV8Zk (single-floor, no post-switch passive) or q3zU7Yy5E5s (stair approach failure pre-switch)."

PREDICTED_SR_DELTA = 0.033

WHY_ALTERNATIVES_REJECTED = """
floor.py (c8 T7): Failed SR=0.433. FloorMixin.on_episode_start passive detection
  wrapper introduced a different regression.
hooks.py (c9 T7): Failed SR=0.433. HooksMixin step-based suppression didn't wire
  correctly into the existing passive detection path.
stair.py (c13 T7): Failed SR=0.367 (regression). T13_POST_SWITCH_HYS=500 via SDP in
  stair.py caused bugs unrelated to the threshold value.
patch.py Fix 10 threshold=350 (c10 T7): SR=0.595 (incumbent). Threshold too low;
  spurious trigger at floor_num_steps=392 > 350 not suppressed.
frontier.py (c15 T7): Failed SR=0.467. Different target mechanism.
"""

WHY_THIS_WILL_WORK = """
candidate_10 (Fix 10, SR=0.595) proved the mechanism: suppressing _detect_passive_stair_entry
for floor_step < 350 raised SR from 0.433 to 0.595. The spurious trigger at ~step 482 was
partially addressed. Telemetry shows floor_num_steps=392 at the remaining spurious trigger,
which slips past threshold=350.

T6 candidate_8 (T6_PASSIVE_HYS=400 in FloorMixin.on_episode_start) successfully targeted
the same XB4GS9ShBRE spurious trigger at step 482. Threshold 400 > 392 covers it.

Fix 11 replicates the 400-step window in patch.py (the code path proven by c10 to work
in T7), avoiding the stair.py SDP regression risk of c13. No other T7 candidate has
used threshold > 350 in the patch.py _detect_passive_stair_entry code path.
"""

FALSIFIABILITY_CHECK = """
Pass: Episode log for XB4GS9ShBRE contains '[T7_PASSIVE_HYS] env=0 ... delta=39x <
      threshold=400' at step ~482. No third floor-switch commit after step 400.
      Episode ends SUCCESS (BLIP-2 crosses threshold on floor 2).

Fail-safe: If floor_num_steps at trigger is > 400 in any run, delta >= 400 and
      _orig_detect_passive fires normally (no suppression, no regression introduced).

Regression check: qyAac8rV8Zk episodes must not be affected (no floor switch →
      switch_ep stays -9999 → no suppression → passive detection unchanged).
      q3zU7Yy5E5s stair approach failure is pre-switch (hysteresis never fires).
"""
