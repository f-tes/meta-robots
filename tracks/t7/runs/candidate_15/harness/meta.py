"""
meta.py — Machine-readable hypothesis metadata for candidate_15.

TARGET_FAILURE_CLASSES: ["post_floor_switch_goal_inaccessibility"]
TARGET_SCENES: ["XB4GS9ShBRE"]
"""

TARGET_FAILURE_CLASSES = ["post_floor_switch_goal_inaccessibility"]
TARGET_SCENES = ["XB4GS9ShBRE"]

HYPOTHESIS = """
Targeting: post_floor_switch_goal_inaccessibility (XB4GS9ShBRE).

dtg_min_achieved=0.74m confirms the agent physically reaches the target bed on floor 2
but never stops. The failure mode is frontier-driven departure: after the floor switch,
the frontier ranker pulls the agent away from the high-BLIP-2 bed region toward
lower-value unexplored frontiers before the stop condition fires.

Fix 10 (candidate_10) suppressed passive stair detection for floor_step < 350, raising
overall SR to 0.595 but leaving XB4GS9ShBRE unresolved (spurious passive detection
at floor_step ~392-402 > 350 threshold still fires).

The untested axis is BLIP-2 peak exploitation inside frontier scoring: if the frontier
ranker positively biases toward the last high-score frontier position seen on the current
floor, the agent will return to — and linger near — the bed rather than drifting to
distant frontiers. Combined with a should_stop override (detection_score > 0.20 AND
distance_to_detection < 1.1m) the agent can declare SUCCESS on its close approach
at 0.74m before the spurious passive detection at floor_step ~392-402 forces
a second stair-climb cycle.
"""

MECHANISM = """
Two-file change: frontier.py (primary state + scoring + stop) + patch.py (Fix 11 wiring).

frontier.py changes:
  1. Peak BLIP-2 state: _peak_blip2_score[env], _peak_blip2_pos[env],
     _peak_stop_triggered[env]. Initialized lazily via _ensure_peak_state(env).
  2. on_frontier_evaluated: after writing base telemetry, updates peak state when
     scores[0] > PEAK_MIN_SCORE=0.20 and the new score exceeds the stored peak.
     Logs [T7_PEAK_UPDATE] on improvement.
  3. peak_exploit_bonus_for_frontier(env, frontier_xy): returns
     PEAK_EXPLOIT_BONUS=0.45 * exp(-dist_to_peak / PEAK_RADIUS_M=4.0) when
     peak state is set; 0.0 otherwise.
  4. should_stop override (shadows HooksMixin.should_stop via MRO): returns True
     when detection_score > PEAK_STOP_MIN=0.20 AND distance_to_detection <
     PEAK_TRIGGER_DIST=1.1m AND step > 50. Logs [T7_PEAK_STOP] on first trigger.
     Falls through to None otherwise.

patch.py changes (Fix 11, added after Fix 10):
  Monkey-patches Ascent_LLM_Planner._sort_frontiers_by_value.
  After the native sort (value_map scores), calls
  harness.peak_exploit_bonus_for_frontier(env, pt) for each frontier, adds it to
  the raw score, and re-sorts. The DP1 distance bonus then acts on the biased scores.
  Logs [T7_PEAK_EXPLOIT] when max_bonus > 0.01.

Constants:
  PEAK_MIN_SCORE = 0.20       threshold to record a peak position
  PEAK_EXPLOIT_BONUS = 0.45   amplitude of Gaussian exploit bonus
  PEAK_RADIUS_M = 4.0         radial decay constant (meters)
  PEAK_TRIGGER_DIST = 1.1     stop condition: distance_to_detection < this
  PEAK_STOP_MIN = 0.20        stop condition: detection_score > this
"""

PREDICTED_CHANGE = """
XB4GS9ShBRE: agent biased toward bed region after first high-BLIP-2 frontier
appears on floor 2. When agent reaches within 1.1m of bed with detection_score > 0.20,
should_stop returns True → episode SUCCESS before step 392-402 passive detection.
Other scenes: peak bonus activates only when a high-score frontier is seen; if no
frontier exceeds 0.20 (e.g., on lower floors before target is visible), the bonus
is zero and behavior is identical to candidate_10.
"""

PREDICTED_SR_DELTA = 0.067

WHY_ALTERNATIVES_REJECTED = """
navmesh_disconnected_stair_centroid (q3zU7Yy5E5s / qyAac8rV8Zk):
  All valid target files exhausted — patch.py (c0,c1), stair.py (c3,c6,c11,c14),
  floor.py (c4), hooks.py (c5), frontier.py (c7), dps.py (c12). Only llm.py remains
  and LLM cannot fix navmesh topology.

post_floor_switch stops already tried:
  hooks.py stop-guard alone (c9): SR=0.433 — stop guard insufficient without
    frontier biasing; agent drifts away before stop fires.
  patch.py passive-detection suppression (c10): SR=0.595 — improved overall but
    XB4GS9ShBRE still fails; floor_step at spurious trigger ~392-402 > 350 threshold.
  stair.py (c13): SR=0.367 — stair mechanics irrelevant after floor switch.
  floor.py (c8): SR=0.433 — hysteresis insufficient.

frontier.py + post_floor_switch is the ONLY untried valid pairing.
The key novelty is COMBINING frontier biasing (keeps agent near bed) WITH a
proximity stop (fires at 0.74m close approach). Neither alone is sufficient.
"""

WHY_THIS_WILL_WORK = """
dtg_min_achieved=0.74m is below PEAK_TRIGGER_DIST=1.1m, so the should_stop condition
fires when the agent is at its closest approach to the bed (distance_to_detection=0.74m).
BLIP-2 score at 0.74m from a bed is expected to exceed PEAK_STOP_MIN=0.20 (HM3D
close-approach BLIP-2 scores for correct object categories typically 0.25-0.60).

Frontier biasing with PEAK_RADIUS_M=4.0 is wide enough to encompass the second-floor
exploration area and will redirect frontiers toward the bed region rather than distant
unexplored zones. PEAK_EXPLOIT_BONUS=0.45 exceeds DP1's distance bonus at 1m
(exp(-1)≈0.37) so biased frontiers near the peak will rank above unbiased frontiers.

This mechanism is additive on candidate_10's Fix 10 passive-detection hysteresis
and does not interfere with the existing fix chains (Fix 0–5, Fix 10).
If the agent triggers should_stop before floor_step ~392-402, the episode resolves
before the spurious second floor switch.
"""

FALSIFIABILITY_CHECK = """
Logs must show [T7_PEAK_UPDATE] lines in XB4GS9ShBRE episode on floor 2 with score > 0.20.
Logs must show [T7_PEAK_EXPLOIT] lines with non-zero bonus values when frontiers exist.
Logs must show [T7_PEAK_STOP] firing in XB4GS9ShBRE episode with dist < 1.1 and score > 0.20.
XB4GS9ShBRE episode outcome must flip from FAILURE to SUCCESS.
Overall SR must rise above 0.595 (incumbent candidate_10).
"""
