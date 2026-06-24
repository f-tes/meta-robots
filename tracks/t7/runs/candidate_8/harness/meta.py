"""
meta.py — Hypothesis metadata for Track7Harness candidate_8.

Machine-read by run_analyzer.py and loop.py. No executable code.
"""

TARGET_FAILURE_CLASSES = ["post_floor_switch_goal_inaccessibility"]

TARGET_SCENES = ["XB4GS9ShBRE"]

HYPOTHESIS = """
In XB4GS9ShBRE, the agent successfully climbs to floor 2 (floor_step resets,
climb_stair_over=True, climb_stair_flag=0). With those two flags in that state,
map_controller.py:487 calls _detect_passive_stair_entry every step. At step ~482,
the agent's position happens to overlap with the stair pixel map for 3 consecutive
steps (PASSIVE_STAIR_DETECTION_THRESHOLD=3), triggering _trigger_stair_climbing a
second time. This pulls the agent off floor 2 just after it achieved dtg_min=0.74m
from the bed, causing the episode to end without SUCCESS.

Confirmed in source: map_controller.py:629-651 shows _detect_passive_stair_entry
accumulates _passive_up_stair_steps and fires when count>=3. The condition at line
486 (if _climb_stair_over and _climb_stair_flag==0) is satisfied as soon as the
first stair climb completes, leaving the passive detector permanently armed.
"""

MECHANISM = """
Fix 7 in floor.py: FloorMixin adds on_episode_start (chaining via super() to
HooksMixin) that lazily installs two monkey-patches on first call.

Patch A — _detect_passive_stair_entry wrapper:
  Checks per-env _psh_countdown[env]. If >0, decrements it, logs
  [T7_PASSIVE_HYS] BLOCKED and returns without calling the original.

Patch B — _update_stair_state wrapper:
  After the original call completes (which sets _climb_stair_over=True,
  _climb_stair_flag=0), sets _psh_countdown[env] = PASSIVE_STAIR_HYSTERESIS (420).
  Logs [T7_PASSIVE_HYS] ARMED with floor_index and countdown.

on_episode_start resets _psh_countdown[env] = 0 before each episode, ensuring
single-floor scenes (no floor transitions, _update_stair_state never armed) are
never blocked.

Only floor.py is changed; all patch.py fixes (Fix 0–5) are preserved unchanged.
"""

PREDICTED_CHANGE = (
    "XB4GS9ShBRE: _update_stair_state fires after successful floor-2 entry "
    "(step ~60-80), arming countdown=420. At step ~482 when spurious passive "
    "detection would have fired, countdown still > 0, [T7_PASSIVE_HYS] BLOCKED "
    "logged, _trigger_stair_climbing suppressed. Agent continues exploring floor 2, "
    "reaches bed at dtg_min=0.74m, triggers SUCCESS. "
    "qyAac8rV8Zk/q3zU7Yy5E5s: single-floor in relevant phase, _update_stair_state "
    "may fire but passive detection is for upstairs; even if armed, 420 steps "
    "covers remaining episode without regression."
)

PREDICTED_SR_DELTA = 0.033

WHY_ALTERNATIVES_REJECTED = """
patch.py+navmesh_disconnected_stair_centroid tried in c0/c1 (SR=0.4, no gain).
stair.py+navmesh_disconnected tried in c2 (SR=0.4) and c6 (SR=0.43, no gain).
floor.py+navmesh_disconnected tried in c4 (SR=0.367, regression).
hooks.py+navmesh_disconnected tried in c5 (SR=0.433, no gain for XB4GS9ShBRE).
frontier.py+navmesh_disconnected tried in c7 (SR=0.433, no gain for XB4GS9ShBRE).

The pair (floor.py, post_floor_switch_goal_inaccessibility) is entirely untried in
T7. Implementing in floor.py (not hooks.py or patch.py) co-locates all floor-
transition logic and avoids per-action hook overhead. hooks.py could wrap the same
function but adds indirection through should_stop which is a different call path.
DP9/DP12 are FORBIDDEN for this cluster.
"""

WHY_THIS_WILL_WORK = """
map_controller.py:629-651 confirms _detect_passive_stair_entry accumulates
_passive_up_stair_steps and fires when count>=3. The condition at line 486
(if _climb_stair_over and _climb_stair_flag==0) is satisfied as soon as the
first stair climb completes, leaving the passive detector permanently armed.

XB4GS9ShBRE dtg_min=0.74m proves the bed was reachable before the spurious
re-trigger at step ~482 removed the agent. PASSIVE_STAIR_HYSTERESIS=420 exceeds
the ~400-step gap between a typical step-60 first climb and the spurious step-482
re-trigger, blocking it while allowing legitimate passive detection in scenes with
>420 steps on one floor before a second staircase is encountered.

T6 candidate_8 (T6_PASSIVE_HYS=400, floor.py only) demonstrated this mechanism
is architecturally sound, but T6 lacked the T7 Fix 3/4/5 base improvements.
T7 candidate_8 builds on candidate_3 (SR=0.433) and adds only the hysteresis fix.
"""

FALSIFIABILITY_CHECK = """
Must see '[T7_PASSIVE_HYS] BLOCKED' log lines in XB4GS9ShBRE env log at step ~482.
Must see '[T7_PASSIVE_HYS] ARMED' immediately after floor transition (~step 60-80).
XB4GS9ShBRE result must flip from FAILURE to SUCCESS.
No regression in qyAac8rV8Zk (single-floor operative phase, _psh_countdown stays
0 or arms but expires before relevant detection window) or q3zU7Yy5E5s.
"""
