"""
meta.py — Hypothesis metadata for Track7Harness candidate_9.

Machine-read by run_analyzer.py and loop.py. No executable code.
"""

TARGET_FAILURE_CLASSES = ["post_floor_switch_goal_inaccessibility"]

TARGET_SCENES = ["XB4GS9ShBRE"]

HYPOTHESIS = """
In XB4GS9ShBRE, the agent successfully climbs to floor 2 (~step 80).
map_controller.py:486 enables _detect_passive_stair_entry every subsequent step
(condition: _climb_stair_over=True, _climb_stair_flag=0). At step ~482, the agent
re-enters the stair pixel map for PASSIVE_STAIR_DETECTION_THRESHOLD=3 consecutive
steps, triggering _trigger_stair_climbing a second time and pulling the agent off
floor 2 just after dtg_min=0.74m from the bed, consuming the remaining budget on a
redundant floor transition instead of reaching the goal.

T7 candidate_8 attempted the same suppression via floor.py (FloorMixin arming from
_update_stair_state). That method fires every step during initialization, resetting
the countdown to 420 each call, which means the effective guard window starts from
the LAST initialization step (not the floor-switch event) and can expire before
step 482 if initialization is slow. Moreover, _update_stair_state is called from
ascent_policy.py:542 inside the climb_stair_initialize mode, which only fires while
_done_initializing is False — a per-episode window that may close before the robot
exits the initialization phase, leaving the countdown unreset.

Candidate_9 moves the patch to hooks.py and arms from _handle_new_floor_initialization
(the exact floor-switch event, called once per successful cross-floor transition)
using a step-based comparison instead of a decrement countdown. The step-based guard
is immune to initialization timing: it checks absolute episode-step difference.
"""

MECHANISM = """
Fix 7b in hooks.py only: HooksMixin overrides on_episode_start and log_step to
install and feed a step-based passive-detection hysteresis guard.

on_episode_start (first call):
  Installs two monkey-patches on Map_Controller via _t7_install_passive_hys().
  Sets _t7_phooks_installed=True so patches are installed only once.
  Resets per-env state: _t7_floor_switch_step[env]=-9999 (sentinel = never switched),
  _t7_cur_step[env]=0.

Patch A — _detect_passive_stair_entry wrapper:
  last = _t7_floor_switch_step.get(env, -9999)
  cur  = _t7_cur_step.get(env, 0)
  If last >= 0 and (cur - last) < T7_PASSIVE_HYS_HOOKS:
      print [T7_PASSIVE_HYS_HOOKS_BLOCKED] and return early.
  Else: call original.

Patch B — _handle_new_floor_initialization wrapper:
  Calls original first (preserves Fix 3 duplicate-init guard from patch.py).
  After return: records _t7_floor_switch_step[env] = _t7_cur_step[env].
  Logs [T7_PASSIVE_HYS_HOOKS_ARMED] with step and new floor_index.

log_step: updates _t7_cur_step[env] = step every step so the patched detect
function always has the current episode step available.

T7_PASSIVE_HYS_HOOKS = 450 (450 > ~402-step gap between floor switch ~80 and
spurious passive detection ~482, with 48-step margin for initialization variation).
"""

PREDICTED_CHANGE = (
    "XB4GS9ShBRE: _handle_new_floor_initialization fires at step ~80 (first floor "
    "switch), arming _t7_floor_switch_step[env]=80. At step ~482, guard checks "
    "482-80=402 < 450 → [T7_PASSIVE_HYS_HOOKS_BLOCKED] logged, "
    "_trigger_stair_climbing suppressed. Agent continues on floor 2, reaches bed "
    "at dtg_min=0.74m, triggers SUCCESS. "
    "q3zU7Yy5E5s / qyAac8rV8Zk: _handle_new_floor_initialization either never "
    "fires (stair disabled by Fix 4) or fires but no spurious passive detection "
    "within 450 steps → no regression."
)

PREDICTED_SR_DELTA = 0.067

WHY_ALTERNATIVES_REJECTED = """
floor.py (candidate_8, T7): same _detect_passive_stair_entry intercept but armed
  from _update_stair_state (fires every initialization step, effective window drifts
  with init duration). SR stayed 0.433 — mechanism did not fire reliably.

patch.py: Fix 3 already patches _handle_new_floor_initialization. Adding a second
  patch in patch.py's apply() would require coordinating with Fix 3's closure,
  increasing regression risk. hooks.py on_episode_start runs after apply() so it
  wraps the already-patched function cleanly.

dps.py (DP12): minimum-interval floor switch cannot suppress a passive detection
  event that fires 400 steps AFTER the interval window has expired. Not applicable.

stair.py / frontier.py: no leverage over passive stair detection callbacks.

The (hooks.py, post_floor_switch_goal_inaccessibility) pairing is untried in T7.
Step-based comparison (vs countdown) is architecturally distinct from candidate_8.
"""

WHY_THIS_WILL_WORK = """
map_controller.py:629-693 confirms _detect_passive_stair_entry fires when
_passive_up_stair_steps[env] >= PASSIVE_STAIR_DETECTION_THRESHOLD (=3). The call
site at line 486 is enabled as soon as _climb_stair_over=True and _climb_stair_flag=0
— both set at the moment of floor-switch success — permanently arming passive
detection for the rest of the episode.

_handle_new_floor_initialization (map_controller.py:566) is the exact function
called at the floor-switch moment (lines 308, 315 of _process_stair_climb_state).
It fires exactly once per successful upward transition, making it the correct and
minimal intercept point for arming the guard.

Step-based guard (cur - last < 450) is immune to initialization timing: it compares
absolute episode-step counters, not decremented values. Floor switch at step 80 plus
window 450 = step 530. Spurious detection at step 482 < 530 → always blocked
regardless of how many steps initialization takes.

T6 c7 (HYSTERESIS=350) and T6 c8 (HYSTERESIS=400) confirmed the suppression concept.
T7 c9 uses 450 to cover initialization duration variation (~20-50 extra steps vs T6).
"""

FALSIFIABILITY_CHECK = """
Log must show [T7_PASSIVE_HYS_HOOKS_ARMED] at step ~80 in XB4GS9ShBRE env log.
Log must show [T7_PASSIVE_HYS_HOOKS_BLOCKED] at step ~482 in XB4GS9ShBRE env log.
The second floor_switch log entry (floor 1→0 at step ~482) must be absent.
XB4GS9ShBRE result must flip from FAILURE to SUCCESS.
No regression in q3zU7Yy5E5s or qyAac8rV8Zk episodes.
"""
