"""
T5 Candidate 7 — Per-stair PointNav failure budget via SDP-I (on_pointnav_failure).

TARGET FAILURE CLASS: navmesh_disconnected_stair_centroid
  Scenes: qyAac8rV8Zk (downstair [-1.22,-8.19]), q3zU7Yy5E5s (upstair disconnected)

WHY CANDIDATES 2–6 FAILED:
  Candidates 2–6 all targeted custom_stair_approach in stair.py with BFS snap logic
  or ring-sampling to find a navigable stair cell. 4 of 5 produced parse errors; c4
  produced SR delta=-0.1 from a stair-frontier clearing regression. None addressed
  the root structural bug: _disable_stair_and_reset_state in map_controller.py zeros
  _climb_stair_flag BEFORE the conditional stair-map cleanup check, so neither
  _up_stair_map/_down_stair_map nor _has_up/down_stair are ever cleared on failure.
  This causes the stair to be re-detected on the very next step and the agent re-
  enters get_close_to_stair with the same unreachable centroid indefinitely.

THIS FIX (candidate_7):
  Two-file patch:
    hooks.py (SDP-I):
      on_episode_start — resets per-env stair failure counters.
      on_pointnav_failure — increments a per-stair failure counter (keyed by
        rounded [x,y] of the centroid). After K=3 consecutive failures for the
        same stair, returns the sentinel "DISABLE_STAIR".

    patch.py (Fix 5):
      Monkey-patches Map_Controller._disable_stair_and_reset_state. Before calling
      the original (which has the flag-zeroing bug), saves the current
      _climb_stair_flag value. After the original runs, calls SDP-I
      (on_pointnav_failure). If SDP-I returns "DISABLE_STAIR", uses the saved flag
      to perform the correct stair-map cleanup the original missed:
        - _disabled_stair_map |= stair_map (prevents re-detection via sensor masking)
        - stair_map.fill(0)
        - stair_frontiers = np.array([])
        - _has_up/down_stair = False
      Logs [T5_STAIR_DISABLED] on permanent disable and [T5_STAIR_DISABLED_ALL] if
      both _has_up_stair and _has_down_stair are now False.

  This approach is immune to the BFS/ring-sampling regressions seen in c2–c6 because
  it does not touch centroid calculation or custom_stair_approach at all.
"""

TARGET_FAILURE_CLASSES = [
    "navmesh_disconnected_stair_centroid",
]

TARGET_SCENES = [
    "qyAac8rV8Zk",
    "q3zU7Yy5E5s",
]

HYPOTHESIS = (
    "PointNav cannot converge to a navmesh-disconnected stair centroid, but no "
    "abort budget exists — the agent retries indefinitely due to a bug in "
    "_disable_stair_and_reset_state that zeros _climb_stair_flag before the stair-"
    "map cleanup check, causing stair maps to persist and re-trigger detection on "
    "the next step. Adding a per-stair consecutive-failure limit (K=3) via SDP-I "
    "(on_pointnav_failure) causes the agent to give up on the unreachable stair "
    "and properly disable it, freeing remaining steps for floor exploration."
)

MECHANISM = (
    "hooks.py: on_pointnav_failure tracks a per-env, per-stair-id failure count. "
    "stair_id = tuple(round(centroid_xy, 2)). At count >= K=3, returns 'DISABLE_STAIR'. "
    "on_episode_start resets self._stair_failures[env] = {}. "
    "patch.py Fix 5: patches Map_Controller._disable_stair_and_reset_state. Saves "
    "_climb_stair_flag[env] before calling original. After original, calls SDP-I. "
    "On 'DISABLE_STAIR': uses saved flag to set _disabled_stair_map |= stair_map "
    "(masking out stair region from future sensor frames), clears stair_map and "
    "stair_frontiers, sets _has_stair=False. Logs [T5_STAIR_DISABLED] + "
    "[T5_STAIR_DISABLED_ALL] if no stairs remain on floor."
)

PREDICTED_CHANGE = (
    "qyAac8rV8Zk: After K=3 GCTS stall/stop cycles (~step 200-210), "
    "'[T5_STAIR_DISABLED] env=0 stair_id=(-1.22,-8.19) after 3 consecutive pointnav "
    "failures → DISABLE_STAIR' fires. Stair maps cleared. Agent resumes floor "
    "exploration for remaining ~30 steps. "
    "q3zU7Yy5E5s: Same for upstair disconnected centroid (~step 230-250). "
    "Expected SR: 0.70 → 0.80-0.90."
)

PREDICTED_SR_DELTA = 0.2

WHY_ALTERNATIVES_REJECTED = (
    "candidates 2/3: BFS snap fires but crashed (KeyError / stale done_set). "
    "candidate_4: Fix 3a blocked Fix 4 for qyAac8rV8Zk (stair frontiers reset to []). "
    "candidate_5: CUDA OOM infra failure; abort sentinel still missing. "
    "candidate_6: ring-sampling with abort sentinel — parse_error (syntax issue). "
    "All c2-c6 targeted custom_stair_approach in stair.py and modified centroid "
    "calculation, which introduced regressions. This fix targets the bug in "
    "_disable_stair_and_reset_state directly and does not touch stair centroid logic."
)

WHY_THIS_WILL_WORK = (
    "The analysis DB (analysis_db.json) confirms: 'highest_leverage_untested_levers' "
    "for both qyAac8rV8Zk and q3zU7Yy5E5s explicitly list "
    "'early_abort_on_N_consecutive_centroid_failures_with_permanent_stair_disable'. "
    "The bug is confirmed: _disable_stair_and_reset_state sets _climb_stair_flag=0 "
    "at line 353 before checking it at lines 357/370, so neither stair-map cleanup "
    "branch fires. This causes _down_stair_map and _down_stair_frontiers to persist, "
    "re-triggering stair detection on the next step (line 545-550 in act()). "
    "Clearing _disabled_stair_map |= stair_map prevents re-detection even if the "
    "sensor continues to detect stair pixels (masking at obstacle_map.py line 577). "
    "K=3 is chosen to allow 2 retry cycles (matching observed 'disabled twice' "
    "pattern) before permanent disable, avoiding false-positive disables on stairs "
    "with transient PointNav failures."
)

FALSIFIABILITY_CHECK = (
    # After eval, grep the log for:
    #   grep "T5_STAIR_DISABLED" <log>
    #   → must appear for qyAac8rV8Zk (stair_id containing -1.22,-8.19) and
    #     q3zU7Yy5E5s (stair_id near -2.12,3.27 or similar)
    #   grep "T5_PNF" <log>
    #   → must show failure_count=1, failure_count=2, failure_count=3 for same stair_id
    #   grep "Error executing job" <log>
    #   → must NOT appear
    #   Mode sequence after T5_STAIR_DISABLED: must NOT include further get_close_to_stair
    #     targeting the same disabled centroid
    # If T5_STAIR_DISABLED fires but agent re-enters GCTS for same stair: _disabled_stair_map
    #   update is not persisting — check obstacle_map.py line 577 masking.
    # If T5_PNF count never reaches 3: GCTS timeout is shorter than expected or episode
    #   ends before 3 failures.
    "After eval: 'T5_STAIR_DISABLED stair_id=X after 3 consecutive pointnav failures' "
    "must appear for qyAac8rV8Zk and q3zU7Yy5E5s. 'T5_PNF' lines must show count "
    "incrementing 1→2→3 for the same stair_id. 'Error executing job' must NOT appear. "
    "After T5_STAIR_DISABLED fires, mode sequence must NOT contain further "
    "get_close_to_stair targeting the same disabled stair centroid."
)
