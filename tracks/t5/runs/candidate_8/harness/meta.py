"""
T5 Candidate 8 — Premature stair success guard with GUARD_STEPS=35 (Fix 4b).

The proposer writes a new version of this file for each candidate.
run_analyzer.py reads this file directly instead of grepping docstrings.
"""

TARGET_FAILURE_CLASSES = [
    "premature_stair_climb_success",
]

TARGET_SCENES = [
    "q3zU7Yy5E5s",
    "XB4GS9ShBRE",
]

HYPOTHESIS = (
    "The stair pixel map covers only the lower ~2/3 of the physical stair. When the "
    "robot climbs into the unmapped upper third, is_robot_in_stair_map_fast flips to "
    "False while paused_step is still low (<30), triggering a false SUCCESS before the "
    "physical floor transition is complete. Candidate_1 attempted this fix with "
    "MIN_STAIR_STEPS=15, but XB4GS9ShBRE fires at paused=20 (>=15), bypassing the "
    "guard entirely. Fix 4b raises the guard to GUARD_STEPS=35, which is ABOVE the "
    "failure threshold (paused=30), suppressing BOTH the premature success branch "
    "(paused<30) AND the stair-disable failure branch (paused=30-34) until the robot "
    "has accumulated 35 low-progress steps after centroid-reach. At paused=35, we "
    "temporarily zero paused_step and call the original, forcing the success path "
    "(paused<30 in original) rather than the failure+disable path (paused>=30)."
)

MECHANISM = (
    "patch.py Fix 4b wraps _process_stair_climb_state. When Phase 2 is active "
    "(reach_centroid=True) and robot is outside the stair pixel map (in_stair_map=False): "
    "  (a) paused < GUARD_STEPS=35: return early without calling original. Logs "
    "      GUARD_HOLDING. Both the success branch (paused<30) and failure branch "
    "      (paused>=30) are suppressed. _climb_stair_over stays False so _climb_stair "
    "      continues issuing the disable_end=True carrot waypoint (forward +1.5m), "
    "      keeping the robot moving upward through the physical stair geometry. "
    "  (b) paused >= GUARD_STEPS=35: temporarily set paused_step=0, call original. "
    "      With paused=0 the original takes the success branch (elif not in_stair_map), "
    "      not the failure+disable branch (paused>=30). Floor transition fires correctly. "
    "      Logs FORCED_SUCCESS with original paused value for classifier analysis. "
    "All other cases (robot still in stair map, or Phase 1) pass through to original "
    "unchanged. Fix 0, Fix 1, Fix 2, Fix 3 from candidate_0 are fully preserved."
)

PREDICTED_CHANGE = (
    "T5_STAIR_CLIMB_EVAL GUARD_HOLDING appears at paused=20..34 for XB4GS9ShBRE. "
    "At paused=35, FORCED_SUCCESS fires and floor transition completes. "
    "No FAILURE_PAUSED entries appear (failure branch suppressed at paused=30-34). "
    "Expected SR: 0.70 → 0.80-0.90."
)

PREDICTED_SR_DELTA = 0.2

WHY_ALTERNATIVES_REJECTED = (
    "candidate_1 (MIN=15): threshold below observed paused=20 in XB4GS9ShBRE — "
    "guard never fires for that scene, delta=0.0. "
    "Raising to MIN=35 (above failure threshold of 30) is the correct fix because "
    "it must suppress BOTH the success branch (paused=20 case) AND the failure "
    "branch (paused=30 case) that would otherwise disable the stair permanently. "
    "Purely raising MIN to, say, 31 would still let the failure branch fire at 30 "
    "before the guard at 31 could engage. GUARD=35 safely covers the full 30-34 gap."
)

WHY_THIS_WILL_WORK = (
    "Root cause confirmed in analysis_db.json for XB4GS9ShBRE: T5_STAIR_CLIMB_EVAL "
    "SUCCESS fires at paused_step=20, floor_step reset confirmed, robot not yet on "
    "destination floor. Stair pixel map ends before physical stair top. "
    "GUARD_STEPS=35 catches paused=20 (20 < 35) and holds the agent in climb mode. "
    "During steps 20-35, _climb_stair issues disable_end=True carrot (+1.5m forward) "
    "since paused>15 already, keeping the robot moving through the physical stair. "
    "By paused=35 the robot is well past the stair end; forced success correctly "
    "initializes the new floor. The failure+disable path (paused=30) is also "
    "suppressed by the guard, preventing false stair disables at paused=30-34."
)

FALSIFIABILITY_CHECK = (
    # After eval, grep the telemetry/log for:
    #   grep "T5_STAIR_CLIMB_EVAL.*SUCCESS" <log> | grep -v "GUARD_HOLDING\|FORCED_SUCCESS"
    #   All lines with plain SUCCESS must have paused_step >= 35.
    #   Lines with paused_step < 35 and SUCCESS (not FORCED_SUCCESS, not GUARD_HOLDING)
    #   mean the guard is not intercepting correctly.
    #
    #   grep "T5_STAIR_CLIMB_EVAL.*GUARD_HOLDING" <log>
    #   Must appear for q3zU7Yy5E5s and XB4GS9ShBRE at paused_step < 35.
    #
    #   grep "T5_STAIR_CLIMB_EVAL.*FORCED_SUCCESS" <log>
    #   Must appear at paused_step=35 for episodes that previously fired premature SUCCESS.
    #
    # If GUARD_HOLDING never appears: candidate_1's issue reproduced — check guard condition.
    # If FORCED_SUCCESS appears but floor transition fails: _handle_new_floor_initialization
    #   call order issue — verify paused=0 trick routes to success branch in original.
    "After eval: grep '[T5_STAIR_CLIMB_EVAL].*SUCCESS' must show NO plain SUCCESS at "
    "paused_step < 35 (only GUARD_HOLDING at paused<35 and FORCED_SUCCESS at paused=35). "
    "GUARD_HOLDING must appear in q3zU7Yy5E5s and XB4GS9ShBRE logs. "
    "FORCED_SUCCESS must appear at paused_step=35 for scenes previously showing premature "
    "SUCCESS at paused=20. FAILURE_PAUSED must NOT appear (failure branch suppressed)."
)
