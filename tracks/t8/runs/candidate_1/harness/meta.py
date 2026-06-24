"""
meta.py — Machine-readable hypothesis metadata for candidate_1.

Read by run_analyzer to correlate SR results with mechanism descriptions.
Do NOT add executable code here.
"""

TARGET_FAILURE_CLASSES = ["frontier_scoring_bias"]

TARGET_SCENES = []  # scene-agnostic; applies to all episodes

HYPOTHESIS = (
    "With no telemetry or failure data yet, the highest-prior risk in zero-shot "
    "object-goal navigation is frontier scoring bias: BLIP-2 semantic scores (Mss) "
    "may be poorly calibrated against the distance bonus (exp(-d) from DP1), causing "
    "the agent to commit to semantically plausible but geometrically distant frontiers "
    "while ignoring closer, equally valid candidates. This manifests as long detour "
    "paths and missed nearby objects."
)

MECHANISM = (
    "Reduce the Mss coefficient in compute_frontier_value (DP1, dps.py) from 1.0 to "
    "0.8 — i.e., the enhanced score becomes (0.8*mss + exp(-d)) for d<=3m, and "
    "0.8*mss for d>3m. This shifts the balance toward proximity without eliminating "
    "semantic guidance. NOTE: target_file in the selected hypothesis was listed as "
    "frontier.py, but diagnosis confirmed the Mss coefficient is only in "
    "dps.py::compute_frontier_value (line 15); frontier.py has no scoring logic."
)

PREDICTED_CHANGE = (
    "Mean selected-frontier distance-to-agent decreases at DP1 decision time. "
    "Successful episode step counts decrease. Frontier selection logs show more "
    "nearby frontiers winning over distant high-Mss frontiers."
)

PREDICTED_SR_DELTA = 0.05

WHY_THIS_WILL_WORK = (
    "Distance-weighted frontier selection is a well-established prior in object-goal "
    "navigation. BLIP-2 semantic scores carry uncertainty for small or occluded "
    "objects, so over-relying on Mss causes the agent to chase high-confidence-but-far "
    "frontiers. Downweighting Mss by 20% reduces this bias while preserving semantic "
    "ranking signal. The 0.8 factor is small enough to be safe: it cannot flip a "
    "frontier from high to low Mss category, only break ties in favor of proximity."
)

WHY_ALTERNATIVES_REJECTED = (
    "stair.py and floor.py are only relevant after the agent commits to a floor-switch "
    "decision. patch.py addresses structural sim issues not scoring logic. llm.py and "
    "hooks.py govern LLM call gating and episode lifecycle. frontier.py SDPs "
    "(build_exploration_memory, on_frontier_exhausted) are no-ops in the baseline. "
    "The DP1 scorer is the earliest and highest-leverage intervention point for "
    "frontier selection bias."
)

FALSIFIABILITY_CHECK = (
    "After the fix, [DP1] log lines should show a measurable decrease in mean "
    "selected-frontier distance-to-agent. If mean selected-frontier distance is "
    "unchanged, the 0.8 coefficient was not applied. If SR decreases, the weight "
    "reduction was too aggressive and should be relaxed toward 0.9."
)
