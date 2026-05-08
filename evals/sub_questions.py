"""Sub-question definitions for VLM physical-law evaluation.

Each physical law is decomposed into observational sub-questions that
detect violations.  The VLM answers yes/no/na/uncertain; the `violation`
field specifies which answer constitutes a violation.  Scoring is computed
in code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubQuestion:
    """A single observational sub-question for a physical law."""
    id: str            # e.g. "gravity_q1"
    question: str      # observational question for VLM
    violation: str     # "yes" → answering "yes" is a violation; "no" → answering "no" is a violation


# ──────────────────────────────────────────────────────────────────────────────
# A. Solid-Body Mechanics
# ──────────────────────────────────────────────────────────────────────────────

GRAVITY = [
    SubQuestion("gravity_q1", "Do any unsupported objects float upward or hover in mid-air?",                                 violation="yes"),
    SubQuestion("gravity_q2", "Does any heavy object drift or float down unrealistically slowly, as if it had no weight?",    violation="yes"),
    SubQuestion("gravity_q3", "Does any object fall sideways or in an unnatural direction instead of downward?",              violation="yes"),
]

INERTIA = [
    SubQuestion("inertia_q1", "Does any object spontaneously start moving without a visible cause?",                            violation="yes"),
    SubQuestion("inertia_q2", "Does any moving object suddenly stop or reverse direction with no visible contact or force?",    violation="yes"),
]

MOMENTUM = [
    SubQuestion("momentum_q1", "Does any object fly off in a direction completely unrelated to how it was hit or pushed?",      violation="yes"),
    SubQuestion("momentum_q2", "Does the recoil direction contradict the direction of the applied force?", violation="yes"),
]

IMPENETRABILITY = [
    SubQuestion("impenetrability_q1", "Does any object pass through another solid object as if it were not there?",            violation="yes"),
    SubQuestion("impenetrability_q2", "Does any part of an object clip into a wall, floor, or another object's interior?",     violation="yes"),
]

COLLISION = [
    SubQuestion("collision_q1", "Does any object remain completely unaffected after being clearly hit by another object?",             violation="yes"),
    SubQuestion("collision_q2", "Does the collision response look wildly too weak or too strong for the visible impact?",              violation="yes"),
    SubQuestion("collision_q3", "Does any object shatter or deform dramatically from a very light touch with no reasonable force?",    violation="yes"),
]

MATERIAL = [
    SubQuestion("material_q1", "Does any rigid material (glass, metal, stone) bend or stretch like rubber?",                           violation="yes"),
    SubQuestion("material_q2", "Does any soft material (cloth, rubber, rope) behave as if it were completely rigid and unbending?",    violation="yes"),
]

# ──────────────────────────────────────────────────────────────────────────────
# B. Fluid Dynamics
# ──────────────────────────────────────────────────────────────────────────────

BUOYANCY = [
    SubQuestion("buoyancy_q1", "Does any heavy object (metal, stone) float on the liquid surface?",            violation="yes"),
    SubQuestion("buoyancy_q2", "Does any light object (wood, cork, plastic) sink to the bottom?",              violation="yes"),
]

DISPLACEMENT = [
    SubQuestion("displacement_q1", "Does the liquid level remain completely unchanged when a large object is submerged?",      violation="yes"),
    SubQuestion("displacement_q2", "Does a full container spill or overflow when more liquid or an object is added?",            violation="no"),
    SubQuestion("displacement_q3", "Does the liquid level behave in a clearly impossible way, such as dropping when volume is added?", violation="yes"),
]

FLOW_DYNAMICS = [
    SubQuestion("flow_dynamics_q1", "Does liquid flow uphill or against gravity without any force pushing it?",                violation="yes"),
    SubQuestion("flow_dynamics_q2", "On a flat surface, does liquid spread outward and become thinner over time?",             violation="no"),
    SubQuestion("flow_dynamics_q3", "Does liquid suddenly stop flowing or freeze in place without an obvious reason?",         violation="yes"),
]

BOUNDARY_INTERACTION = [
    SubQuestion("boundary_q1", "Does liquid ignore a solid boundary and continue moving as if nothing were there?",            violation="yes"),
    SubQuestion("boundary_q2", "Does liquid striking a surface produce visible droplets, spray, or ripples?",                  violation="no"),
    SubQuestion("boundary_q3", "Does liquid accumulate or pool on the wrong side of a barrier?",                               violation="yes"),
]

FLUID_CONTINUITY = [
    SubQuestion("fluid_q1", "Does a continuous pour or flow stay connected as a stream without sudden gaps?",                  violation="no"),
    SubQuestion("fluid_q2", "Does liquid disappear into nothing (not counting brief splashes)?",                               violation="yes"),
    SubQuestion("fluid_q3", "Does liquid appear from nowhere with no visible source?",                                         violation="yes"),
]

# ──────────────────────────────────────────────────────────────────────────────
# C. Optical Physics
# ──────────────────────────────────────────────────────────────────────────────

REFLECTION = [
    SubQuestion("reflection_q1", "Do reflections show completely unrelated content not present in the scene?",                          violation="yes"),
    SubQuestion("reflection_q2", "Does a reflection remain completely static while the scene clearly changes around it?",               violation="yes"),
]

SHADOW = [
    SubQuestion("shadow_q1", "Do different shadows in the same scene point in contradictory directions?",                              violation="yes"),
    SubQuestion("shadow_q2", "Does any shadow remain fixed in place while its object clearly moves?",                                  violation="yes"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Registry: law key → sub-questions
# ──────────────────────────────────────────────────────────────────────────────

SUB_QUESTIONS: dict[str, list[SubQuestion]] = {
    "gravity":              GRAVITY,
    "inertia":              INERTIA,
    "momentum":             MOMENTUM,
    "impenetrability":      IMPENETRABILITY,
    "collision":            COLLISION,
    "material":             MATERIAL,
    "buoyancy":             BUOYANCY,
    "displacement":         DISPLACEMENT,
    "flow_dynamics":        FLOW_DYNAMICS,
    "boundary_interaction": BOUNDARY_INTERACTION,
    "fluid_continuity":     FLUID_CONTINUITY,
    "reflection":           REFLECTION,
    "shadow":               SHADOW,
}
