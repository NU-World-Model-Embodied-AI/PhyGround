"""Shared physical-law criteria definitions — single source of truth.

Every module that needs law names, descriptions, or domain groupings should
import from here instead of defining its own copy.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Full bilingual criteria (Chinese + English) — used by VLM evaluation prompts
# ──────────────────────────────────────────────────────────────────────────────

CRITERIA_EN: dict[str, str] = {
    "gravity": "Do unsupported objects fall downward? Do thrown objects follow a curved trajectory? Does poured liquid fall with gravity?",
    "inertia": "Do stationary objects remain still unless acted upon? Do moving objects maintain their motion unless stopped by friction, collision, or an obstacle?",
    "momentum": "After collision, push, or pull, is the direction of motion reasonable? Ignore speed magnitude.",
    "impenetrability": "Do objects maintain impenetrability — no passing through each other?",
    "collision": "After impact, is there reasonable bounce/shatter/deformation? Does response match impact force?",
    "material": "Does each material respond according to its properties? (glass shatters, rubber bounces, metal is rigid, cloth deforms softly, etc.)",
    "buoyancy": "Do dense objects sink? Do wood/plastic float?",
    "displacement": "When you add more liquid or put an object into it, does the liquid level rise in a realistic way? Does it overflow when full?",
    "flow_dynamics": "Does the liquid's overall motion behave realistically over time — flowing along surfaces, spreading, draining naturally?",
    "boundary_interaction": "When the liquid hits a boundary such as a rock face, container wall, or floor, does it respond realistically? Do local splash, rebound, or split patterns on impact look physically plausible?",
    "fluid_continuity": "Does the liquid avoid disappearing or appearing out of nowhere? Small splashes that briefly break apart are okay.",
    "reflection": "Does the reflection roughly match objects and colors in the scene, and avoid completely unrelated content?",
    "shadow": "Are shadow directions consistent with light source? Do shadows move with objects?",
}

CRITERIA_ZH: dict[str, str] = {
    "gravity": "无支撑的物体（固体或液体）是否向下掉落？抛出的物体轨迹是否呈弧线下落？倾倒的液体是否沿重力方向落下？",
    "inertia": "静止物体是否保持静止？运动物体是否在没有摩擦、碰撞或障碍的情况下保持运动？",
    "momentum": "碰撞、推或拉后，物体运动方向是否合理？忽略速度大小",
    "impenetrability": "物体之间是否保持不可穿透，没有穿模？",
    "collision": "物体撞击后是否有合理的反弹/碎裂/变形？撞击力度与响应程度是否匹配？",
    "material": "每种材料的响应是否符合其属性？玻璃碎裂、橡胶弹回、金属坚硬、布料柔软变形。",
    "buoyancy": "密度大于水的物体是否下沉？木头/塑料等是否浮于水面？",
    "displacement": "当添加液体或物体浸入时，液面是否合理地上升？容器满后是否溢出？",
    "flow_dynamics": "液体整体流动是否符合物理——沿表面流动、铺展、排出是否自然？",
    "boundary_interaction": "液体撞击边界（如岩壁、杯壁、地面）时，局部交互是否合理？撞击后的飞溅、反弹或分流形态是否物理上可信？",
    "fluid_continuity": "液体是否保持物理连续性与质量守恒——无不合理的断裂、消失或凭空生成？短暂飞溅分离可接受。",
    "reflection": "镜子/光滑表面中的反射内容是否大致匹配场景中的物体和颜色，没有完全不相关的内容。",
    "shadow": "阴影方向是否与光源位置一致？物体移动时阴影是否同步移动？",
}

# Backward-compatible alias — defaults to English
CRITERIA = CRITERIA_EN

# Re-export sub-question definitions (defined in evals/sub_questions.py)
from evals.sub_questions import SUB_QUESTIONS, SubQuestion  # noqa: F401

from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────────────────────────────
# Human-eval Rating Instrument — structured criteria for human annotators
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class HumanCriterion:
    """A single criterion in the human-eval rating instrument."""
    key: str                    # e.g. "gravity"
    code: str                   # e.g. "A1"
    domain: str                 # e.g. "Solid-Body Mechanics"
    name: str                   # e.g. "Gravity"
    question: str               # Full question text
    note: str = ""              # Optional guidance note
    name_zh: str = ""           # Chinese name
    question_zh: str = ""       # Chinese question text
    note_zh: str = ""           # Chinese guidance note
    scale: list[int | str] = field(default_factory=lambda: [1, 2, 3, 4, 5, "N/A"])


HUMAN_CRITERIA: list[HumanCriterion] = [
    # ── G. General ───────────────────────────────────────────────────────────
    HumanCriterion(
        key="persistence", code="G1", domain="General", name="Object Persistence",
        question="To what extent do objects maintain consistent appearance, shape, and existence throughout the video?",
        name_zh="物体持久性",
        question_zh="物体在整个视频中在多大程度上保持了一致的外观、形状和存在？",
    ),
    HumanCriterion(
        key="PTV", code="G2", domain="General", name="Temporal Coherence",
        question="To what extent does the temporal sequence of physical events follow a physically plausible order?",
        note="Evaluate whether event ordering is logically consistent (e.g., objects break before scattering, liquid is poured before it flows).",
        name_zh="时序连贯性",
        question_zh="物理事件的时间顺序在多大程度上遵循了物理上合理的顺序？",
        note_zh="例如，物体先碎裂再散开，液体先被倒出再流动。",
    ),
    HumanCriterion(
        key="SA", code="G3", domain="General", name="Prompt Alignment",
        question="To what extent does the video content align with the text prompt?",
        note="Evaluate whether the depicted scene, objects, actions, and overall narrative correspond to the prompt description.",
        name_zh="提示词对齐",
        question_zh="视频内容在多大程度上与文本提示词一致？",
        note_zh="评估所描绘的场景、物体、动作和整体叙事是否与提示词描述相对应。",
    ),
    # ── A. Solid-Body Mechanics ──────────────────────────────────────────────
    HumanCriterion(
        key="gravity", code="A1", domain="Solid-Body Mechanics", name="Gravity",
        question="To what extent do objects and liquids move in accordance with gravity?",
        note="e.g., unsupported objects fall downward, projectiles follow curved trajectories, poured liquids descend.",
        name_zh="重力",
        question_zh="物体和液体在多大程度上按照重力运动？",
        note_zh="例如，无支撑的物体向下掉落，抛射物沿弧线轨迹运动，倾倒的液体向下流。",
    ),
    HumanCriterion(
        key="inertia", code="A2", domain="Solid-Body Mechanics", name="Inertia",
        question="To what extent do objects maintain their state of motion — remaining stationary or continuing to move — in the absence of a plausible external force?",
        note="Penalise significant spontaneous motion or unexplained stopping with no identifiable cause.",
        name_zh="惯性",
        question_zh="在没有合理外力的情况下，物体在多大程度上保持其运动状态——静止的保持静止，运动的继续运动？",
        note_zh="对无明确原因的显著自发运动或无故停止予以扣分。",
    ),
    HumanCriterion(
        key="momentum", code="A3", domain="Solid-Body Mechanics", name="Momentum",
        question="To what extent are post-collision directions of motion physically plausible?",
        note="Evaluate direction only; ignore speed magnitude.",
        name_zh="动量",
        question_zh="碰撞后物体的运动方向在多大程度上是物理合理的？",
        note_zh="仅评估方向；忽略速度大小。",
    ),
    HumanCriterion(
        key="impenetrability", code="A4", domain="Solid-Body Mechanics", name="Impenetrability",
        question="To what extent do solid objects maintain impenetrability, avoiding passage through one another?",
        name_zh="不可穿透性",
        question_zh="固体物体在多大程度上保持不可穿透，避免相互穿过？",
    ),
    HumanCriterion(
        key="collision", code="A5", domain="Solid-Body Mechanics", name="Collision Response",
        question="To what extent do objects exhibit physically plausible responses to impact, proportional to the applied force?",
        note="e.g., bouncing, shattering, deforming.",
        name_zh="碰撞响应",
        question_zh="物体在多大程度上表现出与施加力成比例的物理合理碰撞响应？",
        note_zh="例如，弹跳、碎裂、变形。",
    ),
    HumanCriterion(
        key="material", code="A6", domain="Solid-Body Mechanics", name="Material Properties",
        question="To what extent do objects respond in ways consistent with their apparent material properties?",
        note="e.g., glass shatters, rubber bounces, metal dents, fabric drapes.",
        name_zh="材料属性",
        question_zh="物体在多大程度上以符合其表观材料属性的方式做出响应？",
        note_zh="例如，玻璃碎裂、橡胶弹跳、金属凹陷、布料垂坠。",
    ),
    # ── B. Fluid Dynamics ────────────────────────────────────────────────────
    HumanCriterion(
        key="buoyancy", code="B1", domain="Fluid Dynamics", name="Buoyancy",
        question="To what extent do objects sink or float in a manner consistent with their apparent density?",
        name_zh="浮力",
        question_zh="物体在多大程度上以符合其表观密度的方式下沉或漂浮？",
    ),
    HumanCriterion(
        key="displacement", code="B2", domain="Fluid Dynamics", name="Displacement",
        question="When volume is added or an object is submerged, to what extent does the liquid level rise in a physically plausible manner?",
        note="e.g., level rises proportionally; container overflows when full.",
        name_zh="液面变化",
        question_zh="当添加液体或物体浸入时，液面是否合理地上升？",
        note_zh="例如，液面按比例上升；容器满时溢出。",
    ),
    HumanCriterion(
        key="flow_dynamics", code="B3", domain="Fluid Dynamics", name="Flow Dynamics",
        question="To what extent does liquid flow along surfaces, spread, and drain in a physically plausible manner?",
        note="Ignore brief splash details. Penalise unphysical acceleration, sudden stops, or unsupported uphill flow.",
        name_zh="流体动力学",
        question_zh="液体在多大程度上以物理合理的方式沿表面流动、铺展和排出？",
        note_zh="忽略短暂飞溅细节。对非物理的加速、突然停止或无支撑的上坡流动予以扣分。",
    ),
    HumanCriterion(
        key="boundary_interaction", code="B4", domain="Fluid Dynamics", name="Boundary Interaction",
        question="To what extent does liquid splash, rebound, or split in a physically plausible manner when hitting a boundary?",
        note="e.g., container walls, rock faces, floors.",
        name_zh="边界交互",
        question_zh="液体撞击边界时，其飞溅、反弹或分流在多大程度上是物理合理的？",
        note_zh="例如，容器壁、岩面、地面。",
    ),
    HumanCriterion(
        key="fluid_continuity", code="B5", domain="Fluid Dynamics", name="Fluid Continuity",
        question="Does the liquid avoid disappearing or appearing out of nowhere?",
        note="Small splashes that briefly break apart are okay.",
        name_zh="流体连续性",
        question_zh="液体在多大程度上避免了非物理的碎裂、消失或凭空生成？",
        note_zh="短暂的、物理合理的飞溅分离可以接受。",
    ),
    # ── C. Optical Physics ───────────────────────────────────────────────────
    HumanCriterion(
        key="reflection", code="C1", domain="Optical Physics", name="Reflection",
        question="To what extent do reflective surfaces display content that is physically plausible given the surrounding scene?",
        note="Evaluate whether reflected objects and colours broadly correspond to the scene. Do not require precise geometric mirroring.",
        name_zh="反射",
        question_zh="反射表面在多大程度上显示了与周围场景物理合理的内容？",
        note_zh="评估反射的物体和颜色是否大致对应场景。不要求精确的几何镜像。",
    ),
    HumanCriterion(
        key="shadow", code="C2", domain="Optical Physics", name="Shadow",
        question="To what extent are shadows physically plausible in their direction, placement, and movement relative to light sources and objects?",
        name_zh="阴影",
        question_zh="阴影的方向、位置和运动在多大程度上相对于光源和物体是物理合理的？",
    ),
]

#: Quick lookup by criterion key for human-eval criteria.
HUMAN_CRITERIA_BY_KEY: dict[str, HumanCriterion] = {c.key: c for c in HUMAN_CRITERIA}


def get_criteria_text(law_name: str) -> str:
    """Build full criteria text for a law, preferring HumanCriterion when available."""
    criterion = HUMAN_CRITERIA_BY_KEY.get(law_name)
    if criterion:
        text = criterion.question
        if criterion.note:
            text += f" ({criterion.note})"
        return text
    return CRITERIA.get(law_name, "")

#: Domain grouping for human-eval criteria.
HUMAN_DOMAINS: dict[str, list[HumanCriterion]] = {}
for _c in HUMAN_CRITERIA:
    HUMAN_DOMAINS.setdefault(_c.domain, []).append(_c)

#: Ordered list of all 13 physical laws (stable order for tabular reporting).
ALL_LAWS: list[str] = list(CRITERIA_EN.keys())

# ──────────────────────────────────────────────────────────────────────────────
# Domain → criteria-key groupings
# ──────────────────────────────────────────────────────────────────────────────

def _subs(keys: list[str]) -> list[tuple[str, str]]:
    """Return list of (key, bilingual_description) tuples for the given keys."""
    return [(k, CRITERIA[k]) for k in keys]

#: "fluid" domain covers liquid behaviour only (not smoke, steam, or other gases).
#: Maps each physical domain to its list of (law_key, description) pairs.
DOMAIN_SUBSCORES: dict[str, list[tuple[str, str]]] = {
    "mechanics":      _subs(["gravity", "inertia", "momentum", "impenetrability", "collision", "material"]),
    "collision":      _subs(["collision", "impenetrability", "momentum", "material"]),
    "gravity":        _subs(["gravity", "inertia"]),
    "buoyancy":       _subs(["buoyancy", "displacement"]),
    "fluid":          _subs(["flow_dynamics", "boundary_interaction", "fluid_continuity"]),
    "lighting":       _subs(["reflection", "shadow"]),
}

