"""Human evaluation annotation app."""
import hashlib
import json
import random
import sqlite3
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    g,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
    jsonify,
    abort,
)

from human_eval.config import (
    N_ANNOTATORS_PER_VIDEO,
    ASSIGNMENT_TTL_HOURS,
    VIDEO_DATA_DIR,
    DB_PATH,
    SECRET_KEY,
    DISAGREEMENT_TOP_K,
    STATUS_ASSIGNED,
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    STATUS_SKIPPED,
    COMPARISON_MODELS,
    MODELS_PER_GROUP,
    COMPLETION_SURVEY_URL,
    extract_model,
    get_batch_size,
    VALID_COHORTS,
    TEST_COHORTS,
    COHORT_COMPLETION_CODE,
)
from human_eval.db import init_db, get_db, pending_assignment_sql
from human_eval.assign import assign_comparison_batch

# ---------------------------------------------------------------------------
# Import CRITERIA & DOMAIN_SUBSCORES from physics_criteria (single source of truth)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from evals.physics_criteria import (
    CRITERIA, CRITERIA_EN, CRITERIA_ZH, DOMAIN_SUBSCORES,
    HUMAN_CRITERIA, HUMAN_CRITERIA_BY_KEY, HUMAN_DOMAINS,
)

def _all_criteria_for_lang(lang: str) -> list[tuple[str, str]]:
    """Return a flat, deduplicated list of (law_key, description) for display."""
    src = CRITERIA_ZH if lang == "zh" else CRITERIA_EN
    return [(k, src[k]) for k in src]

def _get_physical_dims(assignment) -> list[tuple[str, str]]:
    """Parse physical_laws JSON from a video/assignment row and return (law, description) pairs."""
    laws = json.loads(assignment["physical_laws"])
    return [(law, CRITERIA.get(law, law)) for law in laws]


# ---------------------------------------------------------------------------
# General dimensions — derived from HumanCriterion (General domain)
# ---------------------------------------------------------------------------
_SCORE_LABELS = {1: "Completely implausible", 2: "Largely implausible", 3: "Partially plausible", 4: "Mostly plausible", 5: "Fully plausible"}
_SCORE_LABELS_ZH = {1: "完全不合理", 2: "大部分不合理", 3: "部分合理", 4: "大部分合理", 5: "完全合理"}

GENERAL_DIMS = [
    (
        c.key, c.name, list(range(1, 6)),
        f"{c.question} ({c.note})" if c.note else c.question,
        _SCORE_LABELS,
    )
    for c in HUMAN_DOMAINS["General"]
]

GENERAL_DIMS_ZH = [
    (
        c.key, c.name_zh, list(range(1, 6)),
        f"{c.question_zh}（{c.note_zh}）" if c.note_zh else c.question_zh,
        _SCORE_LABELS_ZH,
    )
    for c in HUMAN_DOMAINS["General"]
]


def _save_annotation(db, assignment_id, general_scores, physical_scores, physical_dims, meta_ver, note="", play_count=0, stay_seconds=0.0, na_laws=None):
    """Upsert annotation + annotation_items for a single assignment."""
    if na_laws is None:
        na_laws = []
    law_keys = [key for key, _ in physical_dims]
    scores_json = json.dumps({
        "general": general_scores,
        "physical": physical_scores,
        "physical_laws": law_keys,
    })
    na_laws_json = json.dumps(na_laws)

    existing = db.execute(
        "SELECT id FROM annotations WHERE assignment_id = ?", (assignment_id,)
    ).fetchone()

    if existing:
        annotation_id = existing["id"]
        db.execute(
            "UPDATE annotations SET scores_json = ?, metadata_version = ?, note = ?, play_count = ?, stay_seconds = ?, na_laws = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (scores_json, meta_ver, note, play_count, stay_seconds, na_laws_json, annotation_id),
        )
        db.execute("DELETE FROM annotation_items WHERE annotation_id = ?", (annotation_id,))
    else:
        cur = db.execute(
            "INSERT INTO annotations (assignment_id, scores_json, metadata_version, note, play_count, stay_seconds, na_laws) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (assignment_id, scores_json, meta_ver, note, play_count, stay_seconds, na_laws_json),
        )
        annotation_id = cur.lastrowid

    items = [(annotation_id, key, None, general_scores[key]) for key in general_scores]
    items += [(annotation_id, key, key, physical_scores[key]) for key in physical_scores if physical_scores[key] is not None]
    db.executemany(
        "INSERT INTO annotation_items (annotation_id, dimension, law, score) VALUES (?, ?, ?, ?)",
        items,
    )



def _extract_scores(prefix, physical_dims):
    """Parse general + physical scores and NA laws from the current request form."""
    general = {}
    for key, _, _, _, _ in GENERAL_DIMS:
        val = request.form.get(f"{prefix}{key}")
        if val:
            general[key] = int(val)
    physical = {}
    for key, _ in physical_dims:
        val = request.form.get(f"{prefix}{key}")
        if val and val != "null":
            physical[key] = int(val)
    # NA laws come from a separate comma-separated hidden input
    na_raw = request.form.get(f"{prefix}na_laws", "")
    na_laws = [k for k in na_raw.split(",") if k]
    return general, physical, na_laws


# ---------------------------------------------------------------------------
# Demo data (hardcoded, editable)
# ---------------------------------------------------------------------------

DEMO_ENTRIES = [
    {
        "video_url": "/video/wan2.2-i2v-a14b/collision_283.mp4",
        "prompt": "A water balloon is thrown at a large target made of cardboard; it bursts cleanly.",
        "prompt_zh": "一个水气球被投向一个纸板做的大靶子，干脆地炸裂开来。",
        "scores": {"SA": 4, "PTV": 3, "persistence": 2},
        "physical_dims": [
            ("collision", "After impact, is there reasonable bounce/shatter/deformation? Does response match impact force?"),
            ("material", "Do different materials respond according to their properties? Glass shatters? Rubber bounces?"),
            ("momentum", "After collision, is the direction of motion reasonable? Ignore speed magnitude."),
            ("impenetrability", "Do objects maintain impenetrability — no passing through each other?"),
        ],
        "physical_dims_zh": [
            ("collision", "碰撞后是否有合理的弹跳/破碎/变形？响应是否与撞击力匹配？"),
            ("material", "不同材质是否按其属性反应？玻璃碎裂？橡胶弹跳？"),
            ("momentum", "碰撞后运动方向是否合理？忽略速度大小。"),
            ("impenetrability", "物体是否保持不可穿透性——没有互相穿过？"),
        ],
        "physical_scores": {"collision": 4, "material": 3, "momentum": 2, "impenetrability": 1},
        "rationale_general": {
            "SA": "The video shows a water balloon being thrown at a target and bursting. The overall scene matches the prompt well.",
            "PTV": "The balloon approaches and bursts on contact, but because the target behaves like an inflatable object rather than cardboard, the physical interaction sequence is only partially correct.",
            "persistence": "The large target morphs from cardboard into an inflatable object mid-video — a major material/shape inconsistency.",
        },
        "rationale_general_zh": {
            "SA": "视频展示了一个水气球被投向靶子并炸裂的过程，整体场景与提示词匹配良好。",
            "PTV": "气球接近并在接触时炸裂，但由于靶子表现得像充气物体而非纸板，物理交互序列仅部分正确。",
            "persistence": "大靶子在视频中途从纸板变形为充气物体——存在严重的材质/形状不一致。",
        },
        "rationale_physical": {
            "collision": "The balloon deforms and shatters on impact, which is a reasonable response to the collision force.",
            "material": "The water balloon bursts correctly on impact, but the cardboard target expands/inflates unnaturally — real cardboard would not swell like that.",
            "momentum": "After the balloon bursts, the water and debris should stop and fall downward, but they continue moving unnaturally.",
            "impenetrability": "The balloon passes through the target instead of bouncing off or bursting on the surface — objects should not penetrate each other.",
        },
        "rationale_physical_zh": {
            "collision": "气球在撞击时变形并破裂，这是对碰撞力的合理响应。",
            "material": "水气球在撞击时正确地破裂，但纸板靶子不自然地膨胀——真实的纸板不会像那样膨胀。",
            "momentum": "气球破裂后，水和碎片应该停下并向下掉落，但它们继续不自然地移动。",
            "impenetrability": "气球穿过了靶子，而不是在表面弹开或破裂——物体不应互相穿透。",
        },
    },
    {
        "video_url": "/video/cosmos-predict2.5-14b/88H1BBqFzXQ_1_118to222.mp4",
        "prompt": "A heavy car tire rolls over a green bottle on concrete, crushing it flat. The trapped liquid bursts from the broken bottle and rapidly flows outward, spreading across the ground in an expanding puddle.",
        "prompt_zh": "一个沉重的汽车轮胎在混凝土地面上碾过一个绿色瓶子，将其压扁。瓶中的液体从破碎的瓶子中喷出，迅速向外流动，在地面上形成一个不断扩大的水洼。",
        "scores": {"SA": 3, "PTV": 1, "persistence": 3},
        "physical_dims": [
            ("flow_dynamics", "Does the liquid's overall motion behave realistically over time — flowing along surfaces, spreading, draining naturally?"),
            ("material", "Does each material respond according to its properties? (glass shatters, rubber bounces, metal is rigid, cloth deforms softly, etc.)"),
            ("collision", "After impact, is there reasonable bounce/shatter/deformation? Does response match impact force?"),
            ("fluid_continuity", "Does the liquid maintain physical continuity and mass conservation? Brief splash separation is acceptable."),
        ],
        "physical_dims_zh": [
            ("flow_dynamics", "液体整体流动是否符合物理——沿表面流动、铺展、排出是否自然？"),
            ("material", "每种材料的响应是否符合其属性？玻璃碎裂、橡胶弹回、金属坚硬、布料柔软变形。"),
            ("collision", "物体撞击后是否有合理的反弹/碎裂/变形？撞击力度与响应程度是否匹配？"),
            ("fluid_continuity", "液体是否保持物理连续性与质量守恒——无不合理的断裂、消失或凭空生成？短暂飞溅分离可接受。"),
        ],
        "physical_scores": {"flow_dynamics": 4, "material": 1, "collision": 1, "fluid_continuity": 4},
        "rationale_general": {
            "SA": "The liquid bursting from the broken bottle and spreading into a puddle is present, but the first part — a heavy tire rolling over and crushing the bottle flat — is missing.",
            "PTV": "The bottle breaks without any logical cause — there is no visible impact or force that triggers the rupture.",
            "persistence": "The tire and ground maintain consistent appearance, but the text on the bottle label changes mid-video.",
        },
        "rationale_general_zh": {
            "SA": "被困的液体从破碎的瓶子中爆裂而出并在地面形成水洼，这部分有体现；但前半部分——沉重的汽车轮胎碾过绿色瓶子并将其压扁——缺失。",
            "PTV": "瓶子没有任何逻辑原因就破裂了——没有可见的撞击或外力触发破裂。",
            "persistence": "轮胎和地面保持一致，但瓶子的贴纸文字在视频中途发生了变化。",
        },
        "rationale_physical": {
            "flow_dynamics": "The liquid bursts out and rapidly flows outward, forming an expanding puddle on the ground — largely consistent with realistic flow behavior.",
            "material": "The glass bottle does not shatter convincingly; it deforms like a soft object rather than breaking into sharp fragments as real glass would.",
            "collision": "The bottle ruptures without any visible force or impact — it just bursts on its own.",
            "fluid_continuity": "No obvious disappearance, or unexpectedly appear of liquid throughout the video.",
        },
        "rationale_physical_zh": {
            "flow_dynamics": "液体爆裂而出，迅速向外流淌，在地面上形成不断扩大的水洼——基本符合真实的流动行为。",
            "material": "玻璃瓶的碎裂不够逼真，像软物体一样变形，而不是像真正的玻璃那样碎成锋利的碎片。",
            "collision": "瓶子没有受到任何可见的外力或撞击就爆裂了——完全是自行破裂。",
            "fluid_continuity": "整个视频中液体没有明显的碎裂、消失或凭空生成现象。",
        },
    },
    {
        "video_url": "/video/cosmos-predict2.5-2b/collision_144.mp4",
        "prompt": "A baseball is struck by a bat, causing a visible scuff mark on the ball\u2019s surface.",
        "prompt_zh": "一根球棒击中一个棒球，在球的表面留下一个明显的擦痕。",
        "scores": {"SA": 3, "PTV": 1, "persistence": 3},
        "physical_dims": [
            ("collision", "After impact, is there reasonable bounce/shatter/deformation? Does response match impact force?"),
            ("material", "Do different materials respond according to their properties? Glass shatters? Rubber bounces?"),
            ("momentum", "After collision, is the direction of motion reasonable? Ignore speed magnitude."),
        ],
        "physical_dims_zh": [
            ("collision", "碰撞后是否有合理的弹跳/破碎/变形？响应是否与撞击力匹配？"),
            ("material", "不同材质是否按其属性反应？玻璃碎裂？橡胶弹跳？"),
            ("momentum", "碰撞后运动方向是否合理？忽略速度大小。"),
        ],
        "physical_scores": {"collision": 1, "material": 1, "momentum": 1},
        "rationale_general": {
            "SA": "A visible scuff mark appears on the ball, but the bat never clearly strikes the baseball — the striking action is missing.",
            "PTV": "The bat never strikes the ball, yet a scuff mark appears — the temporal sequence is highly inconsistent and misordered.",
            "persistence": "The ball maintains mostly consistent appearance, but the bat only appears for a brief moment before disappearing.",
        },
        "rationale_general_zh": {
            "SA": "球上出现了明显的擦痕，但球棒从未清晰地击中棒球——缺少击球动作。",
            "PTV": "球棒没有击中球，就出现了擦痕，时间顺序极其不一致且顺序错误。",
            "persistence": "球基本保持一致的外观，但球棒只出现了一瞬间就消失了。",
        },
        "rationale_physical": {
            "collision": "No real collision occurs — the bat never makes clear contact with the ball, so there is no physically plausible impact response.",
            "material": "The baseball does not fly away when struck. Instead, a massive, unrealistic brown chunk appears on its surface, resembling mud rather than a scuff mark. The bat also exhibits morphing and clipping issues.",
            "momentum": "The direction of motion is implausible because no real collision occurs — the bat never makes clear contact with the ball.",
        },
        "rationale_physical_zh": {
            "collision": "没有发生真正的碰撞——球棒从未清晰地接触到棒球，因此不存在物理上合理的撞击响应。",
            "material": "棒球被击中后没有飞走。相反，其表面出现了一个巨大的、不真实的棕色块状物，看起来像泥巴而不是擦痕。球棒也表现出变形和穿模问题。",
            "momentum": "运动方向不合理，因为根本没有发生真正的碰撞——球棒从未清晰地接触到棒球。",
        },
    },
]


# ---------------------------------------------------------------------------
# login_required decorator
# ---------------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "annotator_id" not in session:
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


def verify_ownership(assignment):
    if assignment is None:
        abort(404)
    if assignment["annotator_id"] != session["annotator_id"]:
        abort(403)


# ---------------------------------------------------------------------------
# App-level DB helper (stores conn on g, tied to app context)
# ---------------------------------------------------------------------------
def get_app_db() -> sqlite3.Connection:
    from flask import current_app
    # For in-memory databases, use the shared connection stored on the app
    if hasattr(current_app, '_mem_conn'):
        return current_app._mem_conn
    if "db" not in g:
        g.db = get_db(g.db_path)
    return g.db


def close_app_db(exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_app(
    db_path=None,
    video_data_dir=None,
    skip_import=True,
):
    app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"))
    app.secret_key = SECRET_KEY

    resolved_db_path = db_path or str(DB_PATH)
    resolved_video_dir = str(video_data_dir) if video_data_dir else str(VIDEO_DATA_DIR)

    # Initialise schema (for :memory: or first run)
    if resolved_db_path == ":memory:":
        # Keep a single in-memory connection for the lifetime of the app
        mem_conn = sqlite3.connect(":memory:")
        mem_conn.row_factory = sqlite3.Row
        init_db(mem_conn)
        app._mem_conn = mem_conn

        # Don't close the shared in-memory connection on teardown
    else:
        conn = get_db(Path(resolved_db_path))
        init_db(conn)
        if not skip_import:
            from human_eval.import_videos import import_videos
            import_videos(conn, Path(resolved_video_dir))
        conn.close()

        @app.before_request
        def _inject_db_path():
            g.db_path = resolved_db_path

        app.teardown_appcontext(close_app_db)

    # Expose constants to all templates (avoids passing on every render_template call)
    app.jinja_env.globals.update(
        STATUS_ASSIGNED=STATUS_ASSIGNED,
        STATUS_COMPLETED=STATUS_COMPLETED,
        STATUS_PARTIAL=STATUS_PARTIAL,
        STATUS_SKIPPED=STATUS_SKIPPED,
        general_dims=GENERAL_DIMS,
        criteria=CRITERIA,
        domain_subscores=DOMAIN_SUBSCORES,
        human_criteria_by_key=HUMAN_CRITERIA_BY_KEY,
        human_criteria=HUMAN_CRITERIA,
        human_domains=HUMAN_DOMAINS,
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/")
    def login_page():
        if request.args.get("return_url"):
            session["return_url"] = request.args["return_url"]
        cohort = request.args.get("cohort", "others")
        if cohort not in VALID_COHORTS:
            cohort = "others"
        session["cohort"] = cohort
        return render_template("login.html", cohort=cohort)

    @app.route("/guide")
    def guide():
        lang = request.args.get("lang", "en")
        if lang not in ("en", "zh"):
            lang = "en"
        return render_template(
            "guide.html",
            lang=lang,
            general_dims_display=GENERAL_DIMS_ZH if lang == "zh" else GENERAL_DIMS,
            all_criteria=_all_criteria_for_lang(lang),
        )

    @app.route("/demo")
    @login_required
    def demo():
        if request.args.get("return_url"):
            session["return_url"] = request.args["return_url"]
        lang = request.args.get("lang", "en")
        if lang not in ("en", "zh"):
            lang = "en"
        dims = GENERAL_DIMS_ZH if lang == "zh" else GENERAL_DIMS

        # ---- task-list data (merged from /tasks) ----
        db = get_app_db()
        annotator_id = session["annotator_id"]
        user_batch_size = get_batch_size(_get_user_cohort(db, annotator_id))

        rows = db.execute(
            "SELECT a.id, a.status, a.expires_at, a.group_id, "
            "v.filename, v.dataset, v.prompt, v.physical_laws, "
            "cg.prompt AS group_prompt "
            "FROM assignments a "
            "JOIN videos v ON a.video_id = v.id "
            "LEFT JOIN comparison_groups cg ON a.group_id = cg.id "
            "WHERE a.annotator_id = ? AND a.group_id IS NOT NULL "
            "ORDER BY a.group_id, a.id",
            (annotator_id,),
        ).fetchall()

        groups = {}
        for r in rows:
            gid = r["group_id"]
            if gid not in groups:
                groups[gid] = {
                    "group_id": gid,
                    "prompt": r["group_prompt"] or r["prompt"],
                    "assignments": [],
                    "models": [],
                    "status": STATUS_COMPLETED,
                }
            groups[gid]["assignments"].append(r)
            groups[gid]["models"].append(extract_model(r["dataset"]))
            if r["status"] not in (STATUS_COMPLETED, STATUS_PARTIAL, STATUS_SKIPPED):
                groups[gid]["status"] = STATUS_ASSIGNED
            elif r["status"] == STATUS_PARTIAL and groups[gid]["status"] == STATUS_COMPLETED:
                groups[gid]["status"] = STATUS_PARTIAL

        group_list = list(groups.values())
        _done_statuses = (STATUS_COMPLETED, STATUS_PARTIAL)
        user_done_groups = len([g for g in group_list if g["status"] in _done_statuses])

        # Reached quota — redirect to thanks page
        if user_done_groups >= user_batch_size:
            return redirect(url_for("thanks"))

        pending_groups = [g for g in group_list if g["status"] not in _done_statuses]
        if len(group_list) > 0 and len(pending_groups) == 0:
            new_gids = _auto_assign(db, annotator_id)
            if new_gids:
                return redirect(url_for("demo"))

        total_prompts = db.execute(
            "SELECT COUNT(DISTINCT prompt) AS c FROM comparison_groups"
        ).fetchone()["c"]

        group_list.sort(key=lambda g: (g["status"] in (STATUS_COMPLETED, STATUS_PARTIAL), g["group_id"]))

        return render_template(
            "demo.html",
            lang=lang,
            demos=DEMO_ENTRIES,
            general_dims=dims,
            human_criteria_by_key=HUMAN_CRITERIA_BY_KEY,
            groups=group_list,
            annotator_name=session.get("annotator_name"),
            total_prompts=total_prompts,
            user_completed_groups=user_done_groups,
            user_quota=user_batch_size,
        )

    @app.route("/thanks")
    @login_required
    def thanks():
        db = get_app_db()
        cohort = _get_user_cohort(db, session["annotator_id"])
        code = COHORT_COMPLETION_CODE.get(cohort, f"{random.randint(0, 999999):06d}")
        return render_template("thanks.html", code=code, survey_url=COMPLETION_SURVEY_URL)

    @app.route("/login", methods=["POST"])
    def login():
        username = (request.form.get("username") or "").strip()
        if not username:
            return render_template("login.html")

        cohort = (request.form.get("cohort") or session.get("cohort") or "others").strip()
        if cohort not in VALID_COHORTS:
            cohort = "others"

        db = get_app_db()
        row = db.execute("SELECT * FROM annotators WHERE name = ?", (username,)).fetchone()
        if row is None:
            cur = db.execute("INSERT INTO annotators (name, cohort) VALUES (?, ?)", (username, cohort))
            db.commit()
            row = db.execute("SELECT * FROM annotators WHERE id = ?", (cur.lastrowid,)).fetchone()
        elif not row["cohort"] or row["cohort"] != cohort:
            # Update cohort if user re-registers via a different link
            db.execute("UPDATE annotators SET cohort = ? WHERE id = ?", (cohort, row["id"]))
            db.commit()

        session["annotator_id"] = row["id"]
        session["annotator_name"] = row["name"]
        session["cohort"] = cohort

        # Auto-assign comparison groups
        _auto_assign(db, row["id"])

        return redirect(url_for("demographics"))

    @app.route("/demographics", methods=["GET", "POST"])
    @login_required
    def demographics():
        db = get_app_db()

        if request.method == "POST":
            gender = (request.form.get("gender") or "").strip()
            age = (request.form.get("age") or "").strip()
            major = (request.form.get("major") or "").strip()
            education = (request.form.get("education") or "").strip()
            if major == "other":
                major = (request.form.get("major_other") or "").strip() or "other"
            if not gender or not age or not major or not education:
                return redirect(url_for("demographics"))
            db.execute(
                "UPDATE annotators SET gender = ?, age = ?, major = ?, education = ? WHERE id = ?",
                (gender, age, major, education, session["annotator_id"]),
            )
            db.commit()
            return redirect(url_for("demo"))

        # GET — redirect to demo if already filled
        ann_row = db.execute(
            "SELECT gender, age, major, education FROM annotators WHERE id = ?",
            (session["annotator_id"],),
        ).fetchone()
        if ann_row["gender"] and ann_row["age"] and ann_row["major"] and ann_row["education"]:
            return redirect(url_for("demo"))
        return render_template("demographics.html")

    def _get_user_cohort(db, annotator_id) -> str:
        row = db.execute("SELECT cohort FROM annotators WHERE id = ?", (annotator_id,)).fetchone()
        return (row["cohort"] if row and row["cohort"] else "others")

    def _auto_assign(db, annotator_id):
        """Assign a comparison batch with cohort-specific config."""
        cohort = _get_user_cohort(db, annotator_id)
        return assign_comparison_batch(
            db,
            annotator_id,
            n_annotators=N_ANNOTATORS_PER_VIDEO,
            batch_size=get_batch_size(cohort),
            ttl_hours=ASSIGNMENT_TTL_HOURS,
        )

    @app.route("/tasks")
    @login_required
    def task_list():
        return redirect(url_for("demo"))

    def _next_group_url(db, annotator_id, current_group_id=None):
        """Return URL for the next unfinished group, skipping current_group_id."""
        pending_sql = pending_assignment_sql("a")
        row = db.execute(
            "SELECT a.group_id FROM assignments a "
            "WHERE a.annotator_id = ? AND a.group_id IS NOT NULL "
            "AND a.group_id != ? "
            f"AND {pending_sql} "
            "ORDER BY a.group_id LIMIT 1",
            (annotator_id, current_group_id or ""),
        ).fetchone()
        if row:
            return f"/rate_group/{row['group_id']}"
        return None

    @app.route("/tasks/start", methods=["POST"])
    @login_required
    def start_next():
        """Redirect to the first unfinished comparison group."""
        db = get_app_db()
        annotator_id = session["annotator_id"]
        nxt = _next_group_url(db, annotator_id)
        if nxt:
            return redirect(nxt)
        _auto_assign(db, annotator_id)
        nxt = _next_group_url(db, annotator_id)
        if nxt:
            return redirect(nxt)
        return redirect(url_for("task_list"))

    @app.route("/tasks/more", methods=["POST"])
    @login_required
    def request_more():
        db = get_app_db()
        _auto_assign(db, session["annotator_id"])
        return redirect(url_for("task_list"))

    # ------------------------------------------------------------------
    # Comparison group routes
    # ------------------------------------------------------------------

    def _load_group_assignments(db, group_id, annotator_id):
        """Load and verify all assignments in a comparison group."""
        rows = db.execute(
            "SELECT a.*, v.filename, v.dataset, v.prompt, v.physical_laws, v.metadata_version AS v_meta_ver "
            "FROM assignments a JOIN videos v ON a.video_id = v.id "
            "WHERE a.group_id = ? AND a.annotator_id = ? "
            "ORDER BY a.id",
            (group_id, annotator_id),
        ).fetchall()
        if not rows:
            abort(404)
        return rows

    def _shuffle_for_group(items, group_id):
        """Deterministic shuffle based on group_id (same order on page refresh)."""
        seed = int(hashlib.md5(group_id.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        result = list(items)
        rng.shuffle(result)
        return result

    @app.route("/rate_group/<group_id>")
    @login_required
    def rate_group_page(group_id):
        # Persist return_url in session so it survives across group pages
        if request.args.get("return_url"):
            session["return_url"] = request.args["return_url"]
        db = get_app_db()
        annotator_id = session["annotator_id"]
        assignments = _load_group_assignments(db, group_id, annotator_id)

        # Shuffle order (blind evaluation — same model not always in same position)
        assignments = _shuffle_for_group(assignments, group_id)

        # Get physical dims from first assignment (same prompt = same laws)
        physical_dims = _get_physical_dims(assignments[0])

        # Load existing scores for all assignments in one query
        aid_list = [a["id"] for a in assignments]
        placeholders = ",".join("?" * len(aid_list))
        existing_rows = db.execute(
            f"SELECT assignment_id, scores_json, note, na_laws FROM annotations WHERE assignment_id IN ({placeholders})",
            aid_list,
        ).fetchall()
        existing_scores_map = {}
        for r in existing_rows:
            entry = json.loads(r["scores_json"])
            entry["note"] = r["note"] or ""
            entry["na_laws"] = json.loads(r["na_laws"]) if r["na_laws"] else []
            existing_scores_map[r["assignment_id"]] = entry

        # Assign blind labels: Model A, Model B, ...
        labels = [chr(65 + i) for i in range(len(assignments))]  # A, B, C, D, ...

        prompt = assignments[0]["prompt"]

        # Build status map for per-video state display
        status_map = {a["id"]: a["status"] for a in assignments}

        # User progress: completed groups / quota
        user_completed_groups = db.execute(
            "SELECT COUNT(DISTINCT a.group_id) AS c FROM assignments a "
            "WHERE a.annotator_id = ? AND a.group_id IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM assignments a2 WHERE a2.group_id = a.group_id "
            "AND a2.annotator_id = a.annotator_id AND a2.status NOT IN (?, ?, ?))",
            (annotator_id, STATUS_COMPLETED, STATUS_PARTIAL, STATUS_SKIPPED),
        ).fetchone()["c"]

        return render_template(
            "rate_compare.html",
            group_id=group_id,
            prompt=prompt,
            physical_dims=physical_dims,
            video_entries=list(zip(labels, assignments)),
            existing_scores_map=existing_scores_map,
            status_map=status_map,
            user_completed_groups=user_completed_groups,
            user_quota=get_batch_size(_get_user_cohort(db, annotator_id)),
            annotator_name=session.get("annotator_name"),
        )

    @app.route("/rate_group/<group_id>", methods=["POST"])
    @login_required
    def rate_group_submit(group_id):
        db = get_app_db()
        annotator_id = session["annotator_id"]
        assignments = _load_group_assignments(db, group_id, annotator_id)

        physical_dims = _get_physical_dims(assignments[0])

        general_keys = {key for key, _, _, _, _ in GENERAL_DIMS}
        physical_keys = {key for key, _ in physical_dims}

        completed_aids = []
        partial_aids = []
        for a in assignments:
            aid = a["id"]
            general_scores, physical_scores, na_laws = _extract_scores(f"v{aid}_", physical_dims)
            if not general_scores:
                continue
            note = request.form.get(f"v{aid}_note", "").strip()
            play_count = int(request.form.get(f"v{aid}_play_count", 0) or 0)
            stay_seconds = float(request.form.get(f"v{aid}_stay_seconds", 0) or 0)
            _save_annotation(db, aid, general_scores, physical_scores, physical_dims, a["v_meta_ver"], note=note, play_count=play_count, stay_seconds=stay_seconds, na_laws=na_laws)
            covered_physical = set(physical_scores.keys()) | set(na_laws)
            if general_scores.keys() >= general_keys and covered_physical >= physical_keys:
                completed_aids.append(aid)
            else:
                partial_aids.append(aid)

        db.commit()

        action = request.form.get("action", "save")

        if action == "save":
            return jsonify({"ok": True})

        # "Next" — mark assignments by completeness
        all_aids = [a["id"] for a in assignments]
        saved_aids = completed_aids + partial_aids
        if completed_aids:
            db.execute(
                "UPDATE assignments SET status = ?, completed_at = CURRENT_TIMESTAMP "
                "WHERE id IN ({}) AND status NOT IN (?, ?)".format(
                    ",".join("?" * len(completed_aids))),
                [STATUS_COMPLETED, *completed_aids, STATUS_COMPLETED, STATUS_SKIPPED],
            )
        if partial_aids:
            db.execute(
                "UPDATE assignments SET status = ?, completed_at = CURRENT_TIMESTAMP "
                "WHERE id IN ({}) AND status NOT IN (?, ?, ?)".format(
                    ",".join("?" * len(partial_aids))),
                [STATUS_PARTIAL, *partial_aids, STATUS_COMPLETED, STATUS_PARTIAL, STATUS_SKIPPED],
            )
        # Mark remaining (unannotated) assignments as skipped so progress advances
        unsaved_aids = [aid for aid in all_aids if aid not in saved_aids]
        if unsaved_aids:
            db.execute(
                "UPDATE assignments SET status = ?, completed_at = CURRENT_TIMESTAMP "
                "WHERE id IN ({}) AND status NOT IN (?, ?, ?)".format(
                    ",".join("?" * len(unsaved_aids))),
                [STATUS_SKIPPED, *unsaved_aids, STATUS_COMPLETED, STATUS_PARTIAL, STATUS_SKIPPED],
            )
        db.commit()
        return_url = session.pop("return_url", "")
        nxt = _next_group_url(db, annotator_id, current_group_id=group_id)
        if nxt:
            # Keep return_url in session for subsequent groups (re-set it)
            if return_url:
                session["return_url"] = return_url
            return jsonify({"ok": True, "redirect": nxt})
        return jsonify({"ok": True, "redirect": return_url or url_for("task_list")})

    @app.route("/rate_group/<group_id>/skip_video/<int:assignment_id>", methods=["POST"])
    @login_required
    def skip_video(group_id, assignment_id):
        db = get_app_db()
        assignment = db.execute(
            "SELECT * FROM assignments WHERE id = ? AND group_id = ?",
            (assignment_id, group_id),
        ).fetchone()
        verify_ownership(assignment)

        reason = request.form.get("skip_reason", "")
        db.execute(
            "UPDATE assignments SET status = ?, skip_reason = ? WHERE id = ?",
            (STATUS_SKIPPED, reason, assignment_id),
        )
        db.commit()

        # Auto-advance if group is fully done
        pending = db.execute(
            "SELECT COUNT(*) AS c FROM assignments "
            "WHERE group_id = ? AND annotator_id = ? AND status NOT IN (?, ?, ?)",
            (group_id, session["annotator_id"], STATUS_COMPLETED, STATUS_PARTIAL, STATUS_SKIPPED),
        ).fetchone()["c"]
        if pending == 0:
            nxt = _next_group_url(db, session["annotator_id"])
            if nxt:
                return redirect(nxt)
            return redirect(url_for("task_list"))

        return redirect(url_for("rate_group_page", group_id=group_id))

    @app.route("/video/<dataset>/<filename>")
    def serve_video(dataset, filename):
        video_dir = Path(resolved_video_dir) / dataset
        return send_from_directory(str(video_dir), filename)

    @app.route("/api/stats")
    @login_required
    def stats():
        db = get_app_db()

        total_videos = db.execute("SELECT COUNT(*) AS c FROM videos").fetchone()["c"]

        target_n = N_ANNOTATORS_PER_VIDEO

        # Coverage: aggregate in SQL
        coverage = db.execute(
            "SELECT "
            "SUM(CASE WHEN cnt >= ? THEN 1 ELSE 0 END) AS fully, "
            "SUM(CASE WHEN cnt > 0 AND cnt < ? THEN 1 ELSE 0 END) AS partially, "
            "SUM(CASE WHEN cnt = 0 THEN 1 ELSE 0 END) AS uncovered "
            "FROM ("
            "  SELECT COUNT(a.id) AS cnt "
            "  FROM videos v "
            "  LEFT JOIN assignments a ON v.id = a.video_id AND a.status IN (?, ?) "
            "  GROUP BY v.id"
            ")",
            (target_n, target_n, STATUS_COMPLETED, STATUS_PARTIAL),
        ).fetchone()
        fully = coverage["fully"] or 0
        partially = coverage["partially"] or 0
        uncovered = coverage["uncovered"] or 0

        # Per annotator
        per_annotator = db.execute(
            "SELECT an.name, "
            "SUM(CASE WHEN a.status = ? THEN 1 ELSE 0 END) AS completed, "
            "SUM(CASE WHEN a.status = ? THEN 1 ELSE 0 END) AS partial, "
            "SUM(CASE WHEN a.status = ? THEN 1 ELSE 0 END) AS assigned "
            "FROM annotators an "
            "LEFT JOIN assignments a ON an.id = a.annotator_id "
            "GROUP BY an.id",
            (STATUS_COMPLETED, STATUS_PARTIAL, STATUS_ASSIGNED),
        ).fetchall()

        per_annotator_list = [
            {"name": r["name"], "completed": r["completed"] or 0, "partial": r["partial"] or 0, "assigned": r["assigned"] or 0}
            for r in per_annotator
        ]

        # Disagreement top-k
        disagreement_rows = db.execute(
            "SELECT ai.dimension, ai.law, "
            "GROUP_CONCAT(ai.score) AS scores, "
            "MAX(ai.score) - MIN(ai.score) AS max_min_diff, "
            "a2.video_id "
            "FROM annotation_items ai "
            "JOIN annotations ann ON ai.annotation_id = ann.id "
            "JOIN assignments a2 ON ann.assignment_id = a2.id "
            "GROUP BY a2.video_id, ai.dimension, ai.law "
            "HAVING COUNT(ai.score) > 1 "
            "ORDER BY max_min_diff DESC "
            "LIMIT ?",
            (DISAGREEMENT_TOP_K,),
        ).fetchall()

        disagreement_list = [
            {
                "video_id": r["video_id"],
                "dimension": r["dimension"],
                "law": r["law"],
                "scores": [int(s) for s in r["scores"].split(",")],
                "max_min_diff": r["max_min_diff"],
            }
            for r in disagreement_rows
        ]

        return jsonify({
            "total_videos": total_videos,
            "coverage": {
                "fully_covered": fully,
                "partially_covered": partially,
                "uncovered": uncovered,
                "target_n": target_n,
            },
            "per_annotator": per_annotator_list,
            "disagreement_top_k": disagreement_list,
        })

    return app
