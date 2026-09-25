"""
app.py
======
ImmunoVision web application: Flask app factory, all routes (grouped into
Blueprints for auth / RHU / BHW / Admin / API), and the live analytics &
notification helpers the dashboards read from. Data cleaning, feature
engineering, and the ML model live in data_processor.py.

Run: python app.py
"""
import csv
import hashlib
import json
import io
import math
import os
import random
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

from flask import (Flask, Blueprint, render_template, request, redirect, url_for, flash, abort,
                   jsonify, g, has_request_context, Response)
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from sqlalchemy import func, or_

import data_processor as dp
from models import (
    db, login_manager, User, Barangay, Child, VaccineAntigen, VaccineType,
    VaccinationRecord, RiskAssessment, InventoryBatch, VaccineRequest, ActivityLog,
)

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a .env file in the project root, if present, into os.environ
except ImportError:
    pass

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "immunovision.db")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
        "DATABASE_URL", f"sqlite:///{DB_PATH}"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db.init_app(app)
    login_manager.init_app(app)

    app.register_blueprint(auth_bp)
    app.register_blueprint(rhu_bp)
    app.register_blueprint(bhw_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp)

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    @app.context_processor
    def inject_globals():
        return {"today": date.today(), "coverage_color": coverage_color,
                "simulated_barangays": simulated_barangays}

    return app


# ---------------------------------------------------------------------------
# Shared helpers: activity log, role guard, nav builder
# ---------------------------------------------------------------------------

# Activity-log categories. Derived from the action text so no schema change is
# needed; the keyword lists live here rather than in the template so the log
# page and its filter always agree.
LOG_CATEGORIES = (
    ("Login",   ("Logged in", "Logged out")),
    ("Admin",   ("Created user", "Enabled user", "Disabled user", "Reassigned")),
    ("Record",  ("Registered child", "Updated child", "Re-assessed risk")),
    ("Vaccine", ("vaccine stock", "vaccine request", "Approved", "Rejected", "Distributed", "Submitted")),
    ("Report",  ("Exported",)),
)


def log_category(action):
    for name, keywords in LOG_CATEGORIES:
        if any(k.lower() in action.lower() for k in keywords):
            return name
    return "Other"


def child_label(child):
    """Child name for the audit log, with the barangay appended only when the
    name does not already carry it (anonymised registry names look like
    "Unnamed Child #252 (Macayug)")."""
    name = child.display_name
    bgy = child.barangay.name if child.barangay else ""
    return name if (not bgy or f"({bgy})" in name) else f"{name} ({bgy})"


def log_activity(action):
    entry = ActivityLog(user_id=current_user.id if current_user.is_authenticated else None, action=action)
    db.session.add(entry)
    db.session.commit()


def role_required(*roles):
    from functools import wraps

    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            if not current_user.is_authenticated or current_user.role not in roles:
                abort(403)
            return view_func(*args, **kwargs)
        return wrapped
    return decorator


RHU_NAV = [
    ("dashboard", "home", "Dashboard", "rhu.dashboard"),
    ("children", "baby", "Child Records", "rhu.children"),
    ("risk", "heart-pulse", "Risk Classification", "rhu.risk"),
    ("continuation", "target", "Continuation Predictor", "rhu.continuation"),
    ("map", "map", "Municipality Map", "rhu.municipality_map"),
    ("inventory", "package", "Vaccine Inventory", "rhu.inventory"),
    ("requests", "clipboard", "Vaccine Requests", "rhu.requests"),
    ("forecast", "activity", "Vaccine Forecast", "rhu.forecast"),
    ("allocation", "package", "Stock Allocation", "rhu.allocation"),
    ("reports", "file-text", "Reports", "rhu.reports"),
]
BHW_NAV = [
    ("dashboard", "home", "Dashboard", "bhw.dashboard"),
    ("children", "baby", "Child Records", "bhw.children"),
    ("risk", "heart-pulse", "Risk Classification", "bhw.risk"),
    ("continuation", "target", "Continuation Predictor", "bhw.continuation"),
    ("requests", "clipboard", "Vaccine Requests", "bhw.requests"),
    ("reports", "file-text", "Reports", "bhw.reports"),
]
ADMIN_NAV = [
    ("dashboard", "home", "Dashboard", "admin.dashboard"),
    ("users", "users", "User Management", "admin.users"),
    ("assign", "map-pin", "Assign Barangay", "admin.assign_barangay"),
    ("map", "map", "Municipality Map", "admin.municipality_map"),
    ("vaccines", "package", "Vaccine Alerts", "admin.vaccine_alerts"),
    ("logs", "history", "Activity Logs", "admin.logs"),
]


def build_nav(spec):
    return [{"key": k, "icon": i, "label": l, "url": url_for(e)} for k, i, l, e in spec]


# ---------------------------------------------------------------------------
# Analytics: computed live from the database for every dashboard/chart.
# ---------------------------------------------------------------------------

def _children_query(barangay_id=None):
    q = Child.query
    return q.filter(Child.barangay_id == barangay_id) if barangay_id else q


# --- Per-request caches -----------------------------------------------------
# Child.vaccination_records and Child.risk_assessments are lazy="dynamic", so
# every access issues its own SELECT. Walking 563 children one at a time meant
# ~39,000 queries to render a single analytics page. These helpers load each
# table once per request and index it in memory; the analytics functions below
# read from the index instead of re-querying per child. The cache lives on
# flask.g, so it is discarded at the end of every request and never serves
# stale data across page loads.

def _cache(key, builder):
    if not has_request_context():
        return builder()
    store = getattr(g, "_iv_cache", None)
    if store is None:
        store = g._iv_cache = {}
    if key not in store:
        store[key] = builder()
    return store[key]


def _vaccine_types_by_id():
    return _cache("vaccine_types", lambda: {vt.id: vt for vt in VaccineType.query.all()})


def _records_by_child():
    def build():
        out = defaultdict(list)
        for record in VaccinationRecord.query.all():
            out[record.child_id].append(record)
        return out
    return _cache("records", build)


def _latest_risk_by_child():
    """Newest RiskAssessment per child (matches risk_assessments.first(),
    which orders by computed_at descending)."""
    def build():
        out = {}
        for ra in RiskAssessment.query.order_by(
            RiskAssessment.computed_at.asc(), RiskAssessment.id.asc()
        ).all():
            out[ra.child_id] = ra  # later rows overwrite earlier -> newest wins
        return out
    return _cache("risk", build)


def _latest_risk(child):
    return _latest_risk_by_child().get(child.id)


def _doses_given(child_id):
    """{vaccine_type_id: date_administered or None} for one child."""
    def build():
        out = defaultdict(dict)
        for cid, records in _records_by_child().items():
            for record in records:
                out[cid][record.vaccine_type_id] = record.date_administered
        return out
    return _cache("doses", build).get(child_id, {})


def _dose_matrix(children, vt_lookup):
    """{child_id: {schedule_code: date or None}} for the Child Records table.

    The template previously ran one query per child per vaccine (15 doses x
    563 children = ~8,400 queries per page load). Building the whole grid here
    costs nothing extra because the records are already cached."""
    matrix = {}
    for child in children:
        given = _doses_given(child.id)
        matrix[child.id] = {
            code: given.get(vt_lookup[code].id) if code in vt_lookup else None
            for code, *_ in dp.VACCINE_SCHEDULE
        }
    return matrix


def _risk_labels(children):
    """{child_id: risk_label or None} for list templates."""
    risk = _latest_risk_by_child()
    return {c.id: (risk[c.id].risk_label if c.id in risk else None) for c in children}


def _due_and_missed(child, with_codes=False):
    """Doses the child is old enough to have received, and how many are unfilled.

    With with_codes, also returns the dose codes that are unfilled, so the
    diseases those doses protect against can be named."""
    vt_by_id = _vaccine_types_by_id()
    age = child.age_in_days
    due = missed = 0
    codes = []
    for record in _records_by_child().get(child.id, ()):
        vt = vt_by_id.get(record.vaccine_type_id)
        if vt is not None and vt.recommended_age_days <= age:
            due += 1
            if record.date_administered is None:
                missed += 1
                codes.append(vt.code)
    return (due, missed, codes) if with_codes else (due, missed)


def _is_fully_immunized(child):
    due, missed = _due_and_missed(child)
    return True if due == 0 else ((due - missed) / due >= 0.9)


def dashboard_stats(barangay_id=None):
    children = _children_query(barangay_id).all()
    total = len(children)
    fully = sum(1 for c in children if _is_fully_immunized(c))
    risk = _latest_risk_by_child()
    at_risk = sum(1 for c in children
                  if (ra := risk.get(c.id)) and ra.risk_label == "At-Risk")
    return {
        "total_children": total, "fully_immunized": fully, "at_risk": at_risk,
        "coverage_rate": round((fully / total) * 100, 1) if total else 0.0,
    }


def coverage_by_barangay():
    by_barangay = defaultdict(list)
    for c in Child.query.all():
        by_barangay[c.barangay_id].append(c)
    risk = _latest_risk_by_child()
    rows = []
    for b in Barangay.query.order_by(Barangay.name).all():
        children = by_barangay.get(b.id, [])
        total = len(children)
        fully = sum(1 for c in children if _is_fully_immunized(c))
        at_risk = sum(1 for c in children
                      if (ra := risk.get(c.id)) and ra.risk_label == "At-Risk")
        rows.append({"barangay": b.name, "children": total, "fully_immunized": fully, "at_risk": at_risk,
                      "coverage": round((fully / total) * 100, 1) if total else 0.0})
    rows.sort(key=lambda r: r["coverage"], reverse=True)
    return rows


def coverage_by_vaccine(barangay_id=None):
    children = _children_query(barangay_id).all()
    vt_by_code = {vt.code: vt for vt in _vaccine_types_by_id().values()}
    # (child_id, vaccine_type_id) -> record, built once from the cached records
    given_on = {}
    for child_id, records in _records_by_child().items():
        for record in records:
            given_on[(child_id, record.vaccine_type_id)] = record.date_administered

    rows = []
    for code, name, antigen_code, dose_no, rec_days in dp.VACCINE_SCHEDULE:
        vt = vt_by_code.get(code)
        if not vt:
            continue
        eligible = [c for c in children if c.age_in_days >= rec_days]
        if not eligible:
            rows.append({"vaccine": name, "pct": 0})
            continue
        given = sum(1 for c in eligible if given_on.get((c.id, vt.id)) is not None)
        rows.append({"vaccine": name, "pct": round((given / len(eligible)) * 100, 1)})
    return rows


# Colour per antigen for the coverage-by-vaccine chart. Deliberately avoids
# green/amber/red: those carry coverage-status meaning everywhere else in the
# system, and a red bar here would read as "failing" when it only means "IPV".
ANTIGEN_COLOURS = {
    "BCG":  "#2563eb",
    "HEPB": "#0ea5e9",
    "DPT":  "#0d9488",
    "OPV":  "#7c3aed",
    "PCV":  "#a855f7",
    "IPV":  "#4f46e5",
    "MMR":  "#0891b2",
}


def dose_age_label(days):
    """Short age label for a dose, e.g. 'birth', '6wk', '9mo'.

    Ages follow the DOH Expanded Program on Immunization schedule. Shown on the
    chart because the dose names alone do not reveal which doses share a visit -
    DPT 1st, OPV 1st and PCV 1st are all given at the same 6-week appointment."""
    if days <= 0:
        return "birth"
    if days < 180:
        return f"{round(days / 7)}wk"
    return f"{round(days / 30.44)}mo"


def schedule_labels(rows):
    """Chart labels with the due age appended: 'OPV 1st Dose (6wk)'."""
    ages = {name: days for _c, name, _a, _d, days in dp.VACCINE_SCHEDULE}
    out = []
    for r in rows:
        days = ages.get(r["vaccine"])
        out.append(f"{r['vaccine']} ({dose_age_label(days)})" if days is not None else r["vaccine"])
    return out


# Coverage bands. One definition, used by the map, every barangay table and the
# landing page, so the same barangay cannot be amber on one screen and red on
# another. Green is the DOH Expanded Program on Immunization target of 95% -
# the same line the annual trend chart draws - so green means "meets target"
# rather than an arbitrary cut. 75% separates a barangay that is behind from
# one that is seriously behind.
COVERAGE_TARGET = 95          # DOH EPI target, also drawn on the trend charts
COVERAGE_BANDS = ((COVERAGE_TARGET, "green"), (75, "amber"))


def simulated_barangays():
    """Barangays whose records are simulated rather than from the RHU registry.

    Cached per request. The interface marks these so a simulated figure is never
    mistaken for a reported one."""
    cached = getattr(g, "_sim_bgys", None)
    if cached is None:
        rows = (db.session.query(Barangay.name)
                .join(Child, Child.barangay_id == Barangay.id)
                .filter(Child.source == "synthetic").distinct().all())
        cached = {r[0] for r in rows}
        g._sim_bgys = cached
    return cached


def feature_catalog():
    """Every engineered feature, with its weight in the current scorer and how
    it is expected to act once a model is trained.

    The prototype scorer weights six of these; the remaining features are still
    computed for every child and would be available to a classifier. Listing
    all twelve makes the feature set legible without reading the source."""
    rows = []
    for key, label in dp.FEATURE_LABELS.items():
        direction, note = dp.FEATURE_MODEL_ROLE.get(key, ("Context", ""))
        weight = dp.SCORING_RULES.get(key)
        rows.append({
            "key": key, "label": label, "weight": weight,
            "weight_pct": round(weight * 100) if weight else None,
            "direction": direction, "note": note,
            "used": weight is not None,
        })
    # Scored features first, heaviest first; the rest keep their defined order.
    rows.sort(key=lambda r: (not r["used"], -(r["weight"] or 0)))
    return rows


def chart_insights(by_vaccine, by_barangay, trend):
    """One computed sentence per chart.

    Every figure is derived from the same data the chart plots, so an insight
    cannot drift out of step with what is on screen. Barangays holding
    simulated records are excluded: a generated figure must not be reported as
    a finding. Returns {} for a chart when the data cannot support a claim."""
    out = {}

    if by_vaccine:
        lo = min(by_vaccine, key=lambda r: r["pct"])
        hi = max(by_vaccine, key=lambda r: r["pct"])
        short = sum(1 for r in by_vaccine if r["pct"] < COVERAGE_TARGET)
        out["vaccine"] = (
            f"{short} of {len(by_vaccine)} doses are below the {COVERAGE_TARGET}% target. "
            f"{lo['vaccine']} is lowest at {lo['pct']}%, against {hi['pct']}% for {hi['vaccine']}."
        )

    real = [r for r in by_barangay if r["children"]]
    if real:
        # Ranked by how many children are short of target, not by percentage: a
        # small barangay can post the worst rate while a large one holds most of
        # the municipal gap.
        def shortfall(r):
            return max(0, COVERAGE_TARGET - r["coverage"]) / 100 * r["children"]
        worst = max(real, key=shortfall)
        lowest = min(real, key=lambda r: r["coverage"])
        behind = sum(1 for r in real if r["coverage"] < COVERAGE_TARGET)
        out["barangay"] = (
            f"Across the {len(real)} barangays, "
            f"{'all' if behind == len(real) else behind} are below the "
            f"{COVERAGE_TARGET}% target. {lowest['barangay']} has the lowest rate at "
            f"{lowest['coverage']}%, but {worst['barangay']} carries the largest gap "
            f" about {round(shortfall(worst))} of its {worst['children']} children "
            f"short of target."
        )

    if len(trend) >= 2:
        first_y, first_p, _ = trend[0]
        last_y, last_p, last_n = trend[-1]
        prev_p = trend[-2][1]
        step = round(last_p - prev_p, 1)
        direction = "rose" if step > 0 else "fell" if step < 0 else "held steady"
        change = "up" if last_p >= first_p else "down"
        out["trend"] = (
            f"Coverage is {change} from {first_p}% in {first_y} to {last_p}% now "
            f"across {last_n} children, and {direction} "
            f"{abs(step)} points since {trend[-2][0]}."
        )
    return out


def coverage_color(pct, children=None):
    """Colour for a coverage percentage, per COVERAGE_BANDS.

    A barangay with no records is grey, not red: 0% coverage and "no data yet"
    are different statements, and colouring them alike would put a barangay the
    RHU has not yet exported beside one with a genuine coverage problem."""
    if children == 0:
        return "none"
    for floor, colour in COVERAGE_BANDS:
        if pct >= floor:
            return colour
    return "red"


def order_by_schedule(rows):
    """Doses in EPI schedule order, so same-antigen bars sit together."""
    pos = {name: i for i, (_c, name, _a, _d, _r) in enumerate(dp.VACCINE_SCHEDULE)}
    return sorted(rows, key=lambda r: pos.get(r["vaccine"], 99))


def vaccine_colours(rows):
    """Bar colour per dose, grouped by antigen.

    Doses of the same vaccine share a colour, which makes each series visible as
    a group - DPT holding steady across three doses reads differently from OPV
    declining across the same three visits."""
    antigen_of = {name: ant for _c, name, ant, _d, _r in dp.VACCINE_SCHEDULE}
    return [ANTIGEN_COLOURS.get(antigen_of.get(r["vaccine"], ""), "#64748b") for r in rows]


def coverage_change(barangay_id=None):
    """Real month-on-month change in coverage.

    Replaces a hardcoded "+3.2%" badge. Compares coverage as at the end of last
    month with the current figure, using the same point-in-time computation as
    the Reports page, so the two always agree. Returns None when there is no
    prior month to compare against."""
    today = date.today()
    last_month_end = date(today.year, today.month, 1) - timedelta(days=1)
    prev = report_figures(barangay_id, last_month_end)
    if not prev["total_children"]:
        return None
    now = report_figures(barangay_id, today)
    return {
        "delta": round(now["coverage_rate"] - prev["coverage_rate"], 1),
        "since": last_month_end.strftime("%B %Y"),
    }


def stock_levels():
    """Stock position per antigen. Extracted from the inventory route so the
    dashboard panel and the Vaccine Inventory page apply identical thresholds
    and can never disagree about what counts as Low or Critical."""
    rows = []
    for a in VaccineAntigen.query.all():
        batches = InventoryBatch.query.filter_by(antigen_id=a.id, is_archived=False).all()
        total = sum(b.quantity_on_hand for b in batches)
        reorder = min([b.reorder_level for b in batches], default=50)
        expiring = sum(1 for b in batches if b.is_expiring_soon)
        if total <= 0:
            status, color = "Out of Stock", "red"
        elif total <= reorder * 0.5:
            status, color = "Critical", "red"
        elif total <= reorder:
            status, color = "Low", "amber"
        else:
            status, color = "Enough", "green"
        rows.append({"antigen": a, "total": total, "reorder": reorder, "status": status,
                     "color": color, "expiring": expiring, "batches": batches,
                     "pct": min(100, round((total / (reorder * 2)) * 100)) if reorder else 100})
    return rows


def stock_alerts(limit=5):
    """Antigens at or below their reorder level, lowest first.

    The Critical Alerts strip only fires at exactly zero, so a vaccine down to
    its last few vials was invisible until it had already run out."""
    rows = [r for r in stock_levels() if r["status"] != "Enough"]
    rows.sort(key=lambda r: (r["total"], r["antigen"].name))
    return rows[:limit]


# A year needs at least this many registered children before its coverage rate
# is meaningful enough to plot.
TREND_MIN_COHORT = 20


def yearly_trend(barangay_id=None):
    """Coverage at the end of each year, computed from actual dose dates.

    Unlike monthly_trend() this is real, not simulated: coverage as at 31
    December of each year the data covers, using the same point-in-time
    calculation as the Reports page. The last point is capped at today."""
    today = date.today()
    out = []
    for y in sorted(data_years()):
        end = min(date(y, 12, 31), today)
        fig = report_figures(barangay_id, end)
        # A year with only a handful of children registered yields a rate that
        # is noise, not a finding: 2019 held one child, so its 0% made the
        # series look like a rise from nothing. Those years are omitted rather
        # than plotted.
        if fig["total_children"] >= TREND_MIN_COHORT:
            label = str(y) + (" (to date)" if end == today and y == today.year else "")
            out.append((label, fig["coverage_rate"], fig["total_children"]))
    return out


def monthly_trend(barangay_id=None, current_rate=None):
    """SIMULATED 12-month trend.

    The database holds a current snapshot, not month-by-month history, so this
    cannot be measured - it is generated from a fixed seed to give the chart a
    plausible shape. Templates label it as simulated; do not present it as
    recorded data. Once coverage is snapshotted monthly this can be replaced
    with a real series."""
    if current_rate is None:
        current_rate = dashboard_stats(barangay_id)["coverage_rate"]
    seed = int(hashlib.md5(f"trend-{barangay_id or 'all'}".encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    today = date.today()
    months = []
    for i in range(11, -1, -1):
        m = (today.month - i - 1) % 12 + 1
        y = today.year - ((today.month - i - 1) // 12 if today.month - i - 1 < 0 else 0)
        months.append(date(y, m, 1).strftime("%b"))
    start = max(50.0, current_rate - rng.uniform(10, 18))
    values = []
    for i in range(12):
        base = start + (current_rate - start) * (i / 11)
        values.append(round(max(0, min(100, base + rng.uniform(-2.5, 2.5))), 1))
    values[-1] = current_rate
    return list(zip(months, values))


def risk_distribution(barangay_id=None):
    counts = Counter()
    risk = _latest_risk_by_child()
    for c in _children_query(barangay_id).all():
        latest = risk.get(c.id)
        counts[dp.risk_tier(latest.risk_probability) if latest else "Low Risk"] += 1
    total = sum(counts.values()) or 1
    return {
        "high_pct": round(counts["High Risk"] / total * 100, 1), "high_count": counts["High Risk"],
        "medium_pct": round(counts["Medium Risk"] / total * 100, 1), "medium_count": counts["Medium Risk"],
        "low_pct": round(counts["Low Risk"] / total * 100, 1), "low_count": counts["Low Risk"],
    }


def donut_callouts(dist, center=75, ring_r=75, label_r=116, curve_bulge=14):
    """Leader-line geometry for the landing page's risk donut.

    A straight radial line only looks right for the topmost label (see the
    two rounds of overlap bugs this replaced) - everywhere else it needs an
    actual curve, which means real (x, y) points and a quadratic bezier
    control point, not a CSS rotate trick. One curve per segment, from the
    dot on the ring's own edge out to the pill label, bowed sideways so it
    reads as a drawn line rather than a straight spoke.
    """
    segments = [
        ("low", dist["low_pct"], "var(--green-light)"),
        ("medium", dist["medium_pct"], "var(--amber-light)"),
        ("high", dist["high_pct"], "var(--red-light)"),
    ]
    callouts = []
    cursor = 0.0
    for name, pct, colour in segments:
        mid_pct = cursor + pct / 2
        theta = math.radians(mid_pct / 100 * 360)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        dot = (center + ring_r * sin_t, center - ring_r * cos_t)
        anchor = (center + label_r * sin_t, center - label_r * cos_t)
        mid = ((dot[0] + anchor[0]) / 2, (dot[1] + anchor[1]) / 2)
        # Bulge the control point perpendicular to the dot->anchor line so
        # the curve bows to one side instead of running straight through it.
        dx, dy = anchor[0] - dot[0], anchor[1] - dot[1]
        length = math.hypot(dx, dy) or 1
        control = (mid[0] - dy / length * curve_bulge, mid[1] + dx / length * curve_bulge)
        callouts.append({
            "name": name, "pct": pct, "colour": colour,
            "dot": dot, "anchor": anchor, "control": control,
        })
        cursor += pct
    return callouts


# --- Dashboard filtering ----------------------------------------------------
# A BHW works a single list of children, so filtering by risk level and by
# immunization status is genuinely useful there. Both reuse the per-request
# caches, so filtering costs no extra queries.

RISK_LEVELS = [("", "Risk Level: All"), ("At-Risk", "At-Risk only"), ("Not At-Risk", "Not At-Risk only")]
STATUS_LEVELS = [("", "Status: All"), ("complete", "Fully immunized"), ("incomplete", "Incomplete")]


def immunized_as_of(child, as_of=None):
    """Fully immunized as at a date. Doses recorded after `as_of` count as not
    yet given. Shared with report_figures() so the dashboard and the Reports
    page never disagree about the same month."""
    if as_of is None:
        return _is_fully_immunized(child)
    vt_by_id = _vaccine_types_by_id()
    age = (as_of - child.date_of_birth).days
    due = given = 0
    for rec in _records_by_child().get(child.id, ()):
        vtype = vt_by_id.get(rec.vaccine_type_id)
        if vtype is not None and vtype.recommended_age_days <= age:
            due += 1
            if rec.date_administered is not None and rec.date_administered <= as_of:
                given += 1
    return True if due == 0 else (given / due >= 0.9)


def filtered_children(barangay_id=None, risk_level="", status="", as_of=None):
    risk = _latest_risk_by_child()
    kids = _children_query(barangay_id).all()
    if as_of is not None:
        # A child not yet born at the reporting date is not in scope.
        kids = [c for c in kids if c.date_of_birth <= as_of]
    if risk_level in ("At-Risk", "Not At-Risk"):
        kids = [c for c in kids if (ra := risk.get(c.id)) and ra.risk_label == risk_level]
    if status == "complete":
        kids = [c for c in kids if immunized_as_of(c, as_of)]
    elif status == "incomplete":
        kids = [c for c in kids if not immunized_as_of(c, as_of)]
    return kids


def stats_for(children, as_of=None):
    """Same shape as dashboard_stats(), over an explicit child list.

    Note: at_risk uses each child's latest risk assessment. Risk is not stored
    historically, so it is always current even when an earlier period is
    selected - the UI says so rather than implying a back-dated figure."""
    risk = _latest_risk_by_child()
    total = len(children)
    fully = sum(1 for c in children if immunized_as_of(c, as_of))
    at_risk = sum(1 for c in children
                  if (ra := risk.get(c.id)) and ra.risk_label == "At-Risk")
    return {"total_children": total, "fully_immunized": fully, "at_risk": at_risk,
            "coverage_rate": round(fully / total * 100, 1) if total else 0.0}


def risk_distribution_for(children):
    counts = Counter()
    risk = _latest_risk_by_child()
    for c in children:
        latest = risk.get(c.id)
        counts[dp.risk_tier(latest.risk_probability) if latest else "Low Risk"] += 1
    total = sum(counts.values()) or 1
    return {
        "high_pct": round(counts["High Risk"] / total * 100, 1), "high_count": counts["High Risk"],
        "medium_pct": round(counts["Medium Risk"] / total * 100, 1), "medium_count": counts["Medium Risk"],
        "low_pct": round(counts["Low Risk"] / total * 100, 1), "low_count": counts["Low Risk"],
    }


def top_risk_factors(barangay_id=None, limit=6):
    tally = Counter()
    risk = _latest_risk_by_child()
    for c in _children_query(barangay_id).all():
        latest = risk.get(c.id)
        if latest and latest.risk_label == "At-Risk":
            factors = dp.deserialize_factors(latest.top_factors)
            if factors:
                tally[factors[0]["factor"]] += 1
    ranked = tally.most_common(limit)
    if not ranked:
        return []
    max_v = ranked[0][1]
    return [{"label": label, "count": count, "pct": round(count / max_v * 100)} for label, count in ranked]


def at_risk_table(barangay_id=None, limit=None):
    rows = []
    risk = _latest_risk_by_child()
    records = _records_by_child()
    for c in _children_query(barangay_id).all():
        latest = risk.get(c.id)
        if not latest or latest.risk_label != "At-Risk":
            continue
        due, missed, missed_codes = _due_and_missed(c, with_codes=True)
        given_dates = [r.date_administered for r in records.get(c.id, ()) if r.date_administered]
        last_visit = max(given_dates) if given_dates else None
        tier = dp.risk_tier(latest.risk_probability)
        rows.append({"child": c, "risk_score": round(latest.risk_probability * 100), "tier": tier,
                     "color": dp.risk_tier_color(tier), "missed": missed, "due": due, "last_visit": last_visit,
                     # What the missed doses would have protected against, so the
                     # consequence is visible rather than only the dose count.
                     "unprotected": dp.diseases_for_missed(missed_codes),
                     "recommendation": "Home Visit" if tier == "High Risk" else "Schedule Visit"})
    rows.sort(key=lambda r: r["risk_score"], reverse=True)
    return rows[:limit] if limit else rows


def child_detail(child):
    """Everything the read-only child record page shows.

    The only way to inspect one child was the edit form - ~24 date inputs that
    had to be decoded by eye, and which risked altering a clinical record just
    to look at it. This assembles the same data for reading."""
    vt_by_code = {vt.code: vt for vt in _vaccine_types_by_id().values()}
    given = _doses_given(child.id)
    age_days = child.age_in_days
    today = date.today()

    schedule = []
    for code, name, antigen_code, dose_no, rec_days in dp.VACCINE_SCHEDULE:
        vt = vt_by_code.get(code)
        if not vt:
            continue
        when = given.get(vt.id)
        due_on = child.date_of_birth + timedelta(days=rec_days)
        if when:
            status, note = "given", when.strftime("%b %d, %Y")
        elif age_days < rec_days:
            status, note = "upcoming", f"due {due_on.strftime('%b %d, %Y')}"
        elif (today - due_on).days > dp.GRACE_PERIOD_DAYS:
            status, note = "overdue", f"{(today - due_on).days} days overdue"
        else:
            status, note = "due", f"due {due_on.strftime('%b %d, %Y')}"
        schedule.append({"code": code, "name": name, "status": status, "note": note,
                         "date": when, "due_on": due_on})

    latest = _latest_risk(child)
    factors = []
    if latest and latest.top_factors:
        try:
            factors = json.loads(latest.top_factors)
        except (ValueError, TypeError):
            factors = []

    given_dates = [d for d in given.values() if d]
    due, missed = _due_and_missed(child)
    history = (ActivityLog.query
               .filter(ActivityLog.action.contains(child.display_name))
               .order_by(ActivityLog.created_at.desc()).limit(6).all())

    return {
        "schedule": schedule,
        "given_count": sum(1 for r in schedule if r["status"] == "given"),
        "overdue_count": sum(1 for r in schedule if r["status"] == "overdue"),
        "total_doses": len(schedule),
        "due": due, "missed": missed,
        "risk": latest, "factors": factors,
        "risk_tier": dp.risk_tier(latest.risk_probability) if latest else None,
        "last_visit": max(given_dates) if given_dates else None,
        "fully_immunized": _is_fully_immunized(child),
        "history": history,
    }


_MAP_COORDS = {}


# ---------------------------------------------------------------------------
# Report generation: figures recomputed for a chosen barangay and reporting
# period. Kept separate from the dashboard analytics above so the live
# dashboards keep their existing "as of today, all barangays" behaviour.
# ---------------------------------------------------------------------------

def month_options(count=12):
    """[(value, label)] for the Period selector, most recent month first."""
    out = []
    today = date.today()
    y, m = today.year, today.month
    for _ in range(count):
        out.append((f"{y:04d}-{m:02d}", date(y, m, 1).strftime("%B %Y")))
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return out


MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


def data_years():
    """Years the demo data actually covers, newest first. Offering years with
    no data would produce empty dashboards that look like a bug."""
    earliest = db.session.query(func.min(Child.date_of_birth)).scalar()
    first = earliest.year if earliest else date.today().year
    return list(range(date.today().year, first - 1, -1))


def resolve_period(month, year):
    """Turn separate month/year selections into a reporting date.

    Both blank        -> no period filter (live figures).
    Year only         -> as at 31 December of that year.
    Month only        -> that month in the current year.
    Either in future  -> clamped to today, so a forward-dated selection can
                         never show an empty dashboard.
    Returns (as_of|None, label|None, clamped:bool).
    """
    today = date.today()
    try:
        m = int(month) if month else 0
        y = int(year) if year else 0
    except (TypeError, ValueError):
        return None, None, False
    if not (1 <= m <= 12):
        m = 0
    if y and y not in data_years():
        y = 0
    if not m and not y:
        return None, None, False

    if m and y:
        end = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        as_of, label = end - timedelta(days=1), f"{MONTH_NAMES[m - 1]} {y}"
    elif y:
        as_of, label = date(y, 12, 31), f"{y}"
    else:
        y = today.year
        end = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        as_of, label = end - timedelta(days=1), f"{MONTH_NAMES[m - 1]} {y}"

    if as_of > today:
        return today, f"{label} (to date)", True
    return as_of, label, False


def period_end(period):
    """Last day of a 'YYYY-MM' period, never later than today."""
    today = date.today()
    if not period:
        return today
    try:
        y, m = (int(x) for x in period.split("-"))
        nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        return min(nxt - timedelta(days=1), today)
    except (ValueError, TypeError):
        return today


def report_figures(barangay_id=None, as_of=None):
    """Coverage figures for one barangay (or all) as of a date.

    Doses recorded after `as_of` count as not-yet-given, and children born
    after it are excluded, so selecting an earlier period reports the
    situation as it stood then rather than today's totals."""
    as_of = as_of or date.today()
    vt_by_id = _vaccine_types_by_id()
    records = _records_by_child()
    risk = _latest_risk_by_child()
    names = {b.id: b.name for b in Barangay.query.all()}

    children = [c for c in _children_query(barangay_id).all() if c.date_of_birth <= as_of]

    def immunized(child):
        age = (as_of - child.date_of_birth).days
        due = given = 0
        for rec in records.get(child.id, ()):
            vtype = vt_by_id.get(rec.vaccine_type_id)
            if vtype is not None and vtype.recommended_age_days <= age:
                due += 1
                if rec.date_administered is not None and rec.date_administered <= as_of:
                    given += 1
        return True if due == 0 else (given / due >= 0.9)

    grouped = defaultdict(list)
    for c in children:
        grouped[c.barangay_id].append(c)

    rows = []
    for bid, kids in grouped.items():
        fully = sum(1 for c in kids if immunized(c))
        at_risk = sum(1 for c in kids if (ra := risk.get(c.id)) and ra.risk_label == "At-Risk")
        rows.append({"barangay": names.get(bid, "—"), "children": len(kids),
                     "fully_immunized": fully, "at_risk": at_risk,
                     "coverage": round(fully / len(kids) * 100, 1) if kids else 0.0})
    rows.sort(key=lambda r: r["coverage"], reverse=True)

    total = len(children)
    fully_total = sum(r["fully_immunized"] for r in rows)
    at_risk_total = sum(r["at_risk"] for r in rows)
    return {
        "total_children": total,
        "fully_immunized": fully_total,
        "at_risk": at_risk_total,
        "coverage_rate": round(fully_total / total * 100, 1) if total else 0.0,
        "by_barangay": rows,
        "as_of": as_of,
    }


def report_by_vaccine(barangay_id=None, as_of=None):
    """Completion rate per scheduled dose as of a date, for the BHW report
    chart. Mirrors report_figures(): doses recorded after `as_of` count as
    not yet given, and children born after it are excluded."""
    as_of = as_of or date.today()
    vt_by_code = {vt.code: vt for vt in _vaccine_types_by_id().values()}
    given_on = {}
    for child_id, records in _records_by_child().items():
        for record in records:
            given_on[(child_id, record.vaccine_type_id)] = record.date_administered

    children = [c for c in _children_query(barangay_id).all() if c.date_of_birth <= as_of]
    rows = []
    for code, name, antigen_code, dose_no, rec_days in dp.VACCINE_SCHEDULE:
        vt = vt_by_code.get(code)
        if not vt:
            continue
        eligible = [c for c in children if (as_of - c.date_of_birth).days >= rec_days]
        if not eligible:
            rows.append({"vaccine": name, "pct": 0})
            continue
        given = sum(1 for c in eligible
                    if (d := given_on.get((c.id, vt.id))) is not None and d <= as_of)
        rows.append({"vaccine": name, "pct": round(given / len(eligible) * 100, 1)})
    return rows


MAP_GEOMETRY_PATH = os.path.join(BASE_DIR, "static", "data", "san_jacinto_map.json")
_MAP_GEOMETRY = None


def map_geometry():
    """Real barangay boundaries for San Jacinto, Pangasinan.

    Source: PSA PSGC 2023 administrative shapefiles, converted to SVG paths in
    a 1000x1000 viewBox. Loaded from disk once and cached, so the map renders
    with no internet connection and no mapping library."""
    global _MAP_GEOMETRY
    if _MAP_GEOMETRY is None:
        with open(MAP_GEOMETRY_PATH, encoding="utf-8") as fh:
            _MAP_GEOMETRY = json.load(fh)
    return _MAP_GEOMETRY


def _seed_coords():
    """Label anchor per barangay, taken from the real polygon centroids."""
    if not _MAP_COORDS:
        for b in map_geometry()["barangays"]:
            _MAP_COORDS[b["name"]] = (b["cx"] / 1000.0, b["cy"] / 1000.0)
    return _MAP_COORDS


def map_markers(mode="heatmap"):
    coords = _seed_coords()
    shapes = {b["name"]: b for b in map_geometry()["barangays"]}
    cov = {r["barangay"]: r for r in coverage_by_barangay()}
    sim = simulated_barangays()
    markers = []
    for name in dp.BARANGAYS:
        r = cov.get(name, {"children": 0, "at_risk": 0, "coverage": 0})
        x, y = coords[name]
        if mode == "heatmap":
            color = coverage_color(r["coverage"], r["children"])
            if r["children"]:
                value, suffix = r["coverage"], "%"
            else:
                value, suffix = "no data", ""
        elif mode == "at_risk":
            value, suffix = r["at_risk"], ""
            share = (r["at_risk"] / r["children"]) if r["children"] else 0
            color = "red" if share >= 0.30 else "amber" if share >= 0.15 else "green"
        else:
            value, suffix, color = r["children"], "", "blue"
        shape = shapes.get(name, {})
        markers.append({
            "barangay": name, "x": x, "y": y, "value": value, "suffix": suffix, "color": color,
            "simulated": name in sim,
            "d": shape.get("d", ""), "cx": shape.get("cx", 0), "cy": shape.get("cy", 0),
            "area": shape.get("area", 0), "children": r["children"],
            "at_risk": r["at_risk"], "coverage": r["coverage"],
        })
    return markers


# ---------------------------------------------------------------------------
# Notifications: generated live from current DB state (Stock/Expiry/Risk/
# Request/System categories).
# ---------------------------------------------------------------------------

def get_notifications(role, barangay_id=None):
    notes = []
    for antigen in VaccineAntigen.query.all():
        total = (db.session.query(func.coalesce(func.sum(InventoryBatch.quantity_on_hand), 0))
                 .filter(InventoryBatch.antigen_id == antigen.id, InventoryBatch.is_archived.is_(False)).scalar())
        reorder = (db.session.query(func.min(InventoryBatch.reorder_level))
                   .filter(InventoryBatch.antigen_id == antigen.id, InventoryBatch.is_archived.is_(False)).scalar()) or 50
        if total <= 0:
            notes.append({"category": "Stock", "severity": "critical", "icon": "package", "icon_bg": "bg-red",
                          "title": f"{antigen.name} Out of Stock",
                          "body": f"{antigen.name} is completely depleted. Submit a restock request immediately.",
                          "time_label": "Today", "unread": True})
        elif total <= reorder:
            notes.append({"category": "Stock", "severity": "warning", "icon": "package", "icon_bg": "bg-orange",
                          "title": f"{antigen.name} Critically Low",
                          "body": f"Only {total} vials remain. Minimum required is {reorder}.",
                          "time_label": "Today", "unread": True})

    if role == "rhu":
        for batch in InventoryBatch.query.filter(InventoryBatch.is_archived.is_(False)).all():
            if batch.is_expiring_soon:
                days = (batch.expiry_date - date.today()).days
                notes.append({"category": "Expiry", "severity": "warning", "icon": "hourglass", "icon_bg": "bg-amber",
                              "title": f"{batch.antigen.name} Expiring Soon",
                              "body": f"{batch.quantity_on_hand} vials (batch {batch.batch_number}) expire in {days} days.",
                              "time_label": "Today", "unread": True})

    # Runs on every page (the sidebar bell needs the unread count), so it reads
    # from the cached risk/record indexes rather than querying per child.
    at_risk_uncontacted = 0
    risk = _latest_risk_by_child()
    records = _records_by_child()
    today = date.today()
    for child in _children_query(barangay_id if role == "bhw" else None).all():
        latest = risk.get(child.id)
        if latest and latest.risk_label == "At-Risk":
            given = [r.date_administered for r in records.get(child.id, ()) if r.date_administered]
            days_since = (today - max(given)).days if given else 999
            if days_since > 30:
                at_risk_uncontacted += 1
    if at_risk_uncontacted:
        scope = "your barangay" if barangay_id else "the municipality"
        notes.append({"category": "Risk", "severity": "critical", "icon": "alert-octagon", "icon_bg": "bg-red",
                      "title": "High-Risk Children Not Contacted",
                      "body": f"{at_risk_uncontacted} patients in {scope} have not been contacted in over 30 days.",
                      "time_label": "Today", "unread": True})

    if role == "rhu":
        pending = VaccineRequest.query.filter_by(status="pending").count()
        if pending:
            notes.append({"category": "Request", "severity": "info", "icon": "clipboard", "icon_bg": "bg-blue",
                          "title": f"{pending} Vaccine Request{'s' if pending != 1 else ''} Pending",
                          "body": f"{pending} barangay request(s) are awaiting your review.", "time_label": "Today",
                          "unread": True})
    elif role == "bhw":
        recent = (VaccineRequest.query.filter_by(barangay_id=barangay_id, status="approved")
                  .order_by(VaccineRequest.reviewed_at.desc()).first())
        if recent:
            notes.append({"category": "Request", "severity": "info", "icon": "clipboard", "icon_bg": "bg-blue",
                          "title": "Vaccine Request Approved",
                          "body": f"{recent.request_code} ({recent.antigen.name} - {recent.quantity_requested} vials) approved.",
                          "time_label": "Recently", "unread": False})

    notes.append({"category": "System", "severity": "info", "icon": "calendar", "icon_bg": "bg-blue",
                  "title": "Monthly Report Due",
                  "body": f"The {date.today().strftime('%B %Y')} monthly immunization report is due at month end.",
                  "time_label": "This week", "unread": False})
    return notes


def admin_notifications():
    """Account and system-governance alerts for the administrator.

    Deliberately not vaccine stock or clinical risk - those belong to RHU and
    BHW users who can act on them. The administrator's job is accounts, access
    and coverage of staffing, so that is what is surfaced here."""
    notes = []
    today = date.today()
    users = User.query.all()

    disabled = [u for u in users if not u.is_active_flag]
    if disabled:
        notes.append({"category": "Accounts", "severity": "warning", "icon": "x-circle", "icon_bg": "bg-orange",
                      "title": f"{len(disabled)} Disabled Account{'s' if len(disabled) > 1 else ''}",
                      "body": "Disabled accounts cannot sign in: "
                              + ", ".join(u.full_name for u in disabled[:4])
                              + (f" and {len(disabled) - 4} more." if len(disabled) > 4 else "."),
                      "time_label": "Ongoing", "unread": True,
                      "url": url_for("admin.users")})

    unassigned = [u for u in users if u.role == "bhw" and not u.barangay_id]
    if unassigned:
        notes.append({"category": "Accounts", "severity": "critical", "icon": "map-pin", "icon_bg": "bg-red",
                      "title": f"{len(unassigned)} BHW Without a Barangay",
                      "body": "These health workers cannot see any child records until they are assigned: "
                              + ", ".join(u.full_name for u in unassigned[:4]) + ".",
                      "time_label": "Action needed", "unread": True,
                      "url": url_for("admin.assign_barangay")})

    covered = {u.barangay_id for u in users if u.role == "bhw" and u.barangay_id}
    uncovered = [b for b in Barangay.query.order_by(Barangay.name).all() if b.id not in covered]
    if uncovered:
        notes.append({"category": "Coverage", "severity": "warning", "icon": "users", "icon_bg": "bg-orange",
                      "title": f"{len(uncovered)} Barangay Without a BHW",
                      "body": "No health worker is assigned to: "
                              + ", ".join(b.name for b in uncovered[:5])
                              + (f" and {len(uncovered) - 5} more." if len(uncovered) > 5 else "."),
                      "time_label": "Ongoing", "unread": True,
                      "url": url_for("admin.assign_barangay")})

    never = [u for u in users if u.is_active_flag and not u.last_login_at]
    if never:
        notes.append({"category": "Accounts", "severity": "info", "icon": "log-in", "icon_bg": "bg-blue",
                      "title": f"{len(never)} Account{'s' if len(never) > 1 else ''} Never Signed In",
                      "body": "Created but never used - onboarding may be incomplete: "
                              + ", ".join(u.full_name for u in never[:4])
                              + (f" and {len(never) - 4} more." if len(never) > 4 else "."),
                      "time_label": "Ongoing", "unread": True,
                      "url": url_for("admin.users")})

    stale = [u for u in users if u.is_active_flag and u.last_login_at
             and (today - u.last_login_at.date()).days > 30]
    if stale:
        notes.append({"category": "Accounts", "severity": "info", "icon": "clock", "icon_bg": "bg-purple",
                      "title": f"{len(stale)} Account{'s' if len(stale) > 1 else ''} Inactive 30+ Days",
                      "body": "No sign-in for over a month: "
                              + ", ".join(u.full_name for u in stale[:4])
                              + (f" and {len(stale) - 4} more." if len(stale) > 4 else "."),
                      "time_label": "Ongoing", "unread": False,
                      "url": url_for("admin.users")})

    return notes


def unread_count(notes):
    return sum(1 for n in notes if n.get("unread"))


# ---------------------------------------------------------------------------
# Shared child create/predict helper
# ---------------------------------------------------------------------------

def _create_child(form, barangay_locked):
    barangay_id = barangay_locked or int(form.get("barangay_id"))
    child = Child(
        full_name=form.get("full_name"), sex=form.get("sex"),
        date_of_birth=datetime.strptime(form.get("date_of_birth"), "%Y-%m-%d").date(),
        barangay_id=barangay_id, address=form.get("address"), guardian_name=form.get("guardian_name"),
        guardian_contact=form.get("guardian_contact"),
        date_registered=(datetime.strptime(form.get("date_registered"), "%Y-%m-%d").date()
                         if form.get("date_registered") else date.today()),
        vitamin_a_date=(datetime.strptime(form.get("vitamin_a_date"), "%Y-%m-%d").date()
                        if form.get("vitamin_a_date") else None),
        mnp_given=bool(form.get("mnp_given")),
        source="manual", created_by_id=current_user.id,
    )
    db.session.add(child)
    db.session.flush()

    doses_dict = {}
    for code, name, antigen_code, dose_no, rec_days in dp.VACCINE_SCHEDULE:
        vt = VaccineType.query.filter_by(code=code).first()
        date_str = form.get(f"dose_{code}")
        administered = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
        doses_dict[code] = administered
        db.session.add(VaccinationRecord(
            child_id=child.id, vaccine_type_id=vt.id, date_administered=administered,
            status="completed" if administered else "pending", administered_by_id=current_user.id,
        ))
    db.session.commit()

    barangay_name = db.session.get(Barangay, barangay_id).name
    result = dp.predict_for_child(child.date_of_birth, child.date_registered, child.sex, doses_dict, barangay_name)
    db.session.add(RiskAssessment(
        child_id=child.id, risk_label=result["label"], risk_probability=result["probability"],
        model_version=result["model_version"], top_factors=dp.serialize_factors(result["top_factors"]),
    ))
    db.session.commit()
    log_activity(f"Registered child {child_label(child)}")
    return child


def _update_child(child, form):
    """Updates an existing child's basic info and any dose dates submitted
    on the Edit Child form, then recomputes their risk assessment so
    newly-recorded doses are reflected immediately."""
    # An empty string is a deliberate clear, not a missing field.
    if "full_name" in form:
        child.full_name = form.get("full_name", "").strip()
    child.sex = form.get("sex") or child.sex
    if form.get("date_of_birth"):
        child.date_of_birth = datetime.strptime(form.get("date_of_birth"), "%Y-%m-%d").date()
    if form.get("date_registered"):
        child.date_registered = datetime.strptime(form.get("date_registered"), "%Y-%m-%d").date()
    child.guardian_name = form.get("guardian_name")
    child.guardian_contact = form.get("guardian_contact")
    child.address = form.get("address")
    child.vitamin_a_date = (datetime.strptime(form.get("vitamin_a_date"), "%Y-%m-%d").date()
                            if form.get("vitamin_a_date") else None)
    child.mnp_given = bool(form.get("mnp_given"))

    vt_by_code = {vt.code: vt for vt in VaccineType.query.all()}
    existing_by_vt_id = {rec.vaccine_type_id: rec for rec in child.vaccination_records}
    for code in vt_by_code:
        date_str = form.get(f"dose_{code}")
        administered = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
        vt = vt_by_code[code]
        rec = existing_by_vt_id.get(vt.id)
        if rec is None:
            rec = VaccinationRecord(child_id=child.id, vaccine_type_id=vt.id)
            db.session.add(rec)
        rec.date_administered = administered
        rec.status = "completed" if administered else "pending"
    db.session.commit()

    result = _recompute_risk(child)
    log_activity(f"Updated child record {child_label(child)}")
    return result


def _recompute_risk(child):
    vt_by_id = {vt.id: vt for vt in VaccineType.query.all()}
    doses_dict = {vt.code: None for vt in vt_by_id.values()}
    for rec in child.vaccination_records:
        vt = vt_by_id.get(rec.vaccine_type_id)
        if vt:
            doses_dict[vt.code] = rec.date_administered
    result = dp.predict_for_child(child.date_of_birth, child.date_registered, child.sex, doses_dict, child.barangay.name)
    db.session.add(RiskAssessment(
        child_id=child.id, risk_label=result["label"], risk_probability=result["probability"],
        model_version=result["model_version"], top_factors=dp.serialize_factors(result["top_factors"]),
    ))
    db.session.commit()
    return result


def _next_dose_for(child, vt_lookup):
    """Returns the next not-yet-administered dose in the schedule for one
    child (or None if every dose has been given), used by the
    Continuation Predictor list to show what's coming up next."""
    given = _doses_given(child.id)
    for code, name, antigen, dose_no, rec_days in dp.VACCINE_SCHEDULE:
        if given.get(vt_lookup[code].id) is None:
            due_date = child.date_of_birth + timedelta(days=rec_days)
            return {"name": name, "due_date": due_date, "overdue": due_date < date.today()}
    return None


# ---------------------------------------------------------------------------
# Auth blueprint
# ---------------------------------------------------------------------------

auth_bp = Blueprint("auth", __name__)
ROLE_HOME = {"admin": "admin.dashboard", "rhu": "rhu.dashboard", "bhw": "bhw.dashboard"}


@auth_bp.route("/")
def index():
    """Public landing page. Signed-in users go straight to their dashboard."""
    if current_user.is_authenticated:
        return redirect(url_for(ROLE_HOME[current_user.role]))

    # The hero plots the real completion rate for each dose in schedule order.
    # It is the argument for the system: coverage holds through the first year
    # and then falls away at the nine-month doses.
    order = {name: i for i, (_c, name, _a, _d, _r) in enumerate(dp.VACCINE_SCHEDULE)}
    rows = sorted(coverage_by_vaccine(), key=lambda r: order.get(r["vaccine"], 99))
    left, right, top, bottom = 60.0, 985.0, 30.0, 290.0
    step = (right - left) / max(len(rows) - 1, 1)
    curve = []
    for i, r in enumerate(rows):
        curve.append({
            "name": r["vaccine"], "pct": r["pct"],
            "x": round(left + i * step, 1),
            # 100% sits at y=30, 0% at y=290.
            "y": round(top + (100 - r["pct"]) / 100 * (bottom - top), 1),
        })
    dip = min(curve, key=lambda c: c["pct"]) if curve else None

    stats = dashboard_stats()
    return render_template(
        "landing.html",
        curve=curve,
        curve_points=" ".join(f"{c['x']},{c['y']}" for c in curve),
        dip=dip, dip_pct=dip["pct"] if dip else 0, dip_name=dip["name"] if dip else "",
        by_vaccine=rows, vaccine_colours=vaccine_colours(rows), vaccine_labels=schedule_labels(rows),
        # The same polygons and colours the Municipality Map uses.
        map_shapes=map_markers("heatmap"),
        # Lowest coverage first, so the barangays needing attention lead.
        by_barangay=sorted(coverage_by_barangay(), key=lambda r: r["coverage"]),
        trend=yearly_trend(),
        dist=(_dist := risk_distribution()),
        donut_callouts=donut_callouts(_dist),
        stats=stats,
        total_children=stats["total_children"], at_risk=stats["at_risk"],
        fully_immunized=stats["fully_immunized"], coverage_rate=stats["coverage_rate"],
        total_doses=len(dp.VACCINE_SCHEDULE), barangay_count=Barangay.query.count(),
        # Built without %-d: that is a glibc extension and raises
        # ValueError on Windows.
        as_of=f"{date.today():%B} {date.today().day}, {date.today():%Y}",
    )


@auth_bp.route("/coverage")
def public_coverage():
    """Public coverage dashboard. No login required.

    Separate from the landing page on purpose: a homepage has to introduce and
    persuade, a dashboard has to be scannable and current. Every real public
    health dashboard reviewed follows this split (CDC's COVID Data Tracker,
    COVIDVaxView) rather than embedding the data in a marketing page."""
    order = {name: i for i, (_c, name, _a, _d, _r) in enumerate(dp.VACCINE_SCHEDULE)}
    rows = sorted(coverage_by_vaccine(), key=lambda r: order.get(r["vaccine"], 99))
    stats = dashboard_stats()
    with_records = Barangay.query.join(Child, Child.barangay_id == Barangay.id).distinct().count()

    return render_template(
        "public_coverage.html",
        by_vaccine=rows, vaccine_colours=vaccine_colours(rows), vaccine_labels=schedule_labels(rows),
        map_shapes=map_markers("heatmap"),
        by_barangay=sorted(coverage_by_barangay(), key=lambda r: r["coverage"]),
        trend=yearly_trend(),
        stats=stats,
        total_children=stats["total_children"], at_risk=stats["at_risk"],
        fully_immunized=stats["fully_immunized"], coverage_rate=stats["coverage_rate"],
        barangay_count=Barangay.query.count(), barangays_with_records=with_records,
        as_of=f"{date.today():%B} {date.today().day}, {date.today():%Y}",
    )


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for(ROLE_HOME[current_user.role]))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        # The account's own role decides where it lands — the user never picks it.
        if user and user.check_password(password):
            if not user.is_active_flag:
                flash("This account has been disabled. Contact your System Administrator.", "error")
                return render_template("login.html")
            login_user(user)
            user.last_login_at = datetime.utcnow()
            db.session.commit()
            log_activity("Logged in")
            return redirect(request.args.get("next") or url_for(ROLE_HOME[user.role]))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    log_activity("Logged out")
    logout_user()
    return redirect(url_for("auth.login"))


# ---------------------------------------------------------------------------
# RHU blueprint
# ---------------------------------------------------------------------------

rhu_bp = Blueprint("rhu", __name__, url_prefix="/rhu")


def _rhu_ctx(active, **extra):
    notes = get_notifications("rhu")
    return dict(role="rhu", nav_items=build_nav(RHU_NAV), active=active, current_user=current_user,
                settings_url=url_for("rhu.settings"),
                unread_count=unread_count(notes), notif_url=url_for("rhu.notifications"), **extra)


@rhu_bp.before_request
@login_required
@role_required("rhu")
def _rhu_guard():
    pass


@rhu_bp.route("/dashboard")
def dashboard():
    """RHU covers 19 barangays, so the dashboard can be narrowed to one.
    The analytics already take a barangay_id; nothing else changes."""
    barangay_id = request.args.get("barangay_id", type=int)
    selected = db.session.get(Barangay, barangay_id) if barangay_id else None
    if not selected:
        barangay_id = None

    # Validate against the offered months: an unrecognised value must mean
    # "no period filter", not a silent "as of today".
    month = request.args.get("month", "")
    year = request.args.get("year", "")
    as_of, period_label, clamped = resolve_period(month, year)
    kids = filtered_children(barangay_id, as_of=as_of)
    stats = stats_for(kids, as_of)
    # Stock is municipality-wide, so alerts are not narrowed by barangay.
    alerts = [f"{a.name} is completely out of stock" for a in VaccineAntigen.query.all()
              if sum(b.quantity_on_hand for b in InventoryBatch.query.filter_by(antigen_id=a.id, is_archived=False)) <= 0]
    dist = risk_distribution_for(kids)
    # Two actionable panels: who needs a visit, and what stock is running out.
    # Both summarise their dedicated pages rather than reproducing them.
    follow_up = at_risk_table(barangay_id, limit=5)
    low_stock = stock_alerts(limit=5)
    # Approving requests is an RHU responsibility; nothing surfaced them before.
    # Requests belong to a barangay, so they follow the scope filter. (Stock does
    # not: it is held centrally at the RHU, and the panel says so.)
    pending_q = VaccineRequest.query.filter_by(status="pending")
    if barangay_id:
        pending_q = pending_q.filter(VaccineRequest.barangay_id == barangay_id)
    pending = pending_q.order_by(VaccineRequest.requested_at.asc()).limit(5).all()

    # Coverage Analytics merged into this page; its own route was retired.
    # Lowest coverage first: the barangays needing attention lead, matching the
    # admin dashboard's table. Follows the scope filter like every other panel.
    by_barangay = sorted(coverage_by_barangay(), key=lambda r: r["coverage"])
    if selected:
        by_barangay = [r for r in by_barangay if r["barangay"] == selected.name]
    by_vaccine = order_by_schedule(report_by_vaccine(barangay_id, as_of) if as_of else coverage_by_vaccine(barangay_id))
    trend = yearly_trend(barangay_id)
    below_target = len([r for r in by_barangay if r["coverage"] < COVERAGE_TARGET])
    change = coverage_change(barangay_id)
    return render_template("rhu_dashboard.html", **_rhu_ctx(
        "dashboard", page_title="Dashboard", stats=stats, inventory_alerts=alerts,
        at_risk_uncontacted=dist["high_count"], risk_dist=dist, now=datetime.now(),
        barangays=Barangay.query.order_by(Barangay.name).all(),
        selected_barangay=selected, selected_barangay_id=barangay_id,
        months=MONTH_NAMES, years=data_years(), month=month, year=year,
        as_of=as_of, period_label=period_label, period_clamped=clamped,
        follow_up=follow_up, low_stock=low_stock, pending_requests=pending,
        by_barangay=by_barangay, by_vaccine=by_vaccine, trend=trend,
        vaccine_colours=vaccine_colours(by_vaccine), vaccine_labels=schedule_labels(by_vaccine),
        insights=chart_insights(by_vaccine, by_barangay, trend),
        below_target=below_target, change=change,
    ))


@rhu_bp.route("/children", methods=["GET", "POST"])
def children():
    if request.method == "POST":
        _create_child(request.form, barangay_locked=None)
        flash("Child registered successfully.", "success")
        return redirect(url_for("rhu.children"))
    barangay_id = request.args.get("barangay_id", type=int)
    risk_level = request.args.get("risk_level", "")
    search = request.args.get("q", "").strip()
    q = Child.query
    if barangay_id:
        q = q.filter(Child.barangay_id == barangay_id)
    if search:
        q = q.filter(Child.full_name.ilike(f"%{search}%"))
    kids = q.order_by(Child.barangay_id, Child.id).all()
    if risk_level:
        kids = [c for c in kids if (ra := _latest_risk(c)) and ra.risk_label == risk_level]
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    return render_template("rhu_children.html", **_rhu_ctx(
        "children", page_title="Child Records", children=kids,
        barangays=Barangay.query.order_by(Barangay.name).all(), vaccine_schedule=dp.VACCINE_SCHEDULE,
        vt_lookup=vt_lookup, antigens=VaccineAntigen.query.all(), selected_barangay=barangay_id,
        selected_risk=risk_level, search=search,
        dose_matrix=_dose_matrix(kids, vt_lookup), risk_labels=_risk_labels(kids),
    ))


@rhu_bp.route("/children/<int:child_id>/recompute-risk", methods=["POST"])
def recompute_risk(child_id):
    child = db.session.get(Child, child_id) or abort(404)
    result = _recompute_risk(child)
    log_activity(f"Re-assessed risk for {child_label(child)}: {result['label']}")
    flash(f"Risk re-assessed for {child.display_name}: {result['label']}", "success")
    return redirect(request.referrer or url_for("rhu.children"))


@rhu_bp.route("/children/<int:child_id>")
def child_record(child_id):
    child = db.session.get(Child, child_id) or abort(404)
    return render_template("child_record.html", **_rhu_ctx(
        "children", page_title="Child Record", child=child, detail=child_detail(child),
        back_url=url_for("rhu.children"),
        edit_url=url_for("rhu.edit_child", child_id=child.id),
        recompute_url=url_for("rhu.recompute_risk", child_id=child.id),
    ))


@rhu_bp.route("/children/<int:child_id>/edit", methods=["GET", "POST"])
def edit_child(child_id):
    child = db.session.get(Child, child_id) or abort(404)
    if request.method == "POST":
        result = _update_child(child, request.form)
        flash(f"{child.display_name} updated. Risk re-assessed: {result['label']}", "success")
        return redirect(url_for("rhu.children"))
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    doses = {code: (rec.date_administered if (rec := child.vaccination_records.filter_by(
        vaccine_type_id=vt_lookup[code].id).first()) else None) for code, *_ in dp.VACCINE_SCHEDULE}
    return render_template("edit_child.html", **_rhu_ctx(
        "children", page_title="Edit Child Record", child=child, doses=doses,
        back_url=url_for("rhu.children"), form_action=url_for("rhu.edit_child", child_id=child.id),
    ))


@rhu_bp.route("/risk")
def risk():
    dist, factors, rows = risk_distribution(), top_risk_factors(), at_risk_table()
    engine = dp.model_info()
    return render_template("rhu_risk.html", **_rhu_ctx(
        "risk", page_title="Risk Classification", dist=dist, factors=factors, at_risk_rows=rows,
        severity=dp.ANTIGEN_SEVERITY, severity_bands=dp.SEVERITY_BANDS,
        antigen_names={a.code: a.name for a in VaccineAntigen.query.all()},
        engine=engine,
    ))


@rhu_bp.route("/continuation")
def continuation():
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    rows = at_risk_table()
    for r in rows:
        r["continuation_pct"] = 100 - r["risk_score"]
        r["next_dose"] = _next_dose_for(r["child"], vt_lookup)
    engine = dp.model_info()
    return render_template("rhu_continuation.html", **_rhu_ctx(
        "continuation", page_title="Continuation Predictor", rows=rows, stats=dashboard_stats(),
        dist=risk_distribution(), engine=engine, scoring_rules=dp.SCORING_RULES,
        feature_labels=dp.FEATURE_LABELS, features=feature_catalog(),
    ))


@rhu_bp.route("/map")
def municipality_map():
    return render_template("rhu_map.html", **_rhu_ctx(
        "map", page_title="Municipality Map",
        heatmap=map_markers("heatmap"), at_risk=map_markers("at_risk"), density=map_markers("density"),
    ))


@rhu_bp.route("/inventory", methods=["GET", "POST"])
def inventory():
    if request.method == "POST":
        batch = InventoryBatch(
            antigen_id=int(request.form.get("antigen_id")), batch_number=request.form.get("batch_number"),
            quantity_on_hand=int(request.form.get("quantity")), storage_location=request.form.get("storage_location"),
            expiry_date=(datetime.strptime(request.form.get("expiry_date"), "%Y-%m-%d").date()
                        if request.form.get("expiry_date") else None),
            reorder_level=50, added_at=datetime.utcnow(),
        )
        db.session.add(batch)
        db.session.commit()
        log_activity(f"Added vaccine stock: {batch.antigen.name} +{batch.quantity_on_hand} vials")
        flash("Stock added successfully.", "success")
        return redirect(url_for("rhu.inventory"))

    rows = stock_levels()
    stats = {"total_types": len(rows), "out_of_stock": sum(1 for r in rows if r["status"] == "Out of Stock"),
             "low_critical": sum(1 for r in rows if r["status"] in ("Low", "Critical")),
             "expiring_soon": sum(r["expiring"] for r in rows)}
    return render_template("rhu_inventory.html", **_rhu_ctx(
        "inventory", page_title="Vaccine Inventory", rows=rows, stats=stats, antigens=VaccineAntigen.query.all(),
    ))


@rhu_bp.route("/requests")
def requests():
    tab = request.args.get("tab", "pending")
    q = VaccineRequest.query.order_by(VaccineRequest.requested_at.desc())
    if tab == "pending":
        q = q.filter_by(status="pending")
    stats = {"pending": VaccineRequest.query.filter_by(status="pending").count(),
             "approved": VaccineRequest.query.filter_by(status="approved").count(),
             "distributed": VaccineRequest.query.filter_by(status="fulfilled").count(),
             "rejected": VaccineRequest.query.filter_by(status="rejected").count()}
    return render_template("rhu_requests.html", **_rhu_ctx(
        "requests", page_title="Vaccine Requests", reqs=q.all(), stats=stats, tab=tab,
    ))


@rhu_bp.route("/requests/<int:req_id>/<action>", methods=["POST"])
def request_action(req_id, action):
    vr = db.session.get(VaccineRequest, req_id) or abort(404)
    if action == "approve":
        vr.status, vr.reviewed_by_id, vr.reviewed_at = "approved", current_user.id, datetime.utcnow()
        log_activity(f"Approved vaccine request {vr.request_code}")
    elif action == "reject":
        vr.status, vr.reviewed_by_id, vr.reviewed_at = "rejected", current_user.id, datetime.utcnow()
        log_activity(f"Rejected vaccine request {vr.request_code}")
    elif action == "distribute":
        vr.status, vr.distributed_at = "fulfilled", datetime.utcnow()
        batch = (InventoryBatch.query.filter_by(antigen_id=vr.antigen_id, is_archived=False)
                 .order_by(InventoryBatch.expiry_date).first())
        if batch and batch.quantity_on_hand >= vr.quantity_requested:
            batch.quantity_on_hand -= vr.quantity_requested
        log_activity(f"Distributed vaccine request {vr.request_code}")
    db.session.commit()
    flash(f"Request {vr.request_code} {action}d.", "success")
    return redirect(url_for("rhu.requests"))


@rhu_bp.route("/forecast")
def forecast():
    """Projected vaccine demand for the coming years.

    Two components, kept separate on screen because only one is a forecast:
    doses children already in the registry will become due for, and doses for a
    projected birth cohort. RHU only, since ordering stock is their decision."""
    children = _children_query(None).all()
    dobs = [c.date_of_birth for c in children if c.date_of_birth]
    births_by_year = Counter(d.year for d in dobs)
    data = dp.forecast_vaccine_demand(dobs, births_by_year)

    # The forecast covers the barangays that have registry records, not the
    # whole municipality, until every barangay has been exported.
    with_records = len({c.barangay_id for c in children if c.barangay_id})
    antigen_names = {a.code: a.name for a in VaccineAntigen.query.all()}

    return render_template("rhu_forecast.html", **_rhu_ctx(
        "forecast", page_title="Vaccine Forecast",
        forecast=data, cohort=data["cohort"], antigen_names=antigen_names,
        births_by_year=dict(sorted(births_by_year.items())),
        barangays_with_records=with_records,
        total_barangays=Barangay.query.count(),
        registry_children=len(dobs),
    ))


@rhu_bp.route("/allocation")
def allocation():
    """Suggested split of available stock across pending requests.

    Only relevant when requests exceed stock. Three methods are computed rather
    than one, because rationing vaccines is the RHU's decision, not the
    system's. RHU only, since they hold the stock."""
    method = request.args.get("method", "priority")
    if method not in dp.ALLOCATION_METHODS:
        method = "priority"

    stock = {r["antigen"].id: r["total"] for r in stock_levels()}
    coverage = {r["barangay"]: r["coverage"] for r in coverage_by_barangay()}

    pending = (VaccineRequest.query.filter_by(status="pending")
               .order_by(VaccineRequest.requested_at.asc()).all())

    # Grouped by antigen: stock is held per vaccine, so each is rationed
    # separately.
    groups = []
    by_antigen = defaultdict(list)
    for r in pending:
        by_antigen[r.antigen_id].append(r)

    for antigen_id, reqs in by_antigen.items():
        antigen = reqs[0].antigen
        rows = [{"id": r.id, "code": r.request_code, "barangay": r.barangay.name,
                 "requested": r.quantity_requested, "priority": r.priority,
                 "coverage": coverage.get(r.barangay.name)}
                for r in reqs]
        result = dp.allocate_stock(rows, stock.get(antigen_id, 0), method, coverage)
        groups.append({"antigen": antigen, **result})

    # Short vaccines first: those are the decisions that need making.
    groups.sort(key=lambda g: (g["can_meet_all"], -g["shortfall"]))

    return render_template("rhu_allocation.html", **_rhu_ctx(
        "allocation", page_title="Stock Allocation",
        groups=groups, method=method, methods=dp.ALLOCATION_METHODS,
        pending_count=len(pending),
    ))


@rhu_bp.route("/reports")
def reports():
    barangay_id = request.args.get("barangay_id", type=int)
    period = request.args.get("period", "")
    as_of = period_end(period)
    fig = report_figures(barangay_id, as_of)
    return render_template("rhu_reports.html", **_rhu_ctx(
        "reports", page_title="Reports", report_type=request.args.get("type", "coverage"),
        stats=fig, by_barangay=fig["by_barangay"], generated_at=datetime.now(),
        barangays=Barangay.query.order_by(Barangay.name).all(),
        selected_barangay=barangay_id, periods=month_options(), selected_period=period,
        as_of=as_of,
    ))


@rhu_bp.route("/reports/export.csv")
def export_report_csv():
    """Downloads the on-screen report as CSV (opens directly in Excel)."""
    barangay_id = request.args.get("barangay_id", type=int)
    as_of = period_end(request.args.get("period", ""))
    fig = report_figures(barangay_id, as_of)
    scope = next((b.name for b in Barangay.query.filter_by(id=barangay_id)), "All Barangays")

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ImmunoVision - Coverage Report"])
    w.writerow(["San Jacinto Rural Health Unit, Pangasinan"])
    w.writerow(["Scope", scope])
    w.writerow(["Reporting period ending", as_of.strftime("%B %d, %Y")])
    w.writerow(["Generated", datetime.now().strftime("%b %d, %Y %I:%M %p")])
    w.writerow([])
    w.writerow(["Total Children", fig["total_children"]])
    w.writerow(["Fully Immunized", fig["fully_immunized"]])
    w.writerow(["At-Risk Children", fig["at_risk"]])
    w.writerow(["Coverage Rate (%)", fig["coverage_rate"]])
    w.writerow([])
    w.writerow(["Barangay", "Children", "Fully Immunized", "At-Risk", "Coverage %"])
    for r in fig["by_barangay"]:
        w.writerow([r["barangay"], r["children"], r["fully_immunized"], r["at_risk"], r["coverage"]])

    log_activity("Exported coverage report (CSV)")
    filename = f"immunovision_report_{as_of:%Y%m%d}.csv"
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


@rhu_bp.route("/notifications")
def notifications():
    return render_template("rhu_notifications.html", **_rhu_ctx(
        "notifications", page_title="Notifications", notes=get_notifications("rhu"),
    ))


@rhu_bp.route("/settings")
def settings():
    return render_template("rhu_settings.html", **_rhu_ctx("settings", page_title="Settings"))


# ---------------------------------------------------------------------------
# BHW blueprint
# ---------------------------------------------------------------------------

bhw_bp = Blueprint("bhw", __name__, url_prefix="/bhw")


def _bhw_ctx(active, **extra):
    notes = get_notifications("bhw", barangay_id=current_user.barangay_id)
    return dict(role="bhw", nav_items=build_nav(BHW_NAV), active=active, current_user=current_user,
                settings_url=url_for("bhw.settings"),
                unread_count=unread_count(notes), notif_url=url_for("bhw.notifications"), **extra)


@bhw_bp.before_request
@login_required
@role_required("bhw")
def _bhw_guard():
    pass


@bhw_bp.route("/dashboard")
def dashboard():
    bid = current_user.barangay_id
    risk_level = request.args.get("risk_level", "")
    status = request.args.get("status", "")
    month = request.args.get("month", "")
    year = request.args.get("year", "")
    as_of, period_label, clamped = resolve_period(month, year)
    kids = filtered_children(bid, risk_level, status, as_of)
    # The BHW does the home visits, so the "who to chase" list matters most here.
    follow_up = at_risk_table(bid, limit=5)

    # Coverage Analytics merged into this page; its own route was retired.
    by_vaccine = order_by_schedule(report_by_vaccine(bid, as_of) if as_of else coverage_by_vaccine(bid))
    trend = yearly_trend(bid)
    change = coverage_change(bid)
    return render_template("bhw_dashboard.html", **_bhw_ctx(
        "dashboard", page_title="Dashboard", stats=stats_for(kids, as_of),
        risk_dist=risk_distribution_for(kids), now=datetime.now(),
        risk_levels=RISK_LEVELS, status_levels=STATUS_LEVELS,
        risk_level=risk_level, status=status,
        months=MONTH_NAMES, years=data_years(), month=month, year=year,
        as_of=as_of, period_label=period_label, period_clamped=clamped, follow_up=follow_up,
        by_vaccine=by_vaccine, trend=trend, change=change,
        vaccine_colours=vaccine_colours(by_vaccine), vaccine_labels=schedule_labels(by_vaccine),
        insights=chart_insights(by_vaccine, [], trend),
        total_in_barangay=len(filtered_children(bid, as_of=as_of)),
        pending_requests=(VaccineRequest.query.filter_by(barangay_id=bid, status="pending")
                          .order_by(VaccineRequest.requested_at.desc()).first()),
    ))


@bhw_bp.route("/children", methods=["GET", "POST"])
def children():
    bid = current_user.barangay_id
    if request.method == "POST":
        _create_child(request.form, barangay_locked=bid)
        flash("Child registered successfully.", "success")
        return redirect(url_for("bhw.children"))
    risk_level = request.args.get("risk_level", "")
    search = request.args.get("q", "").strip()
    q = Child.query.filter_by(barangay_id=bid)
    if search:
        q = q.filter(Child.full_name.ilike(f"%{search}%"))
    kids = q.order_by(Child.id).all()
    if risk_level:
        kids = [c for c in kids if (ra := _latest_risk(c)) and ra.risk_label == risk_level]
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    return render_template("bhw_children.html", **_bhw_ctx(
        "children", page_title="Child Records", children=kids, barangays=None,
        vaccine_schedule=dp.VACCINE_SCHEDULE, vt_lookup=vt_lookup, selected_risk=risk_level, search=search,
        dose_matrix=_dose_matrix(kids, vt_lookup), risk_labels=_risk_labels(kids),
    ))


@bhw_bp.route("/children/<int:child_id>/recompute-risk", methods=["POST"])
def recompute_risk(child_id):
    child = db.session.get(Child, child_id) or abort(404)
    if child.barangay_id != current_user.barangay_id:
        abort(403)
    result = _recompute_risk(child)
    log_activity(f"Re-assessed risk for {child_label(child)}: {result['label']}")
    flash(f"Risk re-assessed for {child.display_name}: {result['label']}", "success")
    return redirect(request.referrer or url_for("bhw.children"))


@bhw_bp.route("/children/<int:child_id>")
def child_record(child_id):
    child = db.session.get(Child, child_id) or abort(404)
    if child.barangay_id != current_user.barangay_id:
        abort(403)
    return render_template("child_record.html", **_bhw_ctx(
        "children", page_title="Child Record", child=child, detail=child_detail(child),
        back_url=url_for("bhw.children"),
        edit_url=url_for("bhw.edit_child", child_id=child.id),
        recompute_url=url_for("bhw.recompute_risk", child_id=child.id),
    ))


@bhw_bp.route("/children/<int:child_id>/edit", methods=["GET", "POST"])
def edit_child(child_id):
    child = db.session.get(Child, child_id) or abort(404)
    if child.barangay_id != current_user.barangay_id:
        abort(403)
    if request.method == "POST":
        result = _update_child(child, request.form)
        flash(f"{child.display_name} updated. Risk re-assessed: {result['label']}", "success")
        return redirect(url_for("bhw.children"))
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    doses = {code: (rec.date_administered if (rec := child.vaccination_records.filter_by(
        vaccine_type_id=vt_lookup[code].id).first()) else None) for code, *_ in dp.VACCINE_SCHEDULE}
    return render_template("edit_child.html", **_bhw_ctx(
        "children", page_title="Edit Child Record", child=child, doses=doses,
        back_url=url_for("bhw.children"), form_action=url_for("bhw.edit_child", child_id=child.id),
    ))


@bhw_bp.route("/risk")
def risk():
    bid = current_user.barangay_id
    engine = dp.model_info()
    return render_template("bhw_risk.html", **_bhw_ctx(
        "risk", page_title="Risk Classification", dist=risk_distribution(bid), factors=top_risk_factors(bid),
        severity=dp.ANTIGEN_SEVERITY, severity_bands=dp.SEVERITY_BANDS,
        antigen_names={a.code: a.name for a in VaccineAntigen.query.all()},
        at_risk_rows=at_risk_table(bid), engine=engine,
    ))


@bhw_bp.route("/continuation")
def continuation():
    bid = current_user.barangay_id
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    rows = at_risk_table(bid)
    for r in rows:
        r["continuation_pct"] = 100 - r["risk_score"]
        r["next_dose"] = _next_dose_for(r["child"], vt_lookup)
    engine = dp.model_info()
    return render_template("bhw_continuation.html", **_bhw_ctx(
        "continuation", page_title="Continuation Predictor", rows=rows, stats=dashboard_stats(bid),
        dist=risk_distribution(bid), engine=engine, scoring_rules=dp.SCORING_RULES,
        feature_labels=dp.FEATURE_LABELS, features=feature_catalog(),
    ))


@bhw_bp.route("/requests", methods=["GET", "POST"])
def requests():
    bid = current_user.barangay_id
    if request.method == "POST":
        last = VaccineRequest.query.order_by(VaccineRequest.id.desc()).first()
        code = f"REQ-{datetime.now().year}-{(last.id if last else 0) + 1:03d}"
        vr = VaccineRequest(request_code=code, barangay_id=bid, requested_by_id=current_user.id,
                            antigen_id=int(request.form.get("antigen_id")),
                            quantity_requested=int(request.form.get("quantity")),
                            priority=request.form.get("priority", "Normal"), notes=request.form.get("notes"))
        db.session.add(vr)
        db.session.commit()
        log_activity(f"Submitted vaccine request {code} ({vr.antigen.name}, {vr.quantity_requested} vials)")
        flash(f"Request {code} submitted to RHU.", "success")
        return redirect(url_for("bhw.requests"))
    reqs = VaccineRequest.query.filter_by(barangay_id=bid).order_by(VaccineRequest.requested_at.desc()).all()
    stats = {"pending": sum(1 for r in reqs if r.status == "pending"),
             "approved": sum(1 for r in reqs if r.status == "approved"),
             "distributed": sum(1 for r in reqs if r.status == "fulfilled"),
             "rejected": sum(1 for r in reqs if r.status == "rejected")}
    return render_template("bhw_requests.html", **_bhw_ctx(
        "requests", page_title="Vaccine Requests", reqs=reqs, stats=stats, antigens=VaccineAntigen.query.all(),
    ))


@bhw_bp.route("/reports")
def reports():
    bid = current_user.barangay_id
    period = request.args.get("period", "")
    as_of = period_end(period)
    fig = report_figures(bid, as_of)
    return render_template("bhw_reports.html", **_bhw_ctx(
        "reports", page_title="Reports", report_type=request.args.get("type", "coverage"),
        stats=fig, by_vaccine=report_by_vaccine(bid, as_of), generated_at=datetime.now(),
        periods=month_options(), selected_period=period, as_of=as_of,
    ))


@bhw_bp.route("/reports/export.csv")
def export_report_csv():
    """Downloads this barangay's report as CSV (opens directly in Excel)."""
    bid = current_user.barangay_id
    as_of = period_end(request.args.get("period", ""))
    fig = report_figures(bid, as_of)
    scope = current_user.barangay.name if current_user.barangay else "Unassigned"

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ImmunoVision - Barangay Immunization Report"])
    w.writerow(["San Jacinto Rural Health Unit, Pangasinan"])
    w.writerow(["Barangay", scope])
    w.writerow(["Reporting period ending", as_of.strftime("%B %d, %Y")])
    w.writerow(["Prepared by", current_user.full_name])
    w.writerow(["Generated", datetime.now().strftime("%b %d, %Y %I:%M %p")])
    w.writerow([])
    w.writerow(["Total Children", fig["total_children"]])
    w.writerow(["Fully Immunized", fig["fully_immunized"]])
    w.writerow(["At-Risk Children", fig["at_risk"]])
    w.writerow(["Coverage Rate (%)", fig["coverage_rate"]])
    w.writerow([])
    w.writerow(["Vaccine / Dose", "Completion %"])
    for r in report_by_vaccine(bid, as_of):
        w.writerow([r["vaccine"], r["pct"]])

    log_activity("Exported barangay report (CSV)")
    filename = f"immunovision_{scope.split()[0].lower()}_{as_of:%Y%m%d}.csv"
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


@bhw_bp.route("/notifications")
def notifications():
    return render_template("bhw_notifications.html", **_bhw_ctx(
        "notifications", page_title="Notifications", notes=get_notifications("bhw", barangay_id=current_user.barangay_id),
    ))


@bhw_bp.route("/settings")
def settings():
    return render_template("bhw_settings.html", **_bhw_ctx("settings", page_title="Settings"))


# ---------------------------------------------------------------------------
# Admin blueprint
# ---------------------------------------------------------------------------

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def _admin_ctx(active, **extra):
    notes = admin_notifications()
    return dict(role="admin", nav_items=build_nav(ADMIN_NAV), active=active, current_user=current_user,
                settings_url=url_for("admin.settings"),
                unread_count=sum(1 for n in notes if n["unread"]),
                notif_url=url_for("admin.notifications"), **extra)


@admin_bp.before_request
@login_required
@role_required("admin")
def _admin_guard():
    pass


@admin_bp.route("/dashboard")
def dashboard():
    users = User.query.all()
    stats = {"total": len(users), "rhu": sum(1 for u in users if u.role == "rhu"),
             "bhw": sum(1 for u in users if u.role == "bhw"), "active": sum(1 for u in users if u.is_active_flag),
             "disabled": sum(1 for u in users if not u.is_active_flag),
             "logins_today": sum(1 for u in users if u.last_login_at and u.last_login_at.date() == datetime.utcnow().date())}
    role_dist = {"admin": sum(1 for u in users if u.role == "admin"), "rhu": stats["rhu"], "bhw": stats["bhw"]}

    # Municipality-wide oversight. AGGREGATES ONLY - counts and percentages per
    # barangay, never child-level records. The administrator manages accounts and
    # oversees municipal performance, but has no need to see individual children,
    # so the minimum necessary data is exposed here.
    # Deliberately the same functions the RHU dashboard and Coverage Analytics
    # use, so every live view of municipal performance reports identical
    # figures. (report_figures() is reserved for the Reports page, which is
    # explicitly "as of" a chosen period and so can legitimately differ.)
    # Period filter, matching the RHU and BHW dashboards.
    periods = month_options(24)
    # Same scope + period filters as the RHU dashboard: the administrator also
    # oversees all 19 barangays and needs to narrow to one.
    barangay_id = request.args.get("barangay_id", type=int)
    selected = db.session.get(Barangay, barangay_id) if barangay_id else None
    if not selected:
        barangay_id = None
    month = request.args.get("month", "")
    year = request.args.get("year", "")
    as_of, period_label, clamped = resolve_period(month, year)

    if as_of:
        fig = report_figures(barangay_id, as_of)
        muni = {"total_children": fig["total_children"], "fully_immunized": fig["fully_immunized"],
                "at_risk": fig["at_risk"], "coverage_rate": fig["coverage_rate"]}
        rows = fig["by_barangay"]
    else:
        muni = dashboard_stats(barangay_id)
        rows = coverage_by_barangay()
    if selected:
        rows = [r for r in rows if r["barangay"] == selected.name]
    municipality = {
        "total_children": muni["total_children"],
        "fully_immunized": muni["fully_immunized"],
        "at_risk": muni["at_risk"],
        "coverage_rate": muni["coverage_rate"],
        "barangays": len(rows),
        "below_target": sum(1 for r in rows if r["coverage"] < COVERAGE_TARGET),
    }
    # Worst-performing first, so barangays needing attention surface immediately.
    barangay_rows = sorted(rows, key=lambda r: r["coverage"])

    # Oversight is the administrator's job, and 192 of 320 audit entries are from
    # the last week - none of which appeared on this page. Summarises the
    # Activity Logs page rather than reproducing it.
    recent = (ActivityLog.query.order_by(ActivityLog.created_at.desc()).limit(7).all())
    recent_activity = [{"log": r, "category": log_category(r.action)} for r in recent]

    # The admin Coverage Analytics page was retired: its barangay view duplicated
    # the table above, so its two unique panels moved onto this page.
    by_vaccine = order_by_schedule(report_by_vaccine(None, as_of) if as_of else coverage_by_vaccine())
    # Annual rather than monthly: a month/year filter and a 12-month series
    # contradict each other, and yearly coverage can be computed for real.
    trend = yearly_trend()

    return render_template("admin_dashboard.html", **_admin_ctx(
        "dashboard", page_title="Dashboard", stats=stats, role_dist=role_dist, users=users,
        municipality=municipality, barangay_rows=barangay_rows, now=datetime.now(),
        recent_activity=recent_activity, by_vaccine=by_vaccine, trend=trend,
        vaccine_colours=vaccine_colours(by_vaccine), vaccine_labels=schedule_labels(by_vaccine),
        insights=chart_insights(by_vaccine, barangay_rows, trend),
        months=MONTH_NAMES, years=data_years(), month=month, year=year,
        as_of=as_of, period_label=period_label, period_clamped=clamped,
        barangays=Barangay.query.order_by(Barangay.name).all(),
        selected_barangay=selected, selected_barangay_id=barangay_id,
    ))


@admin_bp.route("/users", methods=["GET", "POST"])
def users():
    if request.method == "POST":
        role = request.form.get("role")
        username = request.form.get("username").strip()
        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "error")
            return redirect(url_for("admin.users"))
        u = User(username=username, full_name=f"{request.form.get('first_name')} {request.form.get('last_name')}".strip(),
                 role=role, barangay_id=(int(request.form.get("barangay_id"))
                                         if role == "bhw" and request.form.get("barangay_id") else None))
        u.set_password(request.form.get("password") or "password123")
        db.session.add(u)
        db.session.commit()
        log_activity(f"Created user account: {u.full_name} ({u.role_label})")
        flash(f"User {u.full_name} created.", "success")
        return redirect(url_for("admin.users"))

    all_users = User.query.order_by(User.id).all()
    stats = {"total": len(all_users), "active": sum(1 for u in all_users if u.is_active_flag),
             "bhw": sum(1 for u in all_users if u.role == "bhw"),
             "rhu": sum(1 for u in all_users if u.role in ("rhu", "admin"))}
    return render_template("admin_users.html", **_admin_ctx(
        "users", page_title="User Management", users=all_users, stats=stats,
        barangays=Barangay.query.order_by(Barangay.name).all(),
    ))


@admin_bp.route("/users/<int:user_id>/toggle", methods=["POST"])
def toggle_user(user_id):
    u = db.session.get(User, user_id) or abort(404)
    if u.id == current_user.id:
        abort(400)
    u.is_active_flag = not u.is_active_flag
    db.session.commit()
    log_activity(f"{'Enabled' if u.is_active_flag else 'Disabled'} user account: {u.username}")
    flash(f"User {u.username} {'enabled' if u.is_active_flag else 'disabled'}.", "success")
    return redirect(url_for("admin.users"))


@admin_bp.route("/assign-barangay", methods=["GET", "POST"])
def assign_barangay():
    if request.method == "POST":
        u = db.session.get(User, int(request.form.get("user_id"))) or abort(404)
        u.barangay_id = int(request.form.get("barangay_id"))
        db.session.commit()
        log_activity(f"Reassigned {u.full_name} to {u.barangay.name}")
        flash(f"{u.full_name} reassigned to {u.barangay.name}.", "success")
        return redirect(url_for("admin.assign_barangay"))
    return render_template("admin_assign_barangay.html", **_admin_ctx(
        "assign", page_title="Assign Barangay", bhws=User.query.filter_by(role="bhw").order_by(User.full_name).all(),
        barangays=Barangay.query.order_by(Barangay.name).all(),
    ))


@admin_bp.route("/map")
def municipality_map():
    return render_template("admin_map.html", **_admin_ctx(
        "map", page_title="Municipality Map",
        heatmap=map_markers("heatmap"), at_risk=map_markers("at_risk"), density=map_markers("density"),
    ))


@admin_bp.route("/logs")
def logs():
    """Audit trail. Filterable by category and by user account, and the two
    combine - ?user=7&type=Record is "what this person did to child records"."""
    type_filter = request.args.get("type", "All")
    user_id = request.args.get("user", type=int)

    q = ActivityLog.query
    selected_user = db.session.get(User, user_id) if user_id else None
    if selected_user:
        q = q.filter(ActivityLog.user_id == selected_user.id)
    else:
        user_id = None  # unknown id: fall back to showing everything

    rows = q.order_by(ActivityLog.created_at.desc()).limit(400).all()
    entries = [{"log": r, "category": log_category(r.action)} for r in rows]
    if type_filter != "All":
        entries = [e for e in entries if e["category"] == type_filter]

    return render_template("admin_logs.html", **_admin_ctx(
        "logs", page_title="Activity Logs", entries=entries[:200], type_filter=type_filter,
        categories=["All"] + [name for name, _ in LOG_CATEGORIES],
        users=User.query.order_by(User.full_name).all(),
        selected_user=selected_user, selected_user_id=user_id,
    ))


@admin_bp.route("/notifications")
def notifications():
    notes = admin_notifications()
    return render_template("admin_notifications.html", **_admin_ctx(
        "notifications", page_title="Notifications", notes=notes,
    ))


@admin_bp.route("/settings")
def settings():
    return render_template("admin_settings.html", **_admin_ctx("settings", page_title="Settings"))


@admin_bp.route("/vaccine-alerts")
def vaccine_alerts():
    """Warning levels for vaccine stock.

    Its own page rather than a Settings panel: the RHU reaches Vaccine Inventory
    from the sidebar, and the setting that governs its alerts should be no
    harder to find."""
    return render_template("admin_vaccine_alerts.html", **_admin_ctx(
        "vaccines", page_title="Vaccine Alerts", stock=stock_levels()))


@admin_bp.route("/vaccine-alerts/warning-levels", methods=["POST"])
def update_warning_levels():
    """Set the stock level at which each vaccine is flagged as running low.

    This is a municipal policy setting rather than a daily operation, so it sits
    with the administrator. Current stock is shown alongside each input so the
    value is set against what it affects. Written to every active batch of the
    antigen, since the stock calculation takes the minimum across batches.
    Changes are logged: raising a threshold silences alerts, and that should be
    traceable."""
    changed = []
    for antigen in VaccineAntigen.query.all():
        raw = request.form.get(f"warn_{antigen.id}")
        if raw is None or not raw.strip():
            continue
        try:
            level = int(raw)
        except ValueError:
            flash(f"{antigen.name}: the warning level must be a whole number.", "error")
            continue
        if not 0 <= level <= 100000:
            flash(f"{antigen.name}: the warning level must be between 0 and 100,000.", "error")
            continue

        batches = InventoryBatch.query.filter_by(antigen_id=antigen.id, is_archived=False).all()
        current = min([b.reorder_level for b in batches], default=None)
        if not batches or current == level:
            continue
        for b in batches:
            b.reorder_level = level
        changed.append(f"{antigen.name} {current} to {level}")

    if changed:
        db.session.commit()
        log_activity("Updated vaccine warning levels: " + "; ".join(changed))
        flash(f"Warning level updated for {len(changed)} vaccine(s).", "success")
    else:
        flash("No warning levels were changed.", "info")
    return redirect(url_for("admin.vaccine_alerts"))


# ---------------------------------------------------------------------------
# API blueprint (small JSON endpoints)
# ---------------------------------------------------------------------------

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/health")
def health():
    return jsonify({"status": "ok"})


app = create_app()


def ensure_demo_data():
    """Populate the demo data on first run.

    The prototype ships with a pre-built data/immunovision.db, so normally this
    finds the data already there and does nothing. If that file is missing (or
    was deleted), it rebuilds it automatically instead of failing with an empty
    login page - so `python app.py` is the only command needed to demo."""
    with app.app_context():
        db.create_all()
        if User.query.count() > 0:
            return
        print("No demo data found - building it now (about a minute, one time only)...")
        import seed
        seed.seed()
        print("Demo data ready.\n")


if __name__ == "__main__":
    ensure_demo_data()
    print("ImmunoVision running at http://127.0.0.1:5000")
    print("Log in with  msantos / password123  (RHU)\n")
    app.run(host="0.0.0.0", port=5000, debug=True)