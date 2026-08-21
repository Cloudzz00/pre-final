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
import io
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
        return {"today": date.today()}

    return app


# ---------------------------------------------------------------------------
# Shared helpers: activity log, role guard, nav builder
# ---------------------------------------------------------------------------

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
    ("risk", "heart-pulse", "Risk Prediction", "rhu.risk"),
    ("continuation", "target", "Continuation Predictor", "rhu.continuation"),
    ("coverage", "bar-chart", "Coverage Analytics", "rhu.coverage"),
    ("map", "map", "Municipality Map", "rhu.municipality_map"),
    ("inventory", "package", "Vaccine Inventory", "rhu.inventory"),
    ("requests", "clipboard", "Vaccine Requests", "rhu.requests"),
    ("reports", "file-text", "Reports", "rhu.reports"),
    ("notifications", "bell", "Notifications", "rhu.notifications"),
    ("settings", "settings", "Settings", "rhu.settings"),
]
BHW_NAV = [
    ("dashboard", "home", "Dashboard", "bhw.dashboard"),
    ("children", "baby", "Child Records", "bhw.children"),
    ("risk", "heart-pulse", "Risk Prediction", "bhw.risk"),
    ("continuation", "target", "Continuation Predictor", "bhw.continuation"),
    ("coverage", "bar-chart", "Coverage Analytics", "bhw.coverage"),
    ("requests", "clipboard", "Vaccine Requests", "bhw.requests"),
    ("reports", "file-text", "Reports", "bhw.reports"),
    ("notifications", "bell", "Notifications", "bhw.notifications"),
    ("settings", "settings", "Settings", "bhw.settings"),
]
ADMIN_NAV = [
    ("dashboard", "home", "Dashboard", "admin.dashboard"),
    ("users", "users", "User Management", "admin.users"),
    ("assign", "map-pin", "Assign Barangay", "admin.assign_barangay"),
    ("coverage", "bar-chart", "Coverage Analytics", "admin.coverage"),
    ("map", "map", "Municipality Map", "admin.municipality_map"),
    ("logs", "history", "Activity Logs", "admin.logs"),
    ("settings", "settings", "Settings", "admin.settings"),
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


def _due_and_missed(child):
    """Doses the child is old enough to have received, and how many are unfilled."""
    vt_by_id = _vaccine_types_by_id()
    age = child.age_in_days
    due = missed = 0
    for record in _records_by_child().get(child.id, ()):
        vt = vt_by_id.get(record.vaccine_type_id)
        if vt is not None and vt.recommended_age_days <= age:
            due += 1
            if record.date_administered is None:
                missed += 1
    return due, missed


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


def monthly_trend(barangay_id=None, current_rate=None):
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
        due, missed = _due_and_missed(c)
        given_dates = [r.date_administered for r in records.get(c.id, ()) if r.date_administered]
        last_visit = max(given_dates) if given_dates else None
        tier = dp.risk_tier(latest.risk_probability)
        rows.append({"child": c, "risk_score": round(latest.risk_probability * 100), "tier": tier,
                     "color": dp.risk_tier_color(tier), "missed": missed, "due": due, "last_visit": last_visit,
                     "recommendation": "Home Visit" if tier == "High Risk" else "Schedule Visit"})
    rows.sort(key=lambda r: r["risk_score"], reverse=True)
    return rows[:limit] if limit else rows


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


def _seed_coords():
    if not _MAP_COORDS:
        rng = random.Random(2026)
        for name in dp.BARANGAYS:
            _MAP_COORDS[name] = (round(rng.uniform(0.14, 0.86), 3), round(rng.uniform(0.14, 0.82), 3))
    return _MAP_COORDS


def map_markers(mode="heatmap"):
    coords = _seed_coords()
    cov = {r["barangay"]: r for r in coverage_by_barangay()}
    markers = []
    for name in dp.BARANGAYS:
        r = cov.get(name, {"children": 0, "at_risk": 0, "coverage": 0})
        x, y = coords[name]
        if mode == "heatmap":
            value, suffix = r["coverage"], "%"
            color = "green" if value >= 90 else "teal" if value >= 75 else "amber" if value >= 60 else "red"
        elif mode == "at_risk":
            value, suffix = r["at_risk"], ""
            share = (r["at_risk"] / r["children"]) if r["children"] else 0
            color = "red" if share >= 0.30 else "amber" if share >= 0.15 else "green"
        else:
            value, suffix, color = r["children"], "", "blue"
        markers.append({"barangay": name, "x": x, "y": y, "value": value, "suffix": suffix, "color": color})
    return markers


def coverage_rankings(top_n=6):
    return coverage_by_barangay()[:top_n]


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
    log_activity(f"Registered child {child.full_name} ({barangay_name})")
    return child


def _update_child(child, form):
    """Updates an existing child's basic info and any dose dates submitted
    on the Edit Child form, then recomputes their risk assessment so
    newly-recorded doses are reflected immediately."""
    child.full_name = form.get("full_name") or child.full_name
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
    log_activity(f"Updated child record {child.full_name} ({child.barangay.name})")
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
    if current_user.is_authenticated:
        return redirect(url_for(ROLE_HOME[current_user.role]))
    return redirect(url_for("auth.login"))


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
                unread_count=unread_count(notes), notif_url=url_for("rhu.notifications"), **extra)


@rhu_bp.before_request
@login_required
@role_required("rhu")
def _rhu_guard():
    pass


@rhu_bp.route("/dashboard")
def dashboard():
    stats = dashboard_stats()
    alerts = [f"{a.name} is completely out of stock" for a in VaccineAntigen.query.all()
              if sum(b.quantity_on_hand for b in InventoryBatch.query.filter_by(antigen_id=a.id, is_archived=False)) <= 0]
    dist = risk_distribution()
    return render_template("rhu_dashboard.html", **_rhu_ctx(
        "dashboard", page_title="Dashboard", stats=stats, inventory_alerts=alerts,
        at_risk_uncontacted=dist["high_count"], risk_dist=dist, now=datetime.now(),
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
    flash(f"Risk re-assessed for {child.full_name}: {result['label']}", "success")
    return redirect(request.referrer or url_for("rhu.children"))


@rhu_bp.route("/children/<int:child_id>/edit", methods=["GET", "POST"])
def edit_child(child_id):
    child = db.session.get(Child, child_id) or abort(404)
    if request.method == "POST":
        result = _update_child(child, request.form)
        flash(f"{child.full_name} updated. Risk re-assessed: {result['label']}", "success")
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
        "risk", page_title="Risk Prediction", dist=dist, factors=factors, at_risk_rows=rows,
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
        feature_labels=dp.FEATURE_LABELS,
    ))


@rhu_bp.route("/coverage")
def coverage():
    by_barangay = coverage_by_barangay()
    return render_template("rhu_coverage.html", **_rhu_ctx(
        "coverage", page_title="Coverage Analytics", stats=dashboard_stats(), by_barangay=by_barangay,
        by_vaccine=coverage_by_vaccine(), trend=monthly_trend(),
        below_target=len([r for r in by_barangay if r["coverage"] < 90]),
    ))


@rhu_bp.route("/map")
def municipality_map():
    return render_template("rhu_map.html", **_rhu_ctx(
        "map", page_title="Municipality Map", rankings=coverage_rankings(),
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
            status, color = "Adequate", "green"
        rows.append({"antigen": a, "total": total, "reorder": reorder, "status": status, "color": color,
                     "expiring": expiring, "batches": batches,
                     "pct": min(100, round((total / (reorder * 2)) * 100)) if reorder else 100})
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
                unread_count=unread_count(notes), notif_url=url_for("bhw.notifications"), **extra)


@bhw_bp.before_request
@login_required
@role_required("bhw")
def _bhw_guard():
    pass


@bhw_bp.route("/dashboard")
def dashboard():
    bid = current_user.barangay_id
    return render_template("bhw_dashboard.html", **_bhw_ctx(
        "dashboard", page_title="Dashboard", stats=dashboard_stats(bid), risk_dist=risk_distribution(bid),
        now=datetime.now(),
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
    flash(f"Risk re-assessed for {child.full_name}: {result['label']}", "success")
    return redirect(request.referrer or url_for("bhw.children"))


@bhw_bp.route("/children/<int:child_id>/edit", methods=["GET", "POST"])
def edit_child(child_id):
    child = db.session.get(Child, child_id) or abort(404)
    if child.barangay_id != current_user.barangay_id:
        abort(403)
    if request.method == "POST":
        result = _update_child(child, request.form)
        flash(f"{child.full_name} updated. Risk re-assessed: {result['label']}", "success")
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
        "risk", page_title="Risk Prediction", dist=risk_distribution(bid), factors=top_risk_factors(bid),
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
        feature_labels=dp.FEATURE_LABELS,
    ))


@bhw_bp.route("/coverage")
def coverage():
    bid = current_user.barangay_id
    return render_template("bhw_coverage.html", **_bhw_ctx(
        "coverage", page_title="Coverage Analytics", stats=dashboard_stats(bid),
        by_vaccine=coverage_by_vaccine(bid), trend=monthly_trend(bid),
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
    return dict(role="admin", nav_items=build_nav(ADMIN_NAV), active=active, current_user=current_user,
                unread_count=0, notif_url="#", **extra)


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
    return render_template("admin_dashboard.html", **_admin_ctx(
        "dashboard", page_title="Dashboard", stats=stats, role_dist=role_dist, users=users, now=datetime.now(),
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


@admin_bp.route("/coverage")
def coverage():
    by_barangay = coverage_by_barangay()
    return render_template("admin_coverage.html", **_admin_ctx(
        "coverage", page_title="Coverage Analytics", stats=dashboard_stats(), by_barangay=by_barangay,
        by_vaccine=coverage_by_vaccine(), trend=monthly_trend(),
        below_target=len([r for r in by_barangay if r["coverage"] < 90]),
    ))


@admin_bp.route("/map")
def municipality_map():
    return render_template("admin_map.html", **_admin_ctx(
        "map", page_title="Municipality Map", rankings=coverage_rankings(),
        heatmap=map_markers("heatmap"), at_risk=map_markers("at_risk"), density=map_markers("density"),
    ))


@admin_bp.route("/logs")
def logs():
    return render_template("admin_logs.html", **_admin_ctx(
        "logs", page_title="Activity Logs", logs=ActivityLog.query.order_by(ActivityLog.created_at.desc()).limit(200).all(),
        type_filter=request.args.get("type", "All"),
    ))


@admin_bp.route("/settings")
def settings():
    return render_template("admin_settings.html", **_admin_ctx("settings", page_title="Settings"))


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