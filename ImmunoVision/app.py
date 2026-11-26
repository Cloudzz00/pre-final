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
import json
import math
import io
import os
import time

# Every "today" and "now" in the system is Philippine time, not the server's
# (Render runs on UTC, which put dates a day behind between midnight and 8 AM
# and printed report times 8 hours early). A POSIX zone string needs no tz
# database on the server; the Philippines has no daylight saving time.
os.environ["TZ"] = "PHT-8"
if hasattr(time, "tzset"):
    time.tzset()
import re
import secrets
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone

from flask import (Flask, Blueprint, render_template, request, redirect, url_for, flash, abort,
                   jsonify, g, has_request_context, Response)
from flask_login import login_user, logout_user, login_required, current_user
from sqlalchemy import func

import data_processor as dp
from models import (
    db, login_manager, User, Barangay, Child, VaccineAntigen, VaccineType,
    VaccinationRecord, RiskAssessment, InventoryBatch, VaccineRequest, ActivityLog, Campaign,
    VialSize, BarangayStock, StockTransfer, StockTransferItem, StockMovement,
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
    app.register_blueprint(superadmin_bp)
    app.register_blueprint(api_bp)

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    app.add_template_filter(to_local, "local")

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
LOG_CATEGORIES = (  # log categories
    ("Login",   ("Logged in", "Logged out")),  # login
    ("Admin",   ("Created user", "Enabled user", "Disabled user", "Reassigned", "Reset password for user",  # admin
                 "campaign:", "target population")),
    # Report and Vaccine are checked before Record: "Recorded barangay stock
    # wastage" is a stock change, not a child record.
    ("Report",  ("Exported",)),  # reports
    ("Vaccine", ("vaccine stock", "vaccine request", "Approved", "Rejected", "Distributed", "Submitted",  # vaccines
                 "delivery", "opening stock", "doses per vial", "barangay stock", "physical count",
                 "redistribution")),
    ("Record",  ("Registered child", "Updated child", "Re-assessed risk", " date for ", "Removed",  # records
                 "Recorded")),
)


def log_category(action):  # pick category
    for name, keywords in LOG_CATEGORIES:  # each category
        if any(k.lower() in action.lower() for k in keywords):  # keyword match
            return name
    return "Other"  # no match


PH_TZ = timezone(timedelta(hours=8), "PHT")  # PH time


def to_local(value):  # UTC -> PH
    """A stored timestamp (naive UTC, from datetime.utcnow) as Philippine
    time, for display and for "which day was this" comparisons."""
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc).astimezone(PH_TZ).replace(tzinfo=None)  # convert


def child_label(child):  # child + barangay
    """Child name for the audit log, with the barangay appended only when the
    name does not already carry it (anonymised registry names look like
    "Unnamed Child #252 (Macayug)")."""
    name = child.display_name
    bgy = child.barangay.name if child.barangay else ""
    return name if (not bgy or f"({bgy})" in name) else f"{name} ({bgy})"


def log_activity(action):  # write log
    entry = ActivityLog(user_id=current_user.id if current_user.is_authenticated else None, action=action)  # who + what
    db.session.add(entry)  # add
    db.session.commit()  # save


def role_required(*roles):  # role check
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
    ("vaccine_gaps", "syringe", "Vaccine Gaps", "rhu.vaccine_gaps"),
    ("due_list", "calendar", "Vaccination Due List", "rhu.due_vaccines"),
    ("continuation", "target", "Continuation Predictor", "rhu.continuation"),
    ("map", "map", "Municipality Map", "rhu.municipality_map"),
    ("inventory", "package", "Vaccine Inventory", "rhu.inventory"),
    ("barangay_stock", "refresh", "Barangay Stock", "rhu.barangay_stock"),
    ("requests", "clipboard", "Vaccine Requests", "rhu.requests"),
    ("forecast", "activity", "Vaccine Forecast", "rhu.forecast"),
    ("reports", "file-text", "Reports", "rhu.reports"),
]
BHW_NAV = [
    ("dashboard", "home", "Dashboard", "bhw.dashboard"),
    ("children", "baby", "Child Records", "bhw.children"),
    ("vaccine_gaps", "syringe", "Vaccine Gaps", "bhw.vaccine_gaps"),
    ("due_list", "calendar", "Vaccination Due List", "bhw.due_vaccines"),
    ("continuation", "target", "Continuation Predictor", "bhw.continuation"),
    ("requests", "clipboard", "Vaccine Requests", "bhw.requests"),
    ("stock", "package", "My Stock", "bhw.stock"),
    ("reports", "file-text", "Reports", "bhw.reports"),
]
ADMIN_NAV = [
    ("dashboard", "home", "Dashboard", "admin.dashboard"),
    ("users", "users", "User Management", "admin.users"),
    ("assign", "map-pin", "Assign Barangay", "admin.assign_barangay"),
    ("campaigns", "calendar", "Campaigns", "admin.campaigns"),
    ("map", "map", "Municipality Map", "admin.municipality_map"),
    ("vaccines", "package", "Vaccine Alerts", "admin.vaccine_alerts"),
]
SUPERADMIN_NAV = [
    ("logs", "history", "Activity Logs", "superadmin.logs"),
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

def _cache(key, builder):  # page cache
    if not has_request_context():  # outside a page
        return builder()
    store = getattr(g, "_iv_cache", None)  # page storage
    if store is None:
        store = g._iv_cache = {}
    if key not in store:  # first time
        store[key] = builder()  # load once
    return store[key]  # reuse


def _vaccine_types_by_id():  # vaccine types
    return _cache("vaccine_types", lambda: {vt.id: vt for vt in VaccineType.query.all()})  # by id


def _records_by_child():  # doses by child
    def build():
        out = defaultdict(list)
        for record in VaccinationRecord.query.all():  # all records
            out[record.child_id].append(record)  # group by child
        return out
    return _cache("records", build)  # cached


def _latest_risk_by_child():  # newest risk
    """Newest RiskAssessment per child (matches risk_assessments.first(),
    which orders by computed_at descending)."""
    def build():
        out = {}
        for ra in RiskAssessment.query.order_by(  # oldest first
            RiskAssessment.computed_at.asc(), RiskAssessment.id.asc()
        ).all():
            out[ra.child_id] = ra  # later rows overwrite earlier -> newest wins
        return out
    return _cache("risk", build)  # cached


def _latest_risk(child):  # one child's risk
    return _latest_risk_by_child().get(child.id)


class _RiskAsOf:  # past risk
    """Stands in for a RiskAssessment row when risk is worked out for a past
    date: same risk_label / risk_probability attributes the callers read."""
    __slots__ = ("risk_label", "risk_probability")

    def __init__(self, probability):
        self.risk_probability = probability
        self.risk_label = "At-Risk" if probability >= dp.RISK_THRESHOLD else "Not At-Risk"  # 0.50 cutoff


def risk_as_of(as_of):  # risk on a date
    """{child_id: risk} for the dashboards' period filter.

    No period -> each child's latest stored assessment (live figures).
    A period -> every child born by `as_of` re-scored as they stood on that
    date (doses after it ignored, age measured at it), so the risk cards
    describe the same moment as the coverage figures beside them. Nothing
    is written to the database."""
    if as_of is None:  # today
        return _latest_risk_by_child()  # saved scores

    def build():
        vt_code = {vt.id: vt.code for vt in _vaccine_types_by_id().values()}
        names = {b.id: b.name for b in _cache("all_barangays", lambda: Barangay.query.all())}
        records = _records_by_child()
        items = []
        for c in _cache("children:None", lambda: _children_query(None).all()):  # each child
            if c.date_of_birth > as_of:  # not born yet
                continue
            doses = {}
            for rec in records.get(c.id, ()):  # doses
                code = vt_code.get(rec.vaccine_type_id)
                if code and (rec.campaign_id is None or code not in doses):
                    doses[code] = rec.date_administered
            items.append((c.id, c.date_of_birth, c.date_registered or c.date_of_birth, c.sex,
                          doses, names.get(c.barangay_id, "")))
        return {cid: _RiskAsOf(p) for cid, p in dp.score_children_as_of(items, as_of).items()}  # rescore
    return _cache(f"risk_as_of:{as_of.isoformat()}", build)  # cached


def _doses_given(child_id):  # child's doses
    """{vaccine_type_id: date_administered or None} for one child."""
    def build():
        out = defaultdict(dict)
        for cid, records in _records_by_child().items():
            for record in records:
                out[cid][record.vaccine_type_id] = record.date_administered
        return out
    return _cache("doses", build).get(child_id, {})  # cached


def _dose_matrix(children, vt_lookup):  # dose grid
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


def _risk_labels(children):  # risk labels
    """{child_id: risk_label or None} for list templates."""
    risk = _latest_risk_by_child()
    return {c.id: (risk[c.id].risk_label if c.id in risk else None) for c in children}


def _still_catchable(vt, age):  # catch-up check
    """True unless this dose has its own catch-up deadline (BCG, Rotavirus)
    and the child is past it - at which point it drops out of due/missed
    entirely rather than sitting as permanently "overdue"."""
    return vt.catch_up_deadline_days is None or age <= vt.catch_up_deadline_days  # still allowed


def _due_and_missed(child, with_codes=False):  # due + missed
    """Doses the child is old enough to have received, and how many are unfilled.

    With with_codes, also returns the dose codes that are unfilled, so the
    diseases those doses protect against can be named."""
    vt_by_id = _vaccine_types_by_id()
    age = child.age_in_days  # age
    due = missed = 0
    codes = []
    for record in _records_by_child().get(child.id, ()):  # each dose
        vt = vt_by_id.get(record.vaccine_type_id)
        if vt is not None and not vt.is_optional and vt.recommended_age_days <= age and _still_catchable(vt, age):  # due now
            due += 1  # due
            if record.date_administered is None:  # not given
                missed += 1  # missed
                codes.append(vt.code)
    return (due, missed, codes) if with_codes else (due, missed)


def _due_codes(child):  # due codes
    """(due_codes, missed_codes) for one child - every core dose whose
    recommended age has passed, split into given vs missing. Used to weigh
    severity against what this specific child could possibly be missing,
    not the full 15-dose schedule."""
    vt_by_id = _vaccine_types_by_id()
    age = child.age_in_days
    due_codes, missed_codes = [], []
    for record in _records_by_child().get(child.id, ()):
        vt = vt_by_id.get(record.vaccine_type_id)
        if vt is not None and not vt.is_optional and vt.recommended_age_days <= age and _still_catchable(vt, age):
            due_codes.append(vt.code)
            if record.date_administered is None:
                missed_codes.append(vt.code)
    return due_codes, missed_codes


def _is_fully_immunized(child):  # 90% rule (today)
    due, missed = _due_and_missed(child)  # count
    return True if due == 0 else ((due - missed) / due >= 0.9)  # 90% check


def _target_coverage(fully_immunized, target_population):  # target coverage
    """Coverage against the DOH/PSA projected target population, as opposed
    to coverage against actual registered children. Only meaningful once an
    admin has entered a target for that barangay; None otherwise so the
    template can show "not set" rather than a misleading 0%."""
    if not target_population or target_population <= 0:
        return None
    return round(fully_immunized / target_population * 100, 1)  # vs target


def dashboard_stats(barangay_id=None, max_age_days=None):  # stat cards
    children = _children_query(barangay_id).all()  # children
    if max_age_days is not None:
        children = [c for c in children if c.age_in_days <= max_age_days]
    total = len(children)  # total
    fully = sum(1 for c in children if _is_fully_immunized(c))  # fully immunized
    risk = _latest_risk_by_child()
    at_risk = sum(1 for c in children  # at-risk
                  if (ra := risk.get(c.id)) and ra.risk_label == "At-Risk")
    return {
        "total_children": total, "fully_immunized": fully, "at_risk": at_risk,
        "coverage_rate": round((fully / total) * 100, 1) if total else 0.0,  # coverage %
    }


def _risk_tier_counts(children, risk):  # tier counts
    """High/Medium/Low tier counts for a list of children, same tiering
    risk_distribution() uses, so a barangay row and the donut chart never
    disagree on how a child is banded."""
    counts = Counter()
    for c in children:
        latest = risk.get(c.id)
        counts[dp.risk_tier(latest.risk_probability) if latest else "Low Risk"] += 1  # high/med/low
    return {"high": counts["High Risk"], "medium": counts["Medium Risk"], "low": counts["Low Risk"]}


def coverage_by_barangay():  # per barangay
    by_barangay = defaultdict(list)
    for c in Child.query.all():  # all children
        by_barangay[c.barangay_id].append(c)
    risk = _latest_risk_by_child()  # risk
    rows = []
    for b in Barangay.query.order_by(Barangay.name).all():  # each barangay
        children = by_barangay.get(b.id, [])
        total = len(children)
        fully = sum(1 for c in children if _is_fully_immunized(c))  # fully immunized
        at_risk = sum(1 for c in children  # at-risk
                      if (ra := risk.get(c.id)) and ra.risk_label == "At-Risk")
        rows.append({"barangay": b.name, "barangay_id": b.id, "children": total,
                      "fully_immunized": fully, "at_risk": at_risk,
                      "coverage": round((fully / total) * 100, 1) if total else 0.0,  # coverage %
                      "target_population": b.target_population,
                      "target_coverage": _target_coverage(fully, b.target_population),
                      "tiers": _risk_tier_counts(children, risk)})
    rows.sort(key=lambda r: r["coverage"], reverse=True)  # highest first
    return rows


def campaign_reach(campaign):  # campaign reach
    """Eligible-vs-received counts per barangay for one campaign.

    Eligibility is evaluated against the campaign's own active window (did
    the child's age overlap the target range at any point between
    start_date and end_date), not "today" - a child who was eligible during
    the campaign but has since aged out, or one too young at the time who
    has since aged in, still counts (or doesn't) the same way regardless of
    when this report is viewed."""
    scope = campaign.barangays.all() or Barangay.query.order_by(Barangay.name).all()
    received_child_ids = {
        r.child_id for r in VaccinationRecord.query.filter_by(campaign_id=campaign.id)
        .filter(VaccinationRecord.date_administered.isnot(None)).all()
    }
    rows = []
    totals = {"eligible": 0, "received": 0}
    for b in scope:
        eligible_ids = [
            c.id for c in Child.query.filter_by(barangay_id=b.id).all()
            if (campaign.end_date - c.date_of_birth).days >= campaign.min_age_days
            and (campaign.start_date - c.date_of_birth).days <= campaign.max_age_days
        ]
        received = sum(1 for cid in eligible_ids if cid in received_child_ids)
        rows.append({"barangay": b.name, "eligible": len(eligible_ids), "received": received,
                      "pct": round(received / len(eligible_ids) * 100, 1) if eligible_ids else 0.0})
        totals["eligible"] += len(eligible_ids)
        totals["received"] += received
    rows.sort(key=lambda r: r["pct"])
    totals["pct"] = round(totals["received"] / totals["eligible"] * 100, 1) if totals["eligible"] else 0.0
    return rows, totals


def active_campaigns_summary(barangay_id=None):  # active campaigns
    """One reach total per campaign that's inside its active window right
    now, for the dashboard panel - the full per-barangay breakdown stays on
    the campaign's own detail page rather than being duplicated here.

    With barangay_id set (the BHW dashboard), each total is narrowed to that
    one barangay's row from campaign_reach(), and a campaign out of scope or
    with nobody eligible there is left out rather than shown as a zero."""
    today = date.today()
    campaigns = (Campaign.query.filter(Campaign.start_date <= today, Campaign.end_date >= today)
                 .order_by(Campaign.end_date).all())
    barangay_name = db.session.get(Barangay, barangay_id).name if barangay_id else None
    out = []
    for c in campaigns:
        if barangay_id is not None and not c.barangay_eligible(barangay_id):
            continue
        rows, totals = campaign_reach(c)
        if barangay_id is not None:
            row = next((r for r in rows if r["barangay"] == barangay_name), None)
            if row is None or row["eligible"] == 0:
                continue
            eligible, received, pct = row["eligible"], row["received"], row["pct"]
        else:
            eligible, received, pct = totals["eligible"], totals["received"], totals["pct"]
        out.append({"id": c.id, "name": c.name, "vaccine": c.vaccine_type.name,
                     "eligible": eligible, "received": received, "pct": pct,
                     "days_left": (c.end_date - today).days})
    return out


def eligible_children_for_campaign(campaign, barangay_id=None):  # campaign kids
    """Children eligible for this campaign right now, by today's age and
    barangay - the same test eligible_campaigns_for_child() applies from the
    child's side, so a row here is exactly a child whose Edit Child form
    would offer this campaign's dose field. Used by the campaign worklist,
    a focused alternative to opening each child's full record. Distinct from
    campaign_reach(), which counts anyone eligible at any point during the
    campaign's window for historical reach reporting."""
    q = Child.query
    if barangay_id is not None:
        q = q.filter_by(barangay_id=barangay_id)
    elif campaign.barangays.count():
        q = q.filter(Child.barangay_id.in_([b.id for b in campaign.barangays.all()]))
    kids = [c for c in q.all() if campaign.age_eligible(c.age_in_days)]
    kids.sort(key=lambda c: c.full_name)
    return kids


def coverage_by_vaccine(barangay_id=None):  # per dose
    children = _children_query(barangay_id).all()  # children
    vt_by_code = {vt.code: vt for vt in _vaccine_types_by_id().values()}
    # (child_id, vaccine_type_id) -> record, built once from the cached records
    given_on = {}
    for child_id, records in _records_by_child().items():  # dose dates
        for record in records:
            given_on[(child_id, record.vaccine_type_id)] = record.date_administered

    rows = []
    for code, name, antigen_code, dose_no, rec_days in dp.VACCINE_SCHEDULE:  # each dose
        vt = vt_by_code.get(code)
        if not vt:
            continue
        eligible = [c for c in children if c.age_in_days >= rec_days]  # old enough
        if not eligible:
            rows.append({"vaccine": name, "pct": 0})
            continue
        given = sum(1 for c in eligible if given_on.get((c.id, vt.id)) is not None)  # received
        rows.append({"vaccine": name, "pct": round((given / len(eligible)) * 100, 1)})  # dose %
    return rows


# Colour per antigen for the coverage-by-vaccine chart. Deliberately avoids
# green/amber/red: those carry coverage-status meaning everywhere else in the
# system, and a red bar here would read as "failing" when it only means "IPV".
ANTIGEN_COLOURS = {  # chart colors
    "BCG":   "#2563eb",
    "HEPB":  "#0ea5e9",
    "PENTA": "#0d9488",
    "OPV":   "#7c3aed",
    "PCV":   "#a855f7",
    "IPV":   "#4f46e5",
    "MMR":   "#0891b2",
}


def dose_age_label(days):  # age label
    """Short age label for a dose, e.g. 'birth', '6wk', '9mo'.

    Ages follow the DOH Expanded Program on Immunization schedule. Shown on the
    chart because the dose names alone do not reveal which doses share a visit -
    Penta 1st, OPV 1st and PCV 1st are all given at the same 6-week appointment."""
    if days <= 0:
        return "birth"
    if days < 180:
        return f"{round(days / 7)}wk"
    return f"{round(days / 30.44)}mo"


def schedule_labels(rows):  # chart labels
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

# Single source of truth in data_processor.py - also the ML model's
# training/serving window, so the Continuation Predictor's age scope and the
# model's own never drift apart. Caps at_risk_table() for the Continuation
# Predictor and its dashboard summaries - never for Vaccine Gaps or the
# At-Risk Children report, which stay a complete registry at any age.
CONTINUATION_WINDOW_DAYS = dp.CONTINUATION_WINDOW_DAYS


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


def chart_insights(by_vaccine, by_barangay, trend, trend_year=None):
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
        lede = (
            f"Across the {len(real)} barangays, "
            f"{'all' if behind == len(real) else behind} are below the "
            f"{COVERAGE_TARGET}% target."
        )
        if worst["barangay"] == lowest["barangay"]:
            out["barangay"] = (
                f"{lede} {lowest['barangay']} is worst on both counts: the lowest rate, "
                f"at {lowest['coverage']}%, and the largest gap, about "
                f"{round(shortfall(worst))} of its {worst['children']} children short of target."
            )
        else:
            out["barangay"] = (
                f"{lede} {lowest['barangay']} has the lowest rate at {lowest['coverage']}%, "
                f"but {worst['barangay']} carries the largest gap, about "
                f"{round(shortfall(worst))} of its {worst['children']} children short of target."
            )

    if trend_year:
        # Monthly cohort series: early months hold only a few births, so the
        # sentence uses the months with enough children to mean something.
        trend = [t for t in trend if t[1] is not None and t[2] >= TREND_MIN_COHORT]
    if len(trend) >= 2:
        first_y, first_p, _ = trend[0]
        last_y, last_p, last_n = trend[-1]
        prev_p = trend[-2][1]
        step = round(last_p - prev_p, 1)
        direction = "rose" if step > 0 else "fell" if step < 0 else "held steady"
        change = "up" if last_p >= first_p else "down"
        if trend_year:
            out["trend"] = (
                f"For children born in {trend_year}, coverage went {change} from {first_p}% at the end of "
                f"{first_y} to {last_p}% at the end of {last_y.replace(' (to date)', '')} "
                f"({last_n} children), and {direction} {abs(step)} points in the last month shown."
            )
        else:
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
    a group - Pentavalent holding steady across three doses reads differently
    from OPV declining across the same three visits."""
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


# Vaccines the RHU keeps in stock: the 7 routine antigens plus Rotavirus,
# which BHWs administer and record (an optional dose) and so must be able to
# request and the RHU must be able to stock. Campaign-only vaccines (e.g. MR)
# are not included.
_CORE_ANTIGEN_CODES = ({code for code, _name in dp.VACCINE_ANTIGENS}
                       | {code for code, _name in dp.OPTIONAL_VACCINE_ANTIGENS})


def core_antigens():  # tracked vaccines
    """Antigens in the RHU's inventory and request system: the 7 routine
    vaccines and Rotavirus. Used by Vaccine Inventory, stock status and
    alerts, the BHW request form, warning levels and the allocation view."""
    order = [code for code, _ in dp.VACCINE_ANTIGENS + dp.OPTIONAL_VACCINE_ANTIGENS]
    rows = [a for a in VaccineAntigen.query.all() if a.code in _CORE_ANTIGEN_CODES]  # 8 vaccines
    return sorted(rows, key=lambda a: order.index(a.code))  # schedule order


def stock_levels():  # stock status
    """Stock position per antigen. Extracted from the inventory route so the
    dashboard panel and the Vaccine Inventory page apply identical thresholds
    and can never disagree about what counts as Low or Critical."""
    rows = []
    for a in core_antigens():  # each vaccine
        batches = InventoryBatch.query.filter_by(antigen_id=a.id, is_archived=False).all()  # active batches
        total = sum(b.quantity_on_hand for b in batches)  # total vials
        doses = sum(b.quantity_on_hand * batch_doses_per_vial(b) for b in batches)  # same stock in doses
        reorder = min([b.reorder_level for b in batches], default=50)  # reorder level
        expiring = sum(1 for b in batches if b.is_expiring_soon)  # expiring
        if total <= 0:  # none
            status, color = "Out of Stock", "red"
        elif total <= reorder * 0.5:  # half or less
            status, color = "Critical", "red"
        elif total <= reorder:  # below reorder
            status, color = "Low", "amber"
        else:  # enough
            status, color = "Enough", "green"
        rows.append({"antigen": a, "total": total, "doses": doses, "reorder": reorder, "status": status,
                     "color": color, "expiring": expiring, "batches": batches,
                     "pct": min(100, round((total / (reorder * 2)) * 100)) if reorder else 100})
    return rows


def stock_alerts(limit=5):  # low-stock panel
    """Antigens at or below their reorder level, lowest first.

    The Critical Alerts strip only fires at exactly zero, so a vaccine down to
    its last few vials was invisible until it had already run out."""
    rows = [r for r in stock_levels() if r["status"] != "Enough"]  # not enough
    rows.sort(key=lambda r: (r["total"], r["antigen"].name))  # lowest first
    return rows[:limit]


# A year needs at least this many registered children before its coverage rate
# is meaningful enough to plot.
TREND_MIN_COHORT = 20  # min 20 children


def yearly_trend(barangay_id=None):  # yearly trend
    """Coverage at the end of each year, computed from actual dose dates:
    coverage as at 31 December of each year the data covers, using the same
    point-in-time calculation as the Reports page. The last point is capped
    at today."""
    today = date.today()
    out = []
    for y in sorted(data_years()):  # each year
        end = min(date(y, 12, 31), today)  # Dec 31
        fig = report_figures(barangay_id, end)  # year figures
        # A year with only a handful of children registered yields a rate that
        # is noise, not a finding: 2019 held one child, so its 0% made the
        # series look like a rise from nothing. Those years are omitted rather
        # than plotted.
        if fig["total_children"] >= TREND_MIN_COHORT:  # big enough
            label = str(y) + (" (to date)" if end == today and y == today.year else "")
            out.append((label, fig["coverage_rate"], fig["total_children"]))  # point
    return out


def monthly_cohort_trend(barangay_id, year, risk=None):  # monthly trend
    """Coverage at the end of each month of `year` for the children born in
    that year - the dashboards' trend chart when a single birth year is
    selected. Same calculation as the stat cards (report_figures() with the
    cohort window), so the December point, or the selected month's point,
    equals the card. Months after today are left out; a month with no
    children yet is plotted as a gap (None)."""
    today = date.today()
    born_from = date(year, 1, 1)  # birth year
    out = []
    for m in range(1, 13):  # each month
        first = date(year, m, 1)
        if first > today:  # future
            break
        end = min((date(year + 1, 1, 1) if m == 12 else date(year, m + 1, 1)) - timedelta(days=1), today)  # month end
        fig = report_figures(barangay_id, end, born_from, risk if risk is not None else {})  # month figures
        n = fig["total_children"]
        out.append((first.strftime("%b") + (" (to date)" if end == today else ""),  # point
                    fig["coverage_rate"] if n else None, n))
    return out


def trend_for(barangay_id, period):  # pick trend
    """The dashboards' trend chart: monthly for a single selected birth year,
    annual (all children, every year) otherwise."""
    if period.get("month_active"):  # one year?
        year = int(period["year_from"])
        trend = monthly_cohort_trend(barangay_id, year)  # monthly
        month = int(period["month"]) if period.get("month") else 0
        return {"trend": trend, "trend_monthly": True, "trend_year": year,
                "trend_highlight": month - 1 if 0 < month <= len(trend) else -1,
                "trend_min": TREND_MIN_COHORT,
                "trend_faded": any(n and n < TREND_MIN_COHORT for _l, _v, n in trend)}
    return {"trend": yearly_trend(barangay_id), "trend_monthly": False, "trend_year": None,  # yearly
            "trend_highlight": -1, "trend_min": TREND_MIN_COHORT, "trend_faded": False}


def monthly_doses_administered(barangay_id=None, months=12, as_of=None):  # doses per month
    """Real count of vaccine doses actually administered per calendar month,
    for the `months` months ending at as_of (oldest first; defaults to the
    12 months up to today).

    A rate (coverage %) gets noisy at monthly/per-barangay granularity --
    that is exactly why yearly_trend() above has to drop small cohorts. A
    raw count has no denominator to get noisy, so it stays meaningful even
    for one small barangay in one month. Counts every VaccinationRecord
    with a recorded date_administered, core and optional doses alike.

    Takes as_of so this tracks whatever Period/Year is selected on the
    Reports page, rather than always showing the 12 months up to today
    regardless of what historical snapshot is being viewed."""
    as_of = as_of or date.today()
    y, m = as_of.year, as_of.month
    window = []
    for _ in range(months):
        window.append((y, m))
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    window.reverse()
    start = date(window[0][0], window[0][1], 1)

    q = db.session.query(VaccinationRecord.date_administered).join(  # dated doses
        Child, VaccinationRecord.child_id == Child.id
    ).filter(
        VaccinationRecord.date_administered.isnot(None),
        VaccinationRecord.date_administered >= start,
        VaccinationRecord.date_administered <= as_of,
    )
    if barangay_id:
        q = q.filter(Child.barangay_id == barangay_id)

    counts = Counter()
    for (d,) in q.all():  # count
        counts[(d.year, d.month)] += 1
    return [(date(y, m, 1).strftime("%b %Y"), counts.get((y, m), 0)) for y, m in window]


def risk_distribution(barangay_id=None, max_age_days=None):  # risk mix
    counts = Counter()
    risk = _latest_risk_by_child()
    for c in _children_query(barangay_id).all():
        if max_age_days is not None and c.age_in_days > max_age_days:
            continue
        latest = risk.get(c.id)
        counts[dp.risk_tier(latest.risk_probability) if latest else "Low Risk"] += 1  # tier
    total = sum(counts.values()) or 1
    return {
        "high_pct": round(counts["High Risk"] / total * 100, 1), "high_count": counts["High Risk"],
        "medium_pct": round(counts["Medium Risk"] / total * 100, 1), "medium_count": counts["Medium Risk"],
        "low_pct": round(counts["Low Risk"] / total * 100, 1), "low_count": counts["Low Risk"],
    }


# --- Dashboard filtering ----------------------------------------------------
# A BHW works a single list of children, so filtering by risk level and by
# immunization status is genuinely useful there. Both reuse the per-request
# caches, so filtering costs no extra queries.

RISK_LEVELS = [("", "Risk Level: All"), ("At-Risk", "At-Risk only"), ("Not At-Risk", "Not At-Risk only")]  # risk filter
STATUS_LEVELS = [("", "Status: All"), ("complete", "Fully immunized"), ("incomplete", "Incomplete")]  # status filter


def immunized_as_of(child, as_of=None):  # 90% rule
    """Fully immunized as at a date. Doses recorded after `as_of` count as not
    yet given. Shared with report_figures() so the dashboard and the Reports
    page never disagree about the same month."""
    if as_of is None:  # today
        return _is_fully_immunized(child)  # live check
    vt_by_id = _vaccine_types_by_id()
    age = (as_of - child.date_of_birth).days  # age then
    due = given = 0
    for rec in _records_by_child().get(child.id, ()):  # each dose
        vtype = vt_by_id.get(rec.vaccine_type_id)
        # Same rule as report_figures(): optional doses and doses past their
        # catch-up age are not counted as due.
        if (vtype is not None and not vtype.is_optional and vtype.recommended_age_days <= age  # due then
                and _still_catchable(vtype, age)):
            due += 1  # due
            if rec.date_administered is not None and rec.date_administered <= as_of:  # given by then
                given += 1  # given
    return True if due == 0 else (given / due >= 0.9)  # 90% check


def filtered_children(barangay_id=None, risk_level="", status="", as_of=None, born_from=None,  # pick children
                      risk=None):
    risk = risk if risk is not None else _latest_risk_by_child()  # risk
    kids = _children_query(barangay_id).all()  # barangay
    if as_of is not None:
        # A child not yet born at the reporting date is not in scope.
        kids = [c for c in kids if c.date_of_birth <= as_of]  # born by then
    if born_from is not None:
        kids = [c for c in kids if c.date_of_birth >= born_from]  # birth-year range
    if risk_level in ("At-Risk", "Not At-Risk"):  # risk filter
        kids = [c for c in kids if (ra := risk.get(c.id)) and ra.risk_label == risk_level]
    if status == "complete":  # fully only
        kids = [c for c in kids if immunized_as_of(c, as_of)]
    elif status == "incomplete":  # incomplete only
        kids = [c for c in kids if not immunized_as_of(c, as_of)]
    return kids


def _filter_child_list(kids, risk_level="", status=""):  # list filters
    """Risk and status filters for the Child Records pages. "High Risk" means a
    risk score of 70% or higher, the same rule as the dashboard's High Risk card."""
    if risk_level == "High Risk":
        kids = [c for c in kids if (ra := _latest_risk(c)) and dp.risk_tier(ra.risk_probability) == "High Risk"]
    elif risk_level:
        kids = [c for c in kids if (ra := _latest_risk(c)) and ra.risk_label == risk_level]
    if status == "complete":
        kids = [c for c in kids if immunized_as_of(c)]
    elif status == "incomplete":
        kids = [c for c in kids if not immunized_as_of(c)]
    return kids


def stats_for(children, as_of=None, risk=None):  # count stats
    """Same shape as dashboard_stats(), over an explicit child list.

    at_risk uses `risk` when given (the dashboards pass risk_as_of() for a
    selected period), otherwise each child's latest stored assessment."""
    risk = risk if risk is not None else _latest_risk_by_child()
    total = len(children)  # total
    fully = sum(1 for c in children if immunized_as_of(c, as_of))  # fully immunized
    at_risk = sum(1 for c in children  # at-risk
                  if (ra := risk.get(c.id)) and ra.risk_label == "At-Risk")
    return {"total_children": total, "fully_immunized": fully, "at_risk": at_risk,
            "coverage_rate": round(fully / total * 100, 1) if total else 0.0}  # coverage %


def risk_distribution_for(children, risk=None):  # risk tiers
    counts = Counter()
    risk = risk if risk is not None else _latest_risk_by_child()
    for c in children:
        latest = risk.get(c.id)
        counts[dp.risk_tier(latest.risk_probability) if latest else "Low Risk"] += 1  # tier
    total = sum(counts.values()) or 1
    return {
        "high_pct": round(counts["High Risk"] / total * 100, 1), "high_count": counts["High Risk"],
        "medium_pct": round(counts["Medium Risk"] / total * 100, 1), "medium_count": counts["Medium Risk"],
        "low_pct": round(counts["Low Risk"] / total * 100, 1), "low_count": counts["Low Risk"],
    }


def at_risk_table(barangay_id=None, limit=None, max_age_days=None):  # predictor list
    """max_age_days scopes this to the Continuation Predictor's "early
    warning during the active primary series" framing (the series itself
    runs 0-365 days; the reverted ML model was trained the same way, on
    60-300-day snapshots). Omitted entirely for Vaccine Gaps and the At-Risk
    Children report, which are meant to be a complete current registry
    regardless of age, not a prediction."""
    rows = []
    risk = _latest_risk_by_child()
    records = _records_by_child()
    for c in _children_query(barangay_id).all():  # each child
        if max_age_days is not None and c.age_in_days > max_age_days:  # age limit
            continue
        latest = risk.get(c.id)
        if not latest or latest.risk_label != "At-Risk":  # at-risk only
            continue
        due, missed, missed_codes = _due_and_missed(c, with_codes=True)  # due + missed
        given_dates = [r.date_administered for r in records.get(c.id, ()) if r.date_administered]
        last_visit = max(given_dates) if given_dates else None  # last visit
        tier = dp.risk_tier(latest.risk_probability)  # tier
        rows.append({"child": c, "risk_score": round(latest.risk_probability * 100), "tier": tier,
                     "color": dp.risk_tier_color(tier), "missed": missed, "due": due, "last_visit": last_visit,
                     # What the missed doses would have protected against, so the
                     # consequence is visible rather than only the dose count.
                     "unprotected": dp.diseases_for_missed(missed_codes),
                     "recommendation": "Home Visit" if tier == "High Risk" else "Schedule Visit"})  # visit type
    rows.sort(key=lambda r: r["risk_score"], reverse=True)  # highest first
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------------------
# Vaccine Gaps' own score: deliberately separate from the
# RiskAssessment model above (used by the dashboard stat, Continuation
# Predictor, notifications, and the child record banner). That score blends
# six factors - visit timing, delay history, registration lag - to answer
# "how worrying is this child's overall pattern." This page asks a narrower,
# fully transparent question instead: "of this child's currently-due
# vaccines, how much protection is missing, weighted by how serious each one
# is to go without" (dp.ANTIGEN_SEVERITY - measles/pertussis weigh full,
# polio a bit less, individual-protection-only vaccines less still). No
# model, trained or rule-based blend, is needed for that - it's one number.
# ---------------------------------------------------------------------------

def vaccine_risk_tier(pct):  # gap tier
    if pct >= 60:  # urgent
        return "Visit Urgently", "red"
    if pct >= 25:  # soon
        return "Visit Soon", "amber"
    return "On Track", "green"  # on track


def _child_vaccine_severity(c, antigen_of, records, vt_lookup):  # gap score
    due_codes, missed_codes = _due_codes(c)  # due + missed
    max_severity = dp.weighted_missed(due_codes)  # max weight
    severity = dp.weighted_missed(missed_codes)  # missed weight
    pct = round(severity / max_severity * 100) if max_severity else 0  # % missing
    tier, color = vaccine_risk_tier(pct)  # tier
    given_dates = [r.date_administered for r in records.get(c.id, ()) if r.date_administered]
    return {"child": c, "risk_score": pct, "tier": tier, "color": color,
            "missed": len(missed_codes), "due": len(due_codes),
            "last_visit": max(given_dates) if given_dates else None,
            "missed_codes": missed_codes, "missed_antigens": {antigen_of.get(code) for code in missed_codes},
            "unprotected": dp.diseases_for_missed(missed_codes),
            "next_dose": _next_dose_for(c, vt_lookup),
            # Past the Continuation Predictor's first-year window, a missed
            # dose isn't "about to fall off track" anymore - it's a DOH
            # catch-up case, which is a different kind of visit.
            "is_catch_up": c.age_in_days > CONTINUATION_WINDOW_DAYS,
            "recommendation": "Home Visit" if tier == "Visit Urgently" else "Schedule Visit"}


def vaccine_risk_distribution(barangay_id=None):  # gap mix
    antigen_of = {code: ant for code, _n, ant, _d, _r in dp.VACCINE_SCHEDULE}
    records = _records_by_child()
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    counts = Counter()
    for c in _children_query(barangay_id).all():
        row = _child_vaccine_severity(c, antigen_of, records, vt_lookup)
        counts[row["tier"]] += 1
    total = sum(counts.values()) or 1
    return {
        "high_count": counts["Visit Urgently"], "high_pct": round(counts["Visit Urgently"] / total * 100, 1),
        "medium_count": counts["Visit Soon"], "medium_pct": round(counts["Visit Soon"] / total * 100, 1),
        "low_count": counts["On Track"], "low_pct": round(counts["On Track"] / total * 100, 1),
    }


def vaccine_risk_table(barangay_id=None, vaccine_code=None, limit=None):  # gaps list
    """vaccine_code filters to children missing that antigen specifically
    (e.g. "MMR" shows only children missing an MMR dose), rather than every
    child who needs a visit for any reason.

    Only children tiered "Visit Urgently" or "Visit Soon" appear here - this
    is the same tier vaccine_risk_distribution() counts into those two stat
    cards, and the page both numbers sit on is titled "Children Who Need a
    Visit". A child who merely has one missed-but-not-due dose (tier "On
    Track") used to still show up here, which meant the table could list
    several times more children than its own "need a visit" stat cards and
    subtitle claimed."""
    antigen_of = {code: ant for code, _n, ant, _d, _r in dp.VACCINE_SCHEDULE}
    records = _records_by_child()
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    rows = []
    for c in _children_query(barangay_id).all():  # each child
        row = _child_vaccine_severity(c, antigen_of, records, vt_lookup)
        if row["tier"] == "On Track":  # skip on track
            continue
        if vaccine_code and vaccine_code not in row["missed_antigens"]:  # vaccine filter
            continue
        rows.append(row)
    rows.sort(key=lambda r: r["risk_score"], reverse=True)  # worst first
    return rows[:limit] if limit else rows


def child_detail(child):  # child schedule
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
        elif not _still_catchable(vt, age_days):
            status, note = "past-window", "no longer given at this age"
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

    optional = []
    for code, name, antigen_code, dose_no, rec_days, deadline in dp.OPTIONAL_VACCINE_SCHEDULE:
        vt = vt_by_code.get(code)
        if not vt:
            continue
        when = given.get(vt.id)
        past_window = when is not None and (when - child.date_of_birth).days > deadline
        optional.append({"code": code, "name": name, "date": when, "past_window": past_window})

    given_dates = [d for d in given.values() if d]
    due, missed = _due_and_missed(child)
    history = (ActivityLog.query
               .filter(ActivityLog.action.contains(child.display_name))
               .order_by(ActivityLog.created_at.desc()).limit(6).all())

    # Every campaign dose this child has ever been given - not just the
    # currently-eligible ones the Edit form offers, since a dose given under
    # a campaign that has since closed is still a real, permanent part of
    # this child's record.
    campaign_records = (VaccinationRecord.query.filter_by(child_id=child.id)
                         .filter(VaccinationRecord.campaign_id.isnot(None)).all())
    campaigns = [{"name": r.campaign.name, "vaccine": r.vaccine_type.name, "date": r.date_administered}
                 for r in campaign_records]

    return {
        "schedule": schedule,
        "optional": optional,
        "campaigns": campaigns,
        "given_count": sum(1 for r in schedule if r["status"] == "given"),
        "overdue_count": sum(1 for r in schedule if r["status"] == "overdue"),
        # A dose past its own catch-up window (BCG) and never given drops out
        # of the denominator too - it's no longer part of this child's
        # achievable schedule, not a permanent, uncompletable gap.
        "total_doses": sum(1 for r in schedule if r["status"] != "past-window"),
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


def resolve_range(start_year, end_year, month=""):  # birth-year filter
    """Turn the dashboards' From year / To year / Month selections into a
    birth-cohort window and a reporting date.

    The range selects children BORN from 1 January of the start year, and
    measures them as at 31 December of the end year (or the end of the chosen
    month), so "2020 to 2023" reports the 2020-2023 birth cohorts as they
    stood at the end of 2023. Month applies only when both years are the
    same, i.e. a single year is selected.

    Both blank      -> no filter (live figures).
    To only         -> every child born by the end of that year.
    From only       -> born from that year, measured as at today.
    From > To       -> swapped.
    Past today      -> clamped to today, so a period that has not ended yet
                       never shows an empty dashboard.
    Returns (born_from|None, as_of|None, label|None, clamped:bool, month_active:bool).
    """
    today = date.today()
    years = set(data_years())

    def year_or_none(v):
        try:
            y = int(v)
        except (TypeError, ValueError):
            return None
        return y if y in years else None

    y1, y2 = year_or_none(start_year), year_or_none(end_year)  # years
    if y1 is None and y2 is None:
        return None, None, None, False, False
    if y1 is not None and y2 is not None and y1 > y2:  # swap order
        y1, y2 = y2, y1
    try:
        m = int(month) if month else 0
    except (TypeError, ValueError):
        m = 0
    month_active = y1 is not None and y1 == y2  # same year?
    if not (month_active and 1 <= m <= 12):
        m = 0

    born_from = date(y1, 1, 1) if y1 is not None else None  # born from
    if y2 is None:
        as_of, label = today, f"{y1} to date"  # to date
    elif m:  # month end
        end = date(y2 + 1, 1, 1) if m == 12 else date(y2, m + 1, 1)
        as_of, label = end - timedelta(days=1), f"{MONTH_NAMES[m - 1]} {y2}"
    else:
        as_of = date(y2, 12, 31)  # Dec 31
        label = (f"{y2}" if y1 == y2 else f"{y1}–{y2}" if y1 is not None else f"Up to {y2}")

    clamped = as_of > today  # future
    if clamped:
        as_of, label = today, f"{label} (to date)"
    return born_from, as_of, label, clamped, month_active  # result


def _dashboard_period():  # read filter
    """Reads and resolves the dashboard period filter from the query string."""
    start, end, month = (request.args.get(k, "") for k in ("from", "to", "month"))
    born_from, as_of, label, clamped, month_active = resolve_range(start, end, month)  # resolve
    return {"born_from": born_from, "as_of": as_of, "period_label": label,
            "period_clamped": clamped, "month_active": month_active,
            "year_from": start, "year_to": end, "month": month if month_active else "",
            "months": MONTH_NAMES, "years": data_years()}


def period_end(period):  # month end
    """Last day of a 'YYYY-MM' period, never later than today."""
    today = date.today()
    if not period:
        return today
    try:
        y, m = (int(x) for x in period.split("-"))
        nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)  # next month
        return min(nxt - timedelta(days=1), today)  # last day
    except (ValueError, TypeError):
        return today


def report_date_options():  # dropdown options
    """Choices for the Reports page's single "Report as of" selector:
    the last 12 months, then whole years (the current one runs to today)."""
    this_year = date.today().year
    return {"months": month_options(),
            "years": [(str(y), f"{y} (to date)" if y == this_year else f"{y} (as of Dec 31)")
                      for y in data_years()]}


def resolve_report_date(args):  # report date
    """(as_of date, selected key) from the Reports query string.

    as_of="" -> today; "YYYY-MM" -> end of that month; "YYYY" -> 31 Dec of
    that year (today for the current year). Older links that used the
    separate period= / year= parameters still work."""
    key = args.get("as_of")  # as of
    if key is None:  # older links
        key = args.get("year") or args.get("period") or ""
    key = (key or "").strip()
    today = date.today()
    if len(key) == 4 and key.isdigit() and int(key) in data_years():  # a year
        return min(date(int(key), 12, 31), today), key  # Dec 31
    if key in {v for v, _ in month_options()}:  # a month
        return period_end(key), key  # month end
    return today, ""  # today


def report_figures(barangay_id=None, as_of=None, born_from=None, risk=None):  # report numbers
    """Coverage figures for one barangay (or all) as of a date.

    Doses recorded after `as_of` count as not-yet-given, and children born
    after it are excluded, so selecting an earlier period reports the
    situation as it stood then rather than today's totals."""
    as_of = as_of or date.today()  # date
    vt_by_id = _vaccine_types_by_id()  # vaccine types
    records = _records_by_child()  # doses
    risk = risk if risk is not None else _latest_risk_by_child()  # risk
    # Neither the barangay list nor the (pre-as_of-filter) children list
    # depends on `as_of`, but yearly_trend() calls this once per year of
    # data (~17 calls for a single dashboard render) - caching both per
    # request turns 17 redundant full-table queries into 1.
    all_barangays = _cache("all_barangays", lambda: Barangay.query.all())  # barangays
    names = {b.id: b.name for b in all_barangays}
    targets = {b.id: b.target_population for b in all_barangays}

    all_children = _cache(f"children:{barangay_id}", lambda: _children_query(barangay_id).all())  # children
    children = [c for c in all_children  # born by then
                if c.date_of_birth <= as_of and (born_from is None or c.date_of_birth >= born_from)]

    def immunized(child):  # 90% rule
        age = (as_of - child.date_of_birth).days  # age then
        due = given = 0
        for rec in records.get(child.id, ()):  # each dose
            vtype = vt_by_id.get(rec.vaccine_type_id)
            if (vtype is not None and not vtype.is_optional and vtype.recommended_age_days <= age  # due then
                    and _still_catchable(vtype, age)):
                due += 1  # due
                if rec.date_administered is not None and rec.date_administered <= as_of:  # given by then
                    given += 1
        return True if due == 0 else (given / due >= 0.9)  # 90% check

    grouped = defaultdict(list)  # by barangay
    for c in children:
        grouped[c.barangay_id].append(c)

    rows = []
    for bid, kids in grouped.items():  # each barangay
        fully = sum(1 for c in kids if immunized(c))  # fully immunized
        at_risk = sum(1 for c in kids if (ra := risk.get(c.id)) and ra.risk_label == "At-Risk")  # at-risk
        target_pop = targets.get(bid)
        rows.append({"barangay": names.get(bid, "—"), "barangay_id": bid, "children": len(kids),
                     "fully_immunized": fully, "at_risk": at_risk,
                     "coverage": round(fully / len(kids) * 100, 1) if kids else 0.0,  # coverage %
                     "target_population": target_pop,
                     "target_coverage": _target_coverage(fully, target_pop),
                     "tiers": _risk_tier_counts(kids, risk)})
    rows.sort(key=lambda r: r["coverage"], reverse=True)  # highest first

    total = len(children)  # total
    fully_total = sum(r["fully_immunized"] for r in rows)  # fully total
    at_risk_total = sum(r["at_risk"] for r in rows)  # at-risk total
    # Only counts barangays with a target actually entered -- summing in a
    # 0 for the rest would understate the real target population and make
    # the aggregate rate look better than it should.
    target_rows = [r for r in rows if r["target_population"]]
    target_population_total = sum(r["target_population"] for r in target_rows)
    return {
        "total_children": total,
        "fully_immunized": fully_total,
        "at_risk": at_risk_total,
        "coverage_rate": round(fully_total / total * 100, 1) if total else 0.0,  # coverage %
        "target_population_total": target_population_total,
        "target_coverage_rate": _target_coverage(
            sum(r["fully_immunized"] for r in target_rows), target_population_total),
        "target_barangays_set": len(target_rows),
        "by_barangay": rows,
        "as_of": as_of,
    }


def report_by_vaccine(barangay_id=None, as_of=None, born_from=None):  # per-dose report
    """Completion rate per scheduled dose as of a date, for the BHW report
    chart. Mirrors report_figures(): doses recorded after `as_of` count as
    not yet given, and children born after it are excluded."""
    as_of = as_of or date.today()
    vt_by_code = {vt.code: vt for vt in _vaccine_types_by_id().values()}
    given_on = {}
    for child_id, records in _records_by_child().items():
        for record in records:
            given_on[(child_id, record.vaccine_type_id)] = record.date_administered

    children = [c for c in _children_query(barangay_id).all()
                if c.date_of_birth <= as_of and (born_from is None or c.date_of_birth >= born_from)]
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

def get_notifications(role, barangay_id=None):  # alerts
    if role == "bhw" and not barangay_id:  # unassigned BHW: nothing municipality-wide
        return _apply_seen_state([{"category": "System", "severity": "warning", "icon": "map-pin", "icon_bg": "bg-orange",
                                   "title": "No Barangay Assigned",
                                   "body": "Ask the administrator to assign your barangay to start using ImmunoVision.",
                                   "time_label": "Action needed", "unread": True}], current_user)
    notes = []
    for antigen in core_antigens():  # each vaccine
        total = (db.session.query(func.coalesce(func.sum(InventoryBatch.quantity_on_hand), 0))  # stock total
                 .filter(InventoryBatch.antigen_id == antigen.id, InventoryBatch.is_archived.is_(False)).scalar())
        reorder = (db.session.query(func.min(InventoryBatch.reorder_level))
                   .filter(InventoryBatch.antigen_id == antigen.id, InventoryBatch.is_archived.is_(False)).scalar()) or 50
        if total <= 0:  # out of stock
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
        recent = datetime.utcnow() - timedelta(days=14)
        for t in StockTransfer.query.filter(StockTransfer.kind == "redistribution", StockTransfer.doses_reduced > 0,
                                            StockTransfer.created_at >= recent).all():
            auto_cancelled = t.status == "cancelled" and t.cancelled_by_id is None
            if t.status != "reserved" and not auto_cancelled:
                continue
            notes.append({"category": "Stock", "severity": "warning", "icon": "refresh", "icon_bg": "bg-orange",
                          "title": f"Redistribution {t.code} {'Cancelled' if auto_cancelled else 'Reduced'}",
                          "body": f"{t.from_barangay.name} used {t.doses_reduced} reserved {t.antigen.name} doses "
                                  f"({t.reduce_reason}). "
                                  + (f"{t.to_barangay.name} still needs a source." if auto_cancelled
                                     else f"{t.to_barangay.name} will now get {t.doses_sent} doses."),
                          "time_label": "Recently", "unread": True})
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

    notes.extend(barangay_stock_notes(role, barangay_id))

    notes.append({"category": "System", "severity": "info", "icon": "calendar", "icon_bg": "bg-blue",
                  "title": "Monthly Report Due",
                  "body": f"The {date.today().strftime('%B %Y')} monthly immunization report is due at month end.",
                  "time_label": "This week", "unread": False})

    # Respect this user's notification preferences from Settings. Stock
    # Alerts off hides both Stock and Expiry (both are inventory concerns);
    # At-Risk Alerts off hides the Risk category. Other categories (Request,
    # System) aren't covered by a toggle and always show.
    if not current_user.notify_stock:
        notes = [n for n in notes if n["category"] not in ("Stock", "Expiry")]
    if not current_user.notify_at_risk:
        notes = [n for n in notes if n["category"] != "Risk"]
    return _apply_seen_state(notes, current_user)


def admin_notifications():
    """Staffing, coverage, and account-governance alerts for the RHU
    Administrator - barangay staffing, vaccine stock policy, and municipal
    oversight, plus account provisioning and audit oversight all live under
    this one role."""
    notes = []
    today = date.today()
    users = User.query.all()

    unassigned = [u for u in users if u.role == "bhw" and not u.barangay_id]
    if unassigned:
        notes.append({"category": "Coverage", "severity": "critical", "icon": "map-pin", "icon_bg": "bg-red",
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

    disabled = [u for u in users if not u.is_active_flag]
    if disabled:
        notes.append({"category": "Accounts", "severity": "warning", "icon": "x-circle", "icon_bg": "bg-orange",
                      "title": f"{len(disabled)} Disabled Account{'s' if len(disabled) > 1 else ''}",
                      "body": "Disabled accounts cannot sign in: "
                              + ", ".join(u.full_name for u in disabled[:4])
                              + (f" and {len(disabled) - 4} more." if len(disabled) > 4 else "."),
                      "time_label": "Ongoing", "unread": True,
                      "url": url_for("admin.users")})

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
             and (today - to_local(u.last_login_at).date()).days > 30]
    if stale:
        notes.append({"category": "Accounts", "severity": "info", "icon": "clock", "icon_bg": "bg-purple",
                      "title": f"{len(stale)} Account{'s' if len(stale) > 1 else ''} Inactive 30+ Days",
                      "body": "No sign-in for over a month: "
                              + ", ".join(u.full_name for u in stale[:4])
                              + (f" and {len(stale) - 4} more." if len(stale) > 4 else "."),
                      "time_label": "Ongoing", "unread": False,
                      "url": url_for("admin.users")})

    reset_requests = [u for u in users if u.password_reset_requested_at]
    if reset_requests:
        notes.append({"category": "Accounts", "severity": "critical", "icon": "lock", "icon_bg": "bg-red",
                      "title": f"{len(reset_requests)} Password Reset Request{'s' if len(reset_requests) > 1 else ''}",
                      "body": "Waiting on a new password from you: "
                              + ", ".join(u.full_name for u in reset_requests[:4])
                              + (f" and {len(reset_requests) - 4} more." if len(reset_requests) > 4 else "."),
                      "time_label": "Action needed", "unread": True,
                      "url": url_for("admin.users")})

    return _apply_seen_state(notes, current_user)


def unread_count(notes):
    return sum(1 for n in notes if n.get("unread"))


def _apply_seen_state(notes, user):
    """Downgrades a note to read if its title was already showing the last
    time this user opened their notifications page. Notifications have no
    stored rows to flag read/unread against - they're regenerated from live
    system state on every request - so title is used as the stable key: an
    alert that's still here next visit reads as read, one that cleared and
    later recurred reads as unread again."""
    seen = set()
    if user.last_seen_notification_titles:
        try:
            seen = set(json.loads(user.last_seen_notification_titles))
        except (ValueError, TypeError):
            seen = set()
    for n in notes:
        if n["title"] in seen:
            n["unread"] = False
    return notes


def _mark_notifications_seen(notes, user):
    user.last_seen_notification_titles = json.dumps([n["title"] for n in notes])
    db.session.commit()


# ---------------------------------------------------------------------------
# Shared child create/predict helper
# ---------------------------------------------------------------------------

def _ticked_date(form, checkbox_field, date_field, fallback):
    """For a "given" checkbox paired with a date box that's only shown once
    ticked: the date box is still submitted even while hidden, so the
    checkbox - not the date box's presence - decides whether anything was
    given at all. Falls back to `fallback` (the registration date) if the
    date box was somehow left blank."""
    if not form.get(checkbox_field):
        return None
    date_str = form.get(date_field)
    return datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else fallback


def _parse_form_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date() if value else None
    except ValueError:
        return None


def _doses_before_birth(dob, dated_doses):
    """Names of the doses in `dated_doses` ((name, date|None) pairs) dated
    before the child's date of birth - a day-month reversal or typing error,
    since no vaccine can be given before birth. Checked on the server for
    every form that records a dose, so it holds even if the browser's date
    restrictions are bypassed."""
    if dob is None:
        return []
    return [name for name, given in dated_doses if given is not None and given < dob]


def _schedule_names():
    return {code: name for code, name, *_ in dp.VACCINE_SCHEDULE + dp.OPTIONAL_VACCINE_SCHEDULE}


def _before_birth_message(names):
    return ("A dose cannot be dated before the child's date of birth. Please check: "
            + ", ".join(names) + ". Nothing was saved.")


def _dose_date_error(dob, dated_doses):  # dose date check
    """Error message for dose dates that cannot be right - before the child
    was born, or after today (a dose is only recorded once given) - or None.
    Checked on the server for every form that records a dose, so it holds
    even if the browser's date limits are bypassed."""
    bad = _doses_before_birth(dob, dated_doses)  # before birth
    if bad:
        return _before_birth_message(bad)
    today = date.today()
    future = [name for name, given in dated_doses if given is not None and given > today]  # future
    if future:
        return ("A dose cannot be dated after today. Please check: "
                + ", ".join(future) + ". Nothing was saved.")
    return None  # ok


def _schedule_error(dob, dates_by_code, check_codes):
    """Refuses core dose dates that break the DOH schedule (too young, out of
    order, or under 4 weeks after the previous dose of the same vaccine).
    Only newly entered or changed doses are checked. Message or None."""
    problems = dp.schedule_violations(dob, dates_by_code, set(check_codes))
    if not problems:
        return None
    return "Dose dates must follow the vaccination schedule. " + " ".join(problems) + " Nothing was saved."


NAME_PUNCTUATION = set(" .'-")
PH_MOBILE_RE = re.compile(r"(?:09|\+639)\d{9}")


def _name_error(value, label):
    """Letters (accents and Ñ included), spaces, periods, apostrophes and
    hyphens only, with at least one letter - no digits or other symbols."""
    if not any(ch.isalpha() for ch in value) or any(not (ch.isalpha() or ch in NAME_PUNCTUATION) for ch in value):
        return f"{label} can only contain letters, spaces, periods, apostrophes and hyphens."
    return None


def _contact_error(value):
    digits = re.sub(r"[\s-]", "", value)
    if not PH_MOBILE_RE.fullmatch(digits):
        return "Contact number must be an 11-digit mobile number, e.g. 09171234567 (or +639171234567)."
    return None


def _child_field_errors(form, child=None):
    """Field checks shared by Register Child and Edit Child, mirrored in the
    browser by app.js (data-validate). On the edit form a field is only
    checked when it was changed, so a record imported with an unusual name or
    number can still be saved without retyping it."""
    def changed(name, current):
        return child is None or (form.get(name, "").strip() != (current or "").strip())

    full_name = form.get("full_name", "").strip()
    if child is None and not full_name:
        return "Please enter the child's full name."
    if full_name and changed("full_name", child.full_name if child else None):
        err = _name_error(full_name, "Full name")
        if err:
            return err
    guardian = form.get("guardian_name", "").strip()
    if guardian and changed("guardian_name", child.guardian_name if child else None):
        err = _name_error(guardian, "Guardian name")
        if err:
            return err
    contact = form.get("guardian_contact", "").strip()
    if contact and changed("guardian_contact", child.guardian_contact if child else None):
        err = _contact_error(contact)
        if err:
            return err
    dob = _parse_form_date(form.get("date_of_birth")) or (child.date_of_birth if child else None)
    if dob and dob > date.today():
        return "Date of birth cannot be in the future."
    registered = _parse_form_date(form.get("date_registered"))
    if child is not None and dob == child.date_of_birth and registered == child.date_registered:
        registered = None  # neither date changed: an imported record is saved as it is
    if registered:
        if registered > date.today():
            return "Date of registration cannot be in the future."
        if dob and registered < dob:
            return "Date of registration cannot be before the date of birth."
    return None


def _create_form_errors(form):
    """Validation for the Register Child form, shared by RHU and BHW.
    Returns an error message, or None if the form can be saved."""
    dob = _parse_form_date(form.get("date_of_birth"))
    if dob is None:
        return "Please enter a valid date of birth."
    field_error = _child_field_errors(form)
    if field_error:
        return field_error
    names = _schedule_names()
    dated = [(names[code], _parse_form_date(form.get(f"dose_{code}")))
             for code, *_ in dp.VACCINE_SCHEDULE if form.get(f"dose_{code}_given")]
    new = {code: _parse_form_date(form.get(f"dose_{code}"))
           for code, *_ in dp.VACCINE_SCHEDULE if form.get(f"dose_{code}_given")}
    return _dose_date_error(dob, dated) or _schedule_error(dob, new, new.keys())


def _edit_form_errors(child, form):
    """Validation for the Edit Child form, against the date of birth being
    submitted (it may be corrected in the same save) and every routine,
    optional and campaign dose date on the form."""
    dob = _parse_form_date(form.get("date_of_birth")) if form.get("date_of_birth") else child.date_of_birth
    if dob is None:
        return "Please enter a valid date of birth."
    field_error = _child_field_errors(form, child)
    if field_error:
        return field_error
    names = _schedule_names()
    dated = [(names[code], _parse_form_date(form.get(f"dose_{code}"))) for code in _KNOWN_DOSE_CODES]
    dated += [(c.name, _parse_form_date(form.get(f"dose_campaign_{c.id}")))
              for c in eligible_campaigns_for_child(child)]
    # Schedule check: the dates as they will be saved, judging only the doses
    # whose date this save adds or changes.
    current = _core_dose_dates(child)
    submitted = {code: _parse_form_date(form.get(f"dose_{code}")) for code in dp.VACCINE_CODES}
    changed = [code for code in dp.VACCINE_CODES if submitted[code] != current.get(code)]
    return _dose_date_error(dob, dated) or _schedule_error(dob, submitted, changed)


def _core_dose_dates(child):
    """{dose code: date given} for the child's core schedule doses."""
    codes = {vt.id: vt.code for vt in VaccineType.query.all()}
    return {codes[r.vaccine_type_id]: r.date_administered for r in child.vaccination_records
            if r.campaign_id is None and r.date_administered and codes.get(r.vaccine_type_id) in dp.VACCINE_CODES}


# The core/optional dose codes the Edit Child form's generic per-code loop is
# allowed to touch. A Campaign's VaccineType (e.g. "MR") is a real row in the
# same table but is deliberately excluded here - it's recorded through its
# own dose_campaign_<id> field (see eligible_campaigns_for_child()), never
# by code, since the same vaccine can be reused across different campaign
# runs and a code-keyed lookup can't tell those apart.
_KNOWN_DOSE_CODES = [code for code, *_ in dp.VACCINE_SCHEDULE + dp.OPTIONAL_VACCINE_SCHEDULE]


def eligible_campaigns_for_child(child):
    """Active campaigns this child currently qualifies for, by age and
    barangay. Drives both what the Edit Child form shows/accepts and what
    the child's record displays - never risk scoring, which only ever reads
    dp.VACCINE_SCHEDULE codes regardless of what rows exist in the DB."""
    today = date.today()
    age = child.age_in_days
    candidates = Campaign.query.filter(Campaign.start_date <= today, Campaign.end_date >= today).all()
    return [c for c in candidates if c.age_eligible(age) and c.barangay_eligible(child.barangay_id)]


def _create_child(form, barangay_locked):  # new child
    barangay_id = barangay_locked or int(form.get("barangay_id"))
    dob = datetime.strptime(form.get("date_of_birth"), "%Y-%m-%d").date()  # birth date
    date_registered = (datetime.strptime(form.get("date_registered"), "%Y-%m-%d").date()
                        if form.get("date_registered") else date.today())
    child = Child(  # child row
        full_name=form.get("full_name"), sex=form.get("sex"),
        date_of_birth=dob,
        barangay_id=barangay_id, address=form.get("address"), guardian_name=form.get("guardian_name"),
        guardian_contact=form.get("guardian_contact"),
        date_registered=date_registered,
        # Ticking "Vitamin A given" with no exact date on hand records it as
        # of registration, same as the vaccine checkboxes below.
        vitamin_a_date=(_ticked_date(form, "vita_given", "vitamin_a_date", date_registered)),
        mnp_given=bool(form.get("mnp_given")),
        source="manual", created_by_id=current_user.id,
    )
    db.session.add(child)  # add
    db.session.flush()

    doses_dict = {}
    elsewhere = bool(form.get("given_elsewhere"))  # transferred in: doses given at another center
    for code, name, antigen_code, dose_no, rec_days in dp.VACCINE_SCHEDULE:  # each dose
        vt = VaccineType.query.filter_by(code=code).first()
        administered = _ticked_date(form, f"dose_{code}_given", f"dose_{code}", date_registered)  # ticked dose
        doses_dict[code] = administered
        rec = VaccinationRecord(  # dose row
            child_id=child.id, vaccine_type_id=vt.id, date_administered=administered,
            status="completed" if administered else "pending", administered_by_id=current_user.id,
        )
        db.session.add(rec)
        stock_on_dose_change(child, vt, rec, None, administered, elsewhere)  # barangay stock
    db.session.commit()  # save

    barangay_name = db.session.get(Barangay, barangay_id).name
    result = dp.predict_for_child(child.date_of_birth, child.date_registered, child.sex, doses_dict, barangay_name)  # predict risk
    db.session.add(RiskAssessment(  # save risk
        child_id=child.id, risk_label=result["label"], risk_probability=result["probability"],
        model_version=result["model_version"], top_factors=dp.serialize_factors(result["top_factors"]),
    ))
    db.session.commit()
    log_activity(f"Registered child {child_label(child)}")  # log
    return child  # done


def _update_child(child, form):  # edit child
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
    existing_by_vt_id = {rec.vaccine_type_id: rec for rec in child.vaccination_records if rec.campaign_id is None}
    past_window_warnings = []
    date_changes = []
    elsewhere = bool(form.get("given_elsewhere"))  # transferred in: new doses given at another center
    for code in _KNOWN_DOSE_CODES:  # each dose
        vt = vt_by_code.get(code)
        if vt is None:
            continue
        date_str = form.get(f"dose_{code}")
        administered = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None  # new date
        rec = existing_by_vt_id.get(vt.id)
        if rec is None:
            rec = VaccinationRecord(child_id=child.id, vaccine_type_id=vt.id)
            db.session.add(rec)
        if rec.date_administered is not None and rec.date_administered != administered:  # date changed
            date_changes.append((vt.name, rec.date_administered, administered))
        previous = rec.date_administered
        rec.date_administered = administered  # set date
        rec.status = "completed" if administered else "pending"
        stock_on_dose_change(child, vt, rec, previous, administered, elsewhere)  # barangay stock
        # Optional doses like Rotavirus stop being given past a real age
        # cutoff (rising intussusception risk in older infants), unlike the
        # core schedule which stays catchable indefinitely. A record past
        # that age is still saved as-given - it may be a real dose logged
        # late - but the BHW is warned rather than told nothing.
        if administered and vt.is_optional and vt.catch_up_deadline_days is not None:  # optional dose
            age_at_dose = (administered - child.date_of_birth).days
            if age_at_dose > vt.catch_up_deadline_days:  # too old
                past_window_warnings.append(vt.name)

    # Campaign doses: a separate field per campaign (dose_campaign_<id>), not
    # per vaccine code - the same vaccine can recur across campaign runs, and
    # existing_by_vt_id above only sees non-campaign records, so it can't be
    # reused here. Only ever written for campaigns this child currently
    # qualifies for, and never touches risk scoring (see
    # eligible_campaigns_for_child()'s docstring).
    for campaign in eligible_campaigns_for_child(child):  # campaign doses
        date_str = form.get(f"dose_campaign_{campaign.id}")
        administered = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
        rec = child.vaccination_records.filter_by(campaign_id=campaign.id).first()
        if rec is None:
            if administered is None:
                continue  # nothing to save - don't create an empty row for every eligible campaign
            rec = VaccinationRecord(child_id=child.id, vaccine_type_id=campaign.vaccine_type_id, campaign_id=campaign.id)
            db.session.add(rec)
        if rec.date_administered is not None and rec.date_administered != administered:
            date_changes.append((campaign.name, rec.date_administered, administered))
        previous = rec.date_administered
        rec.date_administered = administered
        rec.status = "completed" if administered else "pending"
        stock_on_dose_change(child, db.session.get(VaccineType, campaign.vaccine_type_id), rec, previous, administered,
                             elsewhere)
    db.session.commit()  # save

    result = _recompute_risk(child)  # re-predict
    log_activity(f"Updated child record {child_label(child)}")  # log
    _log_dose_date_changes(child, date_changes)  # log changes
    result["past_window_warnings"] = past_window_warnings
    return result


def _recompute_risk(child):  # re-predict
    vt_by_id = {vt.id: vt for vt in VaccineType.query.all()}
    doses_dict = {vt.code: None for vt in vt_by_id.values()}
    for rec in child.vaccination_records:  # child's doses
        vt = vt_by_id.get(rec.vaccine_type_id)
        if vt:
            doses_dict[vt.code] = rec.date_administered
    result = dp.predict_for_child(child.date_of_birth, child.date_registered, child.sex, doses_dict, child.barangay.name)  # predict
    db.session.add(RiskAssessment(  # save risk
        child_id=child.id, risk_label=result["label"], risk_probability=result["probability"],
        model_version=result["model_version"], top_factors=dp.serialize_factors(result["top_factors"]),
    ))
    db.session.commit()  # commit
    return result


def _log_dose_date_changes(child, changes):  # log date edits
    """One activity-log entry per changed or removed dose date, so an edit to
    an already-recorded dose is always traceable to who made it and when.
    changes: [(dose name, old date, new date|None)]. First-time recordings are
    not listed here; they are logged by the save itself."""
    for name, old, new in changes:  # each change
        if new is None:  # removed
            log_activity(f"Removed {name} (was {old.isoformat()}) for {child_label(child)}")
        else:  # changed
            log_activity(f"Changed {name} date for {child_label(child)} from "
                         f"{old.isoformat()} to {new.isoformat()}")


def _write_dose(child, vt, rec, administered, given_elsewhere=False):  # save one dose
    """Writes one dose onto a VaccinationRecord (creating it if it doesn't
    exist yet), same per-dose logic _update_child applies when saving the
    full Edit Child form. Returns the vaccine's name as a warning if an
    optional dose was recorded past its usual catch-up age, else None."""
    if rec is None:
        rec = VaccinationRecord(child_id=child.id, vaccine_type_id=vt.id)
        db.session.add(rec)
    previous = rec.date_administered
    rec.date_administered = administered
    rec.status = "completed" if administered else "pending"
    stock_on_dose_change(child, vt, rec, previous, administered, given_elsewhere)  # barangay stock
    if previous is not None and previous != administered:  # date changed
        _log_dose_date_changes(child, [(vt.name, previous, administered)])
    if administered and vt.is_optional and vt.catch_up_deadline_days is not None:
        age_at_dose = (administered - child.date_of_birth).days
        if age_at_dose > vt.catch_up_deadline_days:
            return vt.name
    return None


def _next_dose_for(child, vt_lookup):
    """Returns the next not-yet-administered, still-catchable dose in the
    schedule for one child (or None if every such dose has been given or has
    aged out), used by the Continuation Predictor list to show what's coming
    up next."""
    given = _doses_given(child.id)
    age = child.age_in_days
    for code, name, antigen, dose_no, rec_days in dp.VACCINE_SCHEDULE:
        vt = vt_lookup[code]
        if given.get(vt.id) is None and _still_catchable(vt, age):
            due_date = child.date_of_birth + timedelta(days=rec_days)
            return {"name": name, "due_date": due_date, "overdue": due_date < date.today()}
    return None


def _due_calendar(children, vt_lookup, days_behind=90, days_ahead=60):
    """Every not-yet-administered dose falling within the given window,
    one event per (child, dose), for the BHW dashboard's due-vaccines
    calendar widget. Status buckets match the calendar's legend:
    Overdue (past due), Due Today, Upcoming (future doses).
    Home Visit is reserved for a separately recorded visit date."""
    today = date.today()
    events = []
    for child in children:
        given = _doses_given(child.id)
        for code, name, antigen, dose_no, rec_days in dp.VACCINE_SCHEDULE:
            if given.get(vt_lookup[code].id) is not None:
                continue
            due_date = child.date_of_birth + timedelta(days=rec_days)
            delta = (due_date - today).days
            if delta < -days_behind or delta > days_ahead:
                continue
            if delta < 0:
                status = "Overdue"
            elif delta == 0:
                status = "Due Today"
            else:
                status = "Upcoming"
            events.append({"date": due_date.isoformat(), "child_id": child.id,
                            "child_name": child.display_name, "vaccine": name,
                            "vaccine_code": code, "status": status})
        # Include the BHW's explicitly scheduled home visit, independently
        # of vaccine due dates. Keep it visible even outside the dose window.
        if child.next_visit_date is not None:
            events.append({"date": child.next_visit_date.isoformat(),
                           "child_id": child.id, "child_name": child.display_name,
                           "vaccine": "Home Visit", "status": "Home Visit"})
    return events


DUE_MONTHS_AHEAD = 6       # "This month" plus the next five
DUE_OVERDUE_DAYS = 90      # how far back an unpaid dose still shows as overdue


def _month_options(today=None):
    """[(\"YYYY-MM\", \"November 2026\"), ...] for this month and the next five."""
    today = today or date.today()
    y, m = today.year, today.month
    out = []
    for _ in range(DUE_MONTHS_AHEAD):
        out.append((f"{y:04d}-{m:02d}", date(y, m, 1).strftime("%B %Y")))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _month_bounds(key):
    y, m = (int(x) for x in key.split("-"))
    first = date(y, m, 1)
    nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return first, nxt - timedelta(days=1)


def monthly_due_list(children, month_key, include_overdue=True):
    """Doses still to give, per child, for one calendar month.

    month_key is "YYYY-MM" or "year-YYYY" (a whole calendar year). A month
    shows every dose whose due date falls in it; with include_overdue, doses
    missed before that month (within the last DUE_OVERDUE_DAYS) are carried in
    too, since those children still need them. Without it, no dose already past
    due is listed at all. A whole year never carries overdue doses. Returns (events, summary): summary is one row per antigen with the
    number of doses needed, in schedule order."""
    today = date.today()
    if month_key.startswith("year-"):
        # A whole calendar year, the same period as the Forecast; overdue doses
        # from before it are not carried in, so the two pages count the same doses.
        year = int(month_key[5:])
        first, last = date(year, 1, 1), date(year, 12, 31)
        include_overdue = False
    else:
        first, last = _month_bounds(month_key)
    overdue_floor = today - timedelta(days=DUE_OVERDUE_DAYS)
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    events = []
    for child in children:
        given = _doses_given(child.id)
        for code, name, antigen, _dose_no, rec_days in dp.VACCINE_SCHEDULE:
            vt = vt_lookup.get(code)
            if vt is None or given.get(vt.id) is not None:
                continue
            due = child.date_of_birth + timedelta(days=rec_days)
            in_month = first <= due <= last
            carried = include_overdue and overdue_floor <= due < min(first, today)
            if not (in_month or carried):
                continue
            if not include_overdue and due < today:
                # "Exclude overdue" hides every dose already past due,
                # including ones due earlier in the selected month.
                continue
            delta = (due - today).days
            if delta < 0:
                status = "Overdue"
            elif delta == 0:
                status = "Due Today"
            elif delta <= 7:
                status = "Upcoming"
            else:
                status = "Scheduled"
            events.append({"date": due.isoformat(), "child_id": child.id,
                           "child_name": child.display_name,
                           "barangay": child.barangay.name if child.barangay else "",
                           "vaccine": name, "vaccine_code": code, "antigen": antigen,
                           "status": status, "carried": not in_month})
    events.sort(key=lambda e: (e["date"], e["child_name"]))
    return events


def due_summary(events):
    """[{"antigen", "name", "count", "overdue"}] in schedule order, for the
    "doses needed" strip - what the barangay should have on hand."""
    names = {a.code: a.name for a in VaccineAntigen.query.all()}
    order = []
    for _c, _n, antigen, _d, _r in dp.VACCINE_SCHEDULE:
        if antigen not in order:
            order.append(antigen)
    counts, overdue = Counter(e["antigen"] for e in events), Counter(
        e["antigen"] for e in events if e["status"] == "Overdue")
    return [{"antigen": a, "name": names.get(a, a), "count": counts[a], "overdue": overdue[a]}
            for a in order if counts[a]]


def _year_key(today=None):  # "year-2027": next calendar year, the Forecast's first year
    return f"year-{(today or date.today()).year + 1}"


def _due_list_args():
    """Reads and validates the Due List filters from the query string or,
    after recording a dose, from the posted form."""
    src = request.values
    options = _month_options()
    month = src.get("month") or options[0][0]
    if month != _year_key() and month not in {k for k, _ in options}:
        month = options[0][0]
    return {"month": month, "year_key": _year_key(), "year_label": str(date.today().year + 1),
            "include_overdue": src.get("overdue", "1") != "0",
            "search": src.get("q", "").strip(), "status_filter": src.get("status", ""),
            "month_options": options}


def _render_due_list(template_ctx, children, can_record, **extra):
    args = _due_list_args()
    events = monthly_due_list(children, args["month"], args["include_overdue"])
    summary = due_summary(events)
    if args["search"]:
        events = [e for e in events if args["search"].lower() in e["child_name"].lower()]
    if args["status_filter"]:
        events = [e for e in events if e["status"] == args["status_filter"]]
    whole_year = args["month"] == args["year_key"]
    if whole_year:
        month_label = f"Whole year {args['year_label']} (Jan 1 – Dec 31)"
    else:
        month_label = dict(args["month_options"])[args["month"]]
    if args["include_overdue"] and not whole_year:
        month_label += f", plus overdue from the last {DUE_OVERDUE_DAYS} days"
    return render_template("bhw_due_vaccines.html", **template_ctx(
        "due_list", page_title="Vaccination Due List", events=events, summary=summary,
        summary_total=sum(r["count"] for r in summary), month_label=month_label,
        can_record=can_record, today_iso=date.today().isoformat(), **args, **extra))


# ---------------------------------------------------------------------------
# Auth blueprint
# ---------------------------------------------------------------------------

auth_bp = Blueprint("auth", __name__)
def _safe_next(target):
    """Only follow ?next= when it points to a page on this site. A full URL
    or a //host link would send the user to another website after login."""
    if target and target.startswith("/") and not target.startswith(("//", "/\\")):
        return target
    return None


def _positive_int(value):
    """The value as a whole number of 1 or more, or None."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n >= 1 else None


def _whole_number(value):
    """The value as a whole number of 0 or more, or None."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


def _iso_date(value):
    """A YYYY-MM-DD string as a date, or None if it is not a real date."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


VALID_ROLES = ("rhu", "bhw", "admin", "superadmin")
VALID_PRIORITIES = ("Normal", "High", "Urgent")
MIN_PASSWORD_LENGTH = 8

ROLE_HOME = {"admin": "admin.dashboard", "superadmin": "superadmin.logs",
             "rhu": "rhu.dashboard", "bhw": "bhw.dashboard"}


@auth_bp.route("/")
def index():
    """Public landing page for all visitors, including signed-in users."""

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
        dist=risk_distribution(),
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
                flash("This account has been disabled. Contact your RHU administrator.", "error")
                return render_template("login.html")
            login_user(user)
            user.last_login_at = datetime.utcnow()
            db.session.commit()
            log_activity("Logged in")
            return redirect(_safe_next(request.args.get("next")) or url_for(ROLE_HOME[user.role]))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@auth_bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    """There's no email or SMS in this app, so this can't be a reset-link
    flow. Instead it flags the account for the RHU administrator (who already
    provisions and manages every account) to reset directly - they relay the
    new password to the user offline, the way a small office actually
    handles this."""
    if current_user.is_authenticated:
        return redirect(url_for(ROLE_HOME[current_user.role]))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        user = User.query.filter_by(username=username).first()
        if user and user.is_active_flag:
            user.password_reset_requested_at = datetime.utcnow()
            db.session.commit()
        # Same message either way - confirming or denying a username exists
        # is an account-enumeration leak.
        flash("If that username exists, your RHU administrator has been notified and will reset your password.", "success")
        return redirect(url_for("auth.login"))
    return render_template("forgot_password.html")


@auth_bp.route("/logout")
@login_required
def logout():
    log_activity("Logged out")
    logout_user()
    return redirect(url_for("auth.login"))


ROLE_SETTINGS = {"admin": "admin.settings", "superadmin": "superadmin.settings",
                  "rhu": "rhu.settings", "bhw": "bhw.settings"}


@auth_bp.route("/account/profile", methods=["POST"])
@login_required
def update_profile():
    full_name = request.form.get("full_name", "").strip()
    if full_name:
        current_user.full_name = full_name
        db.session.commit()
        log_activity("Updated profile")
        flash("Profile updated.", "success")
    else:
        flash("Name cannot be empty.", "error")
    return redirect(url_for(ROLE_SETTINGS[current_user.role]))


@auth_bp.route("/account/password", methods=["POST"])
@login_required
def update_password():
    current_password = request.form.get("current_password", "")
    new_password = request.form.get("new_password", "")
    confirm_password = request.form.get("confirm_password", "")
    if not current_user.check_password(current_password):
        flash("Current password is incorrect.", "error")
    elif len(new_password) < 8:
        flash("New password must be at least 8 characters.", "error")
    elif new_password != confirm_password:
        flash("New password and confirmation don't match.", "error")
    else:
        current_user.set_password(new_password)
        db.session.commit()
        log_activity("Changed password")
        flash("Password updated.", "success")
    return redirect(url_for(ROLE_SETTINGS[current_user.role]))


@auth_bp.route("/account/notifications/<pref>", methods=["POST"])
@login_required
def toggle_notification(pref):
    column = {"email": "notify_email", "at_risk": "notify_at_risk", "stock": "notify_stock"}.get(pref) or abort(404)
    setattr(current_user, column, not getattr(current_user, column))
    db.session.commit()
    return redirect(url_for(ROLE_SETTINGS[current_user.role]))


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
    period = _dashboard_period()
    as_of, born_from = period["as_of"], period["born_from"]
    risk = risk_as_of(as_of)
    kids = filtered_children(barangay_id, as_of=as_of, born_from=born_from)
    stats = stats_for(kids, as_of, risk)
    dist = risk_distribution_for(kids, risk)
    # Two actionable panels: who needs a visit, and what stock is running out.
    # Both summarise their dedicated pages rather than reproducing them. The
    # "All N" link points at Continuation Predictor, so N has to be its own
    # (12-month-scoped) count - the dashboard's unrestricted at-risk total
    # would promise more rows than that page actually shows.
    follow_up_all = at_risk_table(barangay_id, max_age_days=CONTINUATION_WINDOW_DAYS)
    follow_up = follow_up_all[:5]
    follow_up_total = len(follow_up_all)
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
    # Follows the period filter too, so this table and the stat cards
    # describe the same children.
    by_barangay = sorted(report_figures(None, as_of, born_from, risk)["by_barangay"] if as_of
                         else coverage_by_barangay(), key=lambda r: r["coverage"])
    if selected:
        by_barangay = [r for r in by_barangay if r["barangay"] == selected.name]
    by_vaccine = order_by_schedule(report_by_vaccine(None, as_of, born_from) if as_of
                                   else coverage_by_vaccine())
    trend_info = trend_for(barangay_id, period)
    trend = trend_info["trend"]
    below_target = len([r for r in by_barangay if r["coverage"] < COVERAGE_TARGET])
    change = coverage_change(barangay_id)
    return render_template("rhu_dashboard.html", **_rhu_ctx(
        "dashboard", page_title="Dashboard", stats=stats,
        risk_dist=dist, now=datetime.now(),
        barangays=Barangay.query.order_by(Barangay.name).all(),
        selected_barangay=selected, selected_barangay_id=barangay_id,
        **period,
        follow_up=follow_up, follow_up_total=follow_up_total, low_stock=low_stock, pending_requests=pending,
        by_barangay=by_barangay, by_vaccine=by_vaccine,
        vaccine_colours=vaccine_colours(by_vaccine), vaccine_labels=schedule_labels(by_vaccine),
        insights=chart_insights(by_vaccine, by_barangay, trend, trend_info["trend_year"]), **trend_info,
        below_target=below_target, change=change, active_campaigns=active_campaigns_summary(),
    ))


@rhu_bp.route("/children", methods=["GET", "POST"])
def children():
    if request.method == "POST":
        error = _create_form_errors(request.form)
        if error:
            flash(error, "error")
            return redirect(url_for("rhu.children", add=1))
        new_child = _create_child(request.form, barangay_locked=None)
        flash("Child registered successfully.", "success")
        return redirect(url_for("rhu.children", new_child=new_child.id))
    barangay_id = request.args.get("barangay_id", type=int)
    risk_level = request.args.get("risk_level", "")
    status = request.args.get("status", "")
    search = request.args.get("q", "").strip()
    selected_year = request.args.get("year", "")
    age_group = request.args.get("age_group", "all")
    if age_group not in ("active", "older", "all"):
        age_group = "all"
    q = Child.query
    if barangay_id:
        q = q.filter(Child.barangay_id == barangay_id)
    if search:
        q = q.filter(Child.full_name.ilike(f"%{search}%"))
    if selected_year:
        if selected_year.isdigit() and len(selected_year) == 4:
            q = q.filter(Child.date_registered >= date(int(selected_year), 1, 1),
                         Child.date_registered < date(int(selected_year) + 1, 1, 1))
        else:
            selected_year = ""
    kids = q.order_by(Child.barangay_id, Child.id).all()
    # Calendar-month cutoff: the 12-month birthday is still in the active view.
    today = date.today()
    def twelve_month_birthday(dob):
        try:
            return dob.replace(year=dob.year + 1)
        except ValueError:  # Feb 29 in a non-leap year
            return dob.replace(year=dob.year + 1, day=28)
    if age_group == "active":
        kids = [c for c in kids if twelve_month_birthday(c.date_of_birth) >= today]
    elif age_group == "older":
        kids = [c for c in kids if twelve_month_birthday(c.date_of_birth) < today]
    kids = _filter_child_list(kids, risk_level, status)
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    registration_years = sorted({str(y[0]) for y in db.session.query(
        func.strftime('%Y', Child.date_registered)).filter(
        Child.date_registered.isnot(None)).distinct().all() if y[0]}, reverse=True)

    # Just registered: show what she can act on today instead of making her
    # open the new record and scan all 15 doses to find out.
    due_now_child, due_now_doses = None, []
    new_child_id = request.args.get("new_child", type=int)
    if new_child_id:
        due_now_child = db.session.get(Child, new_child_id)
        if due_now_child:
            due_now_doses = [d for d in child_detail(due_now_child)["schedule"] if d["status"] in ("due", "overdue")]

    return render_template("rhu_children.html", **_rhu_ctx(
        "children", page_title="Child Records", children=kids,
        barangays=Barangay.query.order_by(Barangay.name).all(), vaccine_schedule=dp.VACCINE_SCHEDULE,
        vt_lookup=vt_lookup, antigens=core_antigens(), selected_barangay=barangay_id,
        selected_risk=risk_level, selected_status=status, search=search, selected_year=selected_year, age_group=age_group,
        registration_years=registration_years,
        dose_matrix=_dose_matrix(kids, vt_lookup), risk_labels=_risk_labels(kids),
        due_now_child=due_now_child, due_now_doses=due_now_doses,
    ))


@rhu_bp.route("/children/<int:child_id>/recompute-risk", methods=["POST"])
def recompute_risk(child_id):
    child = db.session.get(Child, child_id) or abort(404)
    result = _recompute_risk(child)
    log_activity(f"Re-assessed risk for {child_label(child)}: {result['label']}")
    flash(f"Risk re-assessed for {child.display_name}: {result['label']}", "success")
    return redirect(request.referrer or url_for("rhu.children"))


@rhu_bp.route("/due-vaccines")
def due_vaccines():
    """View-only Vaccination Due List across the municipality, filterable to
    one barangay. Doses are recorded by the BHW of each barangay."""
    barangays = Barangay.query.order_by(Barangay.name).all()
    try:
        bid = int(request.args.get("barangay") or 0) or None
    except ValueError:
        bid = None
    chosen = next((b for b in barangays if b.id == bid), None)
    bid = chosen.id if chosen else None
    return _render_due_list(_rhu_ctx, filtered_children(bid), can_record=False,
                            barangays=barangays, barangay_id=bid,
                            scope_label=chosen.name if chosen else "All barangays")


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
        error = _edit_form_errors(child, request.form)
        if error:
            flash(error, "error")
            return redirect(url_for("rhu.edit_child", child_id=child.id))
        result = _update_child(child, request.form)
        flash(f"{child.display_name} updated. Risk re-assessed: {result['label']}", "success")
        for name in result["past_window_warnings"]:
            flash(f"{name} was recorded past its usual eligibility age — please double-check this dose.", "warning")
        return redirect(url_for("rhu.children"))
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    doses = {code: (rec.date_administered if (rec := child.vaccination_records.filter_by(
        vaccine_type_id=vt_lookup[code].id).first()) else None) for code, *_ in dp.VACCINE_SCHEDULE + dp.OPTIONAL_VACCINE_SCHEDULE}
    active_campaigns = eligible_campaigns_for_child(child)
    campaign_doses = {c.id: (rec.date_administered if (rec := child.vaccination_records.filter_by(
        campaign_id=c.id).first()) else None) for c in active_campaigns}
    return render_template("edit_child.html", **_rhu_ctx(
        "children", page_title="Update Child Record", child=child, doses=doses,
        optional_vaccines=dp.OPTIONAL_VACCINE_SCHEDULE,
        active_campaigns=active_campaigns, campaign_doses=campaign_doses,
        back_url=url_for("rhu.children"), form_action=url_for("rhu.edit_child", child_id=child.id),
    ))


@rhu_bp.route("/schedule-visit", methods=["POST"])
def schedule_visit():
    """Save a follow-up visit date for a child, same as the BHW's own
    scheduler - the RHU just isn't limited to one barangay, since they
    oversee all of them."""
    try:
        child_id = int(request.form.get("child_id", ""))
    except (ValueError, TypeError):
        abort(400)
    child = db.session.get(Child, child_id) or abort(404)
    try:
        visit_date = date.fromisoformat(request.form.get("visit_date", ""))
    except (ValueError, TypeError):
        flash("Please select a valid visit date.", "warning")
        return redirect(request.referrer or url_for("rhu.continuation"))
    if visit_date < date.today():
        flash("Visit date cannot be in the past.", "warning")
        return redirect(request.referrer or url_for("rhu.continuation"))
    child.next_visit_date = visit_date
    db.session.commit()
    flash(f"Visit scheduled for {child.display_name} on {visit_date.strftime('%b %d, %Y')}.", "success")
    return redirect(request.referrer or url_for("rhu.continuation"))


@rhu_bp.route("/vaccine-gaps")
def vaccine_gaps():
    vaccine_filter = request.args.get("vaccine", "")
    barangay_id = request.args.get("barangay_id", type=int)
    barangays = Barangay.query.order_by(Barangay.name).all()
    barangay_obj = next((b for b in barangays if b.id == barangay_id), None)
    dist = vaccine_risk_distribution(barangay_id)
    rows = vaccine_risk_table(barangay_id, vaccine_code=vaccine_filter or None)
    return render_template("rhu_vaccine_gaps.html", **_rhu_ctx(
        "vaccine_gaps", page_title="Vaccine Gaps", dist=dist, at_risk_rows=rows,
        severity=dp.ANTIGEN_SEVERITY, severity_bands=dp.SEVERITY_BANDS,
        antigen_names={a.code: a.name for a in VaccineAntigen.query.all()},
        vaccine_filter=vaccine_filter, antigen_options=dp.VACCINE_ANTIGENS,
        barangays=barangays, selected_barangay=barangay_id,
        scope_name=barangay_obj.name if barangay_obj else "all barangays",
    ))


@rhu_bp.route("/continuation")
def continuation():
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    barangay_id = request.args.get("barangay_id", type=int)
    barangays = Barangay.query.order_by(Barangay.name).all()
    barangay_obj = next((b for b in barangays if b.id == barangay_id), None)
    rows = at_risk_table(barangay_id, max_age_days=CONTINUATION_WINDOW_DAYS)
    for r in rows:
        r["continuation_pct"] = 100 - r["risk_score"]
        r["next_dose"] = _next_dose_for(r["child"], vt_lookup)
    engine = dp.model_info()
    return render_template("rhu_continuation.html", **_rhu_ctx(
        "continuation", page_title="Continuation Predictor", rows=rows,
        stats=dashboard_stats(barangay_id, max_age_days=CONTINUATION_WINDOW_DAYS),
        dist=risk_distribution(barangay_id, max_age_days=CONTINUATION_WINDOW_DAYS),
        engine=engine, scoring_rules=dp.SCORING_RULES,
        feature_labels=dp.FEATURE_LABELS, features=feature_catalog(),
        barangays=barangays, selected_barangay=barangay_id,
        scope_name=barangay_obj.name if barangay_obj else "all barangays",
    ))


@rhu_bp.route("/map")
def municipality_map():
    return render_template("rhu_map.html", **_rhu_ctx(
        "map", page_title="Municipality Map",
        heatmap=map_markers("heatmap"), at_risk=map_markers("at_risk"), density=map_markers("density"),
    ))


@rhu_bp.route("/inventory", methods=["GET", "POST"])
def inventory():  # inventory page
    if request.method == "POST":  # add batch
        antigen = db.session.get(VaccineAntigen, _positive_int(request.form.get("antigen_id")) or 0)
        quantity = _positive_int(request.form.get("quantity"))
        batch_number = (request.form.get("batch_number") or "").strip()
        per_vial_raw = request.form.get("doses_per_vial")
        per_vial = _positive_int(per_vial_raw) if per_vial_raw else (default_doses_per_vial(antigen) if antigen else None)
        expiry = None
        if request.form.get("expiry_date"):
            try:
                expiry = datetime.strptime(request.form.get("expiry_date"), "%Y-%m-%d").date()
            except ValueError:
                expiry = False
        error = ("Please choose a vaccine." if antigen is None else
                 "Please enter a batch number." if not batch_number else
                 "Quantity must be a whole number of 1 or more." if quantity is None else
                 "Doses per vial must be a whole number from 1 to 100." if per_vial is None or per_vial > 100 else
                 "Please enter a valid expiry date." if expiry is False else
                 "This batch has already expired and cannot be added." if expiry and expiry < date.today() else None)
        if error:
            flash(error, "error")
            return redirect(url_for("rhu.inventory"))
        batch = InventoryBatch(  # new batch
            antigen_id=antigen.id, batch_number=batch_number,
            quantity_on_hand=quantity, storage_location=request.form.get("storage_location"),
            expiry_date=expiry, reorder_level=50, added_at=datetime.utcnow(), doses_per_vial=per_vial,
        )
        db.session.add(batch)
        db.session.commit()  # save
        log_activity(f"Added vaccine stock: {batch.antigen.name} +{batch.quantity_on_hand} vials "
                     f"({per_vial} doses per vial)")  # log
        flash("Stock added successfully.", "success")
        return redirect(url_for("rhu.inventory"))

    rows = stock_levels()  # stock status
    stats = {"total_types": len(rows), "out_of_stock": sum(1 for r in rows if r["status"] == "Out of Stock"),  # summary cards
             "low_critical": sum(1 for r in rows if r["status"] in ("Low", "Critical")),
             "expiring_soon": sum(r["expiring"] for r in rows)}
    return render_template("rhu_inventory.html", **_rhu_ctx(
        "inventory", page_title="Vaccine Inventory", rows=rows, stats=stats, antigens=core_antigens(),
        today_iso=date.today().isoformat(),
        default_per_vial={a.id: default_doses_per_vial(a) for a in core_antigens()},
        batches=[(b, batch_doses_per_vial(b), batch_distributed(b)) for b in
                 InventoryBatch.query.filter_by(is_archived=False).order_by(InventoryBatch.antigen_id,
                                                                            InventoryBatch.expiry_date).all()],
        wastage_reasons=WASTAGE_REASONS,
    ))


def _rhu_batch(batch_id):
    return db.session.get(InventoryBatch, _positive_int(batch_id) or 0) or abort(404)


def _rhu_movement(batch, kind, vials, note):  # RHU stock change, for the audit trail
    _record_movement(None, batch.antigen_id, kind, vials * batch_doses_per_vial(batch),
                     note=f"RHU batch {batch.batch_number}: {note}")


@rhu_bp.route("/inventory/<int:batch_id>/vial-size", methods=["POST"])
def batch_vial_size(batch_id):  # correct a batch's doses per vial
    batch = _rhu_batch(batch_id)
    per_vial = _positive_int(request.form.get("doses_per_vial"))
    if per_vial is None or per_vial > 100:
        flash("Doses per vial must be a whole number from 1 to 100.", "error")
    elif batch_distributed(batch):
        flash(f"Batch {batch.batch_number} has already been distributed, so its vial size can no longer be changed. "
              "A physical count corrects the barangay stock instead.", "error")
    else:
        old = batch_doses_per_vial(batch)
        batch.doses_per_vial = per_vial
        db.session.commit()
        log_activity(f"Changed doses per vial for vaccine stock batch {batch.batch_number} from {old} to {per_vial}")
        flash(f"Batch {batch.batch_number}: {per_vial} doses per vial.", "success")
    return redirect(url_for("rhu.inventory"))


@rhu_bp.route("/inventory/wastage", methods=["POST"])
def rhu_wastage():  # RHU wastage
    batch = _rhu_batch(request.form.get("batch_id"))
    vials = _positive_int(request.form.get("vials"))
    reason = request.form.get("reason")
    error = ("Enter the number of vials lost (1 or more)." if vials is None else
             f"Only {batch.quantity_on_hand} vials of batch {batch.batch_number} are in stock." if vials > batch.quantity_on_hand else
             "Please choose a reason." if reason not in WASTAGE_REASONS else None)
    if error:
        flash(error, "error")
        return redirect(url_for("rhu.inventory"))
    batch.quantity_on_hand -= vials
    _rhu_movement(batch, "wasted", -vials, f"{vials} vials, {reason}")
    db.session.commit()
    log_activity(f"Recorded vaccine stock wastage: {batch.antigen.name} {vials} vials (batch {batch.batch_number}, {reason})")
    flash(f"Recorded {vials} wasted {batch.antigen.name} vials ({reason}).", "success")
    return redirect(url_for("rhu.inventory"))


@rhu_bp.route("/inventory/count", methods=["POST"])
def rhu_count():  # RHU physical count
    batch = _rhu_batch(request.form.get("batch_id"))
    counted = _whole_number(request.form.get("vials"))
    if counted is None:
        flash("Vials counted must be a whole number (0 or more).", "error")
        return redirect(url_for("rhu.inventory"))
    diff = counted - batch.quantity_on_hand
    if diff:
        _rhu_movement(batch, "count_adjustment", diff, f"system {batch.quantity_on_hand}, counted {counted} vials")
        batch.quantity_on_hand = counted
        db.session.commit()
    log_activity(f"Physical count of vaccine stock: {batch.antigen.name} batch {batch.batch_number}, counted {counted} vials"
                 + (f" (difference {diff:+d})" if diff else ""))
    flash(f"Batch {batch.batch_number}: {counted} vials" + (f" (difference {diff:+d})." if diff else " (matches)."), "success")
    return redirect(url_for("rhu.inventory"))


@rhu_bp.route("/requests")
def requests():  # requests page
    """Pending requests are grouped by vaccine with a suggested split of
    available stock - urgent/high-priority requests filled first, then
    whatever's left shared proportionally - since stock is rationed per
    vaccine, not per request. A starting point only: the RHU can adjust any
    row's quantity before approving it, so rationing stock is still the
    RHU's decision, not the system's. All Requests stays a flat history."""
    tab = request.args.get("tab", "pending")
    stats = {"pending": VaccineRequest.query.filter_by(status="pending").count(),  # tab counts
             "approved": VaccineRequest.query.filter_by(status="approved").count(),
             "distributed": VaccineRequest.query.filter_by(status="fulfilled").count(),
             "rejected": VaccineRequest.query.filter_by(status="rejected").count()}

    groups = None
    reqs = None
    if tab == "pending":  # pending tab
        coverage = {r["barangay"]: r["coverage"] for r in coverage_by_barangay()}  # coverage
        pending = (VaccineRequest.query.filter_by(status="pending")  # pending
                   .order_by(VaccineRequest.requested_at.asc()).all())

        # Grouped by antigen: stock is held per vaccine, so each is rationed
        # separately.
        by_antigen = defaultdict(list)
        for r in pending:  # by vaccine
            by_antigen[r.antigen_id].append(r)

        groups = []
        for antigen_id, antigen_reqs in by_antigen.items():  # each vaccine
            antigen = antigen_reqs[0].antigen
            rows = [{"id": r.id, "code": r.request_code, "barangay": r.barangay.name,
                     "requested_at": r.requested_at, "requested": r.quantity_requested,
                     "priority": r.priority, "coverage": coverage.get(r.barangay.name)}
                    for r in antigen_reqs]
            result = dp.allocate_stock(rows, usable_stock(antigen_id), "priority", coverage)  # suggest split
            # Vaccines outside the RHU's 7 stocked antigens (e.g. an older
            # Rotavirus request) can never be distributed from inventory.
            groups.append({"antigen": antigen, "tracked": antigen.code in _CORE_ANTIGEN_CODES, **result})
        # Short vaccines first: those are the decisions that need making.
        groups.sort(key=lambda g: (g["can_meet_all"], -g["shortfall"]))  # shortest first
    else:
        reqs = VaccineRequest.query.order_by(VaccineRequest.requested_at.desc()).all()  # all requests

    return render_template("rhu_requests.html", **_rhu_ctx(
        "requests", page_title="Vaccine Requests", groups=groups, reqs=reqs, stats=stats, tab=tab,
    ))


def _usable_batches(antigen_id):  # usable batches
    """Batches that can actually be given out: not empty, archived or
    expired, soonest expiry first. Shared by the allocation suggestion and
    by distribution, so a suggestion never counts vials distribution would
    refuse."""
    return (InventoryBatch.query
            .filter(InventoryBatch.antigen_id == antigen_id, InventoryBatch.is_archived.is_(False),  # not empty/expired
                    InventoryBatch.quantity_on_hand > 0, InventoryBatch.expiry_date >= date.today())
            .order_by(InventoryBatch.expiry_date, InventoryBatch.id).all())  # soonest expiry


def usable_stock(antigen_id):  # usable total
    return sum(b.quantity_on_hand for b in _usable_batches(antigen_id))


def _deduct_stock(antigen_id, qty):  # deduct stock
    """Takes `qty` vials from usable stock, first-expiry-first-out across
    batches (skipping empty, archived and expired ones). Returns the
    [(batch, vials taken)] if there was enough; otherwise changes nothing
    and returns an empty list."""
    batches = _usable_batches(antigen_id)  # usable batches
    if sum(b.quantity_on_hand for b in batches) < qty:  # not enough
        return []
    taken, remaining = [], qty  # to take
    for b in batches:  # soonest first
        take = min(b.quantity_on_hand, remaining)  # take
        b.quantity_on_hand -= take  # deduct
        taken.append((b, take))
        remaining -= take
        if not remaining:  # done
            break
    return taken


# ---------------------------------------------------------------------------
# Barangay stock
#
# RHU stock is counted in vials; a barangay's stock is counted in doses, per
# batch (BarangayStock), and changes only through a StockMovement. Stock on
# its way (an RHU delivery or a redistribution) is a StockTransfer and sits in
# nobody's fridge until the receiving BHW confirms what arrived.
# ---------------------------------------------------------------------------

STOCK_NEED_DAYS_AHEAD = 30    # "doses needed soon" = due in the next 30 days ...
STOCK_NEED_DAYS_BEHIND = 90   # ... or overdue by up to 90 days
LOSS_REASONS = ("Fewer vials arrived", "Broken vials", "Cold-chain problem", "Other")


def default_doses_per_vial(antigen):  # vaccine default
    row = db.session.get(VialSize, antigen.id)
    return row.doses_per_vial if row else dp.DEFAULT_DOSES_PER_VIAL.get(antigen.code, 1)


def batch_doses_per_vial(batch):  # batch value
    return batch.doses_per_vial or default_doses_per_vial(batch.antigen)


def vials_label(doses, doses_per_vial):  # "8 vials + 5 doses"
    if doses_per_vial <= 1:
        return f"{doses} vial{'s' if doses != 1 else ''}"
    full, loose = divmod(doses, doses_per_vial)
    if not full and loose:
        return f"{loose} loose dose{'s' if loose != 1 else ''}"
    text = f"{full} vial{'s' if full != 1 else ''}"
    return text + (f" + {loose} dose{'s' if loose != 1 else ''}" if loose else "")


def _next_transfer_code():  # TRF-2026-001
    last = StockTransfer.query.order_by(StockTransfer.id.desc()).first()
    return f"TRF-{datetime.now().year}-{(last.id if last else 0) + 1:03d}"


def _add_to_lot(barangay_id, antigen_id, batch_number, expiry, doses_per_vial, doses):  # stock in
    """Add doses to the barangay's lot for this batch, creating it if needed."""
    lot = BarangayStock.query.filter_by(barangay_id=barangay_id, antigen_id=antigen_id,
                                        batch_number=batch_number, expiry_date=expiry,
                                        doses_per_vial=doses_per_vial).first()
    if lot is None:
        lot = BarangayStock(barangay_id=barangay_id, antigen_id=antigen_id, batch_number=batch_number,
                            expiry_date=expiry, doses_per_vial=doses_per_vial, doses_on_hand=0)
        db.session.add(lot)
        db.session.flush()
    lot.doses_on_hand += doses
    return lot


def _record_movement(barangay_id, antigen_id, kind, doses, lot=None, transfer=None, record=None, note=None):
    db.session.add(StockMovement(
        barangay_id=barangay_id, antigen_id=antigen_id, kind=kind, doses=doses,
        barangay_stock_id=lot.id if lot else None, transfer_id=transfer.id if transfer else None,
        vaccination_record_id=record.id if record else None, note=note,
        user_id=current_user.id if current_user.is_authenticated else None))


def stock_tracking_started(barangay_id, antigen_id):  # tracked yet?
    """A vaccine's stock is tracked in a barangay from its first opening count
    or receipt; before that, recorded doses do not touch stock."""
    return StockMovement.query.filter_by(barangay_id=barangay_id, antigen_id=antigen_id).first() is not None


def reserved_doses(barangay_id):  # {antigen_id: doses}
    """Doses set aside for redistribution to another barangay (still in the fridge)."""
    out = Counter()
    for t in StockTransfer.query.filter_by(from_barangay_id=barangay_id, status="reserved").all():
        out[t.antigen_id] += t.doses_sent
    return out


def _stock_schedule():  # (dose code, antigen code, due age in days), core + optional
    """Every dose barangay stock is kept for: the routine schedule plus
    optional vaccines such as Rotavirus, which is also stocked and requested."""
    return ([(code, antigen, days) for code, _n, antigen, _d, days in dp.VACCINE_SCHEDULE] +
            [(code, antigen, days) for code, _n, antigen, _d, days, _end in dp.OPTIONAL_VACCINE_SCHEDULE])


def doses_needed_soon(barangay_id):  # {antigen code: doses}
    """Doses the barangay's children will need soon: not yet given, still
    catchable, and due in the next 30 days or overdue by up to 90 days."""
    today = date.today()
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    need = Counter()
    for child in filtered_children(barangay_id):
        given = _doses_given(child.id)
        for code, antigen, rec_days in _stock_schedule():
            vt = vt_lookup.get(code)
            if vt is None or given.get(vt.id) is not None or not _still_catchable(vt, child.age_in_days):
                continue
            delta = (child.date_of_birth + timedelta(days=rec_days) - today).days
            if -STOCK_NEED_DAYS_BEHIND <= delta <= STOCK_NEED_DAYS_AHEAD:
                need[antigen] += 1
    return need


def barangay_stock_summary(barangay_id):  # My Stock rows
    """One row per vaccine: what is in the fridge, reserved, available,
    expiring and needed soon, with a status."""
    today = date.today()
    lots = BarangayStock.query.filter_by(barangay_id=barangay_id).all()
    tracked = {a for (a,) in db.session.query(StockMovement.antigen_id)
               .filter_by(barangay_id=barangay_id).distinct()}
    need, reserved = doses_needed_soon(barangay_id), reserved_doses(barangay_id)
    coming = Counter()  # in transit to this barangay, returning to it, or reserved for it elsewhere
    for t in StockTransfer.query.filter(StockTransfer.status.in_(("reserved", "in_transit", "returning"))).all():
        if (t.status == "returning" and t.from_barangay_id == barangay_id) or \
                (t.status != "returning" and t.to_barangay_id == barangay_id):
            coming[t.antigen_id] += t.doses_sent
    rows = []
    for a in core_antigens():
        mine = [lot for lot in lots if lot.antigen_id == a.id and lot.doses_on_hand > 0]
        usable = sorted([lot for lot in mine if not lot.is_expired], key=lambda lot: (lot.expiry_date or date.max))
        in_fridge = sum(lot.doses_on_hand for lot in usable)
        held = min(reserved.get(a.id, 0), in_fridge)
        available = in_fridge - held
        nearest = usable[0].expiry_date if usable else None
        needed = need.get(a.code, 0)
        if a.id not in tracked:
            status, color = "Not tracked yet", "gray"
        elif available <= 0:
            status, color = "Out of stock", "red"
        elif available < needed:
            status, color = ("Low (delivery coming)", "amber") if available + coming[a.id] >= needed else ("Low", "amber")
        elif nearest and (nearest - today).days <= 30:
            status, color = "Expiring soon", "amber"
        else:
            status, color = "OK", "green"
        rows.append({
            "no_stock": StockMovement.query.filter(
                StockMovement.barangay_id == barangay_id, StockMovement.antigen_id == a.id,
                StockMovement.kind == "no_stock",
                StockMovement.created_at >= datetime.utcnow() - timedelta(days=30)).count(),
            "antigen": a, "tracked": a.id in tracked, "lots": usable,
            "in_fridge": in_fridge, "reserved": held, "available": available, "coming": coming[a.id],
            "vials": " + ".join(vials_label(lot.doses_on_hand, lot.doses_per_vial) for lot in usable) or "-",
            "expired": sum(lot.doses_on_hand for lot in mine if lot.is_expired),
            "nearest_expiry": nearest, "needed": needed, "status": status, "color": color,
        })
    return rows


WASTAGE_REASONS = ("Broken vial", "Cold-chain failure", "Opened vial discarded", "Contaminated", "Other")
END_OF_SESSION_REASON = "Opened vial discarded (end of session)"


def discards_after_session(antigen):  # discard opened vials after the session?
    row = db.session.get(VialSize, antigen.id)
    if row is not None and row.discard_after_session is not None:
        return row.discard_after_session
    return antigen.code in dp.DISCARD_AFTER_SESSION


def batch_distributed(batch):  # any vial already sent out?
    return (StockTransferItem.query.join(StockTransfer)
            .filter(StockTransfer.kind == "delivery", StockTransfer.antigen_id == batch.antigen_id,
                    StockTransferItem.batch_number == batch.batch_number).first() is not None)


def tracking_start(barangay_id, antigen_id):  # date stock tracking began
    first = (StockMovement.query.filter_by(barangay_id=barangay_id, antigen_id=antigen_id)
             .order_by(StockMovement.created_at, StockMovement.id).first())
    return to_local(first.created_at).date() if first else None


def usable_lots(barangay_id, antigen_id):  # soonest expiry first
    lots = BarangayStock.query.filter(BarangayStock.barangay_id == barangay_id,
                                      BarangayStock.antigen_id == antigen_id,
                                      BarangayStock.doses_on_hand > 0).all()
    return sorted([lot for lot in lots if not lot.is_expired], key=lambda lot: (lot.expiry_date or date.max, lot.id))


# --- Redistribution between barangays ---------------------------------------
# The RHU reserves doses in one barangay's fridge for another barangay. The
# vials stay where they are (reserved) until the sending BHW marks them
# dispatched (in transit), and only reach the receiver when its BHW counts
# them in. Vials travel whole, so transfers are always in unopened vials.

TRANSFER_STATUS_LABELS = {"reserved": "Reserved for Redistribution", "in_transit": "In Transit",
                          "received": "Received", "returning": "Returning to sender", "cancelled": "Cancelled"}
TRANSFER_STATUS_COLORS = {"reserved": "amber", "in_transit": "blue", "received": "green",
                          "returning": "amber", "cancelled": "gray"}
SURPLUS_FACTOR = 1.5   # holding more than 1.5x the 30-day need is a surplus
RESERVED_USE_REASONS = ("Child due today, no other stock", "Outreach or catch-up session", "Other")


def _queue_log(action):  # activity log entry, saved with the caller's commit
    db.session.add(ActivityLog(user_id=current_user.id if current_user.is_authenticated else None, action=action))


def reserved_by_lot(barangay_id):  # {lot id: reserved doses}
    rows = (db.session.query(StockTransferItem.source_stock_id, func.sum(StockTransferItem.doses_sent))
            .join(StockTransfer, StockTransferItem.transfer_id == StockTransfer.id)
            .filter(StockTransfer.from_barangay_id == barangay_id, StockTransfer.status == "reserved")
            .group_by(StockTransferItem.source_stock_id).all())
    return Counter({lot_id: int(n or 0) for lot_id, n in rows if lot_id})


def free_vials(lot, held):  # unopened vials in a lot that are not reserved
    if lot.is_expired:
        return 0
    return max(0, lot.doses_on_hand // lot.doses_per_vial - held.get(lot.id, 0) // lot.doses_per_vial)


def plan_transfer(barangay_id, antigen_id, doses, round_up=True):  # [(lot, doses)]
    """Whole unopened vials covering `doses`, soonest expiry first, from vials
    not already reserved. round_up: at least `doses` (may be up to one vial
    more), else [] if there is not enough; otherwise at most `doses`."""
    held = reserved_by_lot(barangay_id)
    plan, left = [], doses
    for lot in usable_lots(barangay_id, antigen_id):
        if left <= 0:
            break
        per_vial = lot.doses_per_vial
        want = -(-left // per_vial) if round_up else left // per_vial
        take = min(free_vials(lot, held), want)
        if take:
            plan.append((lot, take * per_vial))
            left -= take * per_vial
    return plan if (left <= 0 or not round_up) else []


def transferable_doses(barangay_id, antigen_id):  # doses in free unopened vials
    held = reserved_by_lot(barangay_id)
    return sum(free_vials(lot, held) * lot.doses_per_vial for lot in usable_lots(barangay_id, antigen_id))


def pick_lot_for_dose(barangay_id, antigen_id):  # (lot or None, dips into reserved?)
    """The batch a dose is taken from: soonest expiry first, but skipping
    doses reserved for another barangay while unreserved ones remain."""
    lots = usable_lots(barangay_id, antigen_id)
    if not lots:
        return None, False
    held = reserved_by_lot(barangay_id)
    for lot in lots:
        if (lot.doses_on_hand - 1) // lot.doses_per_vial * lot.doses_per_vial >= held.get(lot.id, 0):
            return lot, False
    return lots[0], True


def reserved_only_antigens(barangay_id):  # {antigen id: [transfers]} where only reserved doses are left
    out = {}
    for t in StockTransfer.query.filter_by(from_barangay_id=barangay_id, status="reserved").all():
        out.setdefault(t.antigen_id, []).append(t)
    return {aid: ts for aid, ts in out.items() if pick_lot_for_dose(barangay_id, aid)[1]}


def reconcile_reservations(barangay_id, antigen_id, reason):  # reserved doses must still exist
    """After stock goes down (a dose given, wastage, a count), a reservation can
    promise more vials than the fridge holds. The newest transfers are reduced
    first, in whole vials, to what is left; one reduced to nothing is
    cancelled. Returns messages for the BHW; the RHU sees the changes on
    Barangay Stock and in its notifications."""
    transfers = (StockTransfer.query.filter_by(from_barangay_id=barangay_id, antigen_id=antigen_id,
                                               status="reserved").order_by(StockTransfer.id.desc()).all())
    by_lot = defaultdict(list)
    for t in transfers:  # newest first
        for item in t.items:
            if item.doses_sent:
                by_lot[item.source_stock_id].append(item)
    cut_by_transfer = Counter()
    for lot_id, items in by_lot.items():
        lot = db.session.get(BarangayStock, lot_id) if lot_id else None
        capacity = 0 if lot is None or lot.is_expired else lot.doses_on_hand // lot.doses_per_vial * lot.doses_per_vial
        excess = sum(i.doses_sent for i in items) - capacity
        for item in items:
            if excess <= 0:
                break
            cut = min(item.doses_sent, -(-excess // item.doses_per_vial) * item.doses_per_vial)
            item.doses_sent -= cut
            excess -= cut
            cut_by_transfer[item.transfer_id] += cut
    messages = []
    for t in transfers:
        cut = cut_by_transfer.get(t.id)
        if not cut:
            continue
        t.doses_reduced = (t.doses_reduced or 0) + cut
        t.reduce_reason = reason[:255]
        if t.doses_sent == 0:
            t.status, t.cancelled_at = "cancelled", datetime.utcnow()
            text = (f"Redistribution {t.code} to {t.to_barangay.name} cancelled: its {cut} reserved "
                    f"{t.antigen.name} doses were used ({reason}). The RHU has been alerted.")
        else:
            text = (f"Redistribution {t.code} to {t.to_barangay.name} reduced by {cut} to {t.doses_sent} "
                    f"{t.antigen.name} doses ({reason}). The RHU has been alerted.")
        messages.append(text)
        _queue_log(text.replace(" The RHU has been alerted.", ""))
    return messages


def doses_needed_by_barangay(barangay_ids=None):  # {barangay id: {antigen code: doses}}
    """doses_needed_soon() for every barangay (or just barangay_ids) in one pass."""
    today = date.today()
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    need = defaultdict(Counter)
    children = Child.query
    if barangay_ids is not None:
        if not barangay_ids:
            return need
        children = children.filter(Child.barangay_id.in_(list(barangay_ids)))
    for child in children.all():
        given = _doses_given(child.id)
        for code, antigen, rec_days in _stock_schedule():
            vt = vt_lookup.get(code)
            if vt is None or given.get(vt.id) is not None or not _still_catchable(vt, child.age_in_days):
                continue
            delta = (child.date_of_birth + timedelta(days=rec_days) - today).days
            if -STOCK_NEED_DAYS_BEHIND <= delta <= STOCK_NEED_DAYS_AHEAD:
                need[child.barangay_id][antigen] += 1
    return need


def municipal_stock_rows(antigen, need=None):  # RHU Barangay Stock rows for one vaccine
    """Every barangay's stock of one vaccine, with Surplus / OK / Short.
    Short: available + incoming (in transit or reserved for it) is below the 30-day need. Surplus: more than
    1.5x the need, or doses expiring within 30 days that will not be used."""
    today = date.today()
    need = need if need is not None else doses_needed_by_barangay()
    lots = BarangayStock.query.filter(BarangayStock.antigen_id == antigen.id, BarangayStock.doses_on_hand > 0).all()
    tracked = {b for (b,) in db.session.query(StockMovement.barangay_id)
               .filter(StockMovement.antigen_id == antigen.id, StockMovement.barangay_id.isnot(None)).distinct()}
    reserved, incoming = Counter(), Counter()
    for t in StockTransfer.query.filter(StockTransfer.antigen_id == antigen.id,
                                        StockTransfer.status.in_(("reserved", "in_transit", "returning"))).all():
        if t.status == "reserved":  # still in the sender's fridge, already promised to the receiver
            reserved[t.from_barangay_id] += t.doses_sent
            incoming[t.to_barangay_id] += t.doses_sent
        elif t.status == "in_transit":
            incoming[t.to_barangay_id] += t.doses_sent
        else:
            incoming[t.from_barangay_id] += t.doses_sent
    rows = []
    for b in Barangay.query.order_by(Barangay.name).all():
        mine = sorted([lot for lot in lots if lot.barangay_id == b.id and not lot.is_expired],
                      key=lambda lot: (lot.expiry_date or date.max))
        in_fridge = sum(lot.doses_on_hand for lot in mine)
        held = min(reserved[b.id], in_fridge)
        available = in_fridge - held
        needed = need.get(b.id, {}).get(antigen.code, 0)
        expiring = sum(lot.doses_on_hand for lot in mine
                       if lot.expiry_date and (lot.expiry_date - today).days <= STOCK_NEED_DAYS_AHEAD)
        spare = short = 0
        if b.id not in tracked:
            status, color = "Not tracked", "gray"
        elif available + incoming[b.id] < needed:
            status, color, short = "Short", "red", needed - available - incoming[b.id]
        else:
            spare = min(max(available - math.ceil(needed * SURPLUS_FACTOR), expiring - needed, 0),
                        transferable_doses(b.id, antigen.id))
            status, color = ("Surplus", "blue") if spare > 0 else ("OK", "green")
        rows.append({"barangay": b, "in_fridge": in_fridge, "reserved": held, "incoming": incoming[b.id],
                     "available": available, "needed": needed, "expiring": expiring,
                     "nearest_expiry": mine[0].expiry_date if mine else None,
                     "status": status, "color": color, "spare": spare, "short": short})
    return rows


def suggest_transfers(antigen, rows):  # match surplus to shortage
    """Greedy: the largest shortage first, served by the surplus whose doses
    expire soonest, in whole vials."""
    shorts = sorted([r for r in rows if r["short"] > 0], key=lambda r: -r["short"])
    surplus = sorted([dict(r) for r in rows if r["spare"] > 0],
                     key=lambda r: (r["nearest_expiry"] or date.max, -r["spare"]))
    out = []
    for need_row in shorts:
        left = need_row["short"]
        for src in surplus:
            if left <= 0:
                break
            if src["spare"] <= 0:
                continue
            plan = plan_transfer(src["barangay"].id, antigen.id, min(left, src["spare"]), round_up=False)
            doses = sum(d for _lot, d in plan)
            if not doses:
                continue
            out.append({"from": src["barangay"], "to": need_row["barangay"], "doses": doses,
                        "batches": ", ".join(f"{lot.batch_number} (exp {lot.expiry_date.strftime('%b %d, %Y') if lot.expiry_date else '-'})"
                                             for lot, _d in plan)})
            src["spare"] -= doses
            left -= doses
    return out


# --- Barangay stock alerts and CSV ------------------------------------------

STUCK_IN_TRANSIT_DAYS = 2      # in transit longer than this: follow up
STUCK_RESERVED_DAYS = 3        # reserved but not dispatched longer than this
LARGE_COUNT_GAP = 10           # a physical count this many doses short is flagged


def _codes(transfers, limit=4):  # "TRF-2026-001 (Awai), ..."
    text = ", ".join(f"{t.code} ({t.to_barangay.name})" for t in transfers[:limit])
    return text + (f" and {len(transfers) - limit} more" if len(transfers) > limit else "")


def barangay_stock_shortages(barangay_id=None, need=None):  # [(barangay, antigen, available, incoming, needed)]
    """Tracked vaccines whose available + incoming doses are below the 30-day
    need: one barangay (BHW) or all of them (RHU). Kept to a few queries
    because notifications are rebuilt on every page."""
    lots = BarangayStock.query.filter(BarangayStock.doses_on_hand > 0)
    tracked = db.session.query(StockMovement.barangay_id, StockMovement.antigen_id).filter(
        StockMovement.barangay_id.isnot(None))
    if barangay_id:
        lots = lots.filter(BarangayStock.barangay_id == barangay_id)
        tracked = tracked.filter(StockMovement.barangay_id == barangay_id)
    tracked = sorted(set(tracked.distinct().all()))
    if not tracked:  # nothing tracked yet: skip the per-child need calculation
        return []
    if need is None:
        need = doses_needed_by_barangay({bid for bid, _aid in tracked})
    on_hand = Counter()
    for lot in lots.all():
        if not lot.is_expired:
            on_hand[(lot.barangay_id, lot.antigen_id)] += lot.doses_on_hand
    reserved, incoming = Counter(), Counter()
    for t in StockTransfer.query.filter(StockTransfer.status.in_(("reserved", "in_transit", "returning"))).all():
        if t.status == "reserved":
            reserved[(t.from_barangay_id, t.antigen_id)] += t.doses_sent
            incoming[(t.to_barangay_id, t.antigen_id)] += t.doses_sent
        elif t.status == "in_transit":
            incoming[(t.to_barangay_id, t.antigen_id)] += t.doses_sent
        else:
            incoming[(t.from_barangay_id, t.antigen_id)] += t.doses_sent
    antigens = {a.id: a for a in core_antigens()}
    barangays = {b.id: b for b in Barangay.query.all()}
    out = []
    for bid, aid in tracked:
        antigen = antigens.get(aid)
        if antigen is None or bid not in barangays:
            continue
        key = (bid, aid)
        available = on_hand[key] - min(reserved[key], on_hand[key])
        needed = need.get(bid, {}).get(antigen.code, 0)
        if available + incoming[key] < needed:
            out.append((barangays[bid], antigen, available, incoming[key], needed))
    return out


def barangay_stock_notes(role, barangay_id=None):  # alerts for the bell
    now, today = datetime.utcnow(), date.today()
    notes = []
    stuck_cutoff = now - timedelta(days=STUCK_IN_TRANSIT_DAYS)
    reserve_cutoff = now - timedelta(days=STUCK_RESERVED_DAYS)
    soon = today + timedelta(days=STOCK_NEED_DAYS_AHEAD)
    if role == "rhu":
        stock_url = url_for("rhu.barangay_stock")
        transfers_url = url_for("rhu.barangay_stock", tab="transfers")
        stuck = (StockTransfer.query.filter(StockTransfer.status == "in_transit", StockTransfer.dispatched_at < stuck_cutoff)
                 .order_by(StockTransfer.dispatched_at).all())
        if stuck:
            notes.append({"category": "Stock", "severity": "critical", "icon": "clock", "icon_bg": "bg-red",
                          "title": f"{len(stuck)} Transfer{'s' if len(stuck) != 1 else ''} In Transit Over {STUCK_IN_TRANSIT_DAYS} Days",
                          "body": f"Not yet confirmed by the receiving barangay: {_codes(stuck)}. Check the vials reached a fridge.",
                          "time_label": "Follow up", "unread": True, "url": transfers_url})
        returning = (StockTransfer.query.filter(StockTransfer.status == "returning", StockTransfer.cancelled_at < stuck_cutoff)
                     .order_by(StockTransfer.cancelled_at).all())
        if returning:
            notes.append({"category": "Stock", "severity": "warning", "icon": "refresh", "icon_bg": "bg-orange",
                          "title": f"{len(returning)} Cancelled Transfer{'s' if len(returning) != 1 else ''} Not Returned",
                          "body": "Cancelled in transit, but the sender has not counted the vials back in: "
                                  + ", ".join(f"{t.code} ({t.from_barangay.name})" for t in returning[:4]) + ".",
                          "time_label": "Follow up", "unread": True, "url": transfers_url})
        waiting = (StockTransfer.query.filter(StockTransfer.status == "reserved", StockTransfer.created_at < reserve_cutoff)
                   .order_by(StockTransfer.created_at).all())
        if waiting:
            notes.append({"category": "Stock", "severity": "warning", "icon": "refresh", "icon_bg": "bg-orange",
                          "title": f"{len(waiting)} Redistribution{'s' if len(waiting) != 1 else ''} Not Dispatched",
                          "body": f"Reserved over {STUCK_RESERVED_DAYS} days ago and still in the sender's fridge: {_codes(waiting)}.",
                          "time_label": "Follow up", "unread": True, "url": transfers_url})
        week = now - timedelta(days=7)
        lost = [t for t in StockTransfer.query.filter(StockTransfer.received_at >= week).all()
                if any(i.doses_lost for i in t.items)]
        for t in lost:
            n = sum(i.doses_lost or 0 for i in t.items)
            where = t.to_barangay.name if t.status == "received" else t.from_barangay.name
            notes.append({"category": "Stock", "severity": "warning", "icon": "alert-triangle", "icon_bg": "bg-orange",
                          "title": f"{t.code}: {n} Doses Lost in Transit",
                          "body": f"{where} received {t.doses_received} of {t.doses_sent} {t.antigen.name} doses ("
                                  + ", ".join(sorted({i.loss_reason for i in t.items if i.loss_reason})) + ").",
                          "time_label": "This week", "unread": True, "url": url_for("rhu.barangay_stock", tab="discrepancies")})
        gaps = Counter()
        for m in StockMovement.query.filter(StockMovement.kind == "count_adjustment", StockMovement.doses < 0,
                                            StockMovement.created_at >= week, StockMovement.barangay_id.isnot(None)).all():
            gaps[(m.barangay_id, m.antigen_id)] += -m.doses
        for (bid, aid), n in sorted(gaps.items()):
            if n >= LARGE_COUNT_GAP:
                b, a = db.session.get(Barangay, bid), db.session.get(VaccineAntigen, aid)
                notes.append({"category": "Stock", "severity": "warning", "icon": "alert-triangle", "icon_bg": "bg-orange",
                              "title": f"Count Discrepancy: {b.name} {a.name}",
                              "body": f"Physical counts this week found {n} {a.name} doses fewer than the records.",
                              "time_label": "This week", "unread": True,
                              "url": url_for("rhu.barangay_stock", tab="discrepancies")})
        shortages = barangay_stock_shortages()
        if shortages:
            names = sorted({b.name for b, *_ in shortages})
            notes.append({"category": "Stock", "severity": "warning", "icon": "package", "icon_bg": "bg-orange",
                          "title": f"{len(names)} Barangay{'s' if len(names) != 1 else ''} Short on Vaccines",
                          "body": "; ".join(f"{b.name} {a.name} ({avail + inc} of {need} doses)"
                                            for b, a, avail, inc, need in shortages[:4])
                                  + (f"; and {len(shortages) - 4} more." if len(shortages) > 4 else ".")
                                  + " See Barangay Stock for suggested redistributions.",
                          "time_label": "Today", "unread": True, "url": stock_url})
        expiring = (BarangayStock.query.filter(BarangayStock.doses_on_hand > 0, BarangayStock.expiry_date >= today,
                                               BarangayStock.expiry_date <= soon).all())
        if expiring:
            by_barangay = Counter()
            for lot in expiring:
                by_barangay[lot.barangay.name if lot.barangay else "?"] += lot.doses_on_hand
            notes.append({"category": "Expiry", "severity": "warning", "icon": "hourglass", "icon_bg": "bg-amber",
                          "title": f"Barangay Stock Expiring Within {STOCK_NEED_DAYS_AHEAD} Days",
                          "body": ", ".join(f"{name}: {n} doses" for name, n in sorted(by_barangay.items())[:5])
                                  + ". Consider redistributing to where they will be used.",
                          "time_label": "Today", "unread": True, "url": stock_url})
    elif role == "bhw" and barangay_id:
        stock_url = url_for("bhw.stock")
        incoming = StockTransfer.query.filter(
            db.or_(db.and_(StockTransfer.to_barangay_id == barangay_id, StockTransfer.status == "in_transit"),
                   db.and_(StockTransfer.from_barangay_id == barangay_id, StockTransfer.status == "returning"))).all()
        for t in incoming:
            late = (t.dispatched_at or now) < stuck_cutoff
            source = ("returning to you" if t.status == "returning"
                      else "from the RHU" if t.kind == "delivery" else f"from {t.from_barangay.name}")
            notes.append({"category": "Request", "severity": "critical" if late else "info", "icon": "package",
                          "icon_bg": "bg-red" if late else "bg-blue",
                          "title": f"{t.code}: Confirm What Arrived",
                          "body": f"{t.doses_sent} {t.antigen.name} doses {source} are in transit"
                                  + (f" since {to_local(t.dispatched_at).strftime('%b %d')}" if t.dispatched_at else "")
                                  + ". Count them on arrival and confirm on My Stock.",
                          "time_label": "Action needed", "unread": True, "url": stock_url})
        for t in StockTransfer.query.filter_by(from_barangay_id=barangay_id, status="reserved").all():
            late = t.created_at and t.created_at < reserve_cutoff
            notes.append({"category": "Request", "severity": "warning" if late else "info", "icon": "refresh",
                          "icon_bg": "bg-orange" if late else "bg-blue",
                          "title": f"{t.code}: Reserved for {t.to_barangay.name}",
                          "body": f"Keep {t.doses_sent} {t.antigen.name} doses apart for {t.to_barangay.name} and do not use them. "
                                  + ("Reserved over 3 days ago: mark them dispatched once picked up."
                                     if late else "Mark them dispatched when they are picked up."),
                          "time_label": "Action needed", "unread": True, "url": stock_url})
        for b, a, avail, inc, need in barangay_stock_shortages(barangay_id):
            notes.append({"category": "Stock", "severity": "critical" if avail <= 0 else "warning", "icon": "package",
                          "icon_bg": "bg-red" if avail <= 0 else "bg-orange",
                          "title": f"Barangay {a.name} {'Out of Stock' if avail <= 0 else 'Low'}",
                          "body": f"{avail} {a.name} doses available" + (f" (+{inc} on the way)" if inc else "")
                                  + f"; your children need {need} in the next {STOCK_NEED_DAYS_AHEAD} days. Request more from the RHU.",
                          "time_label": "Today", "unread": True, "url": url_for("bhw.requests")})
        expiring = (BarangayStock.query.filter(BarangayStock.barangay_id == barangay_id, BarangayStock.doses_on_hand > 0,
                                               BarangayStock.expiry_date >= today, BarangayStock.expiry_date <= soon)
                    .order_by(BarangayStock.expiry_date).all())
        for lot in expiring:
            notes.append({"category": "Expiry", "severity": "warning", "icon": "hourglass", "icon_bg": "bg-amber",
                          "title": f"{lot.antigen.name} Batch {lot.batch_number} Expiring",
                          "body": f"{lot.doses_on_hand} doses expire on {lot.expiry_date.strftime('%b %d, %Y')}. "
                                  "Use them first; the RHU may redistribute what you cannot use.",
                          "time_label": "Today", "unread": True, "url": stock_url})
        expired = BarangayStock.query.filter(BarangayStock.barangay_id == barangay_id, BarangayStock.doses_on_hand > 0,
                                             BarangayStock.expiry_date < today).all()
        if expired:
            notes.append({"category": "Expiry", "severity": "critical", "icon": "hourglass", "icon_bg": "bg-red",
                          "title": "Expired Doses to Dispose",
                          "body": f"{sum(l.doses_on_hand for l in expired)} expired doses are still counted in your stock. "
                                  "Dispose of them on My Stock.",
                          "time_label": "Action needed", "unread": True, "url": stock_url})
    return notes


def _csv_cell(value):  # dates readable; text that Excel would run as a formula is quoted
    if isinstance(value, datetime):
        return to_local(value).strftime("%Y-%m-%d %H:%M")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return "" if value is None else value


def _csv_response(title, header_rows, columns, rows, filename, log):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([f"ImmunoVision - {title}"])
    w.writerow(["San Jacinto Rural Health Unit, Pangasinan"])
    for row in header_rows:
        w.writerow(row)
    w.writerow(["Generated", datetime.now().strftime("%b %d, %Y %I:%M %p")])
    w.writerow([])
    w.writerow(columns)
    for row in rows:
        w.writerow([_csv_cell(v) for v in row])
    log_activity(log)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


def movement_csv_rows(movements):  # stock history rows
    return [[m.created_at, m.stock.barangay.name if m.stock and m.stock.barangay else
             (db.session.get(Barangay, m.barangay_id).name if m.barangay_id else "RHU"),
             m.antigen.name, MOVEMENT_LABELS.get(m.kind, m.kind), m.doses,
             m.stock.batch_number if m.stock else "",
             db.session.get(StockTransfer, m.transfer_id).code if m.transfer_id else "",
             m.user.full_name if m.user else "", m.note or ""] for m in movements]


MOVEMENT_CSV_COLUMNS = ["Date", "Barangay", "Vaccine", "Change", "Doses", "Batch", "Transfer", "By", "Note"]


def discrepancy_rows(days=90):  # per barangay: where the system and the fridge disagreed
    since = datetime.utcnow() - timedelta(days=days)
    rows = defaultdict(lambda: Counter())
    for m in StockMovement.query.filter(StockMovement.created_at >= since, StockMovement.barangay_id.isnot(None),
                                        StockMovement.kind.in_(("count_adjustment", "no_stock"))).all():
        if m.kind == "no_stock":
            rows[m.barangay_id]["no_stock"] += 1
        elif m.doses < 0:
            rows[m.barangay_id]["count_missing"] += -m.doses
        else:
            rows[m.barangay_id]["count_extra"] += m.doses
    for t in StockTransfer.query.filter(StockTransfer.received_at >= since).all():
        lost = sum(i.doses_lost or 0 for i in t.items)
        if lost:
            rows[t.to_barangay_id if t.status == "received" else t.from_barangay_id]["transit_lost"] += lost
    for t in StockTransfer.query.filter(StockTransfer.created_at >= since, StockTransfer.doses_reduced > 0).all():
        rows[t.from_barangay_id]["reserved_used"] += t.doses_reduced
    names = {b.id: b.name for b in Barangay.query.all()}
    return sorted(({"barangay": names.get(bid, "?"), **counts} for bid, counts in rows.items() if bid),
                  key=lambda r: r["barangay"])


def stock_on_dose_change(child, vt, rec, previous, new, given_elsewhere=False):  # keep stock in step with dose records
    """Called whenever a dose date is saved. A dose recorded by a BHW, dated on
    or after the day stock tracking began, takes 1 dose from the barangay
    (soonest expiry first). Removing a dose, or moving it to before tracking
    began, returns it to the batch it came from. Recording is never blocked:
    with no stock on hand the dose is saved and flagged instead.
    given_elsewhere: the child transferred in and got this dose at another
    health center, so no stock is used here (logged for the audit trail)."""
    if vt is None or child.barangay_id is None or previous == new:
        return
    db.session.flush()  # make sure rec has an id
    bid, antigen_id = child.barangay_id, vt.antigen_id
    taken = StockMovement.query.filter_by(vaccination_record_id=rec.id).filter(
        StockMovement.kind.in_(("administered", "reversal"))).all()
    consumed = sum(m.doses for m in taken) < 0
    start = tracking_start(bid, antigen_id)
    counts = new is not None and start is not None and new >= start
    if consumed and not counts:  # removed, or moved to before tracking: give it back
        last = next(m for m in sorted(taken, key=lambda m: -m.id) if m.kind == "administered")
        if last.stock is not None:
            last.stock.doses_on_hand += 1
        _record_movement(bid, antigen_id, "reversal", 1, lot=last.stock, record=rec,
                         note=f"{vt.name} for {child.display_name} removed or corrected")
    elif not consumed and counts and previous is None and current_user.is_authenticated \
            and current_user.role == "bhw":
        lot, dips_into_reserved = pick_lot_for_dose(bid, antigen_id)
        if given_elsewhere:
            _record_movement(bid, antigen_id, "given_elsewhere", 0, record=rec,
                             note=f"{vt.name} for {child.display_name}: given elsewhere (transferred in)")
        elif lot is not None:
            lot.doses_on_hand -= 1
            _record_movement(bid, antigen_id, "administered", -1, lot=lot, record=rec,
                             note=f"{vt.name} for {child.display_name}"
                                  + (" (reserved dose used)" if dips_into_reserved else ""))
            if dips_into_reserved:  # only reserved doses were left: shrink the transfer
                reason = (request.form.get("reserved_reason") or "").strip() if has_request_context() else ""
                reason = reason if reason else "dose recorded for a child"
                for message in reconcile_reservations(bid, antigen_id, f"{child.display_name}: {reason}"):
                    flash(message, "warning")
        else:
            _record_movement(bid, antigen_id, "no_stock", 0, record=rec,
                             note=f"{vt.name} for {child.display_name}: recorded with no stock on hand")


def delivery_of(vaccine_request):  # request -> its delivery
    """The StockTransfer created when a request was distributed, if any
    (requests distributed before barangay stock existed have none)."""
    return StockTransfer.query.filter_by(request_id=vaccine_request.id, kind="delivery").first()


def preview_request_doses(antigen_id, vials):  # (doses, "5 x 20 + 5 x 10")
    """Doses a request of `vials` would give the barangay, from the batches
    distribution would take (soonest expiry first); vials beyond current
    stock use the vaccine's usual vial size."""
    parts, left = [], vials
    for b in _usable_batches(antigen_id):
        if left <= 0:
            break
        take = min(b.quantity_on_hand, left)
        parts.append((take, batch_doses_per_vial(b)))
        left -= take
    if left > 0:
        parts.append((left, default_doses_per_vial(db.session.get(VaccineAntigen, antigen_id))))
    merged = Counter()
    for n, per_vial in parts:
        merged[per_vial] += n
    return sum(n * pv for pv, n in merged.items()), " + ".join(f"{n} × {pv}" for pv, n in merged.items())


@rhu_bp.route("/barangay-stock")
def barangay_stock():  # every barangay's stock, redistribution and discrepancies
    antigens = core_antigens()
    chosen_id = _positive_int(request.args.get("antigen_id"))
    antigen = next((a for a in antigens if a.id == chosen_id), antigens[0])
    tab = request.args.get("tab") if request.args.get("tab") in ("stock", "transfers", "discrepancies") else "stock"
    need = doses_needed_by_barangay()
    status_counts = {a.id: Counter(r["status"] for r in municipal_stock_rows(a, need)) for a in antigens}
    rows = municipal_stock_rows(antigen, need)
    active_transfers = (StockTransfer.query.filter(StockTransfer.kind == "redistribution",
                                         StockTransfer.status.in_(("reserved", "in_transit", "returning")))
              .order_by(StockTransfer.id.desc()).all())
    recent = (StockTransfer.query.filter(StockTransfer.kind == "redistribution",
                                         StockTransfer.status.in_(("received", "cancelled")))
              .order_by(StockTransfer.id.desc()).limit(30).all())
    return render_template("rhu_barangay_stock.html", **_rhu_ctx(
        "barangay_stock", page_title="Barangay Stock", antigens=antigens, antigen=antigen, tab=tab,
        rows=rows, status_counts=status_counts, suggestions=suggest_transfers(antigen, rows),
        active_transfers=active_transfers, recent=recent, discrepancies=discrepancy_rows(),
        barangays=Barangay.query.order_by(Barangay.name).all(),
        status_labels=TRANSFER_STATUS_LABELS, status_colors=TRANSFER_STATUS_COLORS,
        surplus_factor=SURPLUS_FACTOR, need_ahead=STOCK_NEED_DAYS_AHEAD, need_behind=STOCK_NEED_DAYS_BEHIND))


@rhu_bp.route("/barangay-stock/export.csv")
def barangay_stock_csv():  # CSV of the Barangay Stock page
    kind = request.args.get("kind", "stock")
    stamp = f"{date.today():%Y%m%d}"
    if kind == "transfers":
        rows = []
        for t in StockTransfer.query.order_by(StockTransfer.id).all():
            rows.append([t.code, "Delivery from RHU" if t.kind == "delivery" else "Redistribution", t.antigen.name,
                         t.from_barangay.name if t.from_barangay else "RHU", t.to_barangay.name,
                         TRANSFER_STATUS_LABELS.get(t.status, t.status) + (" (returned)" if t.status == "cancelled" and t.received_at else ""),
                         t.doses_sent, t.doses_received if t.received_at else "",
                         sum(i.doses_lost or 0 for i in t.items) if t.received_at else "",
                         "; ".join(sorted({i.loss_reason for i in t.items if i.loss_reason})),
                         t.doses_reduced or 0, t.reduce_reason or "", t.reason or "",
                         t.created_at, t.dispatched_at, t.received_at, t.cancelled_at])
        return _csv_response("Vaccine Transfers", [], [
            "Code", "Type", "Vaccine", "From", "To", "Status", "Doses sent/reserved", "Doses received", "Doses lost",
            "Loss reason", "Reserved doses used by sender", "Reduction reason", "Reason", "Created", "Dispatched",
            "Received/returned", "Cancelled"], rows, f"immunovision_transfers_{stamp}.csv",
            "Exported vaccine transfers (CSV)")
    if kind == "movements":
        days = request.args.get("days", type=int) or 90
        days = max(1, min(days, 3650))
        since = datetime.utcnow() - timedelta(days=days)
        moves = (StockMovement.query.filter(StockMovement.created_at >= since)
                 .order_by(StockMovement.created_at, StockMovement.id).all())
        return _csv_response("Barangay Stock History", [["Period", f"Last {days} days"]], MOVEMENT_CSV_COLUMNS,
                             movement_csv_rows(moves), f"immunovision_stock_history_{stamp}.csv",
                             "Exported barangay stock history (CSV)")
    if kind == "discrepancies":
        rows = [[d["barangay"], d.get("count_missing", 0), d.get("count_extra", 0), d.get("transit_lost", 0),
                 d.get("no_stock", 0), d.get("reserved_used", 0)] for d in discrepancy_rows()]
        return _csv_response("Stock Discrepancy Report", [["Period", "Last 90 days"]], [
            "Barangay", "Count: doses missing", "Count: doses extra", "Lost in transit", "Doses given with no stock",
            "Reserved doses used"], rows, f"immunovision_discrepancies_{stamp}.csv",
            "Exported stock discrepancy report (CSV)")
    need = doses_needed_by_barangay()
    rows = []
    for a in core_antigens():
        for r in municipal_stock_rows(a, need):
            tracked = r["status"] != "Not tracked"
            rows.append([a.name, r["barangay"].name, r["status"]] +
                        ([r["in_fridge"], r["reserved"], r["incoming"], r["available"]] if tracked else ["", "", "", ""]) +
                        [r["needed"], r["spare"] or "", r["short"] or "", r["expiring"] if tracked else "",
                         r["nearest_expiry"]])
    return _csv_response("Barangay Stock", [["Needed", f"Doses due in the next {STOCK_NEED_DAYS_AHEAD} days or "
                                              f"overdue by up to {STOCK_NEED_DAYS_BEHIND} days"]], [
        "Vaccine", "Barangay", "Status", "In fridge", "Reserved", "Incoming", "Available", "Needed",
        "Can be sent", "Short by", f"Expiring within {STOCK_NEED_DAYS_AHEAD} days", "Nearest expiry"],
        rows, f"immunovision_barangay_stock_{stamp}.csv", "Exported barangay stock (CSV)")


@rhu_bp.route("/transfers/new", methods=["POST"])
def transfer_create():  # reserve doses for redistribution
    """Doses are only reserved: they stay in the sender's fridge, set aside,
    until its BHW marks them dispatched."""
    f = request.form
    src = db.session.get(Barangay, _positive_int(f.get("from_barangay_id")) or 0)
    dst = db.session.get(Barangay, _positive_int(f.get("to_barangay_id")) or 0)
    antigen = next((a for a in core_antigens() if a.id == _positive_int(f.get("antigen_id"))), None)
    doses = _positive_int(f.get("doses"))
    back = url_for("rhu.barangay_stock", antigen_id=antigen.id if antigen else None, tab="transfers")
    error = ("Please choose a vaccine." if antigen is None else
             "Please choose the sending and receiving barangays." if src is None or dst is None else
             "The sending and receiving barangay must be different." if src.id == dst.id else
             "Enter the number of doses to move (1 or more)." if doses is None else None)
    if error:
        flash(error, "error")
        return redirect(back)
    plan = plan_transfer(src.id, antigen.id, doses)
    if not plan:
        flash(f"{src.name} has only {transferable_doses(src.id, antigen.id)} {antigen.name} doses in unopened, "
              "unreserved vials. Vials are moved whole, so opened vials cannot be sent.", "error")
        return redirect(back)
    t = StockTransfer(code=_next_transfer_code(), kind="redistribution", from_barangay_id=src.id,
                      to_barangay_id=dst.id, antigen_id=antigen.id, status="reserved",
                      reason=(f.get("reason") or "").strip()[:255] or None, created_by_id=current_user.id,
                      doses_reduced=0)
    db.session.add(t)
    db.session.flush()
    for lot, d in plan:
        db.session.add(StockTransferItem(transfer_id=t.id, batch_number=lot.batch_number, expiry_date=lot.expiry_date,
                                         doses_per_vial=lot.doses_per_vial, doses_sent=d, source_stock_id=lot.id))
    db.session.commit()
    total = sum(d for _lot, d in plan)
    log_activity(f"Reserved {total} {antigen.name} doses for redistribution {t.code} from {src.name} to {dst.name}")
    flash(f"{t.code}: {total} {antigen.name} doses reserved in {src.name} for {dst.name}"
          + (f" (rounded up from {doses} to whole vials)" if total != doses else "")
          + f". {src.name}'s BHW marks them dispatched when they are picked up.", "success")
    return redirect(back)


def _cancel_transfer(t, reason, back):  # shared by the RHU and the sending BHW
    if not reason:
        flash("Please give a reason for cancelling.", "error")
        return redirect(back)
    t.reason = (f"{t.reason}; " if t.reason else "") + f"Cancelled: {reason}"
    t.reason = t.reason[:255]
    t.cancelled_by_id, t.cancelled_at = current_user.id, datetime.utcnow()
    if t.status == "reserved":
        t.status = "cancelled"
        message = f"{t.code} cancelled; the {t.doses_sent} reserved {t.antigen.name} doses in {t.from_barangay.name} are available again."
    else:  # in transit: the vials go back and the sender counts them in
        t.status = "returning"
        message = (f"{t.code} cancelled while in transit; {t.from_barangay.name}'s BHW will count the "
                   f"{t.doses_sent} {t.antigen.name} doses back in when they return.")
    db.session.commit()
    log_activity(f"Cancelled redistribution {t.code} ({t.antigen.name}, {t.from_barangay.name} to "
                 f"{t.to_barangay.name}): {reason}")
    flash(message, "success")
    return redirect(back)


@rhu_bp.route("/transfers/<int:transfer_id>/cancel", methods=["POST"])
def transfer_cancel(transfer_id):
    t = db.session.get(StockTransfer, transfer_id) or abort(404)
    back = url_for("rhu.barangay_stock", antigen_id=t.antigen_id, tab="transfers")
    if t.kind != "redistribution" or t.status not in ("reserved", "in_transit"):
        flash(f"{t.code} can no longer be cancelled.", "error")
        return redirect(back)
    return _cancel_transfer(t, (request.form.get("reason") or "").strip()[:200], back)


@rhu_bp.route("/transfers/<int:transfer_id>/slip")
def transfer_slip(transfer_id):
    t = db.session.get(StockTransfer, transfer_id) or abort(404)
    return render_template("transfer_slip.html", t=t, status_labels=TRANSFER_STATUS_LABELS)


@rhu_bp.route("/requests/<int:req_id>/<action>", methods=["POST"])
def request_action(req_id, action):  # approve/reject/distribute
    vr = db.session.get(VaccineRequest, req_id) or abort(404)
    # Each step only from the status before it, so a request cannot be
    # approved twice, distributed without approval, or distributed twice
    # (which would deduct the stock twice).
    allowed_from = {"approve": "pending", "reject": "pending", "distribute": "approved"}  # allowed status
    if action not in allowed_from:
        abort(404)
    if vr.status != allowed_from[action]:  # wrong status
        flash(f"Request {vr.request_code} is {vr.status}; it cannot be {action}d.", "error")
        return redirect(request.referrer or url_for("rhu.requests"))
    if action == "approve":  # approve
        # Defaults to the full amount (the plain Approve button on the
        # Requests list never sends a quantity); the Stock Allocation page
        # sends its suggested or RHU-adjusted quantity instead, so stock can
        # actually be rationed rather than only approved in full or refused.
        qty_raw = (request.form.get("quantity") or "").strip()  # quantity
        if not qty_raw:
            qty = vr.quantity_requested       # plain Approve: the full amount
        elif qty_raw.isdigit() and 0 < int(qty_raw) <= vr.quantity_requested:  # 1..requested
            qty = int(qty_raw)
        else:
            # A typed 0, a value above the request, or text is a mistake to
            # report - never silently approve the full request instead.
            flash(f"Enter a quantity from 1 to {vr.quantity_requested} to approve {vr.request_code}.", "error")
            return redirect(request.referrer or url_for("rhu.requests"))
        vr.status, vr.quantity_approved = "approved", qty  # approved
        vr.reviewed_by_id, vr.reviewed_at = current_user.id, datetime.utcnow()
        suffix = f" ({qty} of {vr.quantity_requested} requested)" if qty < vr.quantity_requested else ""
        log_activity(f"Approved vaccine request {vr.request_code}{suffix}")  # log
    elif action == "reject":  # reject
        vr.status, vr.reviewed_by_id, vr.reviewed_at = "rejected", current_user.id, datetime.utcnow()
        log_activity(f"Rejected vaccine request {vr.request_code}")  # log
    elif action == "distribute":  # distribute
        qty = vr.quantity_approved or vr.quantity_requested  # amount
        taken = _deduct_stock(vr.antigen_id, qty)  # deduct
        if not taken:
            flash(f"Not enough usable {vr.antigen.name} stock to distribute {qty} for {vr.request_code}. "
                  "Add stock in Vaccine Inventory first; the request stays approved.", "error")
            return redirect(request.referrer or url_for("rhu.requests"))
        vr.status, vr.distributed_at = "fulfilled", datetime.utcnow()  # fulfilled
        # The vials are now on their way: in transit until the BHW confirms
        # what arrived, never straight into the barangay's stock.
        delivery = StockTransfer(code=_next_transfer_code(), kind="delivery", from_barangay_id=None,
                                 to_barangay_id=vr.barangay_id, antigen_id=vr.antigen_id, request_id=vr.id,
                                 status="in_transit", created_by_id=current_user.id,
                                 dispatched_by_id=current_user.id, dispatched_at=datetime.utcnow())
        db.session.add(delivery)
        db.session.flush()
        for batch, vials in taken:
            per_vial = batch_doses_per_vial(batch)
            db.session.add(StockTransferItem(transfer_id=delivery.id, batch_number=batch.batch_number,
                                             expiry_date=batch.expiry_date, doses_per_vial=per_vial,
                                             doses_sent=vials * per_vial))
        log_activity(f"Distributed vaccine request {vr.request_code} (delivery {delivery.code}, "
                     f"{qty} vials in transit to {vr.barangay.name})")  # log
    db.session.commit()  # save
    flash(f"Request {vr.request_code} {action}d.", "success")
    return redirect(request.referrer or url_for("rhu.requests"))


@rhu_bp.route("/forecast")
def forecast():  # forecast page
    """Projected vaccine demand for the coming years.

    Two components, kept separate on screen because only one is a forecast:
    doses children already in the registry will become due for, and doses for a
    projected birth cohort. RHU only, since ordering stock is their decision."""
    children = _children_query(None).all()
    dobs = [c.date_of_birth for c in children if c.date_of_birth]  # birth dates
    births_by_year = Counter(d.year for d in dobs)  # births per year
    data = dp.forecast_vaccine_demand(dobs, births_by_year)  # forecast

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


@rhu_bp.route("/reports")
def reports():  # reports page
    barangay_id = request.args.get("barangay_id", type=int)
    report_type = request.args.get("type", "coverage")
    as_of, report_key = resolve_report_date(request.args)  # report date
    fig = report_figures(barangay_id, as_of)  # numbers
    barangays = Barangay.query.order_by(Barangay.name).all()
    barangay_obj = next((b for b in barangays if b.id == barangay_id), None)

    extra = {}
    if report_type == "atrisk":  # at-risk report
        extra["at_risk_rows"] = at_risk_table(barangay_id, limit=50)
    elif report_type == "inventory":  # inventory report
        extra["inventory_rows"] = stock_levels()
    elif report_type == "barangay":  # barangay report
        extra["barangay_obj"] = barangay_obj
        if barangay_obj:
            extra["barangay_target_coverage"] = _target_coverage(fig["fully_immunized"], barangay_obj.target_population)
    elif report_type == "monthly":  # trend report
        extra["trend"] = yearly_trend(barangay_id)
        extra["doses_by_month"] = monthly_doses_administered(barangay_id, as_of=as_of)

    return render_template("rhu_reports.html", **_rhu_ctx(
        "reports", page_title="Reports", report_type=report_type,
        stats=fig, by_barangay=fig["by_barangay"], generated_at=datetime.now(),
        barangays=barangays, scope_name=barangay_obj.name if barangay_obj else "All Barangays",
        selected_barangay=barangay_id, date_options=report_date_options(), report_key=report_key,
        as_of=as_of, **extra,
    ))


@rhu_bp.route("/reports/export.csv")
def export_report_csv():  # CSV export
    """Downloads the on-screen report as CSV (opens directly in Excel)."""
    barangay_id = request.args.get("barangay_id", type=int)
    as_of, _key = resolve_report_date(request.args)  # report date
    fig = report_figures(barangay_id, as_of)  # numbers
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


@rhu_bp.route("/campaigns/<int:campaign_id>")
def campaign_worklist(campaign_id):
    """A focused list of just the children eligible for one campaign right
    now, one date field each, so recording a round of campaign doses doesn't
    mean opening each child's full Edit Child form and scrolling past the
    entire core schedule to reach the one field that matters today."""
    campaign = db.session.get(Campaign, campaign_id) or abort(404)
    barangay_id = request.args.get("barangay_id", type=int)
    pending_only = request.args.get("pending") == "1"
    search = request.args.get("q", "").strip()
    kids = eligible_children_for_campaign(campaign, barangay_id)
    if search:
        kids = [c for c in kids if search.lower() in c.full_name.lower()]
    recs = {r.child_id: r.date_administered for r in
            VaccinationRecord.query.filter_by(campaign_id=campaign.id).all()}
    rows = [{"child": c, "dose_date": recs.get(c.id)} for c in kids]
    if pending_only:
        rows = [r for r in rows if not r["dose_date"]]
    rows.sort(key=lambda r: (r["dose_date"] is not None, r["child"].full_name))
    return render_template("campaign_worklist.html", **_rhu_ctx(
        "dashboard", page_title=campaign.name, campaign=campaign, rows=rows,
        barangays=Barangay.query.order_by(Barangay.name).all(),
        selected_barangay_id=barangay_id, pending_only=pending_only, search=search,
    ))


@rhu_bp.route("/campaigns/<int:campaign_id>/record/<int:child_id>", methods=["POST"])
def campaign_record_dose(campaign_id, child_id):
    campaign = db.session.get(Campaign, campaign_id) or abort(404)
    child = db.session.get(Child, child_id) or abort(404)
    if not (campaign.age_eligible(child.age_in_days) and campaign.barangay_eligible(child.barangay_id)):
        abort(400)
    date_str = request.form.get("dose_date")
    administered = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
    rec = child.vaccination_records.filter_by(campaign_id=campaign.id).first()
    date_error = _dose_date_error(child.date_of_birth, [(campaign.name, administered)])
    if date_error:
        flash(date_error, "error")
    else:
        if rec is None and administered is not None:
            rec = VaccinationRecord(child_id=child.id, vaccine_type_id=campaign.vaccine_type_id, campaign_id=campaign.id)
            db.session.add(rec)
        if rec is not None:
            previous = rec.date_administered
            rec.date_administered = administered
            rec.status = "completed" if administered else "pending"
            stock_on_dose_change(child, db.session.get(VaccineType, campaign.vaccine_type_id),
                                 rec, previous, administered)  # barangay stock
            db.session.commit()
            if previous is not None and previous != administered:
                _log_dose_date_changes(child, [(campaign.name, previous, administered)])
            else:
                log_activity(f"Recorded {campaign.name} dose for {child_label(child)}")
            flash(f"Saved {child.full_name}'s dose.", "success")
    return redirect(url_for("rhu.campaign_worklist", campaign_id=campaign.id,
                             barangay_id=request.form.get("barangay_id") or None,
                             pending=request.form.get("pending") or None,
                             q=request.form.get("q") or None))


@rhu_bp.route("/notifications")
def notifications():
    notes = get_notifications("rhu")
    ctx = _rhu_ctx("notifications", page_title="Notifications", notes=notes)
    _mark_notifications_seen(notes, current_user)
    return render_template("rhu_notifications.html", **ctx)


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
    # A BHW sees only their own barangay. Without one (new account, or the
    # admin cleared it) every barangay filter would be empty and pages would
    # show the whole municipality, so only Settings stays open.
    if current_user.barangay_id is None and request.endpoint not in ("bhw.settings", "bhw.notifications"):
        flash("Your account is not assigned to a barangay yet. Ask the administrator to assign one; "
              "until then only Settings is available.", "warning")
        return redirect(url_for("bhw.settings"))


@bhw_bp.route("/dashboard")
def dashboard():
    bid = current_user.barangay_id
    risk_level = request.args.get("risk_level", "")
    status = request.args.get("status", "")
    period = _dashboard_period()
    as_of, born_from = period["as_of"], period["born_from"]
    risk = risk_as_of(as_of)
    kids = filtered_children(bid, risk_level, status, as_of, born_from, risk)
    # The BHW does the home visits, so the "who to chase" list matters most
    # here. "View All (N)" reads straight off follow_up|length in the
    # template, so the full (unlimited) list is what gets passed - slicing
    # it here would silently cap that count at 5 too.
    follow_up_all = at_risk_table(bid, max_age_days=CONTINUATION_WINDOW_DAYS)
    follow_up = follow_up_all[:5]

    # Coverage Analytics merged into this page; its own route was retired.
    by_vaccine = order_by_schedule(report_by_vaccine(bid, as_of, born_from) if as_of
                                   else coverage_by_vaccine(bid))
    trend_info = trend_for(bid, period)
    trend = trend_info["trend"]
    change = coverage_change(bid)
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    due_calendar = _due_calendar(filtered_children(bid), vt_lookup)
    return render_template("bhw_dashboard.html", **_bhw_ctx(
        "dashboard", page_title="Dashboard", stats=stats_for(kids, as_of, risk),
        risk_dist=risk_distribution_for(kids, risk), now=datetime.now(),
        risk_levels=RISK_LEVELS, status_levels=STATUS_LEVELS,
        risk_level=risk_level, status=status,
        **period, follow_up=follow_up,
        follow_up_total=len(follow_up_all),
        by_vaccine=by_vaccine, change=change,
        vaccine_colours=vaccine_colours(by_vaccine), vaccine_labels=schedule_labels(by_vaccine),
        insights=chart_insights(by_vaccine, [], trend, trend_info["trend_year"]), **trend_info,
        total_in_barangay=len(filtered_children(bid, as_of=as_of, born_from=born_from)),
        due_calendar=due_calendar,
        pending_requests=(VaccineRequest.query.filter_by(barangay_id=bid, status="pending")
                          .order_by(VaccineRequest.requested_at.desc()).first()),
        active_campaigns=active_campaigns_summary(bid),
    ))


@bhw_bp.route("/children", methods=["GET", "POST"])
def children():
    bid = current_user.barangay_id
    if request.method == "POST":
        # BHW Add Child only: reject future-age checkboxes even if the browser
        # restrictions are bypassed. Historical administration dates remain editable.
        try:
            dob = datetime.strptime(request.form.get("date_of_birth", ""), "%Y-%m-%d").date()
        except ValueError:
            flash("Please enter a valid date of birth.", "error")
            return redirect(url_for("bhw.children", add=1))
        age_days = (date.today() - dob).days
        if age_days < 0:
            flash("Date of birth cannot be in the future.", "error")
            return redirect(url_for("bhw.children", add=1))
        premature = [name for code, name, _ant, _num, due_days in dp.VACCINE_SCHEDULE
                     if request.form.get(f"dose_{code}_given") and age_days < due_days]
        if premature:
            flash("Not yet age-eligible: " + ", ".join(premature) + ". Please review the date of birth.", "error")
            return redirect(url_for("bhw.children", add=1))
        error = _create_form_errors(request.form)
        if error:
            flash(error, "error")
            return redirect(url_for("bhw.children", add=1))
        new_child = _create_child(request.form, barangay_locked=bid)
        flash("Child registered successfully.", "success")
        return redirect(url_for("bhw.children", new_child=new_child.id))
    risk_level = request.args.get("risk_level", "")
    status = request.args.get("status", "")
    search = request.args.get("q", "").strip()
    selected_year = request.args.get("year", "")
    selected_sort = request.args.get("sort", "newest")
    age_group = request.args.get("age_group", "all")
    if selected_sort not in ("newest", "oldest", "youngest", "oldest_child"):
        selected_sort = "newest"
    if age_group not in ("active", "older", "all"):
        age_group = "active"
    q = Child.query.filter_by(barangay_id=bid)
    if search:
        q = q.filter(Child.full_name.ilike(f"%{search}%"))
    if selected_year:
        if selected_year.isdigit() and len(selected_year) == 4:
            q = q.filter(Child.date_registered >= date(int(selected_year), 1, 1),
                         Child.date_registered < date(int(selected_year) + 1, 1, 1))
        else:
            selected_year = ""
    # Calendar-month cutoff: the 12-month birthday is still in the active view.
    today = date.today()
    def twelve_month_birthday(dob):
        try:
            return dob.replace(year=dob.year + 1)
        except ValueError:  # Feb 29 in a non-leap year
            return dob.replace(year=dob.year + 1, day=28)
    kids = q.all()
    if age_group == "active":
        kids = [c for c in kids if twelve_month_birthday(c.date_of_birth) >= today]
    elif age_group == "older":
        kids = [c for c in kids if twelve_month_birthday(c.date_of_birth) < today]
    if selected_sort == "newest":
        kids.sort(key=lambda c: (c.date_registered or date.min, c.created_at or datetime.min, c.id), reverse=True)
    elif selected_sort == "oldest":
        kids.sort(key=lambda c: (c.date_registered or date.min, c.created_at or datetime.min, c.id))
    elif selected_sort == "youngest":
        kids.sort(key=lambda c: (c.date_of_birth, c.id), reverse=True)
    else:
        kids.sort(key=lambda c: (c.date_of_birth, c.id))
    registration_years = sorted({str(y[0]) for y in db.session.query(
        func.strftime('%Y', Child.date_registered)).filter(
        Child.barangay_id == bid, Child.date_registered.isnot(None)).distinct().all() if y[0]}, reverse=True)

    # Just registered: show what she can act on today instead of making her
    # open the new record and scan all 15 doses to find out.
    due_now_child, due_now_doses = None, []
    new_child_id = request.args.get("new_child", type=int)
    if new_child_id:
        candidate = db.session.get(Child, new_child_id)
        if candidate and candidate.barangay_id == bid:
            due_now_child = candidate
            due_now_doses = [d for d in child_detail(candidate)["schedule"] if d["status"] in ("due", "overdue")]
    kids = _filter_child_list(kids, risk_level, status)
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    return render_template("bhw_children.html", **_bhw_ctx(
        "children", page_title="Child Records", children=kids, barangays=None,
        vaccine_schedule=dp.VACCINE_SCHEDULE, vt_lookup=vt_lookup, selected_risk=risk_level, selected_status=status, search=search,
        selected_year=selected_year, selected_sort=selected_sort, age_group=age_group, registration_years=registration_years,
        dose_matrix=_dose_matrix(kids, vt_lookup), risk_labels=_risk_labels(kids),
        due_now_child=due_now_child, due_now_doses=due_now_doses,
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


@bhw_bp.route("/due-vaccines")
def due_vaccines():
    """Month-by-month list of every dose the barangay's children still need,
    with a date field and Record button on each row, so working through a
    month's doses doesn't mean opening each child's full Edit Child form."""
    bid = current_user.barangay_id
    reserved_only = {}  # dose code -> the transfers its remaining doses are reserved for
    for antigen_id, transfers in reserved_only_antigens(bid).items():
        code = db.session.get(VaccineAntigen, antigen_id).code
        label = ", ".join(f"{t.code} to {t.to_barangay.name}" for t in transfers)
        for dose_code, antigen_code in dp.ANTIGEN_OF.items():
            if antigen_code == code:
                reserved_only[dose_code] = label
    return _render_due_list(_bhw_ctx, filtered_children(bid), can_record=True,
                            scope_label=current_user.barangay.name if current_user.barangay else "",
                            reserved_only=reserved_only, reserved_reasons=RESERVED_USE_REASONS)


def _due_list_return_url():
    f = request.form
    return url_for("bhw.due_vaccines", q=f.get("q") or None, status=f.get("status") or None,
                   month=f.get("month") or None, overdue=f.get("overdue") or None)


@bhw_bp.route("/children/<int:child_id>/record-dose", methods=["POST"])
def record_dose(child_id):
    """Records one dose from the Vaccination Due List, without opening the
    full Edit Child form - the same single-field shortcut the campaign
    worklist gives for campaign doses, but for routine schedule doses,
    which do feed into risk scoring."""
    child = db.session.get(Child, child_id) or abort(404)
    if child.barangay_id != current_user.barangay_id:
        abort(403)
    vt = VaccineType.query.filter_by(code=request.form.get("vaccine_code")).first() or abort(404)
    date_str = request.form.get("date")
    administered = _iso_date(date_str) if date_str else date.today()
    if administered is None:
        flash("Please enter a valid date.", "error")
        return redirect(_due_list_return_url())
    date_error = _dose_date_error(child.date_of_birth, [(vt.name, administered)])
    if not date_error and vt.code in dp.VACCINE_CODES:
        dates = _core_dose_dates(child)
        dates[vt.code] = administered
        date_error = _schedule_error(child.date_of_birth, dates, [vt.code])
    if date_error:
        flash(date_error, "error")
        return redirect(_due_list_return_url())
    rec = child.vaccination_records.filter_by(vaccine_type_id=vt.id, campaign_id=None).first()
    warning = _write_dose(child, vt, rec, administered)
    db.session.commit()
    result = _recompute_risk(child)
    log_activity(f"Recorded {vt.name} dose for {child_label(child)}")
    flash(f"Saved {child.display_name}'s {vt.name}. Risk re-assessed: {result['label']}", "success")
    if warning:
        flash(f"{warning} was recorded past its usual eligibility age — please double-check this dose.", "warning")
    return redirect(_due_list_return_url())


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
        followup_url=url_for("bhw.child_followup_report", child_id=child.id),
    ))


@bhw_bp.route("/children/<int:child_id>/followup-report")
def child_followup_report(child_id):
    """Printable, single-child version of a row on the Children Who Need a
    Visit report - the full picture (why they're flagged, what's overdue,
    what to do) for a home visit, rather than the whole-barangay table."""
    child = db.session.get(Child, child_id) or abort(404)
    if child.barangay_id != current_user.barangay_id:
        abort(403)
    detail = child_detail(child)
    risk_score = round(detail["risk"].risk_probability * 100) if detail["risk"] else 0
    factor_names = [f["factor"] for f in detail["factors"]]
    return render_template("bhw_child_followup_report.html", **_bhw_ctx(
        "reports", page_title="Follow-Up Report", child=child, detail=detail,
        risk_score=risk_score,
        overdue=[r for r in detail["schedule"] if r["status"] == "overdue"],
        due=[r for r in detail["schedule"] if r["status"] == "due"],
        factor_names=factor_names, generated_at=datetime.now(),
        back_url=url_for("bhw.reports", type="atrisk"),
    ))


@bhw_bp.route("/children/<int:child_id>/edit", methods=["GET", "POST"])
def edit_child(child_id):
    child = db.session.get(Child, child_id) or abort(404)
    if child.barangay_id != current_user.barangay_id:
        abort(403)
    if request.method == "POST":
        error = _edit_form_errors(child, request.form)
        if error:
            flash(error, "error")
            return redirect(url_for("bhw.edit_child", child_id=child.id))
        result = _update_child(child, request.form)
        flash(f"{child.display_name} updated. Risk re-assessed: {result['label']}", "success")
        for name in result["past_window_warnings"]:
            flash(f"{name} was recorded past its usual eligibility age — please double-check this dose.", "warning")
        return redirect(url_for("bhw.children"))
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    doses = {code: (rec.date_administered if (rec := child.vaccination_records.filter_by(
        vaccine_type_id=vt_lookup[code].id).first()) else None) for code, *_ in dp.VACCINE_SCHEDULE + dp.OPTIONAL_VACCINE_SCHEDULE}
    active_campaigns = eligible_campaigns_for_child(child)
    campaign_doses = {c.id: (rec.date_administered if (rec := child.vaccination_records.filter_by(
        campaign_id=c.id).first()) else None) for c in active_campaigns}
    return render_template("edit_child.html", **_bhw_ctx(
        "children", page_title="Update Child Record", child=child, doses=doses,
        optional_vaccines=dp.OPTIONAL_VACCINE_SCHEDULE,
        active_campaigns=active_campaigns, campaign_doses=campaign_doses,
        back_url=url_for("bhw.children"), form_action=url_for("bhw.edit_child", child_id=child.id),
    ))


@bhw_bp.route("/vaccine-gaps")
def vaccine_gaps():
    bid = current_user.barangay_id
    vaccine_filter = request.args.get("vaccine", "")
    scope = request.args.get("scope", "")
    rows = vaccine_risk_table(bid, vaccine_code=vaccine_filter or None)
    catchup_count = sum(1 for r in rows if r["is_catch_up"])
    if scope == "catchup":
        rows = [r for r in rows if r["is_catch_up"]]
    return render_template("bhw_vaccine_gaps.html", **_bhw_ctx(
        "vaccine_gaps", page_title="Vaccine Gaps", today=date.today(), dist=vaccine_risk_distribution(bid),
        at_risk_rows=rows, catchup_count=catchup_count, scope=scope,
        vaccine_filter=vaccine_filter, antigen_options=dp.VACCINE_ANTIGENS,
    ))


@bhw_bp.route("/vaccine-gaps/schedule-visit", methods=["POST"])
def schedule_visit():
    """Save a BHW follow-up date for a child in their assigned barangay."""
    try:
        child_id = int(request.form.get("child_id", ""))
    except (ValueError, TypeError):
        abort(400)
    child = db.session.get(Child, child_id)
    if child is None:
        abort(404)
    if child.barangay_id != current_user.barangay_id:
        abort(403)
    try:
        visit_date = date.fromisoformat(request.form.get("visit_date", ""))
    except (ValueError, TypeError):
        flash("Please select a valid visit date.", "warning")
        return redirect(url_for("bhw.vaccine_gaps"))
    if visit_date < date.today():
        flash("Visit date cannot be in the past.", "warning")
        return redirect(url_for("bhw.vaccine_gaps"))
    child.next_visit_date = visit_date
    db.session.commit()
    flash(f"Visit scheduled for {child.display_name} on {visit_date.strftime('%b %d, %Y')}.", "success")
    return redirect(url_for("bhw.vaccine_gaps"))


@bhw_bp.route("/continuation")
def continuation():
    bid = current_user.barangay_id
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    rows = at_risk_table(bid, max_age_days=CONTINUATION_WINDOW_DAYS)
    for r in rows:
        r["continuation_pct"] = 100 - r["risk_score"]
        r["next_dose"] = _next_dose_for(r["child"], vt_lookup)
    engine = dp.model_info()
    return render_template("bhw_continuation.html", **_bhw_ctx(
        "continuation", page_title="Continuation Predictor", rows=rows,
        stats=dashboard_stats(bid, max_age_days=CONTINUATION_WINDOW_DAYS),
        dist=risk_distribution(bid, max_age_days=CONTINUATION_WINDOW_DAYS),
        engine=engine, scoring_rules=dp.SCORING_RULES,
        feature_labels=dp.FEATURE_LABELS, features=feature_catalog(),
    ))


@bhw_bp.route("/stock")
def stock():  # My Stock page
    """The barangay's own vaccine stock, in doses: what is in the fridge, what
    is reserved, what is arriving, and every change."""
    bid = current_user.barangay_id
    rows = barangay_stock_summary(bid)
    incoming = (StockTransfer.query.filter(
        db.or_(db.and_(StockTransfer.to_barangay_id == bid, StockTransfer.status == "in_transit"),
               db.and_(StockTransfer.from_barangay_id == bid, StockTransfer.status == "returning")))
        .order_by(StockTransfer.dispatched_at).all())
    outgoing = (StockTransfer.query.filter(StockTransfer.from_barangay_id == bid,
                                           StockTransfer.status.in_(("reserved", "in_transit")))
                .order_by(StockTransfer.id).all())
    reduced = (StockTransfer.query.filter(StockTransfer.from_barangay_id == bid, StockTransfer.doses_reduced > 0,
                                          StockTransfer.created_at >= datetime.utcnow() - timedelta(days=14))
               .order_by(StockTransfer.id.desc()).all())
    history = (StockMovement.query.filter_by(barangay_id=bid)
               .order_by(StockMovement.created_at.desc(), StockMovement.id.desc()).limit(40).all())
    untracked = [r["antigen"] for r in rows if not r["tracked"]]
    return render_template("bhw_stock.html", **_bhw_ctx(
        "stock", page_title="My Stock", rows=rows, incoming=incoming, history=history,
        untracked=untracked, all_antigens=core_antigens(),
        default_per_vial={a.id: default_doses_per_vial(a) for a in core_antigens()},
        loss_reasons=LOSS_REASONS, wastage_reasons=WASTAGE_REASONS, session_rows=end_of_session_rows(bid),
        today_iso=date.today().isoformat(), movement_labels=MOVEMENT_LABELS, outgoing=outgoing, reduced=reduced,
        status_labels=TRANSFER_STATUS_LABELS, status_colors=TRANSFER_STATUS_COLORS))


def end_of_session_rows(barangay_id):  # leftover doses in today's opened vials
    """For each vaccine given today, the doses left in the last opened vial
    (e.g. 4 MMR doses from 10-dose vials leaves 6). Pre-ticked for vaccines
    discarded after the session; others may be kept (multi-dose vial policy)."""
    today = date.today()
    given = [m for m in StockMovement.query.filter_by(barangay_id=barangay_id, kind="administered").filter(
                 StockMovement.created_at >= datetime.utcnow() - timedelta(days=2)).all()
             if to_local(m.created_at).date() == today]
    already = {m.antigen_id for m in StockMovement.query.filter_by(barangay_id=barangay_id, kind="wasted").filter(
                   StockMovement.note.like(END_OF_SESSION_REASON + "%"),
                   StockMovement.created_at >= datetime.utcnow() - timedelta(days=2)).all()
               if to_local(m.created_at).date() == today}
    out = []
    for antigen_id in sorted({m.antigen_id for m in given} - already):
        mine = [m for m in given if m.antigen_id == antigen_id and m.stock is not None]
        if not mine:
            continue
        lot = max(mine, key=lambda m: m.id).stock  # vial opened most recently
        used = sum(1 for m in mine if m.stock is lot)  # doses taken from that batch today
        leftover = min((-used) % lot.doses_per_vial if lot.doses_per_vial > 1 else 0, lot.doses_on_hand)
        if leftover > 0:
            out.append({"antigen": lot.antigen, "lot": lot, "given_today": len(mine), "leftover": leftover,
                        "discard": discards_after_session(lot.antigen)})
    return out


MOVEMENT_LABELS = {
    "opening": "Opening count", "received": "Received", "administered": "Dose given",
    "reversal": "Dose removed (returned)", "wasted": "Wastage", "expired": "Expired",
    "count_adjustment": "Count adjustment", "transfer_out": "Sent to barangay", "transfer_in": "Received from barangay",
    "no_stock": "Dose given (no stock on hand)",
    "given_elsewhere": "Given elsewhere (no stock used)",
    "transfer_returned": "Returned (transfer cancelled)",
}


def _own_lot(lot_id):  # a batch in the BHW's own barangay, or 404/403
    lot = db.session.get(BarangayStock, _positive_int(lot_id) or 0) or abort(404)
    if lot.barangay_id != current_user.barangay_id:
        abort(403)
    return lot


@bhw_bp.route("/stock/wastage", methods=["POST"])
def stock_wastage():  # log wastage
    lot = _own_lot(request.form.get("lot_id"))
    doses = _positive_int(request.form.get("doses"))
    reason = request.form.get("reason")
    error = ("Enter the number of doses lost (1 or more)." if doses is None else
             f"Only {lot.doses_on_hand} doses of batch {lot.batch_number} are on hand." if doses > lot.doses_on_hand else
             "Please choose a reason." if reason not in WASTAGE_REASONS else None)
    if error:
        flash(error, "error")
        return redirect(url_for("bhw.stock"))
    lot.doses_on_hand -= doses
    _record_movement(lot.barangay_id, lot.antigen_id, "wasted", -doses, lot=lot, note=reason)
    adjusted = reconcile_reservations(lot.barangay_id, lot.antigen_id, f"wastage: {reason}")
    db.session.commit()
    for message in adjusted:
        flash(message, "warning")
    log_activity(f"Recorded barangay stock wastage: {lot.antigen.name} {doses} doses (batch {lot.batch_number}, {reason})")
    flash(f"Recorded {doses} wasted {lot.antigen.name} doses ({reason}).", "success")
    return redirect(url_for("bhw.stock"))


@bhw_bp.route("/stock/discard", methods=["POST"])
def stock_discard():  # end-of-session discard
    saved, adjusted_ids = [], set()
    for row in end_of_session_rows(current_user.barangay_id):
        lot = row["lot"]
        if not request.form.get(f"discard_{lot.id}"):
            continue
        doses = _whole_number(request.form.get(f"doses_{lot.id}"))
        if doses is None or doses > lot.doses_on_hand:
            flash(f"{lot.antigen.name}: enter 0 to {lot.doses_on_hand} doses to discard.", "error")
            return redirect(url_for("bhw.stock"))
        if doses:
            lot.doses_on_hand -= doses
            _record_movement(lot.barangay_id, lot.antigen_id, "wasted", -doses, lot=lot,
                             note=f"{END_OF_SESSION_REASON}, batch {lot.batch_number}")
            saved.append(f"{lot.antigen.name} {doses}")
            adjusted_ids.add(lot.antigen_id)
    if not saved:
        flash("Nothing was discarded.", "info")
        return redirect(url_for("bhw.stock"))
    adjusted = [m for aid in adjusted_ids
                for m in reconcile_reservations(current_user.barangay_id, aid, "end-of-session discard")]
    db.session.commit()
    for message in adjusted:
        flash(message, "warning")
    log_activity("Recorded barangay stock end-of-session discard: " + ", ".join(saved) + " doses")
    flash("End-of-session discard recorded: " + ", ".join(saved) + " doses.", "success")
    return redirect(url_for("bhw.stock"))


@bhw_bp.route("/stock/expired/<int:antigen_id>", methods=["POST"])
def stock_dispose_expired(antigen_id):  # dispose expired doses
    lots = [lot for lot in BarangayStock.query.filter_by(barangay_id=current_user.barangay_id, antigen_id=antigen_id)
            .filter(BarangayStock.doses_on_hand > 0).all() if lot.is_expired]
    if not lots:
        flash("No expired doses to dispose.", "info")
        return redirect(url_for("bhw.stock"))
    total = 0
    for lot in lots:
        total += lot.doses_on_hand
        _record_movement(lot.barangay_id, lot.antigen_id, "expired", -lot.doses_on_hand, lot=lot,
                         note=f"Batch {lot.batch_number} expired {lot.expiry_date.isoformat()}")
        lot.doses_on_hand = 0
    adjusted = reconcile_reservations(current_user.barangay_id, antigen_id, "expired doses disposed")
    db.session.commit()
    for message in adjusted:
        flash(message, "warning")
    log_activity(f"Disposed expired barangay stock: {lots[0].antigen.name} {total} doses")
    flash(f"{total} expired {lots[0].antigen.name} doses disposed.", "success")
    return redirect(url_for("bhw.stock"))


@bhw_bp.route("/stock/count/<int:antigen_id>", methods=["POST"])
def stock_count(antigen_id):  # physical count
    """Correct the balance to what is physically in the fridge, either batch by
    batch (full vials + loose doses) or as one total in doses (spread over the
    batches, soonest expiry first). The difference is recorded."""
    bid = current_user.barangay_id
    antigen = db.session.get(VaccineAntigen, antigen_id) or abort(404)
    lots = usable_lots(bid, antigen_id)
    if not lots:
        flash(f"No {antigen.name} stock to count. Use an opening count or confirm a delivery first.", "error")
        return redirect(url_for("bhw.stock"))
    total_raw = (request.form.get("total_doses") or "").strip()
    counted = {}
    if total_raw:  # one total, spread soonest-expiry first
        total = _whole_number(total_raw)
        if total is None:
            flash("Total doses counted must be a whole number (0 or more).", "error")
            return redirect(url_for("bhw.stock"))
        left = total
        for lot in lots:
            counted[lot.id] = min(lot.doses_on_hand, left)
            left -= counted[lot.id]
        counted[lots[0].id] += left  # any surplus goes to the soonest batch
    else:  # batch by batch
        for lot in lots:
            full = _whole_number(request.form.get(f"full_{lot.id}") or "0")
            loose = _whole_number(request.form.get(f"loose_{lot.id}") or "0")
            if full is None or loose is None:
                flash(f"Batch {lot.batch_number}: full vials and loose doses must be whole numbers.", "error")
                return redirect(url_for("bhw.stock"))
            if lot.doses_per_vial > 1 and loose >= lot.doses_per_vial:
                flash(f"Batch {lot.batch_number}: loose doses must be fewer than one vial ({lot.doses_per_vial}).", "error")
                return redirect(url_for("bhw.stock"))
            counted[lot.id] = full * lot.doses_per_vial + loose
    system = sum(lot.doses_on_hand for lot in lots)
    for lot in lots:
        diff = counted[lot.id] - lot.doses_on_hand
        if diff:
            _record_movement(bid, antigen_id, "count_adjustment", diff, lot=lot,
                             note=f"System {lot.doses_on_hand}, counted {counted[lot.id]}")
            lot.doses_on_hand = counted[lot.id]
    adjusted = reconcile_reservations(bid, antigen_id, "physical count")
    db.session.commit()
    for message in adjusted:
        flash(message, "warning")
    new_total = sum(counted.values())
    log_activity(f"Physical count of barangay stock: {antigen.name} system {system}, counted {new_total} doses")
    flash(f"{antigen.name} count saved: {new_total} doses"
          + (f" (difference {new_total - system:+d})." if new_total != system else " (matches the system)."), "success")
    return redirect(url_for("bhw.stock"))


@bhw_bp.route("/stock/opening", methods=["POST"])
def stock_opening():  # opening count
    """Count a batch that is in the fridge. For a vaccine not tracked yet this
    is its opening count (it starts tracking); for one already tracked it adds
    a batch the records were missing (found during a count), recorded as a
    count adjustment so it shows on the RHU's discrepancy report. Deliveries
    are confirmed under Incoming instead."""
    bid = current_user.barangay_id
    f = request.form
    antigen = db.session.get(VaccineAntigen, _positive_int(f.get("antigen_id")) or 0)
    per_vial = _positive_int(f.get("doses_per_vial"))
    full = _whole_number(f.get("full_vials") or "0")
    loose = _whole_number(f.get("loose_doses") or "0")
    batch_number = (f.get("batch_number") or "").strip()
    expiry = _iso_date(f.get("expiry_date"))
    already = antigen is not None and StockMovement.query.filter(
        StockMovement.barangay_id == bid, StockMovement.antigen_id == antigen.id,
        StockMovement.kind != "opening").first() is not None
    error = ("Please choose a vaccine." if antigen is None or antigen not in core_antigens() else
             "Please enter the batch number." if not batch_number else
             "Please enter a valid expiry date." if expiry is None else
             "This batch has expired; it cannot be added to usable stock." if expiry < date.today() else
             "Doses per vial must be a whole number from 1 to 100." if per_vial is None or per_vial > 100 else
             "Full vials and loose doses must be whole numbers (0 or more)." if full is None or loose is None else
             "Loose doses must be fewer than one full vial." if loose >= per_vial else
             "Enter at least one dose." if full * per_vial + loose == 0 else None)
    if error:
        flash(error, "error")
        return redirect(url_for("bhw.stock"))
    doses = full * per_vial + loose
    lot = _add_to_lot(bid, antigen.id, batch_number, expiry, per_vial, doses)
    if already:
        _record_movement(bid, antigen.id, "count_adjustment", doses, lot=lot,
                         note=f"Batch found during count: {vials_label(doses, per_vial)}")
    else:
        _record_movement(bid, antigen.id, "opening", doses, lot=lot,
                         note=f"{vials_label(doses, per_vial)} counted")
    db.session.commit()
    if already:
        log_activity(f"Physical count of barangay stock: {antigen.name} batch {batch_number} added, {doses} doses")
        flash(f"{antigen.name}: batch {batch_number} added ({doses} doses). It shows as a count difference "
              "on the RHU's discrepancy report.", "success")
    else:
        log_activity(f"Opening stock count for {current_user.barangay.name}: {antigen.name} {doses} doses "
                     f"(batch {batch_number})")
        flash(f"{antigen.name}: {doses} doses added from the opening count.", "success")
    return redirect(url_for("bhw.stock"))


@bhw_bp.route("/stock/receive/<int:transfer_id>", methods=["POST"])
def stock_receive(transfer_id):  # confirm a delivery
    """The BHW confirms what physically arrived. Only good doses enter stock;
    short or damaged doses are recorded with a reason."""
    t = db.session.get(StockTransfer, transfer_id) or abort(404)
    returning = t.status == "returning"  # a cancelled redistribution coming back to the sender
    dest = t.from_barangay_id if returning else t.to_barangay_id
    if dest != current_user.barangay_id:
        abort(403)
    if t.status not in ("in_transit", "returning"):
        flash(f"{t.code} is not waiting for confirmation.", "error")
        return redirect(url_for("bhw.stock"))
    kind = "transfer_returned" if returning else "received" if t.kind == "delivery" else "transfer_in"
    counts = []
    for item in t.items:
        if not item.doses_sent:
            continue
        good = _whole_number(request.form.get(f"good_{item.id}"))
        reason = (request.form.get(f"reason_{item.id}") or "").strip()
        if good is None or good > item.doses_sent:
            flash(f"Batch {item.batch_number}: enter the good doses received, from 0 to {item.doses_sent}.", "error")
            return redirect(url_for("bhw.stock"))
        if good < item.doses_sent and reason not in LOSS_REASONS:
            flash(f"Batch {item.batch_number}: choose why {item.doses_sent - good} doses are missing or unusable.", "error")
            return redirect(url_for("bhw.stock"))
        counts.append((item, good, reason))
    for item, good, reason in counts:
        item.doses_received, item.doses_lost = good, item.doses_sent - good
        item.loss_reason = reason if item.doses_lost else None
        if good:
            lot = _add_to_lot(dest, t.antigen_id, item.batch_number, item.expiry_date,
                              item.doses_per_vial, good)
            _record_movement(dest, t.antigen_id, kind, good, lot=lot, transfer=t,
                             note=f"{t.code}" + (f"; {item.doses_lost} lost: {reason}" if item.doses_lost else ""))
    t.status = "cancelled" if returning else "received"
    t.received_by_id, t.received_at = current_user.id, datetime.utcnow()
    t.received_note = (request.form.get("note") or "").strip()[:255] or None
    db.session.commit()
    lost = t.doses_sent - t.doses_received
    what = "returned redistribution" if returning else "delivery" if t.kind == "delivery" else "redistribution"
    log_activity(f"Received {what} {t.code}: {t.antigen.name} {t.doses_received} of {t.doses_sent} doses"
                 + (f" ({lost} lost)" if lost else ""))
    flash(f"{t.code} received: {t.doses_received} good doses added to stock"
          + (f"; {lost} recorded as lost." if lost else "."), "success")
    return redirect(url_for("bhw.stock"))


def _own_transfer(transfer_id):  # a redistribution leaving the BHW's barangay
    t = db.session.get(StockTransfer, transfer_id) or abort(404)
    if t.kind != "redistribution" or t.from_barangay_id != current_user.barangay_id:
        abort(403)
    return t


@bhw_bp.route("/stock/export.csv")
def stock_csv():  # My Stock as CSV: current stock, then full history
    bid = current_user.barangay_id
    barangay = current_user.barangay.name if current_user.barangay else ""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ImmunoVision - Barangay Stock"])
    w.writerow(["Barangay", barangay])
    w.writerow(["Generated", datetime.now().strftime("%b %d, %Y %I:%M %p")])
    w.writerow([])
    w.writerow(["Vaccine", "Status", "In fridge (doses)", "Vials", "Reserved", "Available", "Needed soon",
                "Nearest expiry", "Expired to dispose"])
    for r in barangay_stock_summary(bid):
        w.writerow([_csv_cell(v) for v in [r["antigen"].name, r["status"], r["in_fridge"] if r["tracked"] else "",
                                           r["vials"] if r["tracked"] else "", r["reserved"] or "",
                                           r["available"] if r["tracked"] else "", r["needed"],
                                           r["nearest_expiry"], r["expired"] or ""]])
    w.writerow([])
    w.writerow(["Stock history"])
    w.writerow(MOVEMENT_CSV_COLUMNS)
    moves = (StockMovement.query.filter_by(barangay_id=bid)
             .order_by(StockMovement.created_at, StockMovement.id).all())
    for row in movement_csv_rows(moves):
        w.writerow([_csv_cell(v) for v in row])
    log_activity("Exported barangay stock (CSV)")
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=immunovision_my_stock_{date.today():%Y%m%d}.csv"})


@bhw_bp.route("/stock/transfers/<int:transfer_id>/dispatch", methods=["POST"])
def transfer_dispatch(transfer_id):  # vials handed to the carrier
    t = _own_transfer(transfer_id)
    if t.status != "reserved":
        flash(f"{t.code} is not waiting to be dispatched.", "error")
        return redirect(url_for("bhw.stock"))
    for message in reconcile_reservations(t.from_barangay_id, t.antigen_id, "checked at dispatch"):
        flash(message, "warning")
    if t.status != "reserved":  # nothing left to send
        db.session.commit()
        return redirect(url_for("bhw.stock"))
    for item in t.items:
        if not item.doses_sent:
            continue
        lot = db.session.get(BarangayStock, item.source_stock_id)
        lot.doses_on_hand -= item.doses_sent
        _record_movement(t.from_barangay_id, t.antigen_id, "transfer_out", -item.doses_sent, lot=lot, transfer=t,
                         note=f"{t.code} to {t.to_barangay.name}")
    t.status, t.dispatched_by_id, t.dispatched_at = "in_transit", current_user.id, datetime.utcnow()
    db.session.commit()
    log_activity(f"Dispatched redistribution {t.code}: {t.antigen.name} {t.doses_sent} doses to {t.to_barangay.name}")
    flash(f"{t.code} dispatched: {t.doses_sent} {t.antigen.name} doses are in transit to {t.to_barangay.name}.", "success")
    return redirect(url_for("bhw.stock"))


@bhw_bp.route("/stock/transfers/<int:transfer_id>/cancel", methods=["POST"])
def transfer_cancel(transfer_id):  # sender cancels before dispatch
    t = _own_transfer(transfer_id)
    if t.status != "reserved":
        flash(f"{t.code} has left your barangay; only the RHU can cancel it now.", "error")
        return redirect(url_for("bhw.stock"))
    return _cancel_transfer(t, (request.form.get("reason") or "").strip()[:200], url_for("bhw.stock"))


@bhw_bp.route("/stock/transfers/<int:transfer_id>/slip")
def transfer_slip(transfer_id):
    t = db.session.get(StockTransfer, transfer_id) or abort(404)
    if current_user.barangay_id not in (t.from_barangay_id, t.to_barangay_id):
        abort(403)
    return render_template("transfer_slip.html", t=t, status_labels=TRANSFER_STATUS_LABELS)


@bhw_bp.route("/requests", methods=["GET", "POST"])
def requests():  # BHW requests
    bid = current_user.barangay_id
    if request.method == "POST":  # submit
        antigen = db.session.get(VaccineAntigen, _positive_int(request.form.get("antigen_id")) or 0)
        quantity = _positive_int(request.form.get("quantity"))
        priority = request.form.get("priority", "Normal")
        error = ("Please choose a vaccine." if antigen is None else
                 "Quantity must be a whole number of 1 or more." if quantity is None else
                 "Please choose a valid priority." if priority not in VALID_PRIORITIES else None)
        if error:
            flash(error, "error")
            return redirect(url_for("bhw.requests"))
        last = VaccineRequest.query.order_by(VaccineRequest.id.desc()).first()
        code = f"REQ-{datetime.now().year}-{(last.id if last else 0) + 1:03d}"  # request code
        vr = VaccineRequest(request_code=code, barangay_id=bid, requested_by_id=current_user.id,  # new request
                            antigen_id=antigen.id, quantity_requested=quantity,
                            priority=priority, notes=request.form.get("notes"))
        db.session.add(vr)
        db.session.commit()  # save
        log_activity(f"Submitted vaccine request {code} ({vr.antigen.name}, {vr.quantity_requested} vials)")  # log
        flash(f"Request {code} submitted to RHU.", "success")
        return redirect(url_for("bhw.requests"))
    reqs = VaccineRequest.query.filter_by(barangay_id=bid).order_by(VaccineRequest.requested_at.desc()).all()  # own requests
    stats = {"pending": sum(1 for r in reqs if r.status == "pending"),
             "approved": sum(1 for r in reqs if r.status == "approved"),
             "distributed": sum(1 for r in reqs if r.status == "fulfilled"),
             "rejected": sum(1 for r in reqs if r.status == "rejected")}
    return render_template("bhw_requests.html", **_bhw_ctx(
        "requests", page_title="Vaccine Requests", reqs=reqs, stats=stats, antigens=core_antigens(),
        per_vial={str(a.id): default_doses_per_vial(a) for a in core_antigens()},
    ))


@bhw_bp.route("/reports")
def reports():
    bid = current_user.barangay_id
    as_of, report_key = resolve_report_date(request.args)
    fig = report_figures(bid, as_of)
    return render_template("bhw_reports.html", **_bhw_ctx(
        "reports", page_title="Reports", report_type=request.args.get("type", "coverage"),
        stats=fig, by_vaccine=report_by_vaccine(bid, as_of), generated_at=datetime.now(),
        at_risk_rows=vaccine_risk_table(bid, limit=50),
        doses_by_month=monthly_doses_administered(bid, as_of=as_of),
        date_options=report_date_options(), report_key=report_key, as_of=as_of,
    ))


@bhw_bp.route("/reports/export.csv")
def export_report_csv():
    """Downloads this barangay's report as CSV (opens directly in Excel)."""
    bid = current_user.barangay_id
    as_of, _key = resolve_report_date(request.args)
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


@bhw_bp.route("/campaigns/<int:campaign_id>")
def campaign_worklist(campaign_id):
    """Same focused worklist as the RHU version, locked to the BHW's own
    barangay - no filter control needed since there is only ever one scope."""
    campaign = db.session.get(Campaign, campaign_id) or abort(404)
    bid = current_user.barangay_id
    if not campaign.barangay_eligible(bid):
        abort(404)
    pending_only = request.args.get("pending") == "1"
    search = request.args.get("q", "").strip()
    kids = eligible_children_for_campaign(campaign, bid)
    if search:
        kids = [c for c in kids if search.lower() in c.full_name.lower()]
    recs = {r.child_id: r.date_administered for r in
            VaccinationRecord.query.filter_by(campaign_id=campaign.id).all()}
    rows = [{"child": c, "dose_date": recs.get(c.id)} for c in kids]
    if pending_only:
        rows = [r for r in rows if not r["dose_date"]]
    rows.sort(key=lambda r: (r["dose_date"] is not None, r["child"].full_name))
    return render_template("campaign_worklist.html", **_bhw_ctx(
        "dashboard", page_title=campaign.name, campaign=campaign, rows=rows,
        barangays=None, selected_barangay_id=bid, pending_only=pending_only, search=search,
    ))


@bhw_bp.route("/campaigns/<int:campaign_id>/record/<int:child_id>", methods=["POST"])
def campaign_record_dose(campaign_id, child_id):
    campaign = db.session.get(Campaign, campaign_id) or abort(404)
    child = db.session.get(Child, child_id) or abort(404)
    if child.barangay_id != current_user.barangay_id:
        abort(403)
    if not (campaign.age_eligible(child.age_in_days) and campaign.barangay_eligible(child.barangay_id)):
        abort(400)
    date_str = request.form.get("dose_date")
    administered = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
    rec = child.vaccination_records.filter_by(campaign_id=campaign.id).first()
    date_error = _dose_date_error(child.date_of_birth, [(campaign.name, administered)])
    if date_error:
        flash(date_error, "error")
    else:
        if rec is None and administered is not None:
            rec = VaccinationRecord(child_id=child.id, vaccine_type_id=campaign.vaccine_type_id, campaign_id=campaign.id)
            db.session.add(rec)
        if rec is not None:
            previous = rec.date_administered
            rec.date_administered = administered
            rec.status = "completed" if administered else "pending"
            stock_on_dose_change(child, db.session.get(VaccineType, campaign.vaccine_type_id),
                                 rec, previous, administered)  # barangay stock
            db.session.commit()
            if previous is not None and previous != administered:
                _log_dose_date_changes(child, [(campaign.name, previous, administered)])
            else:
                log_activity(f"Recorded {campaign.name} dose for {child_label(child)}")
            flash(f"Saved {child.full_name}'s dose.", "success")
    return redirect(url_for("bhw.campaign_worklist", campaign_id=campaign.id,
                             pending=request.form.get("pending") or None,
                             q=request.form.get("q") or None))


@bhw_bp.route("/notifications")
def notifications():
    notes = get_notifications("bhw", barangay_id=current_user.barangay_id)
    ctx = _bhw_ctx("notifications", page_title="Notifications", notes=notes)
    _mark_notifications_seen(notes, current_user)
    return render_template("bhw_notifications.html", **ctx)


@bhw_bp.route("/settings")
def settings():
    return render_template("bhw_settings.html", **_bhw_ctx("settings", page_title="Settings"))


# ---------------------------------------------------------------------------
# Admin blueprint
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Child Records quick panel: open a child's row in the list to record doses,
# correct a date or update guardian details without leaving the page. Shared
# by the RHU and BHW lists; every rule is checked here on the server.
# ---------------------------------------------------------------------------

_VISIT_LABELS = {0: "Birth", 42: "6 wks", 70: "10 wks", 98: "14 wks", 270: "9 mos", 365: "12 mos"}
_SHORT_ANTIGEN = {"BCG": "BCG", "HEPB": "HepB", "PENTA": "Penta", "OPV": "OPV", "PCV": "PCV",
                  "IPV": "IPV", "MMR": "MMR"}


def _short_dose_name(code):
    _c, _n, antigen, dose_no, _d = next(r for r in dp.VACCINE_SCHEDULE if r[0] == code)
    single = sum(1 for r in dp.VACCINE_SCHEDULE if r[2] == antigen) == 1
    return _SHORT_ANTIGEN.get(antigen, antigen) + ("" if single else f" {dose_no}")


def _quick_child(child_id):
    """The child, if this user may update it (a BHW only in their barangay)."""
    child = db.session.get(Child, child_id) or abort(404)
    if current_user.role == "bhw" and child.barangay_id != current_user.barangay_id:
        abort(403)
    return child


def _quick_stock(child):
    """{antigen id: doses available} for a BHW's barangay, for vaccines whose
    stock is tracked; RHU recording does not use barangay stock, so {}."""
    if current_user.role != "bhw" or child.barangay_id is None:
        return {}
    bid = child.barangay_id
    reserved = reserved_doses(bid)
    out = {}
    for a in core_antigens():
        if stock_tracking_started(bid, a.id):
            on_hand = sum(lot.doses_on_hand for lot in usable_lots(bid, a.id))
            out[a.id] = max(0, on_hand - min(reserved.get(a.id, 0), on_hand))
    return out


def _quick_doses(child):
    """Every core dose with its status and, for doses that can be recorded
    now, the earliest date the schedule allows."""
    today = date.today()
    vt_by_code = {vt.code: vt for vt in VaccineType.query.all()}
    detail = {d["code"]: d for d in child_detail(child)["schedule"]}
    dates = _core_dose_dates(child)
    stock = _quick_stock(child)
    previous, following = {}, {}
    last = {}
    for code, _n, antigen, _d, _r in dp.VACCINE_SCHEDULE:
        if antigen in last:
            following[last[antigen]] = code
        last[antigen] = code
    rows = []
    for code, name, antigen, dose_no, rec_days in dp.VACCINE_SCHEDULE:
        d = detail.get(code)
        vt = vt_by_code.get(code)
        if d is None or vt is None:
            continue
        prev = previous.get(antigen)
        previous[antigen] = code
        row = {"code": code, "name": name, "short": _short_dose_name(code), "status": d["status"],
               "date": d["date"], "due_on": d["due_on"], "visit": _VISIT_LABELS.get(rec_days, ""),
               "earliest": None, "latest": None, "lock": None, "stock": stock.get(vt.antigen_id) if stock else None,
               "overdue_days": (today - d["due_on"]).days if d["status"] == "overdue" else 0}
        if d["status"] in ("due", "overdue"):
            earliest = child.date_of_birth + timedelta(days=dp.DOSE_MIN_AGE_DAYS.get(code, 0))
            if prev and dates.get(prev) is None:
                row["lock"] = f"after {_short_dose_name(prev)}"
            else:
                if prev:
                    earliest = max(earliest, dates[prev] + timedelta(days=dp.MIN_DOSE_GAP_DAYS))
                row["earliest"] = earliest
                latest = today
                nxt = following.get(code)
                if nxt and dates.get(nxt):  # a later dose is already recorded: keep 4 weeks before it
                    latest = min(today, dates[nxt] - timedelta(days=dp.MIN_DOSE_GAP_DAYS))
                row["latest"] = latest
                if earliest > today:
                    row["lock"] = f"from {earliest:%m/%d/%y}"
                elif latest < earliest:
                    row["lock"] = f"fix {_short_dose_name(nxt)} first"
        rows.append(row)
    return rows


def _quick_groups(rows):
    groups = []
    for r in rows:
        if not groups or groups[-1]["label"] != r["visit"]:
            groups.append({"label": r["visit"], "doses": []})
        groups[-1]["doses"].append(r)
    for g in groups:
        late = [d["overdue_days"] for d in g["doses"] if d["status"] == "overdue"]
        g["overdue_days"] = max(late) if late else 0
        g["selectable"] = any(d["status"] in ("due", "overdue") and not d["lock"] for d in g["doses"])
    return groups


def _quick_row_state(child):
    """What the list row shows after a change: doses given and risk label."""
    g.pop("_iv_cache", None)  # records changed in this request
    vt_lookup = {vt.code: vt for vt in VaccineType.query.all()}
    doses = _dose_matrix([child], vt_lookup)[child.id]
    latest = _latest_risk(child)
    return {"done": sum(1 for v in doses.values() if v), "total": len(dp.VACCINE_SCHEDULE),
            "risk": latest.risk_label if latest else None,
            "dates": {code: (v.strftime("%m/%d/%y") if v else None) for code, v in doses.items()}}


def _quick_next_steps(child):
    """After recording: doses that can still be given today, and the next
    date more doses become allowed (by age or the 4-week gap)."""
    today = date.today()
    dates = _core_dose_dates(child)
    prev_of = {}
    for code, _n, antigen, _d, _r in dp.VACCINE_SCHEDULE:
        prev_of[code] = prev_of.get(antigen)
        prev_of[antigen] = code
    when = {}
    for r in _quick_doses(child):
        if r["status"] in ("due", "overdue") and r["earliest"]:
            when[r["short"]] = max(today, r["earliest"])
        elif r["status"] == "upcoming":
            prev = prev_of.get(r["code"])
            if prev and dates.get(prev) is None:
                continue  # waits for the previous dose first
            gap = dates[prev] + timedelta(days=dp.MIN_DOSE_GAP_DAYS) if prev else r["due_on"]
            when[r["short"]] = max(r["due_on"], gap)
    now = [n for n, d in when.items() if d <= today]
    later = {n: d for n, d in when.items() if d > today}
    nxt = None
    if later:
        first = min(later.values())
        nxt = {"doses": [n for n, d in later.items() if d == first], "date": f"{first:%m/%d/%y}"}
    return {"still_needed": now, "next_visit": nxt}


def _quick_error(message, status=400):
    return jsonify({"ok": False, "error": message}), status


def _quick_other(child):
    """Optional vaccines, active campaign doses and supplements (Vitamin A,
    MNP): everything else the Edit Child form held."""
    vt_by_code = {vt.code: vt for vt in VaccineType.query.all()}
    given = _doses_given(child.id)
    optional = []
    for code, name, _a, _d, _r, deadline in dp.OPTIONAL_VACCINE_SCHEDULE:
        vt = vt_by_code.get(code)
        if vt:
            when = given.get(vt.id)
            optional.append({"code": code, "name": name, "date": when,
                             "past_window": when is not None and (when - child.date_of_birth).days > deadline})
    campaigns = []
    for c in eligible_campaigns_for_child(child):
        rec = child.vaccination_records.filter_by(campaign_id=c.id).first()
        campaigns.append({"id": c.id, "name": f"{c.vaccine_type.name} - {c.name}",
                          "date": rec.date_administered if rec else None})
    return {"optional": optional, "campaigns": campaigns}


def quick_panel(child_id):
    child = _quick_child(child_id)
    rows = _quick_doses(child)
    return render_template("_child_quick_panel.html", child=child, groups=_quick_groups(rows),
                           other=_quick_other(child), today=date.today(),
                           show_stock=current_user.role == "bhw")


def quick_record(child_id):
    """Records the selected doses on one date. Refused, with the reason, if
    any dose is already given, not yet due, missing its previous dose, or the
    date breaks the schedule; nothing is saved unless every dose passes."""
    child = _quick_child(child_id)
    codes = [c for c in request.form.getlist("codes") if c]
    if not codes:
        return _quick_error("Select at least one dose to record.")
    # one date per dose (date_<code>); a single "date" applies to all
    given_on = {c: _parse_form_date(request.form.get(f"date_{c}") or request.form.get("date")) for c in codes}
    rows = {r["code"]: r for r in _quick_doses(child)}
    for code in codes:
        r = rows.get(code)
        if r is None:
            return _quick_error("Unknown dose.")
        if r["status"] == "given":
            return _quick_error(f"{r['short']} is already recorded.")
        if r["status"] not in ("due", "overdue"):
            return _quick_error(f"{r['short']} is not due yet.")
        if r["lock"] and r["lock"].startswith("after"):
            return _quick_error(f"Record {r['lock'][6:]} before {r['short']}.")
        if given_on[code] is None:
            return _quick_error(f"Please enter a valid date for {r['short']}.")
    error = _dose_date_error(child.date_of_birth, [(rows[c]["name"], given_on[c]) for c in codes])
    if not error:
        dates = _core_dose_dates(child)
        dates.update(given_on)
        error = _schedule_error(child.date_of_birth, dates, codes)
    if error:
        return _quick_error(error)
    vt_by_code = {vt.code: vt for vt in VaccineType.query.all()}
    existing = {r.vaccine_type_id: r for r in child.vaccination_records if r.campaign_id is None}
    for code in codes:
        vt = vt_by_code[code]
        _write_dose(child, vt, existing.get(vt.id), given_on[code], bool(request.form.get("given_elsewhere")))
    db.session.commit()
    result = _recompute_risk(child)
    short = [rows[c]["short"] for c in codes]
    log_activity("Recorded " + ", ".join(f"{rows[c]['name']} ({given_on[c].isoformat()})" for c in codes)
                 + f" for {child_label(child)}")
    return jsonify({"ok": True, "child": child.display_name, "recorded": short, "codes": codes,
                    "dates": {c: given_on[c].isoformat() for c in codes}, "risk": result["label"],
                    "row": _quick_row_state(child), **_quick_next_steps(child)})


def quick_undo(child_id):
    """Removes doses just recorded (same codes, same date), returning their stock."""
    child = _quick_child(child_id)
    codes = request.form.getlist("codes")
    vt_by_code = {vt.code: vt for vt in VaccineType.query.all()}
    existing = {r.vaccine_type_id: r for r in child.vaccination_records if r.campaign_id is None}
    undone = []
    for code in codes:
        vt = vt_by_code.get(code)
        rec = existing.get(vt.id) if vt else None
        when = _parse_form_date(request.form.get(f"date_{code}") or request.form.get("date"))
        if rec is not None and rec.date_administered == when:
            _write_dose(child, vt, rec, None)
            undone.append(vt.name)
    if not undone:
        return _quick_error("Nothing to undo; the doses were already changed.")
    db.session.commit()
    result = _recompute_risk(child)
    return jsonify({"ok": True, "risk": result["label"], "row": _quick_row_state(child)})


def quick_fix_date(child_id):
    """Corrects the date of a dose already recorded; the same schedule rules
    apply to the new date, and the change is logged."""
    child = _quick_child(child_id)
    code = request.form.get("code", "")
    new = _parse_form_date(request.form.get("date"))
    vt = VaccineType.query.filter_by(code=code).first()
    if vt is None or code not in dp.VACCINE_CODES:
        return _quick_error("Unknown dose.")
    if new is None:
        return _quick_error("Please enter a valid date.")
    rec = next((r for r in child.vaccination_records if r.campaign_id is None
                and r.vaccine_type_id == vt.id and r.date_administered), None)
    if rec is None:
        return _quick_error("This dose has not been recorded yet.")
    if rec.date_administered == new:
        return jsonify({"ok": True, "unchanged": True, "row": _quick_row_state(child)})
    error = _dose_date_error(child.date_of_birth, [(vt.name, new)])
    if not error:
        dates = _core_dose_dates(child)
        dates[code] = new
        error = _schedule_error(child.date_of_birth, dates, [code])
    if error:
        return _quick_error(error)
    _write_dose(child, vt, rec, new)
    db.session.commit()
    result = _recompute_risk(child)
    return jsonify({"ok": True, "risk": result["label"], "row": _quick_row_state(child)})


def _child_info_json(child):
    return {"full_name": child.full_name or "", "display_name": child.display_name,
            "sex": child.sex, "sex_label": "Male" if child.sex == "M" else "Female",
            "date_of_birth": child.date_of_birth.isoformat(), "dob_short": f"{child.date_of_birth:%m/%d/%y}",
            "date_registered": child.date_registered.isoformat() if child.date_registered else "",
            "reg_short": f"{child.date_registered:%m/%d/%y}" if child.date_registered else "—",
            "guardian_name": child.guardian_name or "", "guardian_contact": child.guardian_contact or "",
            "address": child.address or ""}


def quick_info(child_id):
    """Saves the child's details from the panel: name, sex, date of birth,
    date of registration, guardian, contact number and address. Validated like
    the Edit Child form; a new date of birth may not put any recorded dose
    before birth or before its scheduled age. Barangay stays admin-only."""
    child = _quick_child(child_id)
    f = {k: (request.form.get(k) or "").strip() for k in
         ("full_name", "sex", "date_of_birth", "date_registered", "guardian_name", "guardian_contact", "address")}
    if not f["full_name"]:
        return _quick_error("Please enter the child's full name.")
    if f["sex"] not in ("M", "F"):
        return _quick_error("Please choose the child's sex.")
    dob = _parse_form_date(f["date_of_birth"])
    registered = _parse_form_date(f["date_registered"])
    if dob is None:
        return _quick_error("Please enter a valid date of birth.")
    if registered is None:
        return _quick_error("Please enter a valid date of registration.")
    error = _child_field_errors(f, child)
    if error:
        return _quick_error(error)
    if dob != child.date_of_birth:
        dates = _core_dose_dates(child)
        names = {code: n for code, n, *_ in dp.VACCINE_SCHEDULE}
        newly_before = [names[c] for c, d in dates.items() if child.date_of_birth <= d < dob]
        if newly_before:
            return _quick_error("With this date of birth, " + ", ".join(newly_before)
                                + " would be dated before the child was born. Check the date of birth or correct those doses first.")
        before = set(dp.schedule_violations(child.date_of_birth, dates, set(dates)))
        after = [m for m in dp.schedule_violations(dob, dates, set(dates)) if m not in before]
        if after:
            return _quick_error("This date of birth doesn't fit the doses already recorded. " + " ".join(after))
    labels = {"full_name": "name", "sex": "sex", "date_of_birth": "date of birth", "date_registered": "date of registration",
              "guardian_name": "guardian", "guardian_contact": "contact number", "address": "address"}
    old = _child_info_json(child)
    child.full_name = f["full_name"]
    child.sex = f["sex"]
    child.date_of_birth = dob
    child.date_registered = registered
    child.guardian_name = f["guardian_name"] or None
    child.guardian_contact = f["guardian_contact"] or None
    child.address = f["address"] or None
    db.session.commit()
    new = _child_info_json(child)
    changed = [labels[k] for k in labels if old[k] != new[k]]
    risk = None
    if old["date_of_birth"] != new["date_of_birth"] or old["sex"] != new["sex"] or old["date_registered"] != new["date_registered"]:
        risk = _recompute_risk(child)["label"]  # age, sex and registration feed the model
    if changed:
        log_activity(f"Updated {', '.join(changed)} for {child_label(child)}")
    return jsonify({"ok": True, "child": new, "risk": risk, "row": _quick_row_state(child)})


def quick_remove(child_id):
    """Removes a recorded core dose (it becomes not given). Its stock goes back
    to the batch it came from and the removal is logged, as on the Edit form."""
    child = _quick_child(child_id)
    code = request.form.get("code", "")
    vt = VaccineType.query.filter_by(code=code).first()
    if vt is None or code not in dp.VACCINE_CODES:
        return _quick_error("Unknown dose.")
    rec = next((r for r in child.vaccination_records if r.campaign_id is None
                and r.vaccine_type_id == vt.id and r.date_administered), None)
    if rec is None:
        return _quick_error("This dose is not recorded.")
    _write_dose(child, vt, rec, None)
    db.session.commit()
    result = _recompute_risk(child)
    return jsonify({"ok": True, "risk": result["label"], "row": _quick_row_state(child)})


def quick_other(child_id):
    """Saves optional vaccines, campaign doses, Vitamin A and MNP - the rest of
    the old Edit Child form - with the same date checks."""
    child = _quick_child(child_id)
    data = _quick_other(child)
    form = request.form
    elsewhere = bool(form.get("given_elsewhere"))
    opt = {o["code"]: _parse_form_date(form.get(f"dose_{o['code']}")) for o in data["optional"]}
    camp = {c["id"]: _parse_form_date(form.get(f"dose_campaign_{c['id']}")) for c in data["campaigns"]}
    vit_a = _parse_form_date(form.get("vitamin_a_date"))
    for key in [f"dose_{c}" for c in opt] + [f"dose_campaign_{c}" for c in camp] + ["vitamin_a_date"]:
        if form.get(key) and _parse_form_date(form.get(key)) is None:
            return _quick_error("Please enter valid dates.")
    names = {o["code"]: o["name"] for o in data["optional"]}
    cnames = {c["id"]: c["name"] for c in data["campaigns"]}
    dated = [(names[c], d) for c, d in opt.items()] + [(cnames[c], d) for c, d in camp.items()] + [("Vitamin A", vit_a)]
    error = _dose_date_error(child.date_of_birth, dated)
    if error:
        return _quick_error(error)
    vt_by_code = {vt.code: vt for vt in VaccineType.query.all()}
    existing = {r.vaccine_type_id: r for r in child.vaccination_records if r.campaign_id is None}
    warnings = []
    for code, when in opt.items():
        vt = vt_by_code[code]
        rec = existing.get(vt.id)
        if (rec.date_administered if rec else None) == when:
            continue
        w = _write_dose(child, vt, rec, when, elsewhere)
        if w:
            warnings.append(w)
    changes = []
    for cid, when in camp.items():
        campaign = db.session.get(Campaign, cid)
        rec = child.vaccination_records.filter_by(campaign_id=cid).first()
        if (rec.date_administered if rec else None) == when:
            continue
        if rec is None:
            rec = VaccinationRecord(child_id=child.id, vaccine_type_id=campaign.vaccine_type_id, campaign_id=cid)
            db.session.add(rec)
        if rec.date_administered is not None:
            changes.append((campaign.name, rec.date_administered, when))
        previous = rec.date_administered
        rec.date_administered = when
        rec.status = "completed" if when else "pending"
        stock_on_dose_change(child, db.session.get(VaccineType, campaign.vaccine_type_id), rec, previous, when, elsewhere)
    child.vitamin_a_date = vit_a
    child.mnp_given = form.get("mnp_given") == "1"
    db.session.commit()
    _log_dose_date_changes(child, changes)
    result = _recompute_risk(child)
    log_activity(f"Updated optional vaccines, campaigns and supplements for {child_label(child)}")
    g.pop("_iv_cache", None)
    fresh = _quick_other(child)
    fmt = lambda d: d.isoformat() if d else ""
    return jsonify({"ok": True, "risk": result["label"], "row": _quick_row_state(child),
                    "warnings": [f"{w} was recorded past its usual eligibility age; please double-check it." for w in warnings],
                    "values": {**{f"dose_{o['code']}": fmt(o["date"]) for o in fresh["optional"]},
                               **{f"dose_campaign_{c['id']}": fmt(c["date"]) for c in fresh["campaigns"]},
                               "vitamin_a_date": fmt(child.vitamin_a_date), "mnp_given": "1" if child.mnp_given else ""}})


for _bp in (rhu_bp, bhw_bp):
    _bp.add_url_rule("/children/<int:child_id>/quick-remove", "quick_remove", quick_remove, methods=["POST"])
    _bp.add_url_rule("/children/<int:child_id>/quick-other", "quick_other", quick_other, methods=["POST"])
    _bp.add_url_rule("/children/<int:child_id>/panel", "quick_panel", quick_panel)
    _bp.add_url_rule("/children/<int:child_id>/quick-record", "quick_record", quick_record, methods=["POST"])
    _bp.add_url_rule("/children/<int:child_id>/quick-undo", "quick_undo", quick_undo, methods=["POST"])
    _bp.add_url_rule("/children/<int:child_id>/quick-fix-date", "quick_fix_date", quick_fix_date, methods=["POST"])
    _bp.add_url_rule("/children/<int:child_id>/quick-info", "quick_info", quick_info, methods=["POST"])


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


@admin_bp.route("/target-population", methods=["POST"])
def set_target_population():
    """DOH/PSA projected target population for one barangay, entered
    manually since it's external demographic data the app has no way to
    derive. Used alongside the actual-registered-children denominator for
    official-standard coverage reporting (see coverage_by_barangay() and
    report_figures())."""
    b = db.session.get(Barangay, request.form.get("barangay_id", type=int)) or abort(404)
    raw = (request.form.get("target_population") or "").strip()
    if not raw:
        b.target_population = None
        log_activity(f"Cleared {b.name}'s target population")
        flash(f"Target population cleared for {b.name}.", "success")
    elif raw.isdigit() and int(raw) > 0:
        b.target_population = int(raw)
        log_activity(f"Set {b.name}'s target population to {int(raw):,}")
        flash(f"Target population for {b.name} set to {int(raw):,}.", "success")
    else:
        flash("Target population must be a positive whole number.", "error")
        return redirect(request.referrer or url_for("admin.dashboard"))
    db.session.commit()
    return redirect(request.referrer or url_for("admin.dashboard"))


@admin_bp.route("/dashboard")
def dashboard():
    # Municipality-wide clinical oversight is AGGREGATES ONLY - counts and
    # percentages per barangay, never child-level records; the RHU
    # administrator has no need to see individual children. Account
    # governance (user stats, recent activity) is included below in full,
    # since that's this role's own data to manage, not clinical data.
    # Deliberately the same functions the RHU dashboard and Coverage Analytics
    # use, so every live view of municipal performance reports identical
    # figures. (report_figures() is reserved for the Reports page, which is
    # explicitly "as of" a chosen period and so can legitimately differ.)
    # Same scope + period filters as the RHU dashboard: the administrator also
    # oversees all 19 barangays and needs to narrow to one.
    barangay_id = request.args.get("barangay_id", type=int)
    selected = db.session.get(Barangay, barangay_id) if barangay_id else None
    if not selected:
        barangay_id = None
    period = _dashboard_period()
    as_of, born_from = period["as_of"], period["born_from"]

    if as_of:
        fig = report_figures(barangay_id, as_of, born_from, risk_as_of(as_of))
        muni = {"total_children": fig["total_children"], "fully_immunized": fig["fully_immunized"],
                "at_risk": fig["at_risk"], "coverage_rate": fig["coverage_rate"]}
        rows = fig["by_barangay"]
    else:
        muni = dashboard_stats(barangay_id)
        rows = coverage_by_barangay()
    if selected:
        rows = [r for r in rows if r["barangay"] == selected.name]
    # Computed from the same rows (whichever path built them) rather than
    # from muni/fig, so this works the same whether or not a period filter
    # is active. Only counts barangays with a target entered, same
    # reasoning as report_figures()'s target_population_total.
    target_rows = [r for r in rows if r.get("target_population")]
    target_population_total = sum(r["target_population"] for r in target_rows)
    municipality = {
        "total_children": muni["total_children"],
        "fully_immunized": muni["fully_immunized"],
        "at_risk": muni["at_risk"],
        "coverage_rate": muni["coverage_rate"],
        "barangays": len(rows),
        "below_target": sum(1 for r in rows if r["coverage"] < COVERAGE_TARGET),
        "target_population_total": target_population_total,
        "target_coverage_rate": _target_coverage(
            sum(r["fully_immunized"] for r in target_rows), target_population_total),
        "target_barangays_set": len(target_rows),
    }
    # Worst-performing first, so barangays needing attention surface immediately.
    barangay_rows = sorted(rows, key=lambda r: r["coverage"])

    # The admin Coverage Analytics page was retired: its barangay view duplicated
    # the table above, so its two unique panels moved onto this page.
    by_vaccine = order_by_schedule(report_by_vaccine(barangay_id, as_of, born_from) if as_of
                                   else coverage_by_vaccine(barangay_id))
    # Annual rather than monthly: a month/year filter and a 12-month series
    # contradict each other, and yearly coverage can be computed for real.
    # Annual view stays municipality-wide; a single selected birth year
    # follows the barangay filter so the monthly line matches the cards.
    trend_info = trend_for(barangay_id, period) if period["month_active"] else trend_for(None, period)
    trend = trend_info["trend"]

    # Barangay staffing: the Assign Barangay question, summarised here rather
    # than making the administrator visit that page to find out whether
    # anything needs attention.
    bhw_users = User.query.filter_by(role="bhw").all()
    covered = {u.barangay_id for u in bhw_users if u.barangay_id}
    all_barangays = Barangay.query.order_by(Barangay.name).all()
    staffing = {
        "covered": sum(1 for b in all_barangays if b.id in covered),
        "total": len(all_barangays),
        "uncovered": [b.name for b in all_barangays if b.id not in covered],
        "unassigned_bhws": [u.full_name for u in bhw_users if not u.barangay_id],
    }

    # Account governance: who has access. What they've been doing with it is
    # Activity Logs' job now, owned solely by the superadmin role - this
    # dashboard only manages accounts, it doesn't audit them.
    all_users = User.query.all()
    user_stats = {"total": len(all_users), "admin": sum(1 for u in all_users if u.role == "admin"),
                  "superadmin": sum(1 for u in all_users if u.role == "superadmin"),
                  "rhu": sum(1 for u in all_users if u.role == "rhu"), "bhw": sum(1 for u in all_users if u.role == "bhw"),
                  "active": sum(1 for u in all_users if u.is_active_flag),
                  "disabled": sum(1 for u in all_users if not u.is_active_flag),
                  "logins_today": sum(1 for u in all_users if u.last_login_at and to_local(u.last_login_at).date() == date.today())}
    role_dist = {"admin": user_stats["admin"], "superadmin": user_stats["superadmin"],
                 "rhu": user_stats["rhu"], "bhw": user_stats["bhw"]}

    return render_template("admin_dashboard.html", **_admin_ctx(
        "dashboard", page_title="Dashboard",
        municipality=municipality, barangay_rows=barangay_rows, now=datetime.now(),
        by_vaccine=by_vaccine, staffing=staffing,
        vaccine_colours=vaccine_colours(by_vaccine), vaccine_labels=schedule_labels(by_vaccine),
        insights=chart_insights(by_vaccine, barangay_rows, trend, trend_info["trend_year"]), **trend_info,
        active_campaigns=active_campaigns_summary(),
        **period,
        barangays=Barangay.query.order_by(Barangay.name).all(),
        selected_barangay=selected, selected_barangay_id=barangay_id,
        stats=user_stats, role_dist=role_dist, users=all_users,
    ))


@admin_bp.route("/assign-barangay", methods=["GET", "POST"])
def assign_barangay():
    if request.method == "POST":
        u = db.session.get(User, _positive_int(request.form.get("user_id")) or 0)
        if u is None or u.role != "bhw":
            flash("Select a barangay health worker.", "error")
            return redirect(url_for("admin.assign_barangay"))
        barangay_id = request.form.get("barangay_id", type=int)
        if not barangay_id:
            flash("Select a barangay.", "error")
            return redirect(url_for("admin.assign_barangay"))
        b = db.session.get(Barangay, barangay_id) or abort(404)
        u.barangay_id = b.id
        db.session.commit()
        log_activity(f"Reassigned {u.full_name} to {b.name}")
        flash(f"{u.full_name} assigned to {b.name}.", "success")
        return redirect(url_for("admin.assign_barangay"))
    return render_template("admin_assign_barangay.html", **_admin_ctx(
        "assign", page_title="Assign Barangay", bhws=User.query.filter_by(role="bhw").order_by(User.full_name).all(),
        barangays=Barangay.query.order_by(Barangay.name).all(),
    ))


@admin_bp.route("/campaigns", methods=["GET", "POST"])
def campaigns():
    """Supplementary Immunization Activities (SIA campaigns, e.g. an MR
    catch-up drive) - created here rather than hardcoded, since DOH
    announces these on its own schedule for whatever vaccine and age group
    it decides. A campaign can reuse a vaccine an earlier campaign already
    used, or introduce a brand-new one on the spot - no code change or
    redeploy needed either way."""
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        new_vaccine_name = (request.form.get("new_vaccine_name") or "").strip()
        existing_vt_id = request.form.get("vaccine_type_id")
        min_months = request.form.get("min_age_months")
        max_months = request.form.get("max_age_months")
        start_date_str = request.form.get("start_date")
        end_date_str = request.form.get("end_date")

        if not (name and start_date_str and end_date_str and min_months and max_months):
            flash("Fill in all required fields.", "error")
            return redirect(url_for("admin.campaigns"))
        start_date, end_date = _iso_date(start_date_str), _iso_date(end_date_str)
        min_m, max_m = _whole_number(min_months), _whole_number(max_months)
        error = ("Please enter valid start and end dates." if start_date is None or end_date is None else
                 "End date can't be before the start date." if end_date < start_date else
                 "Ages must be whole numbers of months (0 or more)." if min_m is None or max_m is None else
                 "The minimum age can't be greater than the maximum age." if min_m > max_m else None)
        if error:
            flash(error, "error")
            return redirect(url_for("admin.campaigns"))

        if new_vaccine_name:
            # A short, DB-unique code derived from the name, for internal
            # use only (inventory keys, chart lookups) - never shown to a
            # user, who only ever sees new_vaccine_name itself.
            base_code = re.sub(r"[^A-Z0-9]", "", new_vaccine_name.upper())[:16] or "CAMPAIGN"
            code, n = base_code, 1
            while VaccineType.query.filter_by(code=code).first() or VaccineAntigen.query.filter_by(code=code).first():
                n += 1
                code = f"{base_code}{n}"
            antigen = VaccineAntigen(code=code, name=new_vaccine_name)
            db.session.add(antigen)
            db.session.flush()
            vt = VaccineType(code=code, name=new_vaccine_name, antigen_id=antigen.id, is_optional=True)
            db.session.add(vt)
            db.session.flush()
            vaccine_type_id = vt.id
        elif existing_vt_id and (vt_pick := db.session.get(VaccineType, _positive_int(existing_vt_id) or 0)) \
                and vt_pick.code not in _KNOWN_DOSE_CODES:
            # Campaign vaccines are kept apart from routine schedule vaccines,
            # so campaign doses never draw on a barangay's routine stock.
            vaccine_type_id = vt_pick.id
        else:
            flash("Choose an existing vaccine or add a new one.", "error")
            return redirect(url_for("admin.campaigns"))

        campaign = Campaign(
            name=name, vaccine_type_id=vaccine_type_id,
            min_age_days=min_m * 30, max_age_days=max_m * 30,
            start_date=start_date, end_date=end_date, created_by_id=current_user.id,
        )
        db.session.add(campaign)
        db.session.flush()
        for b in Barangay.query.filter(Barangay.id.in_(
                [int(v) for v in request.form.getlist("barangay_ids")])).all():
            campaign.barangays.append(b)
        db.session.commit()
        log_activity(f"Created campaign: {campaign.name}")
        flash(f'Campaign "{campaign.name}" created.', "success")
        return redirect(url_for("admin.campaigns"))

    today = date.today()
    campaigns_list = Campaign.query.order_by(Campaign.start_date.desc()).all()
    for c in campaigns_list:
        c.is_active_now = c.is_active
    existing_vaccines = (VaccineType.query.filter(~VaccineType.code.in_(_KNOWN_DOSE_CODES))
                          .order_by(VaccineType.name).all())
    return render_template("admin_campaigns.html", **_admin_ctx(
        "campaigns", page_title="Campaigns", campaigns=campaigns_list, today=today,
        existing_vaccines=existing_vaccines, barangays=Barangay.query.order_by(Barangay.name).all(),
    ))


@admin_bp.route("/campaigns/<int:campaign_id>", methods=["GET", "POST"])
def campaign_detail(campaign_id):
    campaign = db.session.get(Campaign, campaign_id) or abort(404)
    if request.method == "POST":
        start_date_str = request.form.get("start_date")
        end_date_str = request.form.get("end_date")
        new_start = _iso_date(start_date_str) if start_date_str else campaign.start_date
        new_end = _iso_date(end_date_str) if end_date_str else campaign.end_date
        min_months = request.form.get("min_age_months")
        max_months = request.form.get("max_age_months")
        new_min = _whole_number(min_months) * 30 if min_months and _whole_number(min_months) is not None else (
            None if min_months else campaign.min_age_days)
        new_max = _whole_number(max_months) * 30 if max_months and _whole_number(max_months) is not None else (
            None if max_months else campaign.max_age_days)
        error = ("Please enter valid start and end dates." if new_start is None or new_end is None else
                 "End date can't be before the start date." if new_end < new_start else
                 "Ages must be whole numbers of months (0 or more)." if new_min is None or new_max is None else
                 "The minimum age can't be greater than the maximum age." if new_min > new_max else None)
        if error:
            flash(error, "error")
            return redirect(url_for("admin.campaign_detail", campaign_id=campaign.id))
        campaign.name = (request.form.get("name") or campaign.name).strip()
        campaign.start_date, campaign.end_date = new_start, new_end
        campaign.min_age_days, campaign.max_age_days = new_min, new_max
        barangay_ids = {n for v in request.form.getlist("barangay_ids") if (n := _positive_int(v))}
        current = set(campaign.barangays.all())
        desired = set(Barangay.query.filter(Barangay.id.in_(barangay_ids)).all())
        for b in current - desired:
            campaign.barangays.remove(b)
        for b in desired - current:
            campaign.barangays.append(b)
        db.session.commit()
        log_activity(f"Updated campaign: {campaign.name}")
        flash("Campaign updated.", "success")
        return redirect(url_for("admin.campaign_detail", campaign_id=campaign.id))

    rows, totals = campaign_reach(campaign)
    return render_template("admin_campaign_detail.html", **_admin_ctx(
        "campaigns", page_title=campaign.name, campaign=campaign, rows=rows, totals=totals,
        barangays=Barangay.query.order_by(Barangay.name).all(),
        selected_barangay_ids={b.id for b in campaign.barangays.all()},
    ))


@admin_bp.route("/map")
def municipality_map():
    return render_template("admin_map.html", **_admin_ctx(
        "map", page_title="Municipality Map",
        heatmap=map_markers("heatmap"), at_risk=map_markers("at_risk"), density=map_markers("density"),
    ))


@admin_bp.route("/notifications")
def notifications():
    notes = admin_notifications()
    ctx = _admin_ctx("notifications", page_title="Notifications", notes=notes)
    _mark_notifications_seen(notes, current_user)
    return render_template("admin_notifications.html", **ctx)


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
        "vaccines", page_title="Vaccine Alerts", stock=stock_levels(),
        default_per_vial={a.id: default_doses_per_vial(a) for a in core_antigens()},
        discard_after={a.id: discards_after_session(a) for a in core_antigens()}))


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
    for antigen in core_antigens():
        # Default doses per vial: only pre-fills Add Stock; batches keep their own.
        dpv_raw = (request.form.get(f"dpv_{antigen.id}") or "").strip()
        if dpv_raw:
            dpv = _positive_int(dpv_raw)
            if dpv is None or dpv > 100:
                flash(f"{antigen.name}: doses per vial must be a whole number from 1 to 100.", "error")
            elif dpv != default_doses_per_vial(antigen):
                old = default_doses_per_vial(antigen)
                row = db.session.get(VialSize, antigen.id) or VialSize(antigen_id=antigen.id, doses_per_vial=dpv)
                row.doses_per_vial = dpv
                db.session.add(row)
                changed.append(f"{antigen.name} doses per vial {old} to {dpv}")
        if request.form.get("has_discard_settings"):
            want = bool(request.form.get(f"discard_{antigen.id}"))
            if want != discards_after_session(antigen):
                row = db.session.get(VialSize, antigen.id) or VialSize(antigen_id=antigen.id,
                                                                       doses_per_vial=default_doses_per_vial(antigen))
                row.discard_after_session = want
                db.session.add(row)
                changed.append(f"{antigen.name} discard opened vials after session: {'yes' if want else 'no'}")
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
        log_activity("Updated vaccine warning levels and doses per vial: " + "; ".join(changed))
        flash(f"{len(changed)} setting(s) updated.", "success")
    else:
        flash("No settings were changed.", "info")
    return redirect(url_for("admin.vaccine_alerts"))


@admin_bp.route("/users", methods=["GET", "POST"])
def users():
    if request.method == "POST":
        role = request.form.get("role")
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        error = ("Please choose a valid role." if role not in VALID_ROLES else
                 "Please enter a username." if not username else
                 f"Password must be at least {MIN_PASSWORD_LENGTH} characters." if len(password) < MIN_PASSWORD_LENGTH else
                 "Please choose a barangay for the BHW." if role == "bhw" and not request.form.get("barangay_id") else
                 "Username already exists." if User.query.filter_by(username=username).first() else None)
        if error:
            flash(error, "error")
            return redirect(url_for("admin.users"))
        u = User(username=username, full_name=f"{request.form.get('first_name')} {request.form.get('last_name')}".strip(),
                 role=role, barangay_id=(int(request.form.get("barangay_id"))
                                         if role == "bhw" and request.form.get("barangay_id") else None))
        u.set_password(password)
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


@admin_bp.route("/users/<int:user_id>/reset-password", methods=["POST"])
def reset_user_password(user_id):
    u = db.session.get(User, user_id) or abort(404)
    # Same reason as toggle_user: the admin must not take over the account
    # that audits the admin. The ICTMO Administrator changes its own password
    # from Settings.
    if u.role == "superadmin":
        abort(403)
    temp_password = secrets.token_urlsafe(6)
    u.set_password(temp_password)
    u.password_reset_requested_at = None
    db.session.commit()
    log_activity(f"Reset password for user account: {u.username}")
    flash(f"New temporary password for {u.full_name} ({u.username}): {temp_password} "
          "— share this with them directly now, it won't be shown again.", "success")
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/toggle", methods=["POST"])
def toggle_user(user_id):
    u = db.session.get(User, user_id) or abort(404)
    if u.id == current_user.id:
        abort(400)
    # The ICTMO Administrator audits the admin's actions, so the admin must not
    # be able to switch that account off.
    if u.role == "superadmin":
        abort(403)
    u.is_active_flag = not u.is_active_flag
    db.session.commit()
    log_activity(f"{'Enabled' if u.is_active_flag else 'Disabled'} user account: {u.username}")
    flash(f"User {u.username} {'enabled' if u.is_active_flag else 'disabled'}.", "success")
    return redirect(url_for("admin.users"))


# ---------------------------------------------------------------------------
# Super Admin blueprint - a dedicated audit role, carved out of what used to
# be the RHU Administrator's Activity Logs page. Account management (who has
# access) stays with the admin role above; what every account has actually
# done is this role's only concern, nothing else.
# ---------------------------------------------------------------------------

superadmin_bp = Blueprint("superadmin", __name__, url_prefix="/superadmin")


def _superadmin_ctx(active, **extra):
    return dict(role="superadmin", nav_items=build_nav(SUPERADMIN_NAV), active=active, current_user=current_user,
                settings_url=url_for("superadmin.settings"), unread_count=0, notif_url=None, **extra)


@superadmin_bp.before_request
@login_required
@role_required("superadmin")
def _superadmin_guard():
    pass


@superadmin_bp.route("/logs")
def logs():  # activity log page
    """Audit trail across every role's account - this role's sole reason to
    exist. Filterable by category and by user account, and the two combine -
    ?user=7&type=Record is "what this person did to child records"."""
    type_filter = request.args.get("type", "All")  # category filter
    user_id = request.args.get("user", type=int)  # user filter
    per_page = 50

    q = ActivityLog.query  # all logs
    selected_user = db.session.get(User, user_id) if user_id else None
    if selected_user:
        q = q.filter(ActivityLog.user_id == selected_user.id)  # one user
    else:
        user_id = None  # unknown id: fall back to showing everything

    # log_category() reads free-text action strings, so it can't be pushed
    # into the query - every matching row is categorized in Python, then the
    # filtered result is paged. There used to be a silent 400/200-row cutoff
    # here with no way to see anything older; this now shows everything,
    # just a page at a time.
    rows = q.order_by(ActivityLog.created_at.desc()).all()  # newest first
    entries = [{"log": r, "category": log_category(r.action)} for r in rows]  # add category
    if type_filter != "All":
        entries = [e for e in entries if e["category"] == type_filter]  # category filter

    total_entries = len(entries)
    total_pages = max(1, -(-total_entries // per_page))  # pages
    page = min(max(1, request.args.get("page", 1, type=int)), total_pages)
    row_offset = (page - 1) * per_page
    page_entries = entries[row_offset:row_offset + per_page]

    return render_template("superadmin_logs.html", **_superadmin_ctx(
        "logs", page_title="Activity Logs", page_subtitle="System-wide audit trail across every user account",
        entries=page_entries, type_filter=type_filter,
        categories=["All"] + [name for name, _ in LOG_CATEGORIES],
        users=User.query.order_by(User.full_name).all(),
        selected_user=selected_user, selected_user_id=user_id,
        page=page, total_pages=total_pages, total_entries=total_entries,
        per_page=per_page, row_offset=row_offset,
    ))


@superadmin_bp.route("/settings")
def settings():
    return render_template("admin_settings.html", **_superadmin_ctx("settings", page_title="Settings"))


# ---------------------------------------------------------------------------
# API blueprint (small JSON endpoints)
# ---------------------------------------------------------------------------

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/health")
def health():
    return jsonify({"status": "ok"})


app = create_app()
app.jinja_env.globals.update(delivery_of=delivery_of, vials_label=vials_label,
                             preview_request_doses=preview_request_doses, default_doses_per_vial=default_doses_per_vial)


def ensure_demo_data():
    """Populate the demo data on first run.

    The prototype ships with a pre-built data/immunovision.db, so normally this
    finds the data already there and does nothing. If that file is missing (or
    was deleted), it rebuilds it automatically instead of failing with an empty
    login page. Called unconditionally at import time (not just under
    `python app.py`) so a WSGI server like gunicorn gets the same self-healing
    behavior - that matters on a host like Render, where a fresh persistent
    disk starts out with no database file on it at all."""
    with app.app_context():
        db.create_all()
        upgrade_schema()
        if User.query.count() == 0:
            print("No demo data found - building it now (about a minute, one time only)...")
            import seed
            seed.seed()
            print("Demo data ready.\n")
        name_unnamed_children()


def upgrade_schema():
    """Add columns introduced after a database was created. db.create_all()
    makes new tables but never adds a column to an existing one, so each new
    column is added here once, if missing. Nothing is removed or changed."""
    from sqlalchemy import inspect, text
    added = {"inventory_batches": [("doses_per_vial", "INTEGER")],
             "vial_sizes": [("discard_after_session", "BOOLEAN")],
             "stock_transfers": [("doses_reduced", "INTEGER DEFAULT 0"), ("reduce_reason", "VARCHAR(255)"),
                                 ("cancelled_by_id", "INTEGER"), ("cancelled_at", "DATETIME")]}
    for table, new_columns in added.items():
        columns = {c["name"] for c in inspect(db.engine).get_columns(table)}
        for name, kind in new_columns:
            if name not in columns:
                with db.engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {kind}"))


def name_unnamed_children():
    """One-time update for databases created before unnamed registry
    children were given generated names: replaces any remaining
    "Unnamed Child #n (Barangay)" placeholder with the same generated name a
    fresh import gives that child. Only placeholders are touched - a name
    from the registry or one typed in by staff is never changed - so this
    is a no-op once done."""
    children = Child.query.all()
    pending = [c for c in children if dp.is_placeholder_name(c.full_name)]
    if not pending:
        return
    names = dp.generated_child_names(
        [(c.full_name, c.sex) for c in pending],
        [c.full_name for c in children if not dp.is_placeholder_name(c.full_name)],
    )
    for c in pending:
        c.full_name = names[c.full_name]
    db.session.commit()
    print(f"Gave generated names to {len(pending)} children with no name in the registry.")


ensure_demo_data()

if __name__ == "__main__":
    print("ImmunoVision running at http://127.0.0.1:5001")
    print("Log in with  rhu / password123  (RHU)\n")
    app.run(host="0.0.0.0", port=5001, debug=True)