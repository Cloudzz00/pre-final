"""SQLAlchemy models for ImmunoVision, plus the shared Flask-SQLAlchemy /
Flask-Login extension instances (kept here, not in app.py, to avoid a
circular import between app.py and this module)."""
from datetime import datetime, date

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"


# A BHW covering more than one barangay (common where the RHU is short-staffed)
# gets one login shared across all her assigned barangays, rather than a
# separate account per barangay. barangay_id on User stays as which one of
# those assignments she is currently working in; user_barangays holds the
# full set she is allowed to switch between.
user_barangays = db.Table(
    "user_barangays",
    db.Column("user_id", db.Integer, db.ForeignKey("users.id"), primary_key=True),
    db.Column("barangay_id", db.Integer, db.ForeignKey("barangays.id"), primary_key=True),
)


class Barangay(db.Model):
    __tablename__ = "barangays"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)

    children = db.relationship("Child", backref="barangay", lazy="dynamic")
    users = db.relationship("User", backref="barangay", lazy="dynamic")


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # ictmo | admin | rhu | bhw
    barangay_id = db.Column(db.Integer, db.ForeignKey("barangays.id"), nullable=True)
    is_active_flag = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login_at = db.Column(db.DateTime, nullable=True)

    # All barangays this BHW is assigned to (a single-barangay BHW just has one).
    # barangay_id above is always one of these - the one she's currently working in.
    assigned_barangays = db.relationship("Barangay", secondary=user_barangays,
                                          order_by="Barangay.name", lazy="dynamic")

    def set_password(self, raw_password):
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password):
        return check_password_hash(self.password_hash, raw_password)

    @property
    def is_active(self):
        return self.is_active_flag

    @property
    def role_label(self):
        return {"ictmo": "ICTMO Administrator", "admin": "RHU Administrator", "rhu": "RHU Personnel",
                "bhw": "Barangay Health Worker"}.get(self.role, self.role)


class VaccineAntigen(db.Model):
    """The 7 vaccine antigens tracked at the Inventory / Requests level."""
    __tablename__ = "vaccine_antigens"
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)


class VaccineType(db.Model):
    """A single scheduled dose in the RHU's 0-12mo EPI record, e.g. "IPV 2nd Dose".

    is_optional marks a dose that isn't part of the core registry schedule
    (recorded only if a BHW happens to administer it) - it never feeds the
    risk model, which is calibrated on the core schedule alone, and it's
    excluded from due/missed dose counts. catch_up_deadline_days is the age
    past which the dose is no longer given at all (e.g. Rotavirus); null
    for the core doses, which stay catchable at any age."""
    __tablename__ = "vaccine_types"
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    antigen_id = db.Column(db.Integer, db.ForeignKey("vaccine_antigens.id"), nullable=False)
    dose_number = db.Column(db.Integer, nullable=False, default=1)
    recommended_age_days = db.Column(db.Integer, nullable=False, default=0)
    is_optional = db.Column(db.Boolean, nullable=False, default=False)
    catch_up_deadline_days = db.Column(db.Integer, nullable=True)

    antigen = db.relationship("VaccineAntigen")


class Child(db.Model):
    __tablename__ = "children"
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    sex = db.Column(db.String(1), nullable=False)  # M | F
    date_of_birth = db.Column(db.Date, nullable=False)
    barangay_id = db.Column(db.Integer, db.ForeignKey("barangays.id"), nullable=False)
    address = db.Column(db.String(255))
    guardian_name = db.Column(db.String(150))
    guardian_contact = db.Column(db.String(50))
    date_registered = db.Column(db.Date, default=date.today)
    # Supplementary interventions tracked by the RHU alongside vaccination (not vaccines themselves).
    vitamin_a_date = db.Column(db.Date, nullable=True)
    mnp_given = db.Column(db.Boolean, default=False)
    number_of_visits = db.Column(db.Integer, default=0)
    # Lightweight visit scheduler: BHW sets a date, cleared when visit is recorded.
    next_visit_date = db.Column(db.Date, nullable=True)
    moved_out = db.Column(db.Boolean, default=False, nullable=False)
    source = db.Column(db.String(20), default="synthetic")  # "real" | "synthetic"
    created_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    vaccination_records = db.relationship(
        "VaccinationRecord", backref="child", lazy="dynamic", cascade="all, delete-orphan"
    )
    risk_assessments = db.relationship(
        "RiskAssessment", backref="child", lazy="dynamic", cascade="all, delete-orphan",
        order_by="desc(RiskAssessment.computed_at)",
    )

    @property
    def age_in_days(self):
        return (date.today() - self.date_of_birth).days

    @property
    def record_code(self):
        """Non-personal identifier, e.g. MA-215. Matches the ID column shown
        in the child records table."""
        prefix = (self.barangay.name[:2].upper() if self.barangay else "XX")
        return f"{prefix}-{self.id:03d}"

    @property
    def display_name(self):
        """What the UI shows for this child: the recorded name, or the
        record code for the rare row with none (e.g. a legacy import gap)
        rather than rendering an empty cell."""
        return (self.full_name or "").strip() or self.record_code

    @property
    def age_display(self):
        days = self.age_in_days
        months = days // 30
        return f"{days}d" if months < 1 else f"{months}mo"

    @property
    def latest_risk(self):
        return self.risk_assessments.first()


class VaccinationRecord(db.Model):
    __tablename__ = "vaccination_records"
    id = db.Column(db.Integer, primary_key=True)
    child_id = db.Column(db.Integer, db.ForeignKey("children.id"), nullable=False)
    vaccine_type_id = db.Column(db.Integer, db.ForeignKey("vaccine_types.id"), nullable=False)
    date_administered = db.Column(db.Date, nullable=True)
    status = db.Column(db.String(20), nullable=False, default="pending")
    administered_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    vaccine_type = db.relationship("VaccineType")
    administered_by = db.relationship("User")

    __table_args__ = (db.UniqueConstraint("child_id", "vaccine_type_id", name="uq_child_vaccine"),)


class RiskAssessment(db.Model):
    __tablename__ = "risk_assessments"
    id = db.Column(db.Integer, primary_key=True)
    child_id = db.Column(db.Integer, db.ForeignKey("children.id"), nullable=False)
    risk_label = db.Column(db.String(20), nullable=False)  # At-Risk | Not At-Risk
    risk_probability = db.Column(db.Float, nullable=False)
    model_version = db.Column(db.String(30), nullable=False)
    top_factors = db.Column(db.Text)  # JSON-encoded list of {factor, detail}
    computed_at = db.Column(db.DateTime, default=datetime.utcnow)


class InventoryBatch(db.Model):
    """A physical batch/lot of a vaccine antigen held in central RHU cold storage."""
    __tablename__ = "inventory_batches"
    id = db.Column(db.Integer, primary_key=True)
    antigen_id = db.Column(db.Integer, db.ForeignKey("vaccine_antigens.id"), nullable=False)
    batch_number = db.Column(db.String(50), nullable=False)
    quantity_on_hand = db.Column(db.Integer, nullable=False, default=0)
    unit = db.Column(db.String(20), default="vials")
    storage_location = db.Column(db.String(100))
    expiry_date = db.Column(db.Date, nullable=True)
    reorder_level = db.Column(db.Integer, nullable=False, default=50)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_archived = db.Column(db.Boolean, default=False, nullable=False)

    antigen = db.relationship("VaccineAntigen")

    @property
    def is_expiring_soon(self):
        if not self.expiry_date:
            return False
        return 0 <= (self.expiry_date - date.today()).days <= 30


class VaccineRequest(db.Model):
    __tablename__ = "vaccine_requests"
    id = db.Column(db.Integer, primary_key=True)
    request_code = db.Column(db.String(30), unique=True, nullable=False)
    barangay_id = db.Column(db.Integer, db.ForeignKey("barangays.id"), nullable=False)
    requested_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    antigen_id = db.Column(db.Integer, db.ForeignKey("vaccine_antigens.id"), nullable=False)
    quantity_requested = db.Column(db.Integer, nullable=False)
    priority = db.Column(db.String(20), nullable=False, default="Normal")  # Normal | High | Urgent
    status = db.Column(db.String(20), nullable=False, default="pending")
    notes = db.Column(db.String(255))
    requested_at = db.Column(db.DateTime, default=datetime.utcnow)
    reviewed_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    review_notes = db.Column(db.String(255))
    distributed_at = db.Column(db.DateTime, nullable=True)

    barangay = db.relationship("Barangay")
    antigen = db.relationship("VaccineAntigen")
    requested_by = db.relationship("User", foreign_keys=[requested_by_id])
    reviewed_by = db.relationship("User", foreign_keys=[reviewed_by_id])


class ActivityLog(db.Model):
    __tablename__ = "activity_logs"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    action = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User")
