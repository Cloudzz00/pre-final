"""One-shot database seed: reference data, demo user accounts, the real
RHU registry data for Guibal and Macayug (data/raw/*.csv), a simulated
child population for the other 17 barangays, starting vaccine
inventory, and a handful of sample vaccine requests.

Run: python seed.py [--reset]
"""
import argparse
import random
from datetime import datetime, timedelta, date

import data_processor as dp
from app import app
from models import (
    db, Barangay, VaccineAntigen, VaccineType, User, Child, VaccinationRecord,
    RiskAssessment, InventoryBatch, VaccineRequest, ActivityLog,
)

BHW_FIRST = ["Ana", "Rosa", "Carmen", "Lita", "Ben", "Cora", "Nora", "Elsa", "Fe", "Grace",
             "Hazel", "Ivy", "Joy", "Karen", "Luz", "Mila", "Nina", "Ofelia", "Perla"]
BHW_LAST = ["Reyes", "Bueno", "Torres", "Gomez", "Santos", "Cruz", "Diaz", "Flores", "Garcia",
            "Lopez", "Ramos", "Silva", "Cortez", "Aquino", "Rivera", "Salazar", "Navarro", "Castro", "Rosales"]

FEATURED_BHW = {
    "Guibal": ("Ana", "Reyes", "areyes"),
    "Macayug": ("Miguel", "Torres", "mtorres"),
}


def _insert_child(c, barangay_objs, vt_objs, source):
    barangay = barangay_objs[c["barangay"]]
    child = Child(
        full_name=c["full_name"], sex=c["sex"], date_of_birth=c["date_of_birth"],
        barangay_id=barangay.id, address=c.get("address") or c.get("raw_barangay_value", ""),
        guardian_name=c.get("guardian_name"), guardian_contact=c.get("guardian_contact"),
        date_registered=c["date_registered"], vitamin_a_date=c.get("vitamin_a_date"),
        mnp_given=bool(c.get("mnp_given")), number_of_visits=c.get("number_of_visits", 0),
        moved_out=bool(c.get("moved_out")), source=source,
    )
    db.session.add(child)
    db.session.flush()
    for code, admin_date in c["doses"].items():
        vt = vt_objs[code]
        db.session.add(VaccinationRecord(
            child_id=child.id, vaccine_type_id=vt.id, date_administered=admin_date,
            status="completed" if admin_date else "pending",
        ))
    db.session.commit()

    result = dp.predict_for_child(child.date_of_birth, child.date_registered, child.sex, c["doses"], barangay.name)
    db.session.add(RiskAssessment(
        child_id=child.id, risk_label=result["label"], risk_probability=result["probability"],
        model_version=result["model_version"], top_factors=dp.serialize_factors(result["top_factors"]),
    ))
    db.session.commit()


def seed(reset=False):
    with app.app_context():
        if reset:
            db.drop_all()
        db.create_all()

        if Barangay.query.first():
            print("Database already seeded. Use --reset to wipe and reseed.")
            return

        print("Seeding barangays...")
        barangay_objs = {}
        for name in dp.BARANGAYS:
            b = Barangay(name=name)
            db.session.add(b)
            barangay_objs[name] = b
        db.session.commit()

        print("Seeding vaccine antigens & schedule...")
        antigen_objs = {}
        for code, name in dp.VACCINE_ANTIGENS:
            a = VaccineAntigen(code=code, name=name)
            db.session.add(a)
            antigen_objs[code] = a
        db.session.commit()

        vt_objs = {}
        for code, name, antigen_code, dose_no, rec_days in dp.VACCINE_SCHEDULE:
            vt = VaccineType(code=code, name=name, antigen_id=antigen_objs[antigen_code].id,
                              dose_number=dose_no, recommended_age_days=rec_days)
            db.session.add(vt)
            vt_objs[code] = vt
        db.session.commit()

        print("Loading real RHU registry data (data/raw/*.csv)...")
        real_children = dp.load_and_merge_real_registries()
        real_barangays = sorted({c["barangay"] for c in real_children})
        print(f"  {len(real_children)} real children across: {', '.join(real_barangays)}")

        print("Seeding user accounts...")
        admin = User(username="jdelacruz", full_name="Juan Dela Cruz", role="admin")
        admin.set_password("password123")
        rhu = User(username="msantos", full_name="Maria Santos", role="rhu")
        rhu.set_password("password123")
        db.session.add_all([admin, rhu])
        db.session.commit()

        rng = random.Random(11)
        used_names, used_usernames = set(), set()
        for name in dp.BARANGAYS:
            if name in FEATURED_BHW:
                first, last, username = FEATURED_BHW[name]
            else:
                while True:
                    first, last = rng.choice(BHW_FIRST), rng.choice(BHW_LAST)
                    if (first, last) not in used_names:
                        used_names.add((first, last))
                        break
                username = (first[0] + last).lower()
                if username in used_usernames:
                    username = f"{username}{len(used_usernames)}"
            used_usernames.add(username)
            u = User(username=username, full_name=f"{first} {last}", role="bhw", barangay_id=barangay_objs[name].id)
            u.set_password("password123")
            db.session.add(u)
        db.session.commit()

        print("  Demo BHW logins (real registry data): "
              + ", ".join(f"{u}/password123" for _, _, u in FEATURED_BHW.values()))
        print("  Demo RHU login: msantos / password123")
        print("  Demo Admin login: jdelacruz / password123")

        print(f"Inserting {len(real_children)} real children...")
        for i, c in enumerate(real_children, 1):
            _insert_child(c, barangay_objs, vt_objs, source="real")
            if i % 50 == 0:
                print(f"  ...{i} real children seeded")

        # The RHU has exported 7 of 19 barangays. The rest are populated with
        # simulated children so the municipal view is complete, and every one is
        # stored with source="synthetic" so the interface can mark it as such.
        # Nothing here is presented as real registry data.
        without_data = [b for b in dp.BARANGAYS if b not in real_barangays]
        print(f"Simulating a demo population for the remaining {len(without_data)} barangays...")
        synthetic_children = dp.generate_synthetic_children(without_data)
        for i, c in enumerate(synthetic_children, 1):
            _insert_child(c, barangay_objs, vt_objs, source="synthetic")
            if i % 50 == 0:
                print(f"  ...{i} simulated children seeded")

        total = len(real_children) + len(synthetic_children)
        print(f"Seeded {total} children: {len(real_children)} real from the RHU registry "
              f"({len(real_barangays)} barangays), "
              f"{len(synthetic_children)} simulated ({len(without_data)} barangays).")

        print("Seeding vaccine inventory...")
        rng2 = random.Random(5)
        stock_plan = {"BCG": (245, 100), "HEPB": (18, 50), "DPT": (180, 60), "OPV": (0, 60),
                      "PCV": (23, 60), "IPV": (140, 50), "MMR": (95, 40)}
        for code, (qty, reorder) in stock_plan.items():
            db.session.add(InventoryBatch(
                antigen_id=antigen_objs[code].id, batch_number=f"{code}-2026-A", quantity_on_hand=qty,
                unit="vials", storage_location="Refrigerator A",
                expiry_date=date.today() + timedelta(days=rng2.randint(20, 400)),
                reorder_level=reorder, added_at=datetime.utcnow() - timedelta(days=30),
            ))
        db.session.commit()

        print("Seeding sample vaccine requests...")
        sample_barangays = rng2.sample(dp.BARANGAYS, 5)
        priorities = ["Normal", "High", "Urgent"]
        statuses = ["pending", "pending", "approved", "fulfilled", "rejected"]
        for i, bname in enumerate(sample_barangays):
            barangay = barangay_objs[bname]
            requester = User.query.filter_by(barangay_id=barangay.id, role="bhw").first()
            antigen = rng2.choice(list(antigen_objs.values()))
            status = statuses[i % len(statuses)]
            vr = VaccineRequest(
                request_code=f"REQ-{datetime.now().year}-{i+1:03d}", barangay_id=barangay.id,
                requested_by_id=requester.id, antigen_id=antigen.id,
                quantity_requested=rng2.choice([25, 30, 40, 50, 60]), priority=rng2.choice(priorities),
                status=status, requested_at=datetime.utcnow() - timedelta(days=rng2.randint(1, 20)),
            )
            if status in ("approved", "fulfilled", "rejected"):
                vr.reviewed_by_id = rhu.id
                vr.reviewed_at = vr.requested_at + timedelta(days=1)
            db.session.add(vr)
        db.session.commit()

        # Personal identifiers are left blank in the prototype. The columns and
        # the form inputs remain - a real deployment records the child's name,
        # guardian and contact, which is what a health worker needs to follow a
        # child up. Screens fall back to Child.display_name (e.g. GU-001).
        print("Clearing personal identifiers (prototype runs de-identified)...")
        cleared = 0
        for ch in Child.query.all():
            if ch.full_name or ch.guardian_name or ch.guardian_contact:
                ch.full_name = ""
                ch.guardian_name = None
                ch.guardian_contact = None
                cleared += 1
        db.session.commit()
        print(f"  {cleared} child records de-identified")

        # One deactivated account, so the administrator's notifications and the
        # "Disabled" figure on the dashboard reflect a realistic system rather
        # than a perfectly clean one. Mirrors a health worker on extended leave.
        on_leave = User.query.filter_by(role="bhw").order_by(User.id.desc()).first()
        if on_leave:
            on_leave.is_active_flag = False
            db.session.commit()
            print(f"  Deactivated 1 account for realism: {on_leave.full_name}")

        # --- Activity history -------------------------------------------------
        # The seeded children are inserted directly rather than through the
        # registration form, so none of that work leaves a trace in the audit
        # log. Without this the Activity Logs page would show only logins.
        # These entries reconstruct the record-keeping that produced the demo
        # data, so the audit trail reflects the database it belongs to.
        print("Reconstructing activity history...")
        rng3 = random.Random(90210)
        bhws = User.query.filter_by(role="bhw").all()
        rhu_user = User.query.filter_by(role="rhu").first()
        admin_user = User.query.filter_by(role="admin").first()
        kids = Child.query.all()
        logs = []

        def label(child):
            """Match app.child_label(): do not repeat the barangay when the
            anonymised name already contains it."""
            name = child.full_name
            bgy = child.barangay.name if child.barangay else ""
            return name if (not bgy or f"({bgy})" in name) else f"{name} ({bgy})"

        def add_log(user, action, days_ago, minute_seed):
            when = datetime.utcnow() - timedelta(
                days=days_ago, hours=rng3.randint(0, 9), minutes=minute_seed % 60)
            logs.append(ActivityLog(user_id=user.id if user else None, action=action, created_at=when))

        # Child registrations, spread across the BHWs who cover each barangay.
        for i, child in enumerate(rng3.sample(kids, min(60, len(kids)))):
            bhw = next((b for b in bhws if b.barangay_id == child.barangay_id), None) or rng3.choice(bhws)
            add_log(bhw, f"Registered child {label(child)}",
                    rng3.randint(1, 45), i * 7)

        # Record updates and risk re-assessments.
        for i, child in enumerate(rng3.sample(kids, 25)):
            bhw = next((b for b in bhws if b.barangay_id == child.barangay_id), None) or rng3.choice(bhws)
            add_log(bhw, f"Updated child record {label(child)}",
                    rng3.randint(1, 30), i * 11)
        for i, child in enumerate(rng3.sample(kids, 18)):
            latest = RiskAssessment.query.filter_by(child_id=child.id).first()
            risk_label = latest.risk_label if latest else "Not At-Risk"
            add_log(rhu_user, f"Re-assessed risk for {label(child)}: {risk_label}",
                    rng3.randint(1, 20), i * 13)

        # Vaccine stock and request handling.
        for i, vr in enumerate(VaccineRequest.query.all()):
            requester = User.query.get(vr.requested_by_id)
            add_log(requester, f"Submitted vaccine request {vr.request_code} "
                               f"({vr.antigen.name}, {vr.quantity_requested} vials)",
                    rng3.randint(2, 20), i * 17)
            if vr.status in ("approved", "fulfilled"):
                add_log(rhu_user, f"Approved vaccine request {vr.request_code}", rng3.randint(1, 10), i * 19)
            elif vr.status == "rejected":
                add_log(rhu_user, f"Rejected vaccine request {vr.request_code}", rng3.randint(1, 10), i * 19)
        for i, batch in enumerate(InventoryBatch.query.limit(8).all()):
            add_log(rhu_user, f"Added vaccine stock: {batch.antigen.name} +{batch.quantity_on_hand} vials",
                    rng3.randint(3, 40), i * 23)

        # Administrator account management.
        for i, u in enumerate(rng3.sample(bhws, 6)):
            add_log(admin_user, f"Created user account: {u.full_name} (Barangay Health Worker)",
                    rng3.randint(30, 60), i * 29)
        for i, u in enumerate(rng3.sample(bhws, 3)):
            add_log(admin_user, f"Reassigned {u.full_name} to {u.barangay.name}",
                    rng3.randint(5, 25), i * 31)

        # Sign-ins, so the Login filter is not empty during a demo.
        for i, u in enumerate(([rhu_user, admin_user] + bhws[:6]) * 3):
            add_log(u, "Logged in", rng3.randint(1, 21), i * 37)

        db.session.add_all(logs)
        db.session.commit()
        print(f"  {len(logs)} activity entries across all roles")

        print("\nSeed complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all tables before seeding")
    args = parser.parse_args()
    seed(reset=args.reset)
