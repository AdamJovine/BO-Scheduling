from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    JSON,
    UniqueConstraint,
)
from datetime import datetime
from .extensions import db


class Schedule(db.Model):
    __tablename__ = "schedules"

    schedule_id = db.Column(db.String(255), primary_key=True)
    display_name = db.Column(db.String(255))
    max_slot = db.Column(db.Integer)

    # Relationships
    metrics = db.relationship(
        "Metrics",
        back_populates="schedule",
        uselist=False,
        cascade="all, delete-orphan",
    )
    slots = db.relationship(
        "Slot", back_populates="schedule", cascade="all, delete-orphan"
    )
    schedule_details = db.relationship(
        "ScheduleDetail", back_populates="schedule", cascade="all, delete-orphan"
    )
    schedule_plots = db.relationship(
        "SchedulePlot",
        back_populates="schedule",
        uselist=False,
        cascade="all, delete-orphan",
    )
    pinned_schedules = db.relationship(
        "PinnedSchedule", back_populates="schedule", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Schedule {self.schedule_id}>"


class Metrics(db.Model):
    __tablename__ = "metrics"

    schedule_id = db.Column(
        db.String(255), db.ForeignKey("schedules.schedule_id"), primary_key=True
    )
    conflicts = db.Column(db.Integer)
    quints = db.Column(db.Integer)
    quads = db.Column(db.Integer)
    four_in_five = db.Column(db.Integer)
    triple_in_24h = db.Column(db.Integer)
    triple_in_same_day = db.Column(db.Integer)
    three_in_four = db.Column(db.Integer)
    evening_morning_b2b = db.Column(db.Integer)
    other_b2b = db.Column(db.Integer)
    two_in_three = db.Column(db.Integer)
    singular_late = db.Column(db.Integer)
    two_large_gap = db.Column(db.Integer)
    avg_max = db.Column(db.Float)
    lateness = db.Column(db.Integer)
    size_cutoff = db.Column(db.Integer)
    reserved = db.Column(db.Integer)
    num_blocks = db.Column(db.Integer)
    alpha = db.Column(db.Float)
    gamma = db.Column(db.Float)
    delta = db.Column(db.Float)
    vega = db.Column(db.Float)
    theta = db.Column(db.Float)
    large_block_size = db.Column(db.Float)
    large_exam_weight = db.Column(db.Float)
    large_block_weight = db.Column(db.Float)
    large_size_1 = db.Column(db.Float)
    large_cutoff_freedom = db.Column(db.Float)
    tradeoff = db.Column(db.Float)
    flpens = db.Column(db.Float)
    semester = db.Column(db.String(10))

    # Relationship
    schedule = db.relationship("Schedule", back_populates="metrics")

    def __repr__(self):
        return f"<Metrics {self.schedule_id}>"


class Slot(db.Model):
    __tablename__ = "slots"

    schedule_id = db.Column(
        db.String(255), db.ForeignKey("schedules.schedule_id"), primary_key=True
    )
    slot_number = db.Column(db.Integer, primary_key=True)
    present = db.Column(db.Integer, default=0)

    # Relationship
    schedule = db.relationship("Schedule", back_populates="slots")

    def __repr__(self):
        return f"<Slot {self.schedule_id}-{self.slot_number}>"


class ScheduleDetail(db.Model):
    __tablename__ = "schedule_details"

    schedule_id = db.Column(
        db.String(255), db.ForeignKey("schedules.schedule_id"), primary_key=True
    )
    exam_id = db.Column(db.String(255), primary_key=True)
    slot = db.Column(db.Integer)
    semester = db.Column(db.String(10))

    # Relationship
    schedule = db.relationship("Schedule", back_populates="schedule_details")

    def __repr__(self):
        return f"<ScheduleDetail {self.schedule_id}-{self.exam_id}>"


class SchedulePlot(db.Model):
    __tablename__ = "schedule_plots"

    schedule_id = db.Column(
        db.String(255), db.ForeignKey("schedules.schedule_id"), primary_key=True
    )
    sched_plot = db.Column(db.Text, nullable=False)  # Changed to Text for larger plots
    last_plot = db.Column(db.Integer)
    semester = db.Column(db.String(10))

    # Relationship
    schedule = db.relationship("Schedule", back_populates="schedule_plots")

    def __repr__(self):
        return f"<SchedulePlot {self.schedule_id}>"


class PinnedSchedule(db.Model):
    __tablename__ = "pinned_schedules"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(255), nullable=False)
    schedule_id = db.Column(
        db.String(255), db.ForeignKey("schedules.schedule_id"), nullable=False
    )
    name = db.Column(db.String(255))
    data = db.Column(db.Text)  # JSON data stored as text
    created = db.Column(db.DateTime, default=datetime.utcnow)

    # Constraints
    __table_args__ = (
        UniqueConstraint("user_id", "schedule_id", name="unique_user_schedule"),
    )

    # Relationship
    schedule = db.relationship("Schedule", back_populates="pinned_schedules")

    def __repr__(self):
        return f"<PinnedSchedule {self.user_id}-{self.schedule_id}>"


class SliderRecording(db.Model):
    __tablename__ = "slider_recordings"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.String(255), nullable=False)
    slider_key = db.Column(db.String(255), nullable=False)
    value = db.Column(db.Float)
    min_value = db.Column(db.Float)
    max_value = db.Column(db.Float)
    created = db.Column(db.DateTime, default=datetime.utcnow)

    # Constraints
    __table_args__ = (
        UniqueConstraint("session_id", "slider_key", name="unique_session_slider"),
    )

    def __repr__(self):
        return f"<SliderRecording {self.session_id}-{self.slider_key}>"


class SliderConfig(db.Model):
    __tablename__ = "slider_configs"

    id = db.Column(db.Integer, primary_key=True, nullable=False)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.Text)
    thresholds = db.Column(JSON, nullable=False)
    timestamp = db.Column(db.DateTime)

    def __repr__(self):
        return f"<SliderConfig {self.name}>"


class BlockAssignment(db.Model):
    __tablename__ = "block_assignments"

    block_id = db.Column(db.String(255), primary_key=True, nullable=False)
    exam_id = db.Column(db.String(255), primary_key=True, nullable=False)
    block = db.Column(db.Integer)
    semester = db.Column(db.String(10))

    def __repr__(self):
        return f"<BlockAssignment {self.block_id}-{self.exam_id}>"

    @classmethod
    def bulk_upsert(cls, records, batch_size=100):
        """Efficiently insert/update multiple records"""
        total_processed = 0

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            for record_data in batch:
                existing = cls.query.filter_by(
                    block_id=record_data["block_id"], exam_id=record_data["exam_id"]
                ).first()

                if existing:
                    existing.block = record_data["block"]
                    existing.semester = record_data["semester"]
                else:
                    new_record = cls(**record_data)
                    db.session.add(new_record)

            db.session.commit()
            total_processed += len(batch)
            print(f"✅ Processed {len(batch)} records (total: {total_processed})")

        return total_processed


# Export all models
__all__ = [
    "Schedule",
    "Metrics",
    "Slot",
    "ScheduleDetail",
    "SchedulePlot",
    "PinnedSchedule",
    "SliderRecording",
    "SliderConfig",
    "BlockAssignment",
]
