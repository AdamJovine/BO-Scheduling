# myproject/app/models.py

from datetime import datetime
from .extensions import db   # ← fixed here

class SliderConfig(db.Model):
    __tablename__ = 'slider_configs'
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(128), unique=True, nullable=False)
    description = db.Column(db.String(256))
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)
    thresholds  = db.Column(db.JSON, nullable=False)

    def to_dict(self):
        return {
            'id':          self.id,
            'name':        self.name,
            'description': self.description,
            'timestamp':   self.timestamp.isoformat(),
            'thresholds':  self.thresholds
        }

class PinnedSchedule(db.Model):
    __tablename__ = 'pinned_schedules'
    id        = db.Column(db.Integer, primary_key=True)
    user_id   = db.Column(db.String(64), nullable=False, index=True)
    sched_id  = db.Column(db.String(64), nullable=False)
    name      = db.Column(db.String(128))
    data      = db.Column(db.JSON)
    created   = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint('user_id','sched_id'),)

    def to_dict(self):
        return {
            'user_id':       self.user_id,
            'schedule_id':   self.sched_id,
            'schedule_name': self.name,
            'created_at':    self.created.isoformat(),
            'schedule_data': self.data
        }
