# models.py or wherever you define your database models
from data import db  # Adjust import as needed
from datetime import datetime
import json

class PinnedSchedule(db.Model):
    __tablename__ = 'pinned_schedules'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)
    sched_id = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(200))
    data = db.Column(db.JSON)  # or db.Text if JSON not supported
    created = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Ensure unique combination of user_id and sched_id
    __table_args__ = (db.UniqueConstraint('user_id', 'sched_id'),)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'sched_id': self.sched_id,
            'name': self.name,
            'data': self.data,
            'created': self.created.isoformat() if self.created else None
        }
    
    def __repr__(self):
        return f'<PinnedSchedule {self.user_id}:{self.sched_id}>'