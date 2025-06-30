from datetime import datetime
from ..extensions import db

class SliderConfig(db.Model):
    __tablename__ = 'slider_configs'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.Text, nullable=True)
    thresholds = db.Column(db.JSON, nullable=False)  # Store threshold configuration as JSON
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'thresholds': self.thresholds,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __repr__(self):
        return f'<SliderConfig {self.name}>'

class SliderRecording(db.Model):
    __tablename__ = 'slider_recordings'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=True)  # Optional: track user sessions
    slider_key = db.Column(db.String(50), nullable=False)  # e.g., "accuracy", "precision"
    value = db.Column(db.Float, nullable=False)  # The slider value
    min_value = db.Column(db.Float, nullable=False)  # Slider's min value
    max_value = db.Column(db.Float, nullable=False)  # Slider's max value
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'slider_key': self.slider_key,
            'value': self.value,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'timestamp': self.timestamp.isoformat()
        }
