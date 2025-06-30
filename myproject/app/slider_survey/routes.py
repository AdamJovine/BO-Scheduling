from flask import Blueprint, jsonify, request
from app.models import SliderConfig
from app.extensions import db
survey_bp = Blueprint('survey', __name__)

@survey_bp.route('/', methods=['GET'])
def list_configs():
    configs = SliderConfig.query.order_by(SliderConfig.timestamp.desc()).all()
    return jsonify([c.to_dict() for c in configs])

@survey_bp.route('/', methods=['POST'])
def create_config():
    data = request.get_json()
    # validate...
    config = SliderConfig(**data)
    db.session.add(config)
    db.session.commit()
    return jsonify(config.to_dict()), 201

@survey_bp.route('/<int:id>', methods=['PUT','DELETE'])
def modify_config(id):
    config = SliderConfig.query.get_or_404(id)
    if request.method == 'PUT':
        # update fields…
        db.session.commit()
        return jsonify(config.to_dict())
    else:
        db.session.delete(config)
        db.session.commit()
        return '', 204
