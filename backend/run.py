import os
from backend.app import create_app  # ← fixed import
from flask_cors import CORS

app = create_app()
CORS(app)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
