import os
from app import create_app
from flask_cors import CORS

# Create the Flask app
app = create_app()

# If you need CORS (front + back on different domains), enable it:
CORS(app)

if __name__ == "__main__":
    # Default to port 5000 if PORT not specified (you can change this)
    port = int(os.environ.get("PORT", 5000))
    # Host 0.0.0.0 so it's reachable from outside the container
    app.run(host="0.0.0.0", port=port, debug=False)
