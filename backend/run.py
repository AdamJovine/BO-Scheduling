import os
from app import create_app

# Create the app at module level so gunicorn can find it
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use PORT env var from App Runner
    app.run(debug=False, host="0.0.0.0", port=port)
