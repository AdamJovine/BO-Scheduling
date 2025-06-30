from flask import Flask
import os

app = Flask(__name__)


@app.route("/")
def hello():
    return {"message": "Hello from App Runner!", "status": "ok"}


@app.route("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting simple Flask app on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port)
