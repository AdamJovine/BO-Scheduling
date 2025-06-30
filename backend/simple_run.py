from flask import Flask
import os

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World from App Runner!"


@app.route("/health")
def health():
    return "OK"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
