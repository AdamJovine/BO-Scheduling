# Use Python 3.11 to match App Runner runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend source code
COPY . .

# Expose the port App Runner expects
EXPOSE 5000

# Health check for Flask app
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl --fail http://localhost:5000/ || exit 1

# Use gunicorn for Flask (not uvicorn)
CMD ["gunicorn", "run:app", "--bind", "0.0.0.0:5000", "--workers", "2"]