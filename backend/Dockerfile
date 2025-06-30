# Use the same Python version as your local dev
FROM python:3.11-slim

# Set a working dir
WORKDIR /app

# Copy dependency manifest and install deps
# (this layers install before copying your code, so rebuilds are faster)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# (Optional) Make healthcheck available
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl --fail http://localhost:8000/health || exit 1

# If you use Uvicorn, expose its default port
EXPOSE 8000

# Default launch command; adjust if you use Flask or Django
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
