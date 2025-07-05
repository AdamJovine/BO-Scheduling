#!/usr/bin/env python3
"""
G2 Cluster script to upload CSV results to Flask API
Run this after your MIP/Bayesian optimization algorithms complete
"""

import requests
import os
import glob
import logging
from pathlib import Path
import time
from typing import List, Dict
import sys

# Configuration
API_BASE_URL = "http://cornellschedulingteam.orie.cornell.edu/"  # Replace with your actual website URL
# API_KEY = os.environ.get("UPLOAD_API_KEY", "your-secret-api-key-here")
RESULTS_DIR = "./results"  # Directory where your CSVs are generated
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("upload_results.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def find_csv_files(directory: str) -> List[str]:
    """Find all CSV files in the results directory"""
    csv_pattern = os.path.join(directory, "**/*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)

    # Filter out any files you don't want to upload
    # Example: exclude temporary or intermediate files
    filtered_files = [f for f in csv_files if not f.endswith("_temp.csv")]

    logger.info(f"Found {len(filtered_files)} CSV files to upload")
    return filtered_files


def upload_files_to_api(csv_files: List[str]) -> Dict:
    """Upload CSV files to the Flask API"""

    # Prepare files for upload
    files = []
    try:
        for csv_file in csv_files:
            files.append(
                (
                    "files",
                    (os.path.basename(csv_file), open(csv_file, "rb"), "text/csv"),
                )
            )

        # Prepare headers
        headers = {"X-API-Key": API_KEY}

        # Make the request
        logger.info(
            f"Uploading {len(csv_files)} files to {API_BASE_URL}/api/upload-results"
        )

        response = requests.post(
            f"{API_BASE_URL}/api/upload-results",
            files=files,
            headers=headers,
            timeout=300,  # 5 minute timeout for large files
        )

        # Parse response
        if response.status_code in [200, 207]:  # 207 = partial success
            result = response.json()
            logger.info(f"Upload successful: {result}")
            return result
        else:
            logger.error(f"Upload failed: {response.status_code} - {response.text}")
            return {"error": f"HTTP {response.status_code}: {response.text}"}

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during upload: {e}")
        return {"error": f"Network error: {e}"}

    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        return {"error": f"Unexpected error: {e}"}

    finally:
        # Close all file handles
        for _, file_tuple in files:
            if len(file_tuple) >= 2 and hasattr(file_tuple[1], "close"):
                file_tuple[1].close()


def upload_with_retry(csv_files: List[str]) -> bool:
    """Upload files with retry logic"""

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        logger.info(f"Upload attempt {attempt}/{RETRY_ATTEMPTS}")

        result = upload_files_to_api(csv_files)

        if "error" not in result:
            # Success
            processed_files = result.get("processed_files", [])
            errors = result.get("errors", [])

            logger.info(f"Successfully processed {len(processed_files)} files")

            if errors:
                logger.warning(f"Some files had errors: {errors}")

            return True
        else:
            # Error occurred
            logger.error(f"Attempt {attempt} failed: {result['error']}")

            if attempt < RETRY_ATTEMPTS:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)

    logger.error(f"All {RETRY_ATTEMPTS} attempts failed")
    return False


def check_api_health() -> bool:
    """Check if the API is accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/upload-status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            logger.info(f"API health check passed: {status}")
            return True
        else:
            logger.error(f"API health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"API health check error: {e}")
        return False


def main():
    """Main function to run the upload process"""
    logger.info("Starting CSV upload process")

    # Check if results directory exists
    if not os.path.exists(RESULTS_DIR):
        logger.error(f"Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    # Check API health
    if not check_api_health():
        logger.error("API health check failed. Aborting upload.")
        sys.exit(1)

    # Find CSV files
    csv_files = find_csv_files(RESULTS_DIR)

    if not csv_files:
        logger.warning("No CSV files found to upload")
        return

    # Upload files
    success = upload_with_retry(csv_files)

    if success:
        logger.info("Upload process completed successfully")

        # Optional: Archive or cleanup uploaded files
        # archive_uploaded_files(csv_files)

    else:
        logger.error("Upload process failed")
        sys.exit(1)


def archive_uploaded_files(csv_files: List[str]):
    """Optional: Move uploaded files to archive directory"""
    archive_dir = os.path.join(RESULTS_DIR, "uploaded")
    os.makedirs(archive_dir, exist_ok=True)

    for csv_file in csv_files:
        archive_path = os.path.join(archive_dir, os.path.basename(csv_file))
        os.rename(csv_file, archive_path)
        logger.info(f"Archived {csv_file} to {archive_path}")


if __name__ == "__main__":
    main()
