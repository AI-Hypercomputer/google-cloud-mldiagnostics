#!/bin/bash
# Fail on any error
set -e
set -o pipefail

echo "Starting Kokoro release TEST build script"

# Source the common build logic.
source "${KOKORO_PIPER_DIR}/google3/third_party/py/google_cloud_mldiagnostics/kokoro/common_build.sh"

# Define variables specific to this build.
COPYBARA_SKY="${KOKORO_PIPER_DIR}/google3/third_party/py/google_cloud_mldiagnostics/copy.bara.sky"
OUTPUT_DIR=$(mktemp -d)
trap 'rm -rf "${OUTPUT_DIR}"' EXIT

# 1. Setup Java environment.
setup_java

# 2. Run CopyBara to generate OSS source.
run_copybara "${COPYBARA_SKY}" "${OUTPUT_DIR}"

# 3. Build the package and copy artifacts.
build_and_copy_artifacts "${OUTPUT_DIR}"

# TODO: Add automated tests here (e.g., smoke tests) before publishing.

echo "Installing twine and Artifact Registry authentication helpers..."
python -m pip install --default-timeout=100 --retries 5 -U keyring keyrings.google-artifactregistry-auth twine

echo "Publishing to OSS Exit Gate Staging Artifact Registry..."
# Upload the built wheels to the internal staging repository.
# The OSS Exit Gate will automatically pick them up, verify BCID compliance, and release to PyPI.
python -m twine upload \
  --repository-url https://us-python.pkg.dev/oss-exit-gate-prod/google-cloud-mldiagnostics--pypi/ \
  "${KOKORO_ARTIFACTS_DIR}/dist/*"

echo "Successfully uploaded to OSS Exit Gate staging registry!"
echo "You can monitor the release status via the MOSS Dashboard: go/moss-rdp-dash"

echo "Release TEST build script completed successfully"
