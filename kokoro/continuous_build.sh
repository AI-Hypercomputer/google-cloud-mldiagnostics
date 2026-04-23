#!/bin/bash
# Fail on any error
set -e
set -o pipefail

echo "Starting Kokoro continuous build script"

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

echo "Continuous build script completed successfully"
