#!/bin/bash
# Fail on any error
set -e
set -o pipefail

echo "Starting Kokoro build script"

# 1. Run Blaze tests
echo "Running Blaze tests..."
cd "${KOKORO_PIPER_DIR}/google3"
blaze test //third_party/py/google_cloud_mldiagnostics/...

# 2. Run CopyBara to generate OSS source
echo "Running CopyBara..."
COPYBARA_SKY="${KOKORO_PIPER_DIR}/google3/third_party/py/google_cloud_mldiagnostics/copy.bara.sky"

OUTPUT_DIR=$(mktemp -d)
trap 'rm -rf "${OUTPUT_DIR}"' EXIT

if [ -z "$KOKORO_JOB_NAME" ]; then
  echo "Running locally, using system CopyBara"
  /google/bin/releases/copybara/public/copybara/copybara \
       "${COPYBARA_SKY}" \
       piper_to_local \
       --ignore-noop \
       --folder-dir "${OUTPUT_DIR}"
else
  echo "Running in Kokoro, using MPM CopyBara"
  COPYBARA_JAR="${KOKORO_GFILE_DIR}/copybara/copybara_on_kokoro_deploy.jar"
  COPYBARA_RUNFILES="${KOKORO_GFILE_DIR}/copybara/google3"
  
  java -Dcom.google.devtools.copybara.runfiles.path="${COPYBARA_RUNFILES}" \
       -jar "${COPYBARA_JAR}" \
       "${COPYBARA_SKY}" \
       piper_to_local \
       --ignore-noop \
       --folder-dir "${OUTPUT_DIR}"
fi

# 3. Build the package
echo "Building package..."
cd "${OUTPUT_DIR}"

python3 -m venv build-env
source build-env/bin/activate

pip install build
python3 -m build

# Copy artifacts to Kokoro artifacts directory
echo "Copying artifacts to ${KOKORO_ARTIFACTS_DIR}/"
cp -r dist "${KOKORO_ARTIFACTS_DIR}/"

echo "Build script completed successfully"
