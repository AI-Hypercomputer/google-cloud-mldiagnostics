#!/bin/bash
# Fail on any error
set -e
set -o pipefail

# Global configuration
PYTHON_VERSION="3.10.14"

# Setup Java environment.
setup_java() {
  echo "Using hermetic Java JDK from MPM..."
  export JAVA_HOME="${KOKORO_ARTIFACTS_DIR}/mpm/jdk"
  export PATH="$JAVA_HOME/bin:$PATH"
  java -version
}

# Run CopyBara to generate OSS source.
run_copybara() {
  local copybara_sky="$1"
  local output_dir="$2"

  local COPYBARA_JAR="${KOKORO_ARTIFACTS_DIR}/mpm/copybara/copybara_on_kokoro_deploy.jar"
  local COPYBARA_RUNFILES="${KOKORO_ARTIFACTS_DIR}/mpm/copybara/google3"

  echo "Running CopyBara for Kokoro via Java 21..."
  # Run CopyBara in a subshell to avoid changing the global working directory.
  (
    cd "${KOKORO_PIPER_DIR}"
    # Run CopyBara with:
    # - folder_to_local: to copy from Piper to local disk
    # - . : source directory in Piper
    # - --ignore-noop: to avoid failure if no changes
    # - --folder-dir: destination directory on local disk
    java -Dcom.google.devtools.copybara.runfiles.path="${COPYBARA_RUNFILES}" \
         -jar "${COPYBARA_JAR}" \
         --verbose \
         "${copybara_sky}" \
         folder_to_local \
         . \
         --ignore-noop \
         --folder-dir "${output_dir}"
  )
}

# Build the package and copy artifacts.
build_and_copy_artifacts() {
  local output_dir="$1"

  echo "Building package..."
  # Run the build in a subshell to avoid changing the global working directory.
  (
    cd "${output_dir}"

    # NOTE: We use pyenv to install Python because the active Docker image lacks
    # the required Python version or has issues with standard tools like ensurepip.
    # This compiles Python from source, which is slow, but necessary here.
    echo "Installing Python ${PYTHON_VERSION} using pyenv..."
    pyenv install -s "${PYTHON_VERSION}"
    echo "Setting pyenv global version to ${PYTHON_VERSION}"
    pyenv global "${PYTHON_VERSION}"

    echo "Current Python version:"
    python -V
    which python

    # Create and activate a virtual environment to isolate build dependencies.
    echo "Creating virtual environment using $(which python)..."
    python -m venv build-env
    source build-env/bin/activate
    echo "Virtual environment activated."

    # Install dependencies specified for the build process.
    echo "Installing build dependencies..."
    python -m pip install -r kokoro/requirements-build.txt

    # Run the build tool to generate distribution packages (wheel and tarball).
    # Pass --no-isolation to prevent it from fetching dependencies from PyPI.
    echo "Building the package..."
    python -m build --no-isolation

    # Copy artifacts to Kokoro artifacts directory
    # Verify that the build actually produced files before copying.
    echo "Copying artifacts to ${KOKORO_ARTIFACTS_DIR}/dist"
    mkdir -p "${KOKORO_ARTIFACTS_DIR}/dist"
    if [ -n "$(ls -A dist 2>/dev/null)" ]; then
      cp dist/* "${KOKORO_ARTIFACTS_DIR}/dist/"
    else
      echo "WARNING: No artifacts found in dist/ directory."
    fi
  )
}
