# Copyright 2025 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper command to run Vllm with Diagnostics support.

Creates MLRun, run vllm, and finish MLRun.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
from typing import List

from google_cloud_mldiagnostics.api import mlrun
from google_cloud_mldiagnostics.custom_types import mlrun_types

logger = logging.getLogger(__name__)

_DESCRIPTION = """
This command creates MLRun and run vllm with enabled JAX profiling. All
parameters provided will be sent to the vllm command, except MLRun specific ones.
"""

parser = argparse.ArgumentParser(description=_DESCRIPTION)
parser.add_argument(
    "--mlrun_name",
    help="Diagnostic MLRun Name.",
    required=True,
    type=str,
)
parser.add_argument(
    "--mlrun_gcs_path",
    help="Diagnostic MLRun GCS Path.",
    required=True,
    type=str,
)
parser.add_argument("--project", help="Google Project ID.", type=str)
parser.add_argument(
    "--region", default="us-central1", help="Region.", type=str
)
parser.add_argument(
    "--jax_profiler_port",
    default=9999,
    help="JAX profiler server port.",
    type=int,
)



def main(args: List[str] | None):
  """Core main execution method for run_vllm wrapper."""
  diagon_args, rest_args = parser.parse_known_args(args)

  logging.basicConfig(level=logging.DEBUG)
  logger.info(">>> RUN_VLLM.PY VERSION: VERIFYING_BACKGROUND_THREAD_LOGGER_V1 <<<")

  # 1. Update the parent environment for the current process
  os.environ["FORCE_MASTER_HOST"] = "True"
  os.environ["MLRUN_SKIP_LIBTPU"] = "True"
  os.environ["MLRUN_FRAMEWORK"] = "vllm"

  logger.info("Creating mlrun with args: %s", diagon_args)
  run = mlrun.machinelearning_run(
      name=diagon_args.mlrun_name,
      project=diagon_args.project,
      region=diagon_args.region,
      gcs_path=diagon_args.mlrun_gcs_path,
      environment="prod",
      metrics_record_interval_sec=-1,
      serving_engine=mlrun_types.ServingEngine.VLLM,
  )

  # 2. Build a fresh process environment dictionary for the vLLM subprocess
  vllm_env = os.environ.copy()

  # Set JAX profiler variables
  vllm_env["USE_JAX_PROFILER_SERVER"] = "True"
  vllm_env["JAX_PROFILER_SERVER_PORT"] = str(diagon_args.jax_profiler_port)

  # 3. Construct deterministic paths under the dynamic MLRun ID
  base_phased_profiling_dir = os.environ.get("PHASED_PROFILING_DIR")
  if base_phased_profiling_dir:
    phased_prof_dir = (
        f"{base_phased_profiling_dir.rstrip('/')}/{run.name}/plugins/profile/"
    )
    logger.info("Setting PHASED_PROFILING_DIR to: %s", phased_prof_dir)
    vllm_env["PHASED_PROFILING_DIR"] = phased_prof_dir


  logger.info("Running vllm with args: %s", " ".join(rest_args))

  # 4. Handle SIGTERM gracefully and forward to child process
  vllm_proc = None

  def sigterm_handler(unused_signum, unused_frame):
    logger.info("Received SIGTERM, forwarding to child process...")
    try:
      if vllm_proc is not None:
        vllm_proc.terminate()
        try:
          vllm_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
          logger.warning("Timed out waiting for child process. Killing it...")
          vllm_proc.kill()
        logger.info("Child process exited with code %s", vllm_proc.returncode)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.exception("Error in sigterm_handler:")
    finally:
      exit_status = (
          vllm_proc.returncode if (
              vllm_proc is not None and vllm_proc.returncode is not None
          ) else 1
      )
      sys.exit(exit_status)

  signal.signal(signal.SIGTERM, sigterm_handler)

  # 5. Explicitly pass the env dictionary to Popen to bypass spawn issues
  vllm_proc = subprocess.Popen(
      ["vllm"] + rest_args,
      stdout=sys.stdout,
      stderr=sys.stderr,
      env=vllm_env
  )

  try:
    vllm_proc.wait()
  except KeyboardInterrupt:
    logger.info("Shutting down")
    if vllm_proc is not None:
      vllm_proc.terminate()
      try:
        vllm_proc.wait(timeout=15)
      except subprocess.TimeoutExpired:
        logger.warning(
            "Timed out waiting for child process to terminate on "
            "KeyboardInterrupt. Killing it..."
        )
        vllm_proc.kill()
        vllm_proc.wait()

  exit_code = vllm_proc.returncode if vllm_proc is not None else 1
  logger.info("vllm exit code %s", exit_code)
  return exit_code


if __name__ == "__main__":
  main(None)
