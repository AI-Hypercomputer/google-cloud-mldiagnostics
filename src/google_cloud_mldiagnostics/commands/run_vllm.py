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
import json
import logging
import os
import signal
import subprocess
import sys
from typing import List

from google_cloud_mldiagnostics.api import mlrun
from google_cloud_mldiagnostics.custom_types import mlrun_types
from google_cloud_mldiagnostics.utils import run_phase_utils

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
parser.add_argument(
    "--run_group",
    default="",
    help="Diagnostic MLRun Run Group.",
    type=str,
)
parser.add_argument(
    "--configs",
    default=None,
    help="User-defined configurations as a JSON string.",
    type=str,
)
parser.add_argument(
    "--framework",
    default=None,
    choices=["pytorch", "jax", "PYTORCH", "JAX"],
    help="Underlying ML framework for the workload (default: pytorch).",
    type=str,
)
parser.add_argument(
    "--phased_profiling_dir",
    default=None,
    help=(
        "Base GCS destination directory for JAX phased profiling traces. The"
        " MLRun-specific suffix '<run_name>/plugins/profile/' is appended"
        " automatically and exported as PHASED_PROFILING_DIR to vllm. Use"
        " this flag instead of setting PHASED_PROFILING_DIR in the"
        " environment; an environment value is still forwarded, but in"
        " multi-host (Ray) setups it does not propagate the run-scoped path"
        " to remote workers and phased profiles may not show in the XProf UI."
    ),
    type=str,
)



def _update_torch_profiler_args(args: list[str], run_name: str) -> list[str]:
  """Appends dynamic MLRun ID subdirectory to profiler target paths."""
  def append_suffix(path: str) -> str:
    return f"{path.rstrip('/')}/{run_name}/plugins/profile/"

  # If '--' is first, keep it aside to prepend later.
  has_leading_dashdash = False
  if args and args[0] == "--":
    has_leading_dashdash = True
    parse_args = args[1:]
  else:
    parse_args = args

  # If there is another '--' in parse_args, separate them.
  if "--" in parse_args:
    idx = parse_args.index("--")
    real_parse_args = parse_args[:idx]
    post_args = parse_args[idx:]
  else:
    real_parse_args = parse_args
    post_args = []

  # Create a helper parser to parse only the profiler arguments.
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--profiler-config", dest="profiler_config", type=str)
  parser.add_argument(
      "--profiler-config.torch_profiler_dir",
      dest="torch_profiler_dir",
      type=str,
  )

  parsed, remaining = parser.parse_known_args(real_parse_args)
  new_args = list(remaining)

  if parsed.torch_profiler_dir is not None:
    updated_dir = append_suffix(parsed.torch_profiler_dir)
    new_args.append(f"--profiler-config.torch_profiler_dir={updated_dir}")

  if parsed.profiler_config is not None:
    try:
      config_dict = json.loads(parsed.profiler_config)
      if isinstance(config_dict, dict) and "torch_profiler_dir" in config_dict:
        config_dict["torch_profiler_dir"] = append_suffix(
            config_dict["torch_profiler_dir"]
        )
        new_args.append(f"--profiler-config={json.dumps(config_dict)}")
      else:
        new_args.append(f"--profiler-config={parsed.profiler_config}")
    except json.JSONDecodeError:
      new_args.append(f"--profiler-config={parsed.profiler_config}")

  result = new_args + post_args
  if has_leading_dashdash:
    result.insert(0, "--")
  return result


def main(args: List[str] | None):
  """Core main execution method for run_vllm wrapper."""
  diagon_args, rest_args = parser.parse_known_args(args)

  logging.basicConfig(level=logging.DEBUG)
  logger.info(
      ">>> RUN_VLLM.PY VERSION: VERIFYING_BACKGROUND_THREAD_LOGGER_V1 <<<"
  )

  # 1. Update the parent environment for the current process
  os.environ["FORCE_MASTER_HOST"] = "True"
  os.environ["MLRUN_SKIP_LIBTPU"] = "True"

  framework_str = (
      diagon_args.framework
      or os.environ.get("MLRUN_FRAMEWORK")
      or "pytorch"
  ).upper()
  try:
    framework_enum = mlrun_types.Framework[framework_str]
  except KeyError:
    logger.warning(
        "Unknown framework '%s', defaulting to PYTORCH.", framework_str
    )
    framework_enum = mlrun_types.Framework.PYTORCH

  user_configs = None
  if diagon_args.configs:
    try:
      parsed_configs = json.loads(diagon_args.configs)
      if isinstance(parsed_configs, dict):
        user_configs = parsed_configs
      else:
        logger.warning(
            "Expected --configs to be a JSON object (dict), got %s.",
            type(parsed_configs).__name__,
        )
    except json.JSONDecodeError:
      logger.exception(
          "Failed to parse --configs as JSON: %s.", diagon_args.configs
      )

  logger.info("Creating mlrun with args: %s", diagon_args)
  run = mlrun.machinelearning_run(
      name=diagon_args.mlrun_name,
      run_group=diagon_args.run_group,
      configs=user_configs,
      project=diagon_args.project,
      region=diagon_args.region,
      gcs_path=diagon_args.mlrun_gcs_path,
      environment="prod",
      metrics_record_interval_sec=-1,
      serving_engine=mlrun_types.ServingEngine.VLLM,
      framework=framework_enum,
  )

  # 2. Build a fresh process environment dictionary for the vLLM subprocess
  vllm_env = os.environ.copy()

  # Set JAX profiler variables
  vllm_env["USE_JAX_PROFILER_SERVER"] = "True"
  vllm_env["JAX_PROFILER_SERVER_PORT"] = str(diagon_args.jax_profiler_port)

  # 3. Construct deterministic paths under the dynamic MLRun ID
  # --phased_profiling_dir is the supported way to enable phased profiling.
  # A PHASED_PROFILING_DIR found in the OS environment is still forwarded to
  # vllm serve for backward compatibility, but in multi-host (Ray) setups the
  # run-scoped value will not propagate to remote workers: Ray applies driver
  # env vars on workers with os.environ.setdefault(), so a pre-existing value in
  # the worker node environment shadows the path computed here.
  env_phased_profiling_dir = os.environ.get("PHASED_PROFILING_DIR")
  base_phased_profiling_dir = diagon_args.phased_profiling_dir
  if base_phased_profiling_dir:
    if env_phased_profiling_dir:
      logger.warning(
          "PHASED_PROFILING_DIR is set in the OS environment (%s) and"
          " --phased_profiling_dir was also passed; using the flag value. For"
          " multi-host setups, remove PHASED_PROFILING_DIR from the worker"
          " node environment, otherwise workers will ignore the run-scoped"
          " path and their phased profiles may not show in the XProf UI.",
          env_phased_profiling_dir,
      )
  elif env_phased_profiling_dir:
    logger.warning(
        "PHASED_PROFILING_DIR detected in the OS environment (%s); forwarding"
        " it to vllm serve. For multi-host setups, phased profiles may not"
        " show in the XProf UI because the run-scoped path will not"
        " propagate to remote Ray workers. Pass --phased_profiling_dir"
        " instead and do not set PHASED_PROFILING_DIR in the environment.",
        env_phased_profiling_dir,
    )
    base_phased_profiling_dir = env_phased_profiling_dir

  if base_phased_profiling_dir:
    phased_prof_dir = (
        f"{base_phased_profiling_dir.rstrip('/')}/{run.name}/plugins/profile/"
    )
    logger.info("Setting PHASED_PROFILING_DIR to: %s", phased_prof_dir)
    vllm_env["PHASED_PROFILING_DIR"] = phased_prof_dir

  base_torch_profiler_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR")
  if base_torch_profiler_dir:
    torch_prof_dir = (
        f"{base_torch_profiler_dir.rstrip('/')}/{run.name}/plugins/profile/"
    )
    logger.info("Setting VLLM_TORCH_PROFILER_DIR to: %s", torch_prof_dir)
    vllm_env["VLLM_TORCH_PROFILER_DIR"] = torch_prof_dir

  rest_args = _update_torch_profiler_args(rest_args, run.name)


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
    except OSError:
      logger.exception("Error in sigterm_handler:")
    finally:
      monitor = run_phase_utils.RunPhaseMonitor.get_active_monitor()
      if monitor:
        monitor.update_ml_run_with_phase(mlrun_types.RunPhase.PHASE_FAILED)
        monitor.exit_cleanup()
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
  if exit_code != 0:
    monitor = run_phase_utils.RunPhaseMonitor.get_active_monitor()
    if monitor:
      monitor.update_ml_run_with_phase(mlrun_types.RunPhase.PHASE_FAILED)
  return exit_code


if __name__ == "__main__":
  sys.exit(main(None))
