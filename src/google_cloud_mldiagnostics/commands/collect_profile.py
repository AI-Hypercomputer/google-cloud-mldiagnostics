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

"""Collects profiling information from remove hosts running Xprof.

This script connects to a profiler Xprof server started within a program and
collects a trace for a specified duration. The output trace files are saved to a
specified Google Storage path.
"""

from __future__ import annotations

import argparse
from typing import List


_DESCRIPTION = """
To profile running, you first need to start the profiler server in the program
of interest. You can do this via `jax.profiler.start_server(<port>)`. Once the
program is running and the profiler server has started, you can run
`collect_profile` to trace the execution for a provided duration. The trace 
file will be dumped into a directory (determined by `--log_dir`).
"""

DEFAULT_NUM_TRACING_ATTEMPTS = 3
parser = argparse.ArgumentParser(description=_DESCRIPTION)
parser.add_argument(
    "--hosts",
    required=True,
    help="Comma separated list of hosts where to collect trace.",
    type=str,
)
parser.add_argument(
    "--port", required=True, help="Port to collect trace", type=int
)
parser.add_argument(
    "--duration_in_ms",
    required=True,
    help="Duration to collect trace in milliseconds",
    type=int,
)
parser.add_argument(
    "--log_dir",
    required=True,
    help="Google Storage path to store log files.",
    type=str,
)
parser.add_argument(
    "--session_name",
    default="",
    help=(
        "Session name to use for xprof session timestamp for multi host"
        " profiling. Not effect for now. Will be implemented later."
    ),
    type=str,
)
parser.add_argument(
    "--host_tracer_level",
    default=2,
    help="Profiler host tracer level",
    type=int,
)
parser.add_argument(
    "--device_tracer_level",
    default=1,
    help="Profiler device tracer level",
    type=int,
)
parser.add_argument(
    "--python_tracer_level",
    default=1,
    help="Profiler Python tracer level",
    type=int,
)
parser.add_argument(
    "--override_hostnames",
    default="",
    help="Comma separated list of hostnames to use for trace filenames.",
    type=str,
)


def _import_xprof():
  """Import Xprof and raise error if dependency is not present.

  Dedicated method to allow testing.

  Returns:
    The `_pywrap_profiler_plugin` module from `xprof.convert`.
  """
  # pytype: disable=import-error
  try:
    from xprof.convert import _pywrap_profiler_plugin
  except ImportError as exc:
    raise ImportError("This script requires `xprof` to be installed.") from exc
  # pytype: enable=import-error
  return _pywrap_profiler_plugin


def _validate(hostname: str) -> str:
  """Validates and sanitizes a single hostname.

  Checks that the hostname is valid (non-empty, no spaces, no colons).
  Strips leading/trailing whitespace from hostname.

  Args:
    hostname: The hostname to validate.

  Returns:
    A validated and sanitized hostname.

  Raises:
    ValueError: If validation fails.
  """
  stripped_hostname = hostname.strip()
  if not stripped_hostname:
    raise ValueError(
        "Each hostname in override_hostnames must not be empty or contain"
        " only whitespace."
    )
  if " " in stripped_hostname:
    raise ValueError(
        f"Hostname '{stripped_hostname}' in override_hostnames cannot"
        " contain spaces."
    )
  if ":" in stripped_hostname:
    raise ValueError(
        f"Hostname '{stripped_hostname}' in override_hostnames cannot"
        " contain colons (:)."
    )
  return stripped_hostname


def _validate_hostnames(
    hosts: str, override_hostnames: str
) -> str:
  """Validates and sanitizes override_hostnames.

  Checks that number of hostnames matches number of hosts, and that each
  hostname is valid (non-empty, no spaces, no colons). Strips leading/trailing
  whitespace from each hostname.

  Args:
    hosts: Comma-separated list of hostnames or IPs.
    override_hostnames: Comma-separated list of hostnames to use for trace
      filenames.

  Returns:
    A comma-separated string of validated and sanitized hostnames.

  Raises:
    ValueError: If validation fails.
  """
  hostnames = override_hostnames.split(",")
  if len(hosts.split(",")) != len(hostnames):
    raise ValueError(
        "Number of hosts and override_hostnames must be equal if"
        " override_hostnames is provided."
    )

  return ",".join(_validate(hostname) for hostname in hostnames)


def _collect_profile(
    *,
    hosts: str,
    port: int,
    duration_in_ms: int,
    log_dir: str,
    session_name: str,
    host_tracer_level: int,
    device_tracer_level: int,
    python_tracer_level: int,
    override_hostnames: str,
):
  """Collects a profile from the specified hosts and ports.

  Args:
    hosts: Comma-separated list of hostnames or IPs.
    port: The port number of the profiler server on each host.
    duration_in_ms: The duration of the trace collection in milliseconds.
    log_dir: The directory where the profiling logs will be saved.
    session_name: The session name for xprof session. Not effect for now. Will
      be implemented later.
    host_tracer_level: The level of host tracing.
    device_tracer_level: The level of device tracing.
    python_tracer_level: The level of Python tracing.
    override_hostnames: Comma-separated list of hostnames to use for trace
      filenames.
  """
  options = {
      "host_tracer_level": host_tracer_level,
      "device_tracer_level": device_tracer_level,
      "python_tracer_level": python_tracer_level,
  }

  # This script will be used from GKE so we print to guarntee that output
  # will have this messages without additional configuration.
  print(f"Starting remote profile for {hosts} on {port}...")

  if session_name:
    options["session_id"] = session_name
  else:
    print("Session name not provided, xprof will use auto generated")

  if override_hostnames:
    options["override_hostnames"] = _validate_hostnames(
        hosts, override_hostnames
    )

  xprof = _import_xprof()
  xprof.trace(
      _to_hosts_port(hosts, port),
      log_dir,
      "",
      True,
      duration_in_ms,
      DEFAULT_NUM_TRACING_ATTEMPTS,
      options,
  )
  print(f"Dumped profiling information in: {log_dir}")


def _to_hosts_port(hosts: str, port: int):
  hosts_port = []
  for host in hosts.split(","):
    hosts_port.append(f"{host}:{port}")
  return ",".join(hosts_port)


def main(args: List[str] | None):
  parsed_args = parser.parse_args(args)
  _collect_profile(
      hosts=parsed_args.hosts,
      port=parsed_args.port,
      duration_in_ms=parsed_args.duration_in_ms,
      log_dir=parsed_args.log_dir,
      session_name=parsed_args.session_name,
      host_tracer_level=parsed_args.host_tracer_level,
      device_tracer_level=parsed_args.device_tracer_level,
      python_tracer_level=parsed_args.python_tracer_level,
      override_hostnames=parsed_args.override_hostnames,
  )


if __name__ == "__main__":
  main(None)
