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

"""Custom exception classes."""


class MLDiagnosticError(Exception):
  """Base exception for ML Diagnostic SDK."""
  pass


class MLRunConfigurationError(ValueError):
  """Exception raised for ML run configuration errors."""
  pass


class RecordingError(ValueError):
  """Exception raised for recording metrics."""

  pass


class NoActiveRunError(Exception):
  """Raised when no active ML run is found."""

  pass


class ProfilingError(Exception):
  """Raised when profiling fails."""

  pass
