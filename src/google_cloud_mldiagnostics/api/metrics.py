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

"""Module for recording metrics."""

from google_cloud_mldiagnostics.core import metrics
from google_cloud_mldiagnostics.custom_types import metric_types


_metrics_recorder = metrics.metrics_recorder


def record(
    metric_name: metric_types.MetricType or str,
    value: int | float,
    step: int | None = None,
    labels: dict[str, str] | None = None,
    record_on_all_hosts: bool = False,
) -> None:
  """Record a single metric value using the active run.

  Args:
      metric_name: Name of metric to record.
      value: Metric value.
      step: Optional step number (auto-incremented if not provided).
      labels: Optional additional labels.
      record_on_all_hosts: Whether to record metrics on all hosts.

  Raises:
      RecordingError: If no active run or recording fails.

  Example:
      metrics.record(MetricType.TF_FLOPS, per_device_tf_flops)
      metrics.record(MetricType.LEARNING_RATE, learning_rate)
      metrics.record(MetricType.LEARNING_RATE, learning_rate, step=1)
  """
  is_enum = isinstance(metric_name, metric_types.MetricType)
  metric_name = metric_name.value if is_enum else metric_name
  _metrics_recorder.record(
      metric_name, value, step, labels, record_on_all_hosts
  )
