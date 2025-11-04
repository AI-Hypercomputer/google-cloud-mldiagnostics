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

"""Metric type definitions."""

import enum


class MetricType(enum.Enum):
  """Predefined metric types for ML training."""

  # Model quality metrics
  LEARNING_RATE = "learning_rate"
  LOSS = "loss"
  GRADIENT_NORM = "gradient_norm"
  TOTAL_WEIGHTS = "total_weights"

  # Model performance metrics
  STEP_TIME = "step_time"
  THROUGHPUT = "throughput"
  LATENCY = "latency"
  MFU = "mfu"  # Model FLOPs Utilization
  TFLOPS = "tflops"

  # System utilization metrics
  TPU_DUTY_CYCLE = "tpu_duty_cycle"
  TPU_TENSORCORE_UTILIZATION = "tpu_tensorcore_utilization"
  HBM_UTILIZATION = "hbm_utilization"
  HOST_CPU_UTILIZATION = "host_cpu_utilization"
  HOST_MEMORY_UTILIZATION = "host_memory_utilization"

  # Step metrics, system will recocord step metric automatically when invoking
  # other metrics with step information. However, if there is a need to record
  # step separate, use this metric type.
  STEP = "step"
