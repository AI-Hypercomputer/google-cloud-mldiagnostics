<!--
 Copyright 2025 Google LLC
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
      https://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->
# google-cloud-mldiagnostics

## Overview

The `google-cloud-mldiagnostics` library is a Python package designed to help
engineers and researchers monitor and diagnose machine learning training runs
with GCP suite of diagnostic toolings.
It provides tools for tracking workload progress, collecting metrics and
profiling performance.

### Supported Framework

- [jax](https://github.com/jax-ml/jax)
  - any versions
- Other in progress

## How to install

### Install

Install pypi package [link](https://pypi.org/project/google-cloud-mldiagnostics/)

```bash
pip install google-cloud-mldiagnostics
```

This package does not install `libtpu`, `jax` and `xprof` and expects they will
be installed separately.

## How to use

### Monitor training

At the beginning of the training script create a machine learning run:

```python
from google_cloud_mldiagnostics import machinelearning_run

machinelearning_run(
  name=<run-name>,
  gcs_path="gs://<bucket>",
)
```

### Monitor with on-demand profiling

```python
from google_cloud_mldiagnostics import machinelearning_run

machinelearning_run(
  name=<run-name>
  gcs_path="gs://<bucket>",
  on_demand_xprof=True
)
```

### Monitor with programmatic profiling

```python
from google_cloud_mldiagnostics import machinelearning_run
from google_cloud_mldiagnostics import xprof

machinelearning_run(
  name=<run-name>
  gcs_path="gs://<bucket>",
)

xprof=xprof()
xprof.start()
# some code
xprof.stop()
```

### Monitor with predefined metrics

```python
from google_cloud_mldiagnostics import machinelearning_run
from google_cloud_mldiagnostics import metrics
from google_cloud_mldiagnostics import metric_types

machinelearning_run(
  gcs_path="gs://<bucket>",
)

metrics.record(metric_type.MetricType.LOSS, <value>)
```

To pair metric value with current step:

```python
metrics.record(metric_type.MetricType.LOSS, <value>, step=<step>)
```

### Monitor with customer metrics

```python
from google_cloud_mldiagnostics import machinelearning_run
from google_cloud_mldiagnostics import metrics

machinelearning_run(
  gcs_path="gs://<bucket>",
)

metrics.record("<my-metric>", <value>)
```

To pair metric value with current step:

```python
metrics.record("<my-metric>", <value>, step=1)
```