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
<!--
 Copyright 2023-2025 Google LLC

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

# google-cloud-mldiagnostics release notes

## PyPI Package

google-cloud-mldiagnostics is [available in PyPI](https://pypi.org/project/google-cloud-mldiagnostics/) and can be installed through pip. Please see our [Installation Guide](https://pypi.org/project/google-cloud-mldiagnostics/#user-content-install-ml-diagnostics-sdk) for setup instructions.

## Releases

### v1.0.9

#### Changes

- **Application Framework & RL Orchestrator Labels**:
  - Added optional `application_framework` and `rl_orchestrator` parameters to `machinelearning_run` to attach workload metadata directly to your ML run.
  - When specified, non-default values are automatically recorded as labels on the ML run for tracking and filtering.

- **Parameters & Supported Values:**
  - `application_framework`: Supported values are `NONE` (default), `MAXTEXT`, `AXLEARN`, and `MAXDIFFUSION`.
  - `rl_orchestrator`: Supported values are `NONE` (default) and `TUNIX`.

**Example Usage:**

```bash
from google_cloud_mldiagnostics import machinelearning_run
from google_cloud_mldiagnostics.custom_types import mlrun_types


run = machinelearning_run(
    name="<run_name>",
    run_group="<run_group>",
    configs={ "epochs": 100, "batch_size": 32 },
    project="<some_project>",
    region="<some_zone>",
    gcs_path="gs://<some_bucket>",
    on_demand_xprof=True,
    application_framework="MAXTEXT",
    rl_orchestrator="TUNIX",
)
```

### v1.0.8

#### Changes

- **OpenTelemetry Integration**:

  - Added support for configurable Otel exporters in machinelearning_run, allowing users to export SDK metrics and logs to OpenTelemetry Collector and Google Cloud Logging. see the [otel integration](https://pypi.org/project/google-cloud-mldiagnostics/#user-content-configure-telemetry-exporters-for-otel-integration) for details.

- **vLLM Diagnostics Wrapper**:

  - Introduced a wrapper command to run vLLM with ML Diagnostics support, enabling seamless profiling and automatic MLRun lifecycle management. see the [ml-diagnostics-with-vllm](https://pypi.org/project/google-cloud-mldiagnostics/#user-content-using-ml-diagnostics-with-vllm) for details.