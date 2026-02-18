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

- [Overview](#overview)
  - [Github Repo](#github-repo)
  - [Machine Learning Run Intro](#machine-learning-run-intro)
- [Setup](#setup)
  - [Enable API](#enable-api)
  - [IAM Permissions](#iam-permissions)
  - [Google Storage Bucket](#google-storage-bucket)
  - [Configure GKE Cluster](#configure-gke-cluster)
  - [Install SDK](#install-sdk)
- [How to use](#how-to-use)
  - [Creating a machine learning run](#creating-a-machine-learning-run)
  - [Write configs using yaml or json](#write-configs-using-yaml-or-json)
  - [Collect metrics](#collect-metrics)
  - [Programmatic Profile Capture](#programmatic-profile-capture)
  - [Multi-host (process) profiling](#multi-host-process-profiling)
  - [Enable On-Demand Profile Capture](#enable-on-demand-profile-capture)

## Overview

Google Cloud ML Diagnostics is an end-to-end managed platform for ML Engineers
to optimize and diagnose their AI/ML workloads on Google Cloud. The product
allows ML Engineers to collect and visualize all their workload metrics, configs
and profiles with one single platform, all within the same UI. This platform
works for any ML workload (training, inference, etc) including working with
Maxtext/Maxdiffusion as well as any orchestrator on TPU including GKE as well as
custom orchestrator. The current product offering focuses on workloads running
on XLA-based frameworks (JAX, Pytorch XLA, Tensorflow/Keras) on Google Cloud
TPUs and GPUs. Current support is for JAX on TPU only.

Google Cloud ML Diagnostics includes the following features:

- **SDK**: An open source ML Diagnostics SDK to use with your ML workload in order to enable managed ML workload diagnostics experience on Google Cloud
- Integration with JAX and Pytorch framework and libraries (only JAX supported for Preview)
- **Metrics/configs/profiles management**:
  - Track workload metrics, including model quality, model performance and system metrics.
  - Track workload configs including software configs, system configs as well as user-defined configs
  - Manage profile sessions
- **Managed XProf**: Managed profiling with XProf, which allows faster loading
of large profiles, supports multiple users simultaneously accessing profiles and
supports easy to use out-of-the-box features such as multi-host profiling and
on-demand profiling.
- Visualization of metrics/configs/profiles in both Cluster Director and Google **Kubernetes Engine** on the Google Cloud console
- Link sharing for ML runs and profiles for easy collaboration

### Github Repo

This repo contains the following components for the Google Cloud MLDiagnostics
platform:

1. **google-cloud-mldiagnostics** SDK: a Python package designed for ML
Engineers to integrate with their ML workload to help track metrics and diagnose
performance of their machine learning runs. It provides functions for tracking
workload configs, collecting metrics and profiling performance.
1. mldiagnostics-injection-webhook: A Helm chart to inject metadata into JobSet, RayJob, and LWS pods, which is needed by the MLDiagnostics SDK.
1. mldiagnostics-connection-operator: A Helm chart to capture profiler traces based on the MLDiagnosticsConnection Custom Resource in frameworks like JAX.
1. MLDiagnosticsConnection CRD: The Custom Resource Definition for `mldiagnosticsconnections.diagon.gke.io`.

### Machine Learning Run Intro

ML Diagnostics represents each ML workload run by Machine Learning Run
(a.k.a MLRun). Metrics and configs collected by ML Diagnostics will be attached
to the MLRun.

MLRun can have zero or more profiling sessions. Each session represents a single
start and stop of XProf. Users can trigger XProf programmatically from their
workload code as well as on demand from the UI. Each XProf session will be
attached to the MLRun.

## Setup

### Enable API

Enable Cluster Director API https://docs.cloud.google.com/endpoints/docs/openapi/enable-api

### IAM Permissions

The Google Service Account used by your workload requires the following IAM
roles assigned on the project:

1. `roles/clusterdirector.editor`: For full access to create and manage MLRun resources and view the UI.
1. `roles/logging.logWriter`: To write logs and metrics to Google Cloud Logging.
1. `roles/storage.objectUser`: To save profiles to the GCS bucket specified in `machinelearning_run`.

For read-only access (viewing UI only, not creating MLRuns),
`roles/clusterdirector.viewer` is sufficient.

**GKE**: If your workload runs on GKE, we recommend using [Workload Identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) to associate a Kubernetes Service Account with a Google Service Account that has been granted the roles above.

### Google Storage Bucket

A Google Cloud Storage bucket to store profile data.

### Configure GKE Cluster

If GKE will be used for the ML workload, user needs to install the following to
their GKE cluster. Please ensure the GKE cluster is configured as a regional
cluster with Workload Identity enabled.

#### GKE: Install injection-webhook in the cluster

For workloads running in GKE, injection-webhook is needed to provide SDK needed
metadata. It supports these common ML kubernetes workloads:
JobSet/RayJob/LeaderWorkerSet.

#### Install cert-manager if not already installed
Cert-manager is a prerequisite for the injection-webhook. If it’s not installed, follow this to install. After installing cert-manager, it may take up to two minutes for the certificate to become ready.

#### Install cert-manager

Install helm for Debian/Ubuntu. For other distributions,
follow https://helm.sh/docs/intro/install/ to install.

```bash
sudo apt-get install curl gpg apt-transport-https --yes

curl -fsSL https://packages.buildkite.com/helm-linux/helm-debian/gpgkey | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/helm.gpg] https://packages.buildkite.com/helm-linux/helm-debian/any/ any main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list

sudo apt-get update
sudo apt-get install helm
```

Then install cert-manager

```bash
helm repo add jetstack https://charts.jetstack.io
helm repo update

helm install \
  cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.13.0 \
  --set installCRDs=true \
  --set global.leaderElection.namespace=cert-manager \
  --timeout 10m
```

Or you can use kubectl

```bash
kubectl create namespace cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml -n cert-manager

kubectl delete -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml -n cert-manager
```

#### Install injection-webhook

```bash
helm install mldiagnostics-injection-webhook --namespace=gke-mldiagnostics --create-namespace oci://us-docker.pkg.dev/ai-on-gke/mldiagnostics-webhook-and-operator-helm/mldiagnostics-injection-webhook
## Uninstall. First, uninstall MutatingWebhookConfiguration, then uninstall helm charts.
# kubectl delete MutatingWebhookConfiguration mldiagnostics-injection-webhook-mutating-webhook-config
# helm uninstall mldiagnostics-injection-webhook -n gke-mldiagnostics
```

The above command can be edited with `-f` or `--set` flags to pass in a custom
values file or key-value pair respectively for the chart.

Or you can use gcloud and kubectl

```bash
gcloud artifacts generic download --repository=mldiagnostics-webhook-and-operator-yaml --location=us --package=mldiagnostics-injection-webhook --version=v0.5.0 --destination=./ --project=ai-on-gke
kubectl create namespace gke-mldiagnostics
# it needs to be installed inside namespace gke-mldiagnostics. If not, need to change mldiagnostics-injection-webhook-v0.5.0.yaml
kubectl apply -f mldiagnostics-injection-webhook-v0.5.0.yaml -n gke-mldiagnostics

## Uninstall. First, uninstall MutatingWebhookConfiguration, then delete yaml.
# kubectl delete MutatingWebhookConfiguration mldiagnostics-injection-webhook-mutating-webhook-config
# kubectl delete -f  mldiagnostics-injection-webhook-v0.5.0.yaml -n gke-mldiagnostics
```

#### Label workload

To trigger the injection-webhook to inject metadata into pods, you need to label
either the workload itself or its namespace with
`managed-mldiagnostics-gke=true` before deploying the workload. You have two
options:

1.  **Label a namespace:** This will enable the webhook for all Jobset/LWS/RayJob workloads within that namespace.

    ```bash
    kubectl create namespace ai-workloads
    kubectl label namespace ai-workloads managed-mldiagnostics-gke=true
    ```

2.  **Label a Jobset/LWS/RayJob workload:** This will enable the webhook only for the specific workload.

    ```yaml
    # Example for JobSet
    apiVersion: jobset.x-k8s.io/v1alpha2
    kind: JobSet
    metadata:
      name: single-host-tpu-v3-jobset2
      namespace: default
      labels:
        managed-mldiagnostics-gke: "true"
    ```

#### GKE: Install connection-operator in the cluster

For seamless on-demand profiling on GKE, we recommend deploying GKE connection
operator along with injection webhook into the GKE cluster. This will ensure
that your machine learning run knows which GKE nodes it is running on and so
on-demand capture drop down can auto populate these nodes automatically.

##### Compatibility Matrix

| JAX Version | `mldiagnostics-connection-operator` Helm Chart Version |
| :--- | :--- |
| 0.8.2 | 0.14.0 |
| 0.8.1 / 0.8.3 | 0.15.0 |

```bash
helm install mldiagnostics-connection-operator \
   --namespace=gke-mldiagnostics \
   --create-namespace \
   --version <version> \
oci://us-docker.pkg.dev/ai-on-gke/mldiagnostics-webhook-and-operator-helm/mldiagnostics-connection-operator

## use this to uninstall
# helm uninstall mldiagnostics-connection-operator -n gke-mldiagnostics
```

The above command can be edited with `-f` or `--set` flags to pass in a custom
values file or key-value pair respectively for the chart. Or you can use gcloud
and kubectl

```bash
gcloud artifacts generic download --repository=mldiagnostics-webhook-and-operator-yaml --location=us --package=mldiagnostics-connection-operator --version=v0.5.0 --destination=./ --project=ai-on-gke
kubectl create namespace gke-mldiagnostics
kubectl apply -f mldiagnostics-connection-operator-v0.5.0.yaml -n gke-mldiagnostics

## use this to uninstall
# kubectl delete -f mldiagnostics-connection-operator-v0.5.0.yaml -n gke-mldiagnostics
```

### Install SDK

Pip install [SDK](https://pypi.org/project/google-cloud-mldiagnostics/)

```bash
pip install google-cloud-mldiagnostics
```

This package does not install `libtpu`, `jax`, and `xprof`; you are expected to
install these separately if needed for your workload.

## How to use

### Enable Cloud Logging

The SDK uses Python's standard `logging` module to output information. To route
these logs to Google Cloud Logging, you need to install and configure the
`google-cloud-logging` library. This allows you to view SDK logs, metrics
written as logs, and your own application logs in the Google Cloud console.

1.  **Install the library:**

    ```bash
    pip install google-cloud-logging
    ```

2.  **Configure logging in your script:**
    Add the following lines to the beginning of your Python script to attach
    the Cloud Logging handler to the Python root logger:

    ```python
    import logging
    import google.cloud.logging

    # Instantiate a Cloud Logging client
    logging_client = google.cloud.logging.Client()

    # Attaches the Cloud Logging handler to the Python root logger
    logging_client.setup_logging()

    # Now, standard logging calls will go to Cloud Logging
    logging.info("SDK logs and application logs will appear in Cloud Logging.")
    ```

### Enable Debug Logging
By default, the logging level is set to `INFO`. To see more detailed logs from
the SDK, such as MLRun details, you can set the logging level to `DEBUG` *after*
calling `setup_logging()`:

```python
import logging
import google.cloud.logging

logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.getLogger().setLevel(logging.DEBUG) # Enable DEBUG level logs

logging.debug("This is a debug message.")
logging.info("This is an info message.")
```
With `DEBUG` level enabled, you will see additional SDK diagnostics in Cloud
Logging, for example:
```
DEBUG:google_cloud_mldiagnostics.core.global_manager:current run details: {'name': 'projects/my-gcp-project/locations/us-central1/mlRuns/my-run-12345', 'gcs_path': 'gs://my-bucket/profiles', ...}
```

### Creating a machine learning run

In order to use Google Cloud ML Diagnostics platform, you will need to create a
machine learning run. This requires instrumenting your ML workload with the SDK
to perform logging, metric collection, and profile tracing.

Below is a basic example of how to initialize Cloud Logging, create an MLRun,
record metrics, and capture a profile:

```python
import logging
import os
import google.cloud.logging
from google_cloud_mldiagnostics import machinelearning_run, metrics, xprof, metric_types

# 1. Set up Cloud Logging
# Make sure to pip install google-cloud-logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
# Optional: Set logging level to DEBUG for more detailed SDK logs
# logging.getLogger().setLevel(logging.DEBUG)

# 2. Define and start machinelearning run
try:
    run = machinelearning_run(
          name="<run_name>",
          run_group="<run_group>",
          configs={ "epochs": 100, "batch_size": 32 },
          project="<some_project>",
          region="<some_zone>",
          gcs_path="gs://<some_bucket>",
          on_demand_xprof=True,
        )
    logging.info(f"MLRun created: {run.name}")

    # 3. Collect metrics during your run
    metrics.record(metric_types.MetricType.LOSS, 0.123, step=1)
    logging.info("Loss metric recorded.")

    # 4. Capture profiles programmatically
    with xprof():
        # ... your code to profile here ...
        pass
    logging.info("Profile captured.")

except Exception as e:
    logging.error(f"Error during MLRun: {e}", exc_info=True)

```

`name` Required. A unique identifier for this specific run. SDK will
automatically add a timestamp at the end of the name for GKE to make each run
unique every time it is run. For GCE, the user needs to ensure this name is
unique every time they run.

`run_group` Optional. An identifier that can help group multiple runs that
belong to the same experiment/ml objective. Example: all runs associated with a
tpu slice size sweep can be labeled with `run_group=”tpuslicesizesweep”`

`project` Optional. If not specified, the project will be extracted from gcloud
CLI.

`region` Optional, automatically assigned to `us-central1`. Currently only
`us-central1` is available.

`configs` Optional. Key-value pairs containing configuration parameters for the
run. Note: If configs are not defined, default software and system configs will
show up in UI but none of the user configs for ML workload will be seen in UI.
For configs, there are some configs that are automatically collected by SDK and
the user does not need to write them:

1. Software configs - framework, framework version, XLA flags
1. System configs - device type, # slices, slice size, # hosts

The project and region information are where the machine learning run metadata
information will be stored. Note that the region used for machinelearning run
does not have to be the same as the region used for your actual workload run,
example: you can run your workload in `europe_west4-a` but have your
machinelearning run information stored in `us-central1-a`.

`gcs_path` Required only if SDK will be used for profile capture. The Google
Cloud Storage location where all profiles will be saved. Example
`gs://my-bucket`. Could include folder path if needed like
`gs://my-bucket/folder1`. If capturing profile programmatically or on-demand,
this is required or else profile capture will error.

`on-demand-xprof` Optional, if you want to enable on demand profiling, starts
xprofz daemon on port `9999`. Note that you can enable on-demand profiling and
also do programmatic profiling in the same code, but user needs to make sure
that the on-demand capture time does not happen at the same time as the
programmatic profile capture.

`environment` Optional, defaults to `prod`. Used to specify the environment
where the run metadata is stored.

To test on a different environment, you can specify it using an environment
variable:

```python
machinelearning_run(
    # ...
    environment=os.environ.get("ENV", "env")
)
```

### Write configs using yaml or json

For many workloads, there are too many configs to define directly in your
machinelearning run definition. Instead, you can write configs to your
machinelearning run using json or yaml.

```python
import yaml
import json

# Read the YAML file
with open('config.yaml', 'r') as yaml_file:
  # Parse YAML into a Python dictionary
  yaml_data = yaml.safe_load(yaml_file)

# Define machinelearning run
machinelearning_run(
  name="<run_name>",
  run_group="<run_group>",
  configs=yaml_data,
  project="<some_project>",
  region="<some_zone>",
  gcs_path="gs://<some_bucket>",
)
```

### Collect metrics

The SDK allows users to collect model metrics, model perf metrics and system
metrics and visualize these as both average values as well as time series
charts.

The SDK provides two functions for recording metrics: `metrics.record()` for
capturing individual data points, and `metrics.record_metrics()` for recording
multiple metrics in a single batch. Both functions write metrics to Cloud
Logging, enabling subsequent visualization and analysis.

For example, to record a single metric:

```python
metrics.record(metric_types.MetricType.LOSS, 0.123, step=1)
```

To record multiple metrics efficiently, use `record_metrics`:

```python
from google_cloud_mldiagnostics import metric_types
# User codes
# machinelearning_run should be called
# ......

for step in range(num_steps):
  if (step + 1) % 10 == 0:
    metrics.record_metrics([
        # Model quality metrics
        {"metric_name": metric_types.MetricType.LEARNING_RATE, "value": step_size},
        {"metric_name": metric_types.MetricType.LOSS, "value": loss},
        {"metric_name": metric_types.MetricType.GRADIENT_NORM, "value": gradient},
        {"metric_name": metric_types.MetricType.TOTAL_WEIGHTS, "value": total_weights},
        # Model performance metrics
        {"metric_name": metric_types.MetricType.STEP_TIME, "value": step_time},
        {"metric_name": metric_types.MetricType.THROUGHPUT, "value": throughput},
        {"metric_name": metric_types.MetricType.LATENCY, "value": latency},
        {"metric_name": metric_types.MetricType.TFLOPS, "value": tflops},
        {"metric_name": metric_types.MetricType.MFU, "value": mfu},
    ], step=step+1)
```

There are some metrics that are automatically collected by SDK from libTPU,
psutil and JAX libraries and the user does not need to write them:

1. System metrics - TPU tensorcore utilization, TPU duty cycle, HBM utilization, Host CPU utilization, Host memory utilization

These system metrics will by default have “time” as the x-axis only.
We also have some predefined key-value pairs for certain metrics so these can
be collected easily and will show up in the Pantheon UI automatically. Note
that these metrics aren’t calculated automatically, these are just predefined
keys that the user can write metrics values to these keys by themselves.

1. Model quality metric keys - `LEARNING_RATE`, `LOSS`, `GRADIENT_NORM`, `TOTAL_WEIGHTS`
1. Model perf metric keys - `STEP_TIME`, `THROUGHPUT`, `LATENCY`, `MFU`, `TFLOPS`

These predefined metrics as well as other user-defined metrics can be recorded
with x-axis as `time` or as `step`.

### Programmatic Profile Capture

In order to capture XProf profiles of your ML workload, you have two options:

1. Programmatic capture
1. On-demand capture (aka manual capture)

With programmatic capture, you need to annotate your model code in order to
specify where in your code you want to capture profiles. Typically, you capture
a profile for a few training steps, or profile a specific block of code within
your model. For programmatic profile capture within ML Diagnostics SDK we offer
3 options:

-   API-based Collection: control profiling with `start()` and `stop()` methods
-   Decorator-based Collection: annotate functions with `@xprof(run)` for
    automatic profiling
-   Context Manager: Use with `xprof()` for clean, scope-based profiling that
    automatically handles start/stop operations

These methods are abstracted out from the framework-level APIs (JAX, Pytorch
XLA, Tensorflow) for profile collection so you can use the same profile capture
code across all frameworks. All the profile sessions will be captured in the GCS
bucket defined in the machine learning run.

**Note** For preview, only JAX is supported.

```python
# Support collection via APIs
prof = xprof()  # Updates metadata and starts xprofz collector
prof.start()  # Collects traces to GCS bucket
# ..... Your code execution here
# ....
prof.stop()

# Also supports collection via decorators
@xprof()
def abc(self):
    # does something
    Pass

# Use xprof as a context manager to automatically start and stop collection
with xprof() as prof:
    # Your training or execution code here
    train_model()
    evaluate_model()
```

### Multi-host (process) profiling

For programmatic profiling, the SDK starts profiling on each host (process)
where ML workload code is executing. If the list of nodes is not provided, we
will automatically collect all hosts.

```python
# starts profiling on all nodes
prof = xprof()
prof.start()
# ...
prof.stop()
```

By default, calling `prof.start()` without a `session_id` on multiple hosts will
result in separate trace sessions—one for each host. To group traces from
different hosts into a single, unified multi-host session in XProf, you must
ensure that `prof.start()` is called with the *same* `session_id` on all
participating hosts.

You can achieve this by passing a consistent session ID string, for example:

```python
# Use the same session_id on all hosts to group traces
prof = xprof()
prof.start(session_id="profiling_session")
# ...
prof.stop()
```

Additionally SDK provides way to enable profiling only for specific hosts
(processes):

```python
# starts profiling on node with index 0 and 2
prof = xprof(process_index_list=[0,2])
prof.start()
# ...
prof.stop()
```

So, for the typical case of collecting profiles on just host 0, the user will
need to specify just index 0 in the list.

### Enable On-Demand Profile Capture

You can use on-demand profile capture when you want to capture profiles in an
ad hoc manner, or when you don't enable programmatic profile capture. This can
be helpful when you see a problem with your model metrics during the run and
want to capture profiles at that instant for some period in order to diagnose
the problem.

To enable feature customer needs to configure ML Run with on demand support,
example:

```python
# Define machinelearning run
machinelearning_run(
    name="<run_name>",
    # specify where profiling data will be stored
    gcs_path="gs://<bucket>",
    ...
    # enable on demand profiling, starts xprofz daemon on port 9999
    on_demand_xprof=True
)
```

This method is abstracted out from the framework-level APIs (JAX, Pytorch XLA,
Tensorflow) for profile collection so you can use the same profile capture code
across all frameworks. All the profile sessions will be captured in the GCS
bucket defined in the machine learning run.

For seamless on-demand profiling on GKE, we recommend deploying GKE connection
operator along with injection webhook into the GKE cluster (see prereq section).
This will ensure that your machine learning run knows which GKE nodes it is
running on and so on-demand capture drop down can auto populate these nodes
automatically.

### Viewing Logs, Metrics and Profiles

Once your workload is running with the SDK and Cloud Logging configured:

*   **Logs and Metrics**: Can be viewed in the Google Cloud Console under
**Logging > Logs Explorer**. Metrics recorded via `metrics.record()` are written
as log entries and can be filtered or used to create log-based metrics.
*   **Profiles and Run Details**: Can be viewed in the **Cluster Director**
section of the Google Cloud Console. The `machinelearning_run()` function will
output a log message containing a direct link to the run in the UI.

### Dockerfile Example

Below is an example Dockerfile snippet for packaging an application that uses
the `google-cloud-mldiagnostics` SDK. Remember to include `google-cloud-logging`
for Cloud Logging integration.

```dockerfile
# Base image (user's choice, e.g., python:3.10-slim, or a base with ML frameworks)
FROM python:3.11-slim

# Install base utilities
RUN pip install --no-cache-dir --upgrade pip

# Install SDK and Logging client
# psutil is installed as a dependency of google-cloud-mldiagnostics
RUN pip install --no-cache-dir \
    google-cloud-mldiagnostics \
    google-cloud-logging

# Optional: For JAX/TPU workloads
# RUN pip install --no-cache-dir "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
#     pip install --no-cache-dir libtpu xprof

# Add your application code
COPY ./app /app
WORKDIR /app

# Run your script
CMD ["python", "your_train_script.py"]
```

## Deploy Workload with SDK integrated

After integrating the SDK with your workload, you need to package the workload
in an image and then create your yaml file as `<yaml_name>.yaml` with the image
specified. Then you can deploy your workload using GKE.

For GKE:

```bash
kubectl apply -f <yaml_name>.yaml
```

For GCE, just SSH into your VM and run the python code for your workload

```python
source venv/bin/activate
python3.11 <workload>.py
```

When user deploys their workload with SDK, they will get a link to Console
similar to below:

To find this link as well as your MLrun name, first find your job name with
namespace `diagon` (or your workload's namespace):

```bash
kubectl get job -n <your-namespace>
```

Then, find the MLrun name and link in your kubectl logs by passing this job name
and namespace. Note: You must specify the workload container (e.g., `-c workload`)
because the Diagon sidecar handles its own logging.

```bash
kubectl logs jobs/s5-tpu-slice-0 -n <your-namespace> -c workload
```