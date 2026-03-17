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
- [Prerequisites](#prerequisites)
  - [Enable Cluster Director API](#enable-api)
  - [IAM Permissions](#iam-permissions)
  - [Configure GKE Cluster](#configure-gke-cluster)
  - [Install ML Diagnostics SDK](#install-ml-diagnostics-sdk)
- [How to use](#how-to-use)
  - [Enable Cloud Logging](#enable-cloud-logging)
  - [Enable Debug Logging](#enable-debug-logging)
  - [Creating a machine learning run](#creating-a-machine-learning-run)
  - [Write configs using yaml or json](#write-configs-using-yaml-or-json)
  - [Collect metrics](#collect-metrics)
  - [Programmatic Profile Capture](#programmatic-profile-capture)
  - [Multi-host (process) profiling](#multi-host-process-profiling)
  - [Enable On-Demand Profile Capture](#enable-on-demand-profile-capture)
  - [Viewing Logs, Metrics and Profiles](#viewing-logs-metrics-and-profiles)
  - [Package Workload with SDK with Dockerfile for GKE](#package-workload-with-sdk-with-dockerfile-for-gke)
  - [Deploy Workload with SDK integrated](#deploy-workload-with-sdk-integrated)
- [Using ML Diagnostics with Maxtext](#using-ml-diagnostics-with-maxtext)

## Overview

**Note:** Google Cloud ML Diagnostics supports only JAX on Google Cloud TPUs today.

Google Cloud ML Diagnostics is an end-to-end managed platform for optimizing
and diagnosing AI/ML workloads on Google Cloud. The platform lets you collect
and visualize all workload metrics, configs and profiles within a single
platform. ML Diagnostics is applicable to both training and inference
workloads, and is compatible with all orchestrators on TPU, including Google
Kubernetes Engine and custom orchestrators.

ML Diagnostics includes the following features:

- **Create and track Machine Learning runs**: Use ML Diagnostics platform to
create and register your machine learning runs either through Google Cloud CLI or by
integrating ML Diagnostics SDK with your workload. Creating Machine learning
runs allows users to deploy managed XProf instances as well as collect/manage
workload metrics, configs and profile sessions.
- **Google Cloud CLI experience**: Use ML Diagnostics APIs through Google Cloud CLI to
register and manage ML runs, deploy managed XProf resources, visualize already
captured XProf profile sessions in GCS bucket and trigger profile capture from CLI
- **Python SDK**: An open-source [ML Diagnostics SDK](https://github.com/AI-Hypercomputer/google-cloud-mldiagnostics) that users can integrate
with their ML workload to get the complete ML workload diagnostics experience to
collect/manage workload metrics, configs and profiles on Google Cloud.
- **Managed profiling**: ML Diagnostics deploys a Managed instance of [XProf](https://openxla.org/xprof)
with a scalable backend into the customer's account which allows faster loading of
large profiles, supports multiple users simultaneously accessing profiles and
supports easy to use out-of-the-box features such as multi-host profiling and
on-demand profiling.
- **Workload metrics**: Track workload metrics, including model quality, model
performance and system metrics.
- **Workload Config management**: Track workload configs including software
configs, system configs as well as user-defined configs.
- **Visualizations in Cluster Director and GKE**: Visualize metrics, configs,
and profiles in Cluster Director and Google Kubernetes Engine in the Google
Cloud console.
- **Link sharing**: ML Diagnostics allows easy collaboration with shareable
links for profiles and machine learning run information.

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

## Prerequisites

Before using ML Diagnostics, enable the Cluster Director API and add the required IAM permissions.

### Enable Cluster Director API {#enable-api}

**Note:** You do not need to use the Cluster Director for deploying and managing your clusters in order to use the ML Diagnostics product. ML Diagnostics product works with clusters managed by GKE and Cluster Director or even clusters using custom orchestrators. ML Diagnostics is part of the Cluster Director family of APIs, but doesn't depend on users using the Cluster Director product itself.

For more on enabling Cluster Director API, see [Enabling an API in your Google Cloud project](https://docs.cloud.google.com/endpoints/docs/openapi/enable-api).

### IAM Permissions

The Google Service Account used by your workload requires the following IAM roles assigned on the project:

1. `roles/clusterdirector.editor`: For full access to create and manage MLRun resources and view the user interface.
1. `roles/logging.logWriter`: To write logs and metrics to Google Cloud Logging.
1. `roles/storage.objectUser`: To save profiles to the GCS bucket specified in `machinelearning_run`.

For read-only access (viewing UI only, not creating MLRuns), `roles/clusterdirector.viewer` is sufficient.

For workloads on Google Kubernetes Engine, use [Workload Identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) to associate a Kubernetes Service Account with a Google Service Account that has been granted the required roles.

### Configure GKE Cluster

If GKE will be used for the ML workload, user needs to install the following to
their GKE cluster. Please ensure the GKE cluster is configured as a regional
cluster with Workload Identity enabled.

#### Install injection-webhook in the cluster

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



##### Install or upgrade

To list all available versions for the helm charts in the Google Cloud Artifact Registry, run:

```bash
gcloud artifacts tags list \
  --package=mldiagnostics-injection-webhook \
  --repository=mldiagnostics-webhook-and-operator-helm \
  --location=us --project=ai-on-gke
```

We recommend using `helm upgrade --install` to both install for the first time or upgrade an existing installation.

```bash
helm upgrade --install mldiagnostics-injection-webhook \
  --namespace=gke-mldiagnostics \
  --create-namespace \
  --version 0.23.0 \
  oci://us-docker.pkg.dev/ai-on-gke/mldiagnostics-webhook-and-operator-helm/mldiagnostics-injection-webhook
```

The above command can be edited with `-f` or `--set` flags to pass in a custom
values file or key-value pair respectively for the chart.

##### Uninstall

Since google-cloud-mldiagnostics SDK depends on the METADATA injected by the
mldiagnostics-injection-webhook, make sure your workloads are not using
google-cloud-mldiagnostics SDK before uninstall it.

To completely remove the injection webhook, follow these steps:

```bash
helm uninstall mldiagnostics-injection-webhook -n gke-mldiagnostics
```

Or you can use gcloud and kubectl. To list all available YAML versions:

```bash
gcloud artifacts versions list \
  --package=mldiagnostics-injection-webhook \
  --repository=mldiagnostics-webhook-and-operator-yaml \
  --location=us --project=ai-on-gke
```

```bash
gcloud artifacts generic download --repository=mldiagnostics-webhook-and-operator-yaml --location=us --package=mldiagnostics-injection-webhook --version=v0.23.0 --destination=./ --project=ai-on-gke
kubectl create namespace gke-mldiagnostics
# it needs to be installed inside namespace gke-mldiagnostics. If not, need to change mldiagnostics-injection-webhook-v0.23.0.yaml
kubectl apply -f mldiagnostics-injection-webhook-v0.23.0.yaml -n gke-mldiagnostics

## Uninstall. First, uninstall MutatingWebhookConfiguration, then delete yaml.
# kubectl delete MutatingWebhookConfiguration mldiagnostics-injection-webhook-mutating-webhook-config
# kubectl delete -f  mldiagnostics-injection-webhook-v0.23.0.yaml -n gke-mldiagnostics
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

#### Install connection-operator in the cluster

##### Install or upgrade

To list all available versions for the helm charts in the Google Cloud Artifact Registry, run:

```bash
gcloud artifacts tags list \
  --package=mldiagnostics-connection-operator \
  --repository=mldiagnostics-webhook-and-operator-helm \
  --location=us --project=ai-on-gke
```

For seamless on-demand profiling on GKE, we recommend deploying the GKE connection
operator along with the injection webhook into the GKE cluster. This will ensure
that your machine learning run knows which GKE nodes it is running on and so the
on-demand capture drop-down can auto-populate these nodes automatically.

We recommend using `helm upgrade --install` to both install for the first time or upgrade an existing installation.

##### Compatibility Matrix

| JAX Version | `mldiagnostics-connection-operator` Helm Chart Version |
| :--- | :--- |
| 0.8.x | 0.21.0 |
| 0.9.x+ | 0.21.0+ |

###### For JAX 0.8.x:
```bash
helm upgrade --install mldiagnostics-connection-operator \
  --namespace=gke-mldiagnostics \
  --create-namespace \
  --version 0.21.0 \
  oci://us-docker.pkg.dev/ai-on-gke/mldiagnostics-webhook-and-operator-helm/mldiagnostics-connection-operator \
  --set 'mldiagnosticsConnectionOperator.controller.args={--metrics-bind-address=:8443,--leader-elect,--health-probe-bind-address=:8081,--sidecar-timeout=65m,--disable-hostname-override}'
```

###### For JAX 0.9.x+:
```bash
helm upgrade --install mldiagnostics-connection-operator \
  --namespace=gke-mldiagnostics \
  --create-namespace \
  --version 0.21.0 \
  oci://us-docker.pkg.dev/ai-on-gke/mldiagnostics-webhook-and-operator-helm/mldiagnostics-connection-operator
```

The above command can be edited with `-f` or `--set` flags to pass in a custom
values file or key-value pair respectively for the chart.

##### Uninstall

```bash
helm uninstall mldiagnostics-connection-operator -n gke-mldiagnostics
```

Or you can use gcloud and kubectl. To list all available YAML versions:

```bash
gcloud artifacts versions list \
  --package=mldiagnostics-connection-operator \
  --repository=mldiagnostics-webhook-and-operator-yaml \
  --location=us --project=ai-on-gke
```

```bash
gcloud artifacts generic download --repository=mldiagnostics-webhook-and-operator-yaml --location=us --package=mldiagnostics-connection-operator --version=v0.21.0 --destination=./ --project=ai-on-gke
kubectl create namespace gke-mldiagnostics
kubectl apply -f mldiagnostics-connection-operator-v0.21.0.yaml -n gke-mldiagnostics

## use this to uninstall
# kubectl delete -f mldiagnostics-connection-operator-v0.21.0.yaml -n gke-mldiagnostics
```


### Install ML Diagnostics SDK

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
to perform logging, metrics collection, and profile tracing.

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

#### Single metric recording

```python
metrics.record(metric_types.MetricType.LOSS, 0.123, step=1)
```

#### Multiple metric recording

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
with x-axis as `time`, or both `time` and `step`.

User can record any custom metric in the workload as shown below:

```python
metrics.record("custom_metrics_1", step_size, step=step + 1)
```

This will capture a new custom_metrics_1 for workload and the user can view it in the Model Metrics tab for this specific machine learning run.

To record multiple metrics in one call, user can use record_metrics method as shown below:

```python
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
        # Custom metrics
        {"custom_metrics_1", "value":<value>},
        {"custom_metrics_2", "value":<value>},
        {"avg_mtp_acceptance_rate_percent", "value":<value>},
        {"dpo_reward_accuracy", "value":<value>},
    ], step=step+1)
```

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

**Note:** Google Cloud ML Diagnostics primarily supports JAX on Google Cloud TPUs (support for other frameworks like vllm, sglang, Torch TPU, etc will come in the future).

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
adhoc manner, or when you don't enable programmatic profile capture. This can
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

Once your workload is running with the SDK and Cloud Logging configured you will
see all machine learning runs on Google Cloud Pantheon console in both Cluster
Director and GKE UI.

1. In Cluster Director, you can find all your machine learning runs created by ML Diagnostics under Cluster Director -> Diagnostics tab
2. In GKE, you can find all your machine learning runs created by ML Diagnostics under GKE -> AI/ML -> Diagnostics tab

In both Cluster Director and GKE, you will find the following pages:

1. List view table with summary information of all your machine learning runs
2. Run details for each run with details of configs and run information
3. Time series charts for all metrics: model metrics, performance metrics, system
   metrics. You can also view these metrics Cloud Logging under Logging > Logs
   Explorer. Metrics recorded via `metrics.record()` are written as log entries
   and can be filtered or used to create log-based metrics.
4. Profiles tab with all profile sessions (programmatic or on-demand) for that particular run, with links to the Xprof viewer (Xprof UI will open in a separate browser tab). In this profiles tab, you can also capture an on-demand profile session directly from the UI.

### Package Workload with SDK with Dockerfile for GKE

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

### Deploy Workload with SDK integrated

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

## Using ML Diagnostics with Maxtext

For users who use Maxtext as their ML workload, ML Diagnostics SDK is already pre-integrated with Maxtext. You can enable ML Diagnostics with Maxtext with the `managed_mldiagnostics` flag. If this is enabled, it will:
- Create a managed MachineLearning run with all the MaxText configs.
- Upload profiling traces, if the profiling is enabled by `profiler="xplane"`.
- Upload training metrics, at the defined `log_period` interval.

These are the new flags related to this feature:

```yaml
managed_mldiagnostics: True  # Whether to enable the managed diagnostics
managed_mldiagnostics_run_group: "<some-name>"  # Optional. Used to group multiple runs.
```

To enable ML Diagnostics in Maxtext, you can either change the configuration file of your run, or pass the flags from the command line. 

When you run `MaxText.train`, you can pass these flags:

```bash
python3 -m MaxText.train src/MaxText/configs/base.yml run_name="demo-mldiagnostics-run-2" model_name="<your_chosen_model>" base_output_directory=gs://<your_gcs_folder>/  dataset_type=synthetic steps=100 log_period=10 profiler=xplane upload_all_profiler_results=True managed_mldiagnostics=True managed_mldiagnostics_run_group="demo-mldiagnostics-group"
```

`upload_all_profiler_results=True` captures multihost profiles from all hosts.