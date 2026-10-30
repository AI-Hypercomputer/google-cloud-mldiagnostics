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
  - [Enable Cluster Director API](#enable-cluster-director-api)
  - [IAM Permissions](#iam-permissions)
  - [Configure GKE Cluster](#configure-gke-cluster)
  - [Install ML Diagnostics SDK](#install-ml-diagnostics-sdk)
  - [Upgrading & Known Limitations in Older Versions](#upgrading-known-limitations-in-older-versions)
- [How to use](#how-to-use)
  - [Enable Cloud Logging](#enable-cloud-logging)
  - [Enable Debug Logging](#enable-debug-logging)
  - [Configure Telemetry Exporters for OTel Integration](#configure-telemetry-exporters-for-otel-integration)
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
- [Using ML Diagnostics for RL (MaxText + Tunix + Pathways)](#using-ml-diagnostics-for-rl-maxtext--tunix--pathways)
- [Using ML Diagnostics with vLLM](#using-ml-diagnostics-with-vllm)
- [Using ML Diagnostics with PyTorch on TPU](#using-ml-diagnostics-with-pytorch-on-tpu)
- [Using ML Diagnostics with Cluster Toolkit (CTK)](#using-ml-diagnostics-with-cluster-toolkit-ctk)

## Overview

[**Release Notes**](https://github.com/AI-Hypercomputer/google-cloud-mldiagnostics/blob/main/docs/release_notes.md)


**Note:** Google Cloud ML Diagnostics supports JAX, PyTorch, vLLM, and RL (Maxtext + Tunix + Pathways) on Google Cloud TPUs and GPUs today.

Google Cloud ML Diagnostics is an end-to-end managed platform for optimizing
and diagnosing AI/ML workloads on Google Cloud. The platform lets you collect
and visualize all workload metrics, configs and profiles within a single
platform. ML Diagnostics is applicable to both training and inference
workloads, and is compatible with all orchestrators on TPU and GPU, including Google
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
- **Universal Telemetry Stack Integration**: Log and metric collection setup (OTel, Prometheus, Loki, Grafana).


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

### Enable Cluster Director API

**Note:** You do not need to use the Cluster Director for deploying and managing your clusters in order to use the ML Diagnostics product. ML Diagnostics product works with clusters managed by GKE and Cluster Director or even clusters using custom orchestrators. ML Diagnostics is part of the Cluster Director family of APIs, but doesn't depend on users using the Cluster Director product itself.

For more on enabling Cluster Director API, see [Enabling an API in your Google Cloud project](https://docs.cloud.google.com/endpoints/docs/openapi/enable-api).

### IAM Permissions

The Google Service Account used by your workload requires the following IAM roles assigned on the project:

1. `roles/clusterdirector.editor`: For full access to create and manage MLRun resources and view the user interface.
1. `roles/logging.logWriter`: To write logs and metrics to Google Cloud Logging.
1. `roles/storage.objectUser`: To save profiles to the GCS bucket specified in `machinelearning_run`.
1. `roles/storage.bucketViewer`: To view the GCS bucket metadata for uploading profiles.

For read-only access (viewing UI only, not creating MLRuns), `roles/clusterdirector.viewer` is sufficient.

For workloads on Google Kubernetes Engine, use [Workload Identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) to associate a Kubernetes Service Account with a Google Service Account that has been granted the required roles.

### Enable Log Analytics on Cloud Logging

The model-metrics and performance-metrics time series charts render their data with a Log Analytics query against the project's `_Default` log bucket. If Log Analytics is not enabled on that bucket, these charts fail (the widget shows a generic "Something went wrong in the backend" / "0 time series"), even though the metric data is present in Cloud Logging and Profiles / system metrics render fine. Enable it once per project (non-destructive upgrade — existing logs stay queryable, no added cost, cannot be disabled):

```bash
gcloud logging buckets update _Default --location=global \
  --enable-analytics --project=PROJECT_ID
```

This requires the `logging.buckets.update` permission (for example `roles/logging.configWriter`).

**Note:** Log Analytics does not backfill. Only logs ingested *after* it is enabled are queryable, so runs that completed before enabling it will still show empty charts — launch a new run to verify.

### Configure GKE Cluster

> **Tip:** If you provision your GKE cluster using
> [Google Cloud Cluster Toolkit (CTK)](#using-ml-diagnostics-with-cluster-toolkit-ctk),
> setting `enable_ml_diagnostics: true` in your cluster blueprint automatically
> configures the ML Diagnostics webhook, connection operator, namespace labels,
> and Workload Identity service accounts—you can skip the manual cluster setup
> steps below.

If GKE will be used for the ML workload, user needs to install the following to
their GKE cluster. Please ensure the GKE cluster is configured as a regional
cluster with Workload Identity enabled.

#### Install injection-webhook in the cluster

For workloads running in GKE, injection-webhook is needed to provide SDK needed
metadata. It supports these common ML kubernetes workloads:
JobSet/RayJob/LeaderWorkerSet.

**Note:** If `mldiagnostics-injection-webhook` is not installed in your GKE
cluster (or if your workload does not have the `managed-mldiagnostics-gke=true`
label), you can pass GKE workload metadata directly to `machinelearning_run()`
using the `workload_details` parameter (see
[Creating a machine learning run](#creating-a-machine-learning-run)).

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
  --version 0.25.0 \
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
gcloud artifacts generic download --repository=mldiagnostics-webhook-and-operator-yaml --location=us --package=mldiagnostics-injection-webhook --version=v0.25.0 --destination=./ --project=ai-on-gke
kubectl create namespace gke-mldiagnostics
# it needs to be installed inside namespace gke-mldiagnostics. If not, need to change mldiagnostics-injection-webhook-v0.25.0.yaml
kubectl apply -f mldiagnostics-injection-webhook-v0.25.0.yaml -n gke-mldiagnostics

## Uninstall. First, uninstall MutatingWebhookConfiguration, then delete yaml.
# kubectl delete MutatingWebhookConfiguration mldiagnostics-injection-webhook-mutating-webhook-config
# kubectl delete -f  mldiagnostics-injection-webhook-v0.25.0.yaml -n gke-mldiagnostics
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
| 0.8.x | 0.24.0 |
| 0.9.x+ | 0.24.0+ |

###### For JAX 0.8.x:
```bash
helm upgrade --install mldiagnostics-connection-operator \
  --namespace=gke-mldiagnostics \
  --create-namespace \
  --version 0.24.0 \
  oci://us-docker.pkg.dev/ai-on-gke/mldiagnostics-webhook-and-operator-helm/mldiagnostics-connection-operator \
  --set 'mldiagnosticsConnectionOperator.controller.args={--metrics-bind-address=:8443,--health-probe-bind-address=:8081,--sidecar-timeout=65m,--disable-hostname-override}'
```

###### For JAX 0.9.x+:
```bash
helm upgrade --install mldiagnostics-connection-operator \
  --namespace=gke-mldiagnostics \
  --create-namespace \
  --version 0.24.0 \
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
gcloud artifacts generic download --repository=mldiagnostics-webhook-and-operator-yaml --location=us --package=mldiagnostics-connection-operator --version=v0.24.0 --destination=./ --project=ai-on-gke
kubectl create namespace gke-mldiagnostics
kubectl apply -f mldiagnostics-connection-operator-v0.24.0.yaml -n gke-mldiagnostics

## use this to uninstall
# kubectl delete -f mldiagnostics-connection-operator-v0.24.0.yaml -n gke-mldiagnostics
```


### Install ML Diagnostics SDK

Pip install [SDK](https://pypi.org/project/google-cloud-mldiagnostics/)

```bash
pip install google-cloud-mldiagnostics
```

This package does not install `libtpu`, `jax`, and `xprof`; you are expected to
install these separately if needed for your workload.

### Upgrading & Known Limitations in Older Versions

We strongly recommend upgrading to **SDK version `1.0.6` or later** (`pip install --upgrade google-cloud-mldiagnostics`).

**Known limitations in SDK versions `< 1.0.6`:**

*   **GKE Workload Monitoring Compatibility:** If your GKE cluster version is `1.36.0-gke.4681000` or later and the cluster is present in a supported Workload Monitoring GA region (`us-central1`, `us-east1`, `us-east4`, `us-east5`, `us-south1`, `us-west1`, `europe-west1`, `europe-west2`, `europe-west4`, `southamerica-west1`), Diagon Workload Monitoring automatically pre-creates a Machine Learning Run for the workload. Using MLDiagnostics SDK version `< 1.0.6` will log an informational message similar to the following and skip updating it, which prevents profiling from being enabled on the run:

    ```log
    INFO:google_cloud_mldiagnostics.core.global_manager:ML run 'projects/12345/locations/us-central1/machineLearningRuns/my-job' already exists, skipping creation.
    ```

    The mitigation is to upgrade MLDiagnostics to the latest SDK version (`1.0.6` or higher), which automatically updates existing runs and enables full profile capture.


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
### Configure Telemetry Exporters for OTel Integration

**New Feature**: The SDK now supports dual-routing and pluggable telemetry! In addition to standard Google Cloud Logging, you can configure the SDK to export metrics and logs to an OpenTelemetry (OTel) Collector. This is extremely useful for integration with localized monitoring stacks like **Prometheus, Loki, and Grafana**.

#### Prerequisites for OTel Integration

- OpenTelemetry Collector running and accessible from your workload.
- (Optional) Prometheus, Loki, and Grafana set up to consume data from the OTel Collector.

#### Configuring Exporters in SDK

You can configure the metrics exporter mode and user configuration using the `metrics_exporter_config` parameter in `machinelearning_run`.

> [!TIP]
> If you are using the **Google Cloud Exporter** in your OpenTelemetry Collector, make sure to configure `default_log_name` in the collector configuration. The Google Cloud exporter requires a log name; if the SDK doesn't provide the `gcp.log_name` attribute, logs will be dropped unless `default_log_name` is set in the collector. See [OpenTelemetry Google Cloud Exporter docs](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/exporter/googlecloudexporter#log-configuration) for more details.

Example:

```python
machinelearning_run(
    # ... other parameters ...
    metrics_exporter_config={
        "mode": ["otel", "cloud_logging"], # Export system metrics to both OTel and Cloud Logging
        "userConfig": ["otel", "cloud_logging"], # Export user metrics to both OTel and Cloud Logging
    },
)
```

By default, if `metrics_exporter_config` is not provided, it defaults to `["cloud_logging"]`.

#### Environment Variables

You can also use environment variables to control OTel integration:

- `DIAGON_ENABLE_OTEL`: Set to `'true'` to enable OTel exporter.
- `OTEL_EXPORTER_OTLP_ENDPOINT`: The endpoint of the OTel Collector (e.g., `http://otel-collector.diagon.svc.cluster.local:4317`).
- `OTEL_METRIC_EXPORT_INTERVAL`: Metric export interval in milliseconds (default is `5000`).
- `OTEL_PYTHON_LOG_LEVEL`: Log level for OTel Python SDK (e.g., `DEBUG`).
- `OTEL_EXPORTER_OTLP_INSECURE`: Set to `'true'` to disable TLS for OTLP exporter.

> [!IMPORTANT]
> Ensure the `opentelemetry-sdk` and `opentelemetry-exporter-otlp` packages are installed (or install via `pip install google_cloud_mldiagnostics[otel]`).


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

`workload_details` Optional (supported for GKE workloads only). A dictionary of
GKE workload metadata used when `mldiagnostics-injection-webhook` is not
installed or enabled on the workload. Supported keys are:

- `id` (**required**): Workload name (for example, the `JobSet` name).
- `kind` (*optional*): Kubernetes workload kind (for example, `"JobSet"`,
  `"Job"`, `"LeaderWorkerSet"`, or `"RayJob"`). Defaults to `"JobSet"`.
- `namespace` (**required** unless running in-cluster): Kubernetes namespace of
  the workload. In-cluster, falls back to
  `/var/run/secrets/kubernetes.io/serviceaccount/namespace` or `POD_NAMESPACE`.
- `cluster` (**required** unless running in-cluster): GKE cluster name or full
  resource path (`projects/PROJECT/locations/LOCATION/clusters/CLUSTER`). Bare
  cluster names are supported in-cluster only (where `cluster-location` is
  resolved from GCE instance metadata); outside GKE, provide the full resource
  path.
- `parent_workload` (*optional*): Parent workload name.
- `creation-timestamp` (*optional*): RFC 3339 workload creation timestamp.
- `labels` (*optional*): Dictionary of string key-value workload labels.

Example of creating a run on GKE without `mldiagnostics-injection-webhook`:

```python
run = machinelearning_run(
    name="<run_name>",
    run_group="<run_group>",
    project="<some_project>",
    region="<some_region>",
    gcs_path="gs://<some_bucket>",
    on_demand_xprof=True,
    workload_details={
        "id": "<jobset_name>",
        "kind": "JobSet",
        "namespace": "<k8s_namespace>",
        "cluster": "<gke_cluster_name>",
    },
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
multiple metrics in a single batch. These functions dispatch metrics to the
configured exporters (Cloud Logging by default, or OpenTelemetry if enabled).

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

There are some metrics that are automatically collected by the SDK (from libTPU, pynvml, psutil, and JAX libraries) depending on the accelerator type:

1. **For TPUs**: TPU tensorcore utilization, TPU duty cycle, HBM utilization, Host CPU utilization, Host memory utilization.
2. **For GPUs**: GPU utilization, GPU Tensor Core utilization, VRAM (HBM) utilization, Host CPU utilization, Host memory utilization.

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

**Note:** Google Cloud ML Diagnostics supports JAX, PyTorch, vLLM, and RL (Maxtext + Tunix + Pathways) on Google Cloud TPUs and GPUs (support for other frameworks like sglang will come in the future).

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

### Multi-host JAX Initialization

When running multi-host JAX workloads (such as GPU clusters or TPU
multi-slice configurations), you must initialize the JAX distributed system
to coordinate between hosts and avoid race conditions during startup.

Unlike single-slice TPU setups where JAX automatically detects the
coordinator setup, these environments require explicit configuration of the
coordinator address, number of processes, and process ID.

#### Option 1: Using Environment Variables (Recommended for general setups)

You can configure this by setting the following environment variables on all hosts before running your script:

*   `JAX_COORDINATOR_ADDRESS`: The IP address and port (e.g., `192.168.0.1:29400`) of the host with rank 0 (the coordinator).
*   `JAX_NUM_PROCESSES`: The total number of hosts/nodes in the cluster.
*   `JAX_PROCESS_ID`: The rank/ID of the current host (from `0` to `num_processes - 1`).

In your Python script, initialize JAX:

```python
import jax

# Initialize JAX distributed using the environment variables
jax.distributed.initialize()
```

#### Option 2: Explicit Initialization (Recommended for GKE/JobSet)

If you are deploying on GKE using **JobSet**, you can leverage Kubernetes environment variables (like `JOB_COMPLETION_INDEX` for the process ID) and internal Headless Service DNS for the coordinator address.

Here is an example of how to initialize JAX in your training script:

```python
import os
import logging
import jax

# 1. Determine process ID from Kubernetes JobSet completion index
process_id = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))

# 2. Define the number of processes (matches JobSet replicatedJob replicas)
num_processes = 2

# 3. Define the coordinator address using the headless service DNS of the first pod (rank 0)
# Format: <pod-name>.<service-name>.<namespace>.svc.cluster.local:<port>
coordinator_address = "multihost-gpu-jobset-gpu-slice-0-0.multihost-gpu-jobset.default.svc.cluster.local:8471"

logging.info(
    f"Initializing JAX distributed: coordinator={coordinator_address}, "
    f"num_processes={num_processes}, process_id={process_id}"
)

jax.distributed.initialize(
    coordinator_address=coordinator_address,
    num_processes=num_processes,
    process_id=process_id,
)
```

**Important:**
*   You can choose any open port (e.g., `1234` or `8471`) for the JAX
    distributed service, as long as it is available and does not conflict
    with other services.
*   Ensure that the coordinator DNS address is correctly resolvable from all pods in the JobSet.

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
   *(Note: If model and performance metric charts show "0 time series" or fail to load, ensure Log Analytics is enabled on your `_Default` log bucket as described in [Enable Log Analytics on Cloud Logging](#enable-log-analytics-on-cloud-logging).)*
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

# Optional: For JAX/GPU workloads (adjust CUDA version as needed)
# RUN pip install --no-cache-dir "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

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

## Using ML Diagnostics for RL (MaxText + Tunix + Pathways)

Google Cloud ML Diagnostics is natively supported with the Reinforcement Learning (RL) post-training stack on Cloud TPUs:

- **[MaxText](https://github.com/AI-Hypercomputer/maxtext)**: Provides the model definitions, configuration system (`src/maxtext/configs/post_train/rl.yml`), and RL training entrypoint (`maxtext.trainers.post_train.rl.train_rl`).
- **[Tunix](https://github.com/google/tunix)**: Executes the RL post-training loop (such as GRPO and GSPO-Token), coordinating policy updates on the `actor` role and inference generation on the `rollout` role.
- **Pathways**: Provides single-controller distributed execution across TPU worker hosts for the trainer and sampler meshes.

When `managed_mldiagnostics: True` is enabled on an RL workload, ML Diagnostics automatically:

- Creates a managed `MachineLearningRun` (`MLRun`) and records the workload configurations.
- Streams RL training, reward, completion, and performance metrics at the configured `log_period` interval.
- Captures per-role XProf traces (`actor` and `rollout`) by counting role invocations (`actor_invocation_<n>` and `rollout_invocation_<n>`) across Pathways worker hosts when `profiler="xplane"` is enabled, saving traces under `<base_output_directory>/<run_name>/managed-mldiagnostics/<mlrun_name>/plugins/profile/<role>_invocation_<n>/`.

### 1. RL Configuration Flags (`rl.yml`)

Configure ML Diagnostics and per-role invocation profiling in `src/maxtext/configs/post_train/rl.yml` or via command-line overrides:

```yaml
run_name: "<your_run_name>"
base_output_directory: "gs://<your_gcs_bucket_dir>"

# Enable managed ML Diagnostics run registration and metric streaming
managed_mldiagnostics: True
enable_tunix_perf_metrics: True

# Enable per-role invocation XProf profiling
profiler: "xplane"
rl:
  profiler_start_invocation: 20
  profiler_num_invocations: 1
  profiler_max_num_hosts: 2
```

- `run_name` & `base_output_directory`: Required Cloud Storage (`gs://`) output path and run identifier.
- `managed_mldiagnostics`: Enables `MLRun` registration, RL scalar metric streaming, and profile session uploads.
- `enable_tunix_perf_metrics`: Enables Tunix step-timing and performance metrics (`perf/train/*`).
- `profiler: "xplane"`: Enables per-role (`actor` and `rollout`) XProf trace capture.
- `rl.profiler_start_invocation`: Zero-indexed role invocation to begin profiling (past initial compilation and warmup).
- `rl.profiler_num_invocations`: Number of consecutive invocations to profile for each role (`actor` and `rollout`).
- `rl.profiler_max_num_hosts`: Number of Pathways worker hosts to trace. Defaults to `1` (which traces only worker 0); set this to the number of worker hosts in your TPU topology (for example, `2` on a 2-host `v5litepod-8` slice) so both trainer and sampler hosts are captured in the trace.

> **Note:**
> - **Pathways multi-host profiling**: By default, `pathwaysutils` only traces worker 0 (`rl.profiler_max_num_hosts: 1`). When the `actor` trainer and `rollout` sampler are disaggregated across multiple TPU worker hosts on Pathways (which requires at least 2 VMs, such as `v5litepod-8` with `2x4` topology), leaving `rl.profiler_max_num_hosts` at `1` omits the sampler host from rollout traces. Set `rl.profiler_max_num_hosts` to at least the total number of TPU worker hosts in your slice (for example, `2`).
> - **Invocation counting vs. step flags**: RL profiling counts **role invocations** (`rl.profiler_start_invocation`, `rl.profiler_num_invocations`), where `actor` runs once per policy update and `rollout` runs once per generation batch. If legacy `skip_first_n_steps_for_profiler` or `profiler_steps` flags are passed without `rl.profiler_*` overrides, `RLConfig` maps them to `rl.profiler_start_invocation` and `rl.profiler_num_invocations`.

### 2. Prepare Docker Image for RL

Package the MaxText post-training container image with `google-cloud-mldiagnostics`, `xprof`, and `google-cloud-logging`:

```dockerfile
FROM us-docker.pkg.dev/cloud-tpu-images/maxtext-images/tpu_post_training:latest

RUN pip install --no-cache-dir --upgrade \
    google-cloud-mldiagnostics \
    xprof \
    google-cloud-logging
```

### 3. Run RL with ML Diagnostics on GKE (Pathways)

Ensure `mldiagnostics-injection-webhook` is installed in your GKE cluster (see [Configure GKE Cluster](#configure-gke-cluster)) so the `MLRun` metadata is automatically injected into your JobSet pods, then submit the RL workload with Pathways enabled (for example, on a 2-host `v5litepod-8` / `2x4` slice using Cluster Toolkit `gcluster job submit --pathways`):

```bash
MAXTEXT_COMMAND="ENABLE_PATHWAYS_PERSISTENCE=1 \
python3 -m maxtext.trainers.post_train.rl.train_rl \
  run_name=<your_run_name> \
  model_name=<your_model_name> \
  base_output_directory=gs://<your_gcs_bucket_dir> \
  num_batches=500 \
  managed_mldiagnostics=True \
  enable_tunix_perf_metrics=True \
  profiler=xplane \
  rl.profiler_start_invocation=20 \
  rl.profiler_num_invocations=1 \
  rl.profiler_max_num_hosts=2"

gcluster job submit \
  --image="<your_rl_image>" \
  --name="<your_run_name>" \
  --pathways \
  --compute-type="v5litepod-8" \
  --topology="2x4" \
  --num-slices=1 \
  --pathways-gcs-location="gs://<your_gcs_bucket_dir>" \
  --command="${MAXTEXT_COMMAND}"
```

### 4. Viewing RL Metrics and Per-Role Profiles

When the workload starts, the SDK logs the registered `MLRun` name, the direct Google Cloud Console link, and the Cloud Storage profiling output path in the workload container logs (see [Deploy Workload with SDK integrated](#deploy-workload-with-sdk-integrated) and [Viewing Logs, Metrics and Profiles](#viewing-logs-metrics-and-profiles)):

```bash
kubectl logs -l jobset.sigs.k8s.io/jobset-name=<your_run_name> -c jax-tpu
```

Open the Google Cloud Console link (or navigate to **Cluster Director -> Diagnostics** or **GKE -> AI/ML -> Diagnostics**) and select your `MLRun`:

1. **Model Metrics**: View time-series charts for RL training metrics (`actor/train/*`), reward metrics (`rewards/train/*`), completion lengths (`completions/train/*`), and step timing (`perf/train/*`).
2. **Profiles**: View the captured per-role XProf sessions—listed separately as `actor_invocation_<n>` (policy update) and `rollout_invocation_<n>` (rollout generation)—in the **Profiles** tab of your `MLRun` and click to open each trace in the managed XProf viewer. Per-host `*.xplane.pb` trace files are organized in Cloud Storage under `<base_output_directory>/<run_name>/managed-mldiagnostics/<mlrun_name>/plugins/profile/<role>_invocation_<n>/`.

## Using ML Diagnostics with vLLM

ML Diagnostics provides a built-in wrapper command, `google_cloud_mldiagnostics.commands.run_vllm`, to run vLLM serving and inference workloads with ML Diagnostics telemetry and profiling enabled on Google Cloud TPUs and GPUs.

The `run_vllm` wrapper:
- Creates and registers a managed Machine Learning Run (`MLRun`) with `serving_engine="VLLM"`.
- Configures profiler environment variables (`USE_JAX_PROFILER_SERVER`, `JAX_PROFILER_SERVER_PORT`, `PHASED_PROFILING_DIR`, and `VLLM_TORCH_PROFILER_DIR`) and updates profiler output paths to organize profiles under `<mlrun_gcs_path>/<run_name>/plugins/profile/`.
- Forwards all remaining command-line arguments directly to `vllm`.

### `run_vllm` Command-Line Arguments

- `--mlrun_name` (Required): Unique identifier for the MLRun.
- `--mlrun_gcs_path` (Required): Google Cloud Storage bucket path where profiler traces and artifacts are saved (e.g., `gs://my-bucket/vllm-profiles`).
- `--project` (Optional): Google Cloud Project ID.
- `--region` (Optional): Region where the MLRun metadata is stored (default: `us-central1`).
- `--run_group` (Optional): Identifier used to group multiple related runs.
- `--configs` (Optional): User-defined configurations passed as a JSON object string (e.g., `'{"user":"my-team", "running_on":"TPU"}'`).
- `--framework` (Optional): Underlying ML framework for the workload (`pytorch` or `jax`, default: `pytorch`).
- `--jax_profiler_port` (Optional): JAX profiler server port (default: `9999`).
- `--phased_profiling_dir` (Optional): Base GCS directory for JAX phased profiling traces. The wrapper appends `<run_name>/plugins/profile/` and exports the result as `PHASED_PROFILING_DIR` to the vLLM process. This is the recommended way to enable phased profiling.

> **Note (multi-host setups):** If `PHASED_PROFILING_DIR` is detected in the OS environment and `--phased_profiling_dir` is not passed, the wrapper still forwards it to `vllm serve` (with the run-scoped suffix) and logs a warning. In multi-host setups (e.g. Ray backend on GKE LeaderWorkerSet), phased profiles may not show in the XProf UI in this case: vLLM's Ray executor applies driver environment variables on remote workers with `os.environ.setdefault()`, so the run-scoped path does not propagate to workers that already have `PHASED_PROFILING_DIR` in their environment, and their traces end up outside the MLRun directory. Use `--phased_profiling_dir` and do not set `PHASED_PROFILING_DIR` in the environment.

### 1. Prepare Docker Image for vLLM

Below is an example `Dockerfile` for packaging a vLLM TPU image with the ML Diagnostics SDK:

```dockerfile
FROM vllm/vllm-tpu:latest
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        google-cloud-mldiagnostics \
        xprof \
        google-cloud-logging \
        packaging \
        tensorflow-cpu
EXPOSE 8000 9999
ENTRYPOINT ["python3", "-m", "google_cloud_mldiagnostics.commands.run_vllm"]
CMD ["--help"]
```

### 2. Run vLLM with ML Diagnostics

You can run the container on GKE or GCE by passing the MLRun arguments along with your standard `vllm serve` arguments:

```bash
docker run --name vllm_diagnostics -d --privileged --net=host \
  -e HF_HUB_DISABLE_XET=1 \
  -e USE_MOE_EP_KERNEL=1 \
  -e PJRT_DEVICE=TPU \
  -e ENABLE_GOOGLE_DIAGON_ML_DIAGNOSTICS=True \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -e VLLM_TORCH_PROFILER_DIR="gs://<your_gcs_bucket>/vllm_profiles" \
  -e PHASED_PROFILING_DIR="gs://<your_gcs_bucket>/vllm_profiles" \
  -e PYTHONUNBUFFERED=1 \
  -e PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR=100 \
  -e HF_TOKEN="<your_hf_token>" \
  -e HUGGING_FACE_HUB_TOKEN="<your_hf_token>" \
  <your_vllm_image> \
  --mlrun_name="vllm-tpu-run" \
  --run_group="vllm-benchmark" \
  --configs='{"user":"ml-team", "running_on":"TPU"}' \
  --mlrun_gcs_path="gs://<your_gcs_bucket>/vllm_profiles" \
  --project="<your_project_id>" \
  --region="us-central1" \
  serve Qwen/Qwen2-1.5B \
  --kv-cache-dtype=fp8 \
  --tensor-parallel-size=1 \
  --data-parallel-size=1 \
  --max-model-len=2048 \
  --max-num-batched-tokens=2048 \
  --max-num-seqs=64 \
  --no-enable-prefix-caching \
  --block-size=256 \
  --gpu-memory-utilization=0.95 \
  --port=8000 \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir="gs://<your_gcs_bucket>/vllm_profiles"
```

### 3. Profiling vLLM Workloads

You can capture profiler traces from your vLLM workload using any of the following methods:

1. **Phased Profiling**: When `ENABLE_GOOGLE_DIAGON_ML_DIAGNOSTICS=True` and `PHASED_PROFILING_DIR` are configured, profiling is automatically captured across different inference workload phases (such as `PREFILL_ONLY`, `PREFILL_HEAVY`, `BALANCED`, and `DECODE_HEAVY`) for the number of steps specified by `PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR`.
2. **Manual (API-Triggered) Profiling**: You can programmatically trigger and stop profiling sessions by sending HTTP requests to vLLM's `/start_profile` and `/stop_profile` API endpoints while generating load against the server.
3. **On-Demand UI Profiling**: Capture profile sessions directly from the **Profiles** tab of your MLRun in the Google Cloud Console (Cluster Director or GKE AI/ML Diagnostics UI).

## Using ML Diagnostics with PyTorch on TPU

Google Cloud ML Diagnostics natively supports distributed PyTorch TPU workloads (`torch_tpu`) across Cloud TPU generations (**v5e, v6e, v4, and v5p**).

### Distributed Training Architecture

PyTorch TPU workloads use a multi-process distributed architecture where **one independent process runs per TPU chip** (for example, 4 processes per host on TPU v5e/v6e). The SDK manages multi-process synchronization across both programmatic and live on-demand profiling:

#### Programmatic Profiling Synchronization

For programmatic profiling (`prof.start()` / `prof.stop()` or `with xprof():`), the SDK automatically synchronizes all distributed PyTorch ranks by executing `torch.distributed.barrier()` before starting and stopping trace collection. This collective barrier ensures all worker processes begin and end profiling at the exact same execution step, unifying multi-rank traces under a single cohesive session in XProf.

#### Live On-Demand Profiling

To support live on-demand profiling across multi-process workloads without port conflicts:

1.  Each worker process binds its internal loopback profiler server to `127.0.0.1:(10000 + local_rank)`.
2.  The host master process launches the `OnDemandXprofProxyServer` listening on external port `9999`.
3.  When an on-demand profile is triggered from the Cloud Console, the proxy intercepts the request, fans out in parallel to all local rank servers, and assigns `override_hostname = f"{base_host}_{rank}"`.
4.  Each rank writes its own trace file (`<hostname>_<rank>.xplane.pb`) to Cloud Storage without file collisions.

### 1. Worker Script Entry Point (`worker_main`)

Define a `worker_main()` function executed by each local TPU rank process to initialize distributed communication and run training. All worker ranks must call `machinelearning_run(...)` (do not wrap it inside `if dist.get_rank() == 0:`)—the SDK internally coordinates registration so only rank 0 contacts the control plane, while non-zero ranks initialize local managers and profiler servers:

```python
import torch
import torch.distributed as dist
import torch.tpu
import torch.tpu.distributed

from google_cloud_mldiagnostics import machinelearning_run
from google_cloud_mldiagnostics import xprof
from google_cloud_mldiagnostics.custom_types.mlrun_types import Framework


def worker_main():
  # 1. Initialize PyTorch Distributed with the TPU backend
  dist.init_process_group(backend="tpu_dist")
  device = torch.accelerator.current_accelerator()

  # 2. Register Machine Learning Run (called on all ranks; do not guard with rank == 0)
  machinelearning_run(
      name="<run_name>",
      run_group="<run_group>",
      project="<project_id>",
      region="us-central1",
      gcs_path="gs://<bucket>/profiles",
      on_demand_xprof=True,
      log_system_metrics=True,
      framework=Framework.PYTORCH,
  )

  # 3. Programmatic profiling session
  prof = xprof()
  prof.start("step_50_trace")

  # ... model training loop ...

  prof.stop()
```

*   `backend="tpu_dist"`: Initializes PyTorch distributed communication via the TPU runtime.
*   `device = torch.accelerator.current_accelerator()`: Automatically resolves the local TPU device assigned to the current process (e.g., `tpu:0`).
*   `machinelearning_run`: **All worker ranks must call `machinelearning_run(...)`** (do not guard with `if dist.get_rank() == 0:`). While the SDK internally skips control-plane registration on non-zero ranks, non-master ranks must still initialize `GlobalRunManager` (otherwise `prof.start()` raises `ProfilingError: No active ML run found for profiling` and hangs rank 0 at `torch.distributed.barrier()`) and start their local loopback profiler servers (`127.0.0.1:(10000 + local_rank)`) when `on_demand_xprof=True`.
*   `framework=Framework.PYTORCH`: Informs the SDK that the workload is running PyTorch, ensuring that process ranks and device topologies are discovered using PyTorch accelerator APIs.
*   `on_demand_xprof=True`: Starts the host-level `OnDemandXprofProxyServer` on port `9999` on the host master process. When an on-demand profile is triggered from the Cloud Console, the proxy multiplexes the request to internal loopback profilers (`127.0.0.1:(10000 + local_rank)`) across all local TPU ranks, writing distinct `<hostname>_<rank>.xplane.pb` traces to Cloud Storage without port or file collisions.
*   `log_system_metrics=True`: Periodically logs TPU duty cycle, TensorCore utilization, and HBM memory usage to Cloud Logging.
*   **Programmatic Profiling Synchronization**: When calling `prof.start()` / `prof.stop()` (or `with xprof():`), the SDK automatically executes `torch.distributed.barrier()` before starting and stopping trace collection. This collective barrier ensures all worker processes begin and end profiling at the exact same execution step, unifying multi-rank traces under a single cohesive session in XProf.

### 2. Multi-Process Elastic Launch (`main`)

On Cloud TPU, workloads launch one worker process per TPU chip (e.g., 4 processes on a `2x2` slice). Coordinate local worker process launching using `torch.distributed.launcher`:

```python
import os
import torch
from torch.distributed import launcher


def main():
  if "LOCAL_RANK" in os.environ:
    worker_main()
  else:
    # Bootstrap TPU launch environment
    torch.tpu.distributed.set_tpu_launch_env()

    nproc_per_node = torch.accelerator.device_count()
    if nproc_per_node == 0:
      raise RuntimeError("No TPU devices found on this node.")

    nnodes = int(os.getenv("NNODES", "1"))
    node_rank = int(os.getenv("NODE_RANK", "0"))
    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "29500")

    config = launcher.LaunchConfig(
        min_nodes=nnodes,
        max_nodes=nnodes,
        nproc_per_node=nproc_per_node,
        run_id="<run_id>",
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0" if nnodes == 1 else f"{master_addr}:{master_port}",
        rdzv_configs={"rank": node_rank} if nnodes > 1 else {},
        max_restarts=0,
    )
    launcher.elastic_launch(config, worker_main)()


if __name__ == "__main__":
  main()
```

*   `torch.tpu.distributed.set_tpu_launch_env()`: Automatically derives `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` from GKE/GCE TPU environment variables.
*   `torch.accelerator.device_count()`: Detects available TPU chips on the host (e.g., 4).
*   `launcher.elastic_launch`: Spawns the worker processes executing `worker_main()`.

### 3. Package Workload with Dockerfile

Package the PyTorch TPU application and dependencies into a container image:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl wget libc++-dev libc++abi-dev libunwind8 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1" \
    ALLOW_MULTIPLE_LIBTPU_LOAD="true"

RUN pip install --no-cache-dir \
    torch \
    libtpu \
    torch_tpu \
    google_cloud_mldiagnostics \
    xprof

COPY train.py ./

ENTRYPOINT ["python3", "train.py"]
```

### 4. Deploy on GKE with JobSet

When configuring your Kubernetes **JobSet** for PyTorch TPU, enable DNS hostnames for inter-pod communication and specify the accelerator nodeSelector and container configurations:

```yaml
spec:
  network:
    enableDNSHostnames: true
  # ...
  nodeSelector:
    cloud.google.com/gke-tpu-accelerator: tpu-v5-lite-podslice
    cloud.google.com/gke-tpu-topology: 2x2
  containers:
    - name: torch-tpu-worker
      image: <IMAGE_URI>
      ports:
        - name: xprof-proxy
          containerPort: 9999
      env:
        - name: TORCH_TPU_TOPOLOGY
          value: "2x2"
      resources:
        limits:
          google.com/tpu: 4
        requests:
          google.com/tpu: 4
```

*   `network.enableDNSHostnames: true`: Enables deterministic pod DNS hostnames, required for distributed worker discovery and multi-host communication.
*   `containerPort: 9999`: Exposes the `OnDemandXprofProxyServer` port for on-demand profiling captures from the Cloud Console.
*   `TORCH_TPU_TOPOLOGY`: Informs the PyTorch TPU runtime of the slice topology (e.g. `2x2`).
*   `google.com/tpu: 4`: Allocates 4 TPU chips to the container.

### 5. Deploy on GCE (TPU VM)

For standalone development directly on a single-host TPU VM:

```bash
# 1. Provision TPU VM
gcloud compute tpus tpu-vm create my-tpu-vm \
    --project=<project_id> \
    --zone=us-central1-a \
    --accelerator-type=v5litepod-4 \
    --version=v2-alpha-tpuv5-lite

# 2. Install dependencies & copy script
gcloud compute tpus tpu-vm ssh my-tpu-vm --zone=us-central1-a \
    --command="pip install --no-cache-dir torch libtpu torch_tpu google_cloud_mldiagnostics xprof"
gcloud compute tpus tpu-vm scp train.py my-tpu-vm:~/train.py --zone=us-central1-a

# 3. Execute workload
gcloud compute tpus tpu-vm ssh my-tpu-vm --zone=us-central1-a \
    --command="python3 train.py"
```

## Using ML Diagnostics with Cluster Toolkit (CTK)

[Google Cloud Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit)
(`gcluster`) natively supports Google Cloud ML Diagnostics on Google Kubernetes
Engine (GKE). When you deploy a GKE cluster using Cluster Toolkit blueprints
(such as
[`gke-tpu-v6e`](https://github.com/GoogleCloudPlatform/cluster-toolkit/tree/main/examples/gke-tpu-v6e)
or
[`gke-tpu-7x`](https://github.com/GoogleCloudPlatform/cluster-toolkit/tree/main/examples/gke-tpu-7x)),
you can enable GKE-managed ML Diagnostics directly in your cluster blueprint,
automating all manual cluster and IAM setup steps described in
[IAM Permissions](#iam-permissions) and
[Configure GKE Cluster](#configure-gke-cluster).

> **Note:** Ensure the
> [Cluster Director API](#enable-cluster-director-api) and
> [Log Analytics on Cloud Logging](#enable-log-analytics-on-cloud-logging) are
> enabled on your project before running workloads.

### What Gets Enabled Automatically

When `enable_ml_diagnostics: true` is set in a Cluster Toolkit GKE blueprint,
Cluster Toolkit automatically provisions and configures:

1.  **GKE-Managed ML Diagnostics Control Plane & Operators**: Configures
    `managed_machine_learning_diagnostics_config { enabled = true }` on the GKE
    cluster, which automatically deploys and manages the in-cluster components
    (`mldiagnostics-injection-webhook`, `mldiagnostics-connection-operator`, and
    `MLDiagnosticsConnection` CRDs) without manual `helm` or `cert-manager`
    installation.
2.  **Automatic Namespace Labeling**: Creates the target workload namespace
    (`vars.user_namespace`) and labels it with
    `managed-mldiagnostics-gke: "true"` so any `JobSet`, `LeaderWorkerSet`
    (LWS), or `RayJob` deployed in that namespace automatically receives pod
    metadata injection.
3.  **Workload Identity & IAM Service Accounts**: Provisions a Google Cloud
    Service Account (default: `gke-wl-sa`) with the required project roles
    (`roles/hypercomputecluster.editor`, `roles/logging.logWriter`,
    `roles/monitoring.metricWriter`, `roles/monitoring.viewer`,
    `roles/stackdriver.resourceMetadata.writer`, `roles/storage.admin`,
    `roles/artifactregistry.reader`, and `roles/container.admin`) and binds it
    via Workload Identity to a Kubernetes Service Account (default:
    `workload-identity-k8s-sa`) in your workload namespace.
4.  **GKE Version Validation**: Enforces a Terraform precondition check
    verifying that the GKE cluster version is `1.35.0-gke.3065000` or higher.

### How to Enable ML Diagnostics in Cluster Toolkit

ML Diagnostics is **enabled by default** across all standard GKE TPU blueprints
in Cluster Toolkit (`gke-tpu-v4`, `gke-tpu-v5e`, `gke-tpu-v5p`, `gke-tpu-v6e`,
and `gke-tpu-7x`, including the `-advanced` variants). For other GKE blueprints
(such as GPU or custom clusters), `enable_ml_diagnostics` defaults to `false` in
the `modules/scheduler/gke-cluster` module.

To enable or customize ML Diagnostics in your Cluster Toolkit blueprint:

#### 1. Configure Blueprint Settings

In your blueprint YAML (for example, `examples/gke-tpu-v6e/gke-tpu-v6e.yaml`),
ensure the `workload_service_account` module includes
`hypercomputecluster.editor` and the `modules/scheduler/gke-cluster` module sets
`enable_ml_diagnostics: true`, `configure_workload_identity_sa: true`, and
`version_prefix: "1.35."`:

```yaml
vars:
  # The namespace to be configured and labeled for running ML workloads
  user_namespace: default

deployment_groups:
- group: primary
  modules:
  - id: workload_service_account
    source: modules/project/service-account
    settings:
      name: gke-wl-sa
      project_roles:
      - logging.logWriter
      - monitoring.metricWriter
      - monitoring.viewer
      - stackdriver.resourceMetadata.writer
      - storage.admin
      - artifactregistry.reader
      - container.admin
      - hypercomputecluster.editor

  - id: gke-tpu-v6e-cluster
    source: modules/scheduler/gke-cluster
    use: [gke-tpu-v6e-net-0, workload_service_account]
    settings:
      configure_workload_identity_sa: true
      namespace: $(vars.user_namespace)
      enable_ml_diagnostics: true
      version_prefix: "1.35."
      release_channel: REGULAR
```

#### 2. Deploy the Cluster

From the root of the
[`cluster-toolkit`](https://github.com/GoogleCloudPlatform/cluster-toolkit)
repository, build `gcluster` and deploy your blueprint:

```bash
test -x ./gcluster || make

./gcluster deploy \
  -d examples/gke-tpu-v6e/gke-tpu-v6e-deployment.yaml \
  examples/gke-tpu-v6e/gke-tpu-v6e.yaml
```

#### 3. Connect to the Cluster and Verify Configuration

Fetch cluster credentials and verify that the workload namespace is labeled with
`managed-mldiagnostics-gke=true` and the `workload-identity-k8s-sa` service
account exists:

```bash
gcloud container clusters get-credentials <DEPLOYMENT_NAME> \
  --region=<REGION> \
  --project=<PROJECT_ID>

kubectl get namespace <user_namespace> --show-labels
kubectl get serviceaccount -n <user_namespace>
```

### Sample Workload Example with Cluster Toolkit

Cluster Toolkit provides a ready-to-run sample workload in
[`examples/gke-tpu-v6e/ml-diagnostics-sample-workload-test`](https://github.com/GoogleCloudPlatform/cluster-toolkit/tree/main/examples/gke-tpu-v6e/ml-diagnostics-sample-workload-test)
to validate your GKE TPU cluster's ML Diagnostics setup:

- [`sample_workload/sample_workload.py`](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/examples/gke-tpu-v6e/ml-diagnostics-sample-workload-test/sample_workload/sample_workload.py):
  Initializes JAX distributed execution, registers a
  [`machinelearning_run`](#creating-a-machine-learning-run) using `PROJECT_ID`,
  `REGION`, and `GCS_PATH` environment variables, captures an
  [`xprof`](#programmatic-profile-capture) trace session, and records
  [model quality and performance metrics](#collect-metrics).
- [`Dockerfile`](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/examples/gke-tpu-v6e/ml-diagnostics-sample-workload-test/Dockerfile):
  Packages `sample_workload.py` with `jax[tpu]`, `libtpu`,
  `google-cloud-mldiagnostics`, and `xprof` (see
  [Package Workload with SDK with Dockerfile for GKE](#package-workload-with-sdk-with-dockerfile-for-gke)).

#### 1. Build and Push the Sample Image

From the sample test directory in your cloned `cluster-toolkit` repository,
create an Artifact Registry repository and build and push the container image:

```bash
export REGION=<region>           # e.g., us-central1
export PROJECT_ID=<project_id>   # Your Google Cloud Project ID
export REPO_NAME=mldiagnostics-tests
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/sample-workload:latest"

gcloud artifacts repositories create ${REPO_NAME} \
    --repository-format=docker \
    --project="${PROJECT_ID}" \
    --location="${REGION}"

docker build -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"
```

#### 2. Configure and Submit `sample_job.yaml`

Update the placeholders in
[`sample_job.yaml`](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/examples/gke-tpu-v6e/ml-diagnostics-sample-workload-test/sample_job.yaml)
to match your Cluster Toolkit blueprint deployment:

- `<user_namespace>`: Workload namespace from your blueprint
  (`vars.user_namespace`, default: `default`).
- `<k8s-service-account-name>`: Workload Identity Kubernetes service account
  created by Cluster Toolkit (default: `workload-identity-k8s-sa`).
- `<tpu_accelerator>` & `<tpu_topology>`: Accelerator and topology labels from
  your blueprint (for example, `tpu-v6e-slice` with `4x4` for TPU v6e, or
  `tpu7x` with `2x2x1` for TPU 7x).
- `<workload_image>`: The `${IMAGE_URI}` pushed in Step 1.
- `PROJECT_ID`, `REGION`, and `GCS_PATH`: Your project ID, cluster region, and
  an existing GCS bucket path (for example,
  `gs://your-ml-diagnostics-bucket/tests`).
- `google.com/tpu`: Chips per node matching your `machine_type` (for example,
  `4`).

```yaml
apiVersion: v1
kind: Service
metadata:
  name: tpu-discovery-service
  namespace: <user_namespace>
spec:
  clusterIP: None
  selector:
    jobset.sigs.k8s.io/jobset-name: sample-tpu-jobset
---
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: sample-tpu-jobset
  namespace: <user_namespace>
  labels:
    diagon-enabled: "true"
    kueue.x-k8s.io/queue-name: user-queue
spec:
  network:
    enableDNSHostnames: true
    subdomain: tpu-discovery-service
  replicatedJobs:
  - name: tpu-slice
    replicas: 1
    template:
      spec:
        parallelism: 1
        completions: 1
        completionMode: Indexed
        backoffLimit: 100
        template:
          spec:
            serviceAccountName: <k8s-service-account-name> # default: workload-identity-k8s-sa
            automountServiceAccountToken: true
            restartPolicy: Never
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: <tpu_accelerator> # e.g., tpu-v6e-slice or tpu7x
              cloud.google.com/gke-tpu-topology: <tpu_topology>       # e.g., 2x2, 4x4, or 2x2x1
            containers:
            - name: jax-job
              image: <workload_image>
              env:
              - name: PROJECT_ID
                value: <project_id>
              - name: REGION
                value: <region>
              - name: GCS_PATH
                value: <gcs_bucket_path> # e.g., "gs://your-ml-diagnostics-bucket/tests"
              ports:
              - containerPort: 8471
              - containerPort: 8431
              securityContext:
                privileged: true
              resources:
                requests:
                  google.com/tpu: 4
                limits:
                  google.com/tpu: 4
```

Submit the job and inspect the pod logs (`sample-tpu-jobset-tpu-slice-...`) to
retrieve the Google Cloud Console `MLRun` URL:

```bash
kubectl apply -f sample_job.yaml
kubectl get all -n <user_namespace>
kubectl logs <pod_name> -n <user_namespace>
```

Once running, open the `MLRun` URL to inspect your workload configs, metrics,
and XProf profiles in the Console as described in
[Viewing Logs, Metrics and Profiles](#viewing-logs-metrics-and-profiles).
