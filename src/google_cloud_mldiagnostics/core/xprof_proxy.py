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

"""gRPC proxy server for on-demand XProf profiling in multi-process PyTorch."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import types
from typing import Any

from google_cloud_mldiagnostics.utils import host_utils
import grpc

logger = logging.getLogger(__name__)

PROFILE_METHOD = "/tensorflow.ProfilerService/Profile"
DEFAULT_BASE_INTERNAL_PORT = 10000


def get_internal_port(
    local_rank: int,
    base_port: int = DEFAULT_BASE_INTERNAL_PORT,
) -> int:
  """Returns deterministic internal loopback port for a local rank."""
  return base_port + local_rank


_cached_pb2 = None


def _get_profiler_service_pb2() -> Any:
  """Builds minimal protobuf classes needed for ProfilerService/Profile RPC."""
  global _cached_pb2
  if _cached_pb2 is not None:
    return _cached_pb2

  from google.protobuf import descriptor_pb2  # pylint: disable=g-import-not-at-top
  from google.protobuf import descriptor_pool  # pylint: disable=g-import-not-at-top
  from google.protobuf import message_factory  # pylint: disable=g-import-not-at-top

  file_descriptor_proto_cls = getattr(descriptor_pb2, "FileDescriptorProto")
  field_descriptor_proto_cls = getattr(descriptor_pb2, "FieldDescriptorProto")

  file_proto = file_descriptor_proto_cls()
  file_proto.name = "profiler_service.proto"
  file_proto.package = "tensorflow"

  opts_msg = file_proto.message_type.add()
  opts_msg.name = "ProfileOptions"
  f = opts_msg.field.add()
  f.name = "session_id"
  f.number = 4
  f.type = field_descriptor_proto_cls.TYPE_STRING

  f = opts_msg.field.add()
  f.name = "override_hostname"
  f.number = 15
  f.type = field_descriptor_proto_cls.TYPE_STRING

  req_msg = file_proto.message_type.add()
  req_msg.name = "ProfileRequest"
  f = req_msg.field.add()
  f.name = "duration_ms"
  f.number = 1
  f.type = field_descriptor_proto_cls.TYPE_UINT64

  f = req_msg.field.add()
  f.name = "max_events"
  f.number = 2
  f.type = field_descriptor_proto_cls.TYPE_UINT64

  f = req_msg.field.add()
  f.name = "tools"
  f.number = 3
  f.type = field_descriptor_proto_cls.TYPE_STRING
  f.label = field_descriptor_proto_cls.LABEL_REPEATED

  f = req_msg.field.add()
  f.name = "opts"
  f.number = 4
  f.type = field_descriptor_proto_cls.TYPE_MESSAGE
  f.type_name = "tensorflow.ProfileOptions"

  f = req_msg.field.add()
  f.name = "repository_root"
  f.number = 5
  f.type = field_descriptor_proto_cls.TYPE_STRING

  f = req_msg.field.add()
  f.name = "session_id"
  f.number = 6
  f.type = field_descriptor_proto_cls.TYPE_STRING

  f = req_msg.field.add()
  f.name = "host_name"
  f.number = 7
  f.type = field_descriptor_proto_cls.TYPE_STRING

  tool_msg = file_proto.message_type.add()
  tool_msg.name = "ProfileToolData"
  f = tool_msg.field.add()
  f.name = "name"
  f.number = 1
  f.type = field_descriptor_proto_cls.TYPE_STRING
  f = tool_msg.field.add()
  f.name = "data"
  f.number = 2
  f.type = field_descriptor_proto_cls.TYPE_BYTES

  resp_msg = file_proto.message_type.add()
  resp_msg.name = "ProfileResponse"
  f = resp_msg.field.add()
  f.name = "tool_data"
  f.number = 6
  f.type = field_descriptor_proto_cls.TYPE_MESSAGE
  f.type_name = "tensorflow.ProfileToolData"
  f.label = field_descriptor_proto_cls.LABEL_REPEATED

  f = resp_msg.field.add()
  f.name = "empty_trace"
  f.number = 7
  f.type = field_descriptor_proto_cls.TYPE_BOOL

  pool = descriptor_pool.DescriptorPool()
  pool.Add(file_proto)

  profile_request_cls = message_factory.GetMessageClass(
      pool.FindMessageTypeByName("tensorflow.ProfileRequest")
  )
  profile_response_cls = message_factory.GetMessageClass(
      pool.FindMessageTypeByName("tensorflow.ProfileResponse")
  )

  _cached_pb2 = types.SimpleNamespace(
      ProfileRequest=profile_request_cls,
      ProfileResponse=profile_response_cls,
  )
  return _cached_pb2


class OnDemandXprofProxyHandler(grpc.GenericRpcHandler):
  """gRPC generic handler that intercepts Profile requests and fans out to all local ranks."""

  def __init__(
      self,
      local_world_size: int,
      base_internal_port: int = DEFAULT_BASE_INTERNAL_PORT,
  ):
    self._local_world_size = local_world_size
    self._base_internal_port = base_internal_port
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(4, local_world_size)
    )

  def service(
      self, handler_call_details: grpc.HandlerCallDetails
  ) -> grpc.RpcMethodHandler | None:
    if handler_call_details.method == PROFILE_METHOD:
      pb2 = _get_profiler_service_pb2()
      return grpc.unary_unary_rpc_method_handler(
          self._handle_profile,
          request_deserializer=pb2.ProfileRequest.FromString,
          response_serializer=pb2.ProfileResponse.SerializeToString,
      )
    return None

  def _handle_profile(
      self,
      request: Any,
      context: grpc.ServicerContext,
  ) -> Any:
    """Fans out the ProfileRequest to all local rank C++ profiler servers."""
    pb2 = _get_profiler_service_pb2()

    req_opts = getattr(request, "opts", None)
    raw_host = (
        getattr(req_opts, "override_hostname", None)
        or getattr(request, "host_name", None)
        or host_utils.get_hostname()
    )
    base_host = raw_host.partition(":")[0]

    duration_ms = getattr(request, "duration_ms", 4000) or 4000
    duration_sec = duration_ms / 1000.0
    call_timeout = max(30.0, duration_sec * 2.0 + 15.0)

    logger.info(
        "OnDemandXprofProxy received ProfileRequest for host %r (%d ms, timeout"
        " %.1fs), fanning out to %d local ranks with base hostname %r.",
        request.host_name,
        duration_ms,
        call_timeout,
        self._local_world_size,
        base_host,
    )

    futures = []
    for i in range(self._local_world_size):
      rank_request = pb2.ProfileRequest()
      rank_request.CopyFrom(request)
      rank_hostname = f"{base_host}_{i}"
      rank_request.host_name = rank_hostname
      if hasattr(rank_request, "opts"):
        rank_request.opts.override_hostname = rank_hostname
        req_opts = getattr(request, "opts", None)
        if req_opts and hasattr(req_opts, "session_id") and req_opts.session_id:
          rank_request.opts.session_id = req_opts.session_id

      port = get_internal_port(i, self._base_internal_port)
      target_addr = f"127.0.0.1:{port}"
      fut = self._executor.submit(
          self._call_rank_profile, target_addr, rank_request, call_timeout
      )
      futures.append((i, fut))

    responses = []
    errors = []
    for i, fut in futures:
      try:
        resp = fut.result(timeout=call_timeout + 5.0)
        responses.append((i, resp))
        logger.info(
            "Local rank %d profiling succeeded (empty_trace=%s, tool_data=%d).",
            i,
            getattr(resp, "empty_trace", False),
            len(resp.tool_data) if hasattr(resp, "tool_data") else 0,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Local rank %d profiling failed or timed out: %s. "
            "Gracefully degrading and continuing with remaining healthy ranks.",
            i,
            e,
        )
        errors.append((i, e))

    if not responses:
      err_summary = "; ".join(f"rank {r}: {err}" for r, err in errors)
      logger.error("All local ranks failed profiling: %s", err_summary)
      context.abort(
          grpc.StatusCode.INTERNAL,
          f"Failed to profile any local ranks on host: {err_summary}",
      )

    responses.sort(key=lambda x: x[0])
    consolidated_response = pb2.ProfileResponse()
    all_empty = True
    for _, resp in responses:
      if hasattr(resp, "tool_data"):
        consolidated_response.tool_data.extend(resp.tool_data)
      if hasattr(resp, "empty_trace") and not resp.empty_trace:
        all_empty = False

    if hasattr(consolidated_response, "empty_trace"):
      consolidated_response.empty_trace = all_empty

    if errors:
      logger.warning(
          "Profile completed with graceful degradation: %d/%d ranks succeeded,"
          " %d ranks failed (%s).",
          len(responses),
          self._local_world_size,
          len(errors),
          ", ".join(f"rank {r}" for r, _ in errors),
      )

    return consolidated_response

  def _call_rank_profile(
      self,
      target_addr: str,
      request: Any,
      timeout: float = 60.0,
  ) -> Any:
    """Sends ProfileRequest to a single worker rank on loopback with timeout."""
    pb2 = _get_profiler_service_pb2()
    with grpc.insecure_channel(target_addr) as channel:
      method = channel.unary_unary(
          PROFILE_METHOD,
          request_serializer=pb2.ProfileRequest.SerializeToString,
          response_deserializer=pb2.ProfileResponse.FromString,
      )
      return method(request, timeout=timeout)


class OnDemandXprofProxyServer:
  """Manages the lifecycle of the on-demand XProf gRPC proxy on LOCAL_RANK 0."""

  def __init__(
      self,
      port: int = 9999,
      local_world_size: int = 1,
      base_internal_port: int = DEFAULT_BASE_INTERNAL_PORT,
  ):
    self._port = port
    self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    self._server = grpc.server(self._executor)
    handler = OnDemandXprofProxyHandler(
        local_world_size=local_world_size,
        base_internal_port=base_internal_port,
    )
    self._server.add_generic_rpc_handlers((handler,))
    try:
      self._server.add_insecure_port(f"0.0.0.0:{port}")
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Failed to bind proxy server to 0.0.0.0:%d: %s", port, e)
    try:
      self._server.add_insecure_port(f"[::]:{port}")
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  def start(self) -> None:
    self._server.start()
    logger.info("OnDemandXprofProxyServer started on port %d.", self._port)

  def stop(self, grace: float = 1.0) -> None:
    self._server.stop(grace=grace)
    self._executor.shutdown(wait=False)
    logger.info("OnDemandXprofProxyServer stopped.")
