import logging
import threading
import asyncio
from dataclasses import dataclass

import numpy as np
from pylsl import (
    StreamInfo,
    StreamOutlet,
    StreamInlet,
    resolve_streams,
    resolve_bypred,
    cf_float32,
    cf_double64,
    cf_int32,
    cf_int16,
    cf_int8,
    cf_int64,
    cf_string,
    local_clock,
)

from nonoha_core.data_type import (
    DataType,
    PortValueType,
    PropertyDataType,
    name_to_data_type,
)
from nonoha_core.port import (
    PortValue,
    StaticTypeInputPort,
    StaticTypeOutputPort,
)
from nonoha_core.node import (
    BaseNode,
    NodeDoc,
    replace_node_input_port,
    replace_node_output_port,
)

logger = logging.getLogger(__name__)


# DataType to LSL channel format mapping
_DATA_TYPE_TO_LSL_FORMAT = {
    DataType.Float32: cf_float32,
    DataType.Float64: cf_double64,
    DataType.Int32: cf_int32,
    DataType.Int16: cf_int16,
    DataType.Int8: cf_int8,
    DataType.Int64: cf_int64,
    DataType.String: cf_string,
}

# DataType to numpy dtype mapping
_DATA_TYPE_TO_NUMPY_DTYPE = {
    DataType.Float32: np.float32,
    DataType.Float64: np.float64,
    DataType.Int32: np.int32,
    DataType.Int16: np.int16,
    DataType.Int8: np.int8,
    DataType.Int64: np.int64,
}


@dataclass(frozen=True)
class LslInletConfig:
    """Immutable configuration snapshot for LslInletNode pull worker thread."""

    stream_name: str
    stream_type: str
    channel_count: int
    channel_format: int  # LSL format constant (cf_float32, etc.)
    numpy_dtype: type
    max_samples: int
    pull_interval_ms: int


class LslOutletNode(BaseNode):
    """Send data from nonoha to LSL stream."""

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Publish data to the network as an LSL (Lab Streaming Layer) stream. "
                "Other LSL-compatible applications can discover and receive this stream. "
                "Supports configurable channel count, sample rate, and data format. "
                "The outlet is automatically (re)created when properties change."
            ),
            category="LSL",
            inputs={
                "in": "",
            },
            outputs={},
        )

    PROPERTIES = {
        "stream_name": {
            "description": "LSL stream name.",
            "data_type": "str",
            "default_value": "",
        },
        "stream_type": {
            "description": "LSL stream type.",
            "data_type": "str",
            "default_value": "",
        },
        "channel_count": {
            "description": "Number of channels.",
            "data_type": "i32",
            "default_value": 1,
        },
        "sample_rate": {
            "description": "Nominal sample rate (0 = irregular).",
            "data_type": "f64",
            "default_value": 0.0,
        },
        "channel_format": {
            "description": "Data type name (f32, f64, i32, i16, i8, i64, str).",
            "data_type": "str",
            "default_value": "f32",
        },
        "source_id": {
            "description": "Unique source identifier.",
            "data_type": "str",
            "default_value": "",
        },
    }

    def __init__(self, name: str):
        super().__init__(name, "LslOutlet")

        # Default configuration
        channel_format = "f32"
        channel_count = 1

        # Input port - type determined by channel_format and channel_count
        value_type = PortValueType(name_to_data_type(channel_format), (channel_count,))
        input_in = StaticTypeInputPort("in", [value_type])
        input_in.subscribe(id(self), self._on_input_in)

        self.inputs[input_in.name] = input_in
        self._input_in = input_in

        self._stream_name = ""
        self._stream_type = ""
        self._channel_count = channel_count
        self._sample_rate = 0.0  # IRREGULAR_RATE
        self._channel_format = channel_format
        self._source_id = ""

        # Cached numpy dtype for hot path optimization
        self._numpy_dtype = _DATA_TYPE_TO_NUMPY_DTYPE[name_to_data_type(channel_format)]

        # LSL objects (lazy initialization)
        self._info: StreamInfo | None = None
        self._outlet: StreamOutlet | None = None

    def finalize(self):
        """Clean up LSL resources."""
        if self._outlet is not None:
            del self._outlet
            self._outlet = None
        if self._info is not None:
            del self._info
            self._info = None

    def _create_outlet(
        self,
        stream_name: str,
        stream_type: str,
        channel_count: int,
        sample_rate: float,
        data_type: DataType,
        source_id: str,
    ) -> tuple[StreamInfo, StreamOutlet]:
        """Create LSL outlet with specified configuration.

        Returns:
            Tuple of (StreamInfo, StreamOutlet)

        Raises:
            Exception if outlet creation fails
        """
        # Get LSL channel format from DataType
        lsl_format = _DATA_TYPE_TO_LSL_FORMAT[data_type]

        # Create StreamInfo
        info = StreamInfo(
            name=stream_name,
            type=stream_type,
            channel_count=channel_count,
            nominal_srate=sample_rate,
            channel_format=lsl_format,
            source_id=source_id,
        )

        # Create outlet (may raise exception)
        outlet = StreamOutlet(info)

        logger.info(
            f"{self.name}: Created LSL outlet '{stream_name}' "
            f"({channel_count} ch, {data_type.name})"
        )

        return info, outlet

    def set_properties(self, properties: dict[str, PropertyDataType]):
        # 1. Extract and validate properties
        stream_name = self._stream_name
        if "stream_name" in properties:
            stream_name = str(properties["stream_name"])

        stream_type = self._stream_type
        if "stream_type" in properties:
            stream_type = str(properties["stream_type"])

        channel_count = self._channel_count
        if "channel_count" in properties:
            channel_count = int(properties["channel_count"])
            if channel_count < 1:
                raise ValueError("channel_count must be >= 1")

        sample_rate = self._sample_rate
        if "sample_rate" in properties:
            sample_rate = float(properties["sample_rate"])
            if sample_rate < 0:
                raise ValueError("sample_rate must be >= 0")

        channel_format = self._channel_format
        numpy_dtype = self._numpy_dtype
        if "channel_format" in properties:
            channel_format = str(properties["channel_format"])
            try:
                data_type = name_to_data_type(channel_format)
                if data_type not in _DATA_TYPE_TO_LSL_FORMAT:
                    raise KeyError()
                numpy_dtype = _DATA_TYPE_TO_NUMPY_DTYPE[data_type]
            except KeyError:
                raise ValueError(
                    f"Invalid channel_format: {channel_format}. "
                    f"Must be one of: f32, f64, i32, i16, i8, i64, str"
                )

        source_id = self._source_id
        if "source_id" in properties:
            source_id = str(properties["source_id"])

        # 2. Check if outlet needs recreation
        need_recreate = (
            stream_name != self._stream_name
            or stream_type != self._stream_type
            or channel_count != self._channel_count
            or sample_rate != self._sample_rate
            or channel_format != self._channel_format
            or source_id != self._source_id
        )

        # 3. Try to create new outlet BEFORE updating state (fail-fast)
        new_info: StreamInfo | None = None
        new_outlet: StreamOutlet | None = None
        if need_recreate:
            # This may raise an exception - state remains unchanged on failure
            new_info, new_outlet = self._create_outlet(
                stream_name,
                stream_type,
                channel_count,
                sample_rate,
                name_to_data_type(channel_format),
                source_id,
            )

        # 4. Replace input port if channel_format or channel_count changed
        new_input_in = self._input_in
        if (
            channel_format != self._channel_format
            or channel_count != self._channel_count
        ):
            value_type = PortValueType(
                name_to_data_type(channel_format), (channel_count,)
            )
            new_input_in = StaticTypeInputPort("in", [value_type])

        if new_input_in != self._input_in:
            replace_node_input_port(
                self, self._input_in, new_input_in, self._on_input_in
            )
            self._input_in = new_input_in

        # 5. Update instance variables (only after successful outlet creation)
        self._stream_name = stream_name
        self._stream_type = stream_type
        self._channel_count = channel_count
        self._sample_rate = sample_rate
        self._channel_format = channel_format
        self._numpy_dtype = numpy_dtype
        self._source_id = source_id

        # 6. Replace outlet if recreated
        if need_recreate:
            # Clean up old outlet
            if self._outlet is not None:
                del self._outlet
            if self._info is not None:
                del self._info
            # Assign new outlet
            self._info = new_info
            self._outlet = new_outlet

        # 7. Update properties dictionary
        self.properties["stream_name"] = stream_name
        self.properties["stream_type"] = stream_type
        self.properties["channel_count"] = channel_count
        self.properties["sample_rate"] = sample_rate
        self.properties["channel_format"] = channel_format
        self.properties["source_id"] = source_id

        # 8. Notify graph
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

    def _on_input_in(self, values: list[PortValue]):
        if self._outlet is None:
            return

        value_array = values[0]

        try:
            # Cast if dtype or byte order mismatch (dtype comparison includes byte order)
            # astype() produces native byte order, so a single check handles both cases
            if value_array.dtype != self._numpy_dtype:
                value_array = value_array.astype(self._numpy_dtype)

            self._outlet.push_chunk(value_array)
        except Exception as e:
            logger.error(f"{self.name}: Failed to send LSL data: {e}")


class LslInletNode(BaseNode):
    """Receive data from LSL stream into nonoha."""

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Receive data from an LSL (Lab Streaming Layer) stream on the network. "
                "Resolves the stream by name/type, validates channel count and format at connection, "
                "then continuously pulls samples in a background thread. "
                "Outputs timestamped sample chunks. Automatically reconnects on stream loss."
            ),
            category="LSL",
            inputs={},
            outputs={
                "out": "",
            },
        )

    PROPERTIES = {
        "stream_name": {
            "description": "LSL stream name to connect to.",
            "data_type": "str",
            "default_value": "",
        },
        "stream_type": {
            "description": "LSL stream type filter.",
            "data_type": "str",
            "default_value": "",
        },
        "channel_format": {
            "description": "Data type name (f32, f64, i32, i16, i8, i64).",
            "data_type": "str",
            "default_value": "f32",
        },
        "channel_count": {
            "description": "Expected channel count.",
            "data_type": "i32",
            "default_value": 1,
        },
        "pull_interval_ms": {
            "description": "Pull interval in milliseconds.",
            "data_type": "i32",
            "default_value": 10,
        },
    }

    def __init__(self, name: str):
        super().__init__(name, "LslInlet")

        # Default configuration
        channel_format = "f32"
        channel_count = 1

        # Output port - type determined by channel_format and channel_count
        data_type = name_to_data_type(channel_format)
        timestamp_type = PortValueType(DataType.Float64)
        value_type = PortValueType(data_type, (channel_count,))
        output_out = StaticTypeOutputPort(
            "out", [timestamp_type, value_type], ["timestamp", "sample"]
        )

        self.outputs[output_out.name] = output_out
        self._output_out = output_out

        # Configuration state
        self._stream_name = ""
        self._stream_type = ""
        self._channel_format = channel_format
        self._channel_count = channel_count
        self._pull_interval_ms = 10
        self._max_samples = 1024

        # Cached numpy dtype for hot path optimization
        self._numpy_dtype = _DATA_TYPE_TO_NUMPY_DTYPE[data_type]

        # Runtime state (inlet is managed by worker thread locally)
        self._pull_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._main_loop: asyncio.AbstractEventLoop | None = None

    def activate(self):
        self._main_loop = asyncio.get_running_loop()
        # Start pulling if stream_name was already configured via set_properties
        if self._stream_name:
            self._start_pulling()

    def finalize(self):
        """Clean up LSL resources and stop thread."""
        self._stop_pulling()

    def _resolve_and_connect(self, config: LslInletConfig) -> StreamInlet | None:
        """Resolve LSL stream and create inlet.

        Channel count and channel format are verified at connection time only.
        If either doesn't match the configured value, connection is rejected.
        This eliminates per-pull validation overhead in the hot path and prevents
        buffer overflow from format mismatch.

        Args:
            config: Immutable configuration snapshot.

        Returns:
            StreamInlet on success (channel count and format verified), None on failure or mismatch.
        """
        try:
            # Build resolve predicate
            predicates = []
            if config.stream_name:
                predicates.append(f"name='{config.stream_name}'")
            if config.stream_type:
                predicates.append(f"type='{config.stream_type}'")

            pred = " and ".join(predicates) if predicates else ""

            # Resolve streams
            # logger.info(f"{self.name}: Resolving LSL streams (predicate: '{pred}')")
            streams = resolve_bypred(pred, timeout=1.0)

            if not streams:
                # logger.warning(f"{self.name}: No LSL streams found")
                return None

            # Use first matching stream
            stream_info = streams[0]
            actual_channel_count = stream_info.channel_count()
            actual_channel_format = stream_info.channel_format()
            logger.info(
                f"{self.name}: Found stream '{stream_info.name()}' "
                f"({actual_channel_count} ch, format={actual_channel_format}, {stream_info.nominal_srate()} Hz)"
            )

            # Reject connection if channel count mismatch (validated once at connection)
            if actual_channel_count != config.channel_count:
                logger.warning(
                    f"{self.name}: Channel count mismatch - "
                    f"configured: {config.channel_count}, stream: {actual_channel_count}. "
                    f"Connection rejected. Update configuration to match stream."
                )
                return None

            # Reject connection if channel format mismatch (prevents buffer overflow)
            if actual_channel_format != config.channel_format:
                logger.warning(
                    f"{self.name}: Channel format mismatch - "
                    f"configured: {config.channel_format}, stream: {actual_channel_format}. "
                    f"Connection rejected. Update configuration to match stream."
                )
                return None

            # Create inlet (channel count and format guaranteed to match)
            inlet = StreamInlet(stream_info)
            return inlet

        except Exception as e:
            logger.error(f"{self.name}: Failed to resolve LSL stream: {e}")
            return None

    def _start_pulling(self):
        """Start background thread for continuous data pulling."""
        if self._pull_thread is not None and self._pull_thread.is_alive():
            return

        if self._main_loop is None:
            return  # activate() not called yet; will be started from activate()

        # Create immutable config snapshot for worker thread
        config = LslInletConfig(
            stream_name=self._stream_name,
            stream_type=self._stream_type,
            channel_count=self._channel_count,
            channel_format=_DATA_TYPE_TO_LSL_FORMAT[
                name_to_data_type(self._channel_format)
            ],
            numpy_dtype=self._numpy_dtype,
            max_samples=self._max_samples,
            pull_interval_ms=self._pull_interval_ms,
        )

        self._stop_event.clear()
        self._pull_thread = threading.Thread(
            target=self._pull_worker,
            args=(config, self._main_loop),
            daemon=True,
        )
        self._pull_thread.start()
        logger.info(f"{self.name}: Started LSL pull thread")

    def _stop_pulling(self):
        """Stop background pulling thread."""
        self._stop_event.set()
        if self._pull_thread is not None and self._pull_thread.is_alive():
            self._pull_thread.join(timeout=2.0)
        self._pull_thread = None

    def _pull_worker(
        self, config: LslInletConfig, main_loop: asyncio.AbstractEventLoop
    ):
        """Background thread for continuous data pulling.

        Performance optimizations:
        - Channel count validated at connection time only (not per-pull)
        - Pre-allocated sample buffer with dest_obj (avoids Python list creation)

        Args:
            config: Immutable configuration snapshot (thread-safe, no locks needed).
            main_loop: Event loop for dispatching data to main thread.
        """
        # Worker-local state (not shared with main thread)
        inlet: StreamInlet | None = None

        # Pre-allocated buffer for hot path optimization
        # dest_obj avoids pylsl creating Python list objects on each pull
        sample_buffer = np.zeros(
            (config.max_samples, config.channel_count), dtype=config.numpy_dtype
        )

        while not self._stop_event.is_set():
            try:
                # Connect if needed (channel count validated here, not in hot path)
                if inlet is None:
                    inlet = self._resolve_and_connect(config)
                    if inlet is None:
                        self._stop_event.wait(2.0)
                        continue

                # Pull available data into pre-allocated buffer
                # Channel count is guaranteed to match (validated at connection)
                _, timestamps = inlet.pull_chunk(
                    timeout=0.0,
                    max_samples=config.max_samples,
                    dest_obj=sample_buffer,
                )

                num_samples = len(timestamps)
                if num_samples > 0:
                    # Slice view and copy (buffer is reused for next pull)
                    samples = sample_buffer[:num_samples].copy()
                    timestamp_array = np.array(timestamps, dtype=np.float64)

                    # Dispatch on main thread
                    main_loop.call_soon_threadsafe(
                        self._dispatch_out, samples, timestamp_array
                    )
            except Exception as e:
                logger.error(f"{self.name}: Pull error: {e}")
                inlet = None  # Force reconnect

            # Wait for next poll
            self._stop_event.wait(config.pull_interval_ms / 1000.0)

        # Cleanup on exit
        if inlet is not None:
            del inlet

    def _dispatch_out(self, samples: np.ndarray, timestamps: np.ndarray):
        """Dispatch samples on main thread."""
        self._output_out.dispatch([timestamps, samples])

        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()

    def set_properties(self, properties: dict[str, PropertyDataType]):
        # 1. Extract and validate properties
        stream_name = self._stream_name
        if "stream_name" in properties:
            stream_name = str(properties["stream_name"])

        stream_type = self._stream_type
        if "stream_type" in properties:
            stream_type = str(properties["stream_type"])

        channel_format = self._channel_format
        if "channel_format" in properties:
            channel_format = str(properties["channel_format"])
            try:
                data_type = name_to_data_type(channel_format)
                if data_type not in _DATA_TYPE_TO_NUMPY_DTYPE:
                    raise KeyError()
            except KeyError:
                raise ValueError(
                    f"Invalid channel_format: {channel_format}. "
                    f"Must be one of: f32, f64, i32, i16, i8, i64"
                )

        channel_count = self._channel_count
        if "channel_count" in properties:
            channel_count = int(properties["channel_count"])
            if channel_count < 1:
                raise ValueError("channel_count must be >= 1")

        pull_interval_ms = self._pull_interval_ms
        if "pull_interval_ms" in properties:
            pull_interval_ms = int(properties["pull_interval_ms"])
            if pull_interval_ms < 1:
                raise ValueError("pull_interval_ms must be >= 1")

        # 2. Check if worker restart needed (any config change requires restart)
        need_restart = (
            stream_name != self._stream_name
            or stream_type != self._stream_type
            or channel_format != self._channel_format
            or channel_count != self._channel_count
            or pull_interval_ms != self._pull_interval_ms
        )

        # 4. Update cached numpy dtype if channel_format changed
        if channel_format != self._channel_format:
            data_type = name_to_data_type(channel_format)
            self._numpy_dtype = _DATA_TYPE_TO_NUMPY_DTYPE[data_type]

        # 5. Replace output port if channel_format or channel_count changed
        if (
            channel_format != self._channel_format
            or channel_count != self._channel_count
        ):
            timestamp_type = PortValueType(DataType.Float64)
            value_type = PortValueType(
                name_to_data_type(channel_format), (channel_count,)
            )
            new_output_out = StaticTypeOutputPort(
                "out", [timestamp_type, value_type], ["timestamp", "sample"]
            )
            replace_node_output_port(self, self._output_out, new_output_out)
            self._output_out = new_output_out

        # 6. Update instance variables
        self._stream_name = stream_name
        self._stream_type = stream_type
        self._channel_format = channel_format
        self._channel_count = channel_count
        self._pull_interval_ms = pull_interval_ms

        # 7. Update properties dictionary
        self.properties["stream_name"] = stream_name
        self.properties["stream_type"] = stream_type
        self.properties["channel_format"] = channel_format
        self.properties["channel_count"] = channel_count
        self.properties["pull_interval_ms"] = pull_interval_ms

        # 8. Restart worker if it was running
        if need_restart:
            self._stop_pulling()
            self._start_pulling()

        # 9. Notify graph
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)


class LslStreamResolverNode(BaseNode):
    """Discover available LSL streams on the network.

    Uses run_in_executor for non-blocking async execution.
    Fixed 1.0 second timeout (not user-configurable).
    """

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Discover available LSL streams on the network. "
                "Send any value to the 'resolve' input to trigger a scan (1s timeout). "
                "Outputs a table of found streams with name, type, channel count, format, "
                "sample rate, source ID, and hostname. Optionally filter by stream name/type."
            ),
            category="LSL",
            inputs={
                "resolve": "",
            },
            outputs={
                "streams": "",
            },
        )

    PROPERTIES = {
        "stream_name": {
            "description": "Filter by stream name (empty = any).",
            "data_type": "str",
            "default_value": "",
        },
        "stream_type": {
            "description": "Filter by stream type (empty = any).",
            "data_type": "str",
            "default_value": "",
        },
    }

    # Fixed timeout for resolve operations
    _RESOLVE_TIMEOUT = 1.0

    def __init__(self, name: str):
        super().__init__(name, "LslStreamResolver")

        # Configuration
        self._stream_name = ""
        self._stream_type = ""

        # Event loop for async dispatch
        self._main_loop: asyncio.AbstractEventLoop | None = None

        # Cancellation and concurrency control
        self._cancel_event = threading.Event()
        self._resolving = False

        # Input port (trigger)
        self._input_resolve = StaticTypeInputPort(
            "resolve", [PortValueType(DataType.Any)]
        )
        self._input_resolve.subscribe(id(self), self._on_resolve_trigger)

        # Output port (7 fields for stream info)
        self._output_streams = StaticTypeOutputPort(
            "streams",
            [
                PortValueType(DataType.String),  # name
                PortValueType(DataType.String),  # type
                PortValueType(DataType.Int32),  # channel_count
                PortValueType(DataType.Int32),  # channel_format
                PortValueType(DataType.Float64),  # nominal_srate
                PortValueType(DataType.String),  # source_id
                PortValueType(DataType.String),  # hostname
            ],
            [
                "name",
                "type",
                "channel_count",
                "channel_format",
                "nominal_srate",
                "source_id",
                "hostname",
            ],
        )

        self.inputs["resolve"] = self._input_resolve
        self.outputs["streams"] = self._output_streams

    def activate(self):
        self._main_loop = asyncio.get_running_loop()

    def finalize(self):
        """Signal cancellation for any pending resolve."""
        self._cancel_event.set()

    def set_properties(self, properties: dict[str, PropertyDataType]):
        # 1. Extract and validate properties
        stream_name = self._stream_name
        if "stream_name" in properties:
            stream_name = str(properties["stream_name"])

        stream_type = self._stream_type
        if "stream_type" in properties:
            stream_type = str(properties["stream_type"])

        # 2. Update instance variables (after all validation)
        self._stream_name = stream_name
        self._stream_type = stream_type

        # 3. Update properties dictionary
        self.properties["stream_name"] = stream_name
        self.properties["stream_type"] = stream_type

        # 4. Notify graph
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

    def _on_resolve_trigger(self, _values: list[PortValue]):
        """Handle resolve trigger - start async resolve."""
        # Skip if already resolving (concurrent trigger)
        if self._resolving:
            logger.info(f"{self.name}: Resolve already in progress, skipping")
            return

        # Skip if cancelled (finalize called)
        if self._cancel_event.is_set():
            logger.info(f"{self.name}: Cancelled, skipping resolve")
            return

        self._resolving = True

        # Snapshot current config for thread-safe access
        stream_name = self._stream_name
        stream_type = self._stream_type

        # Run resolve in executor (non-blocking)
        if self._main_loop is None:
            return
        self._main_loop.run_in_executor(
            None, self._resolve_worker, stream_name, stream_type
        )

    def _resolve_worker(self, stream_name: str, stream_type: str):
        """Background worker for LSL stream resolution."""
        try:
            # Check cancellation before resolve
            if self._cancel_event.is_set():
                return

            # Build resolve predicate
            predicates = []
            if stream_name:
                predicates.append(f"name='{stream_name}'")
            if stream_type:
                predicates.append(f"type='{stream_type}'")

            pred = " and ".join(predicates) if predicates else ""

            # Resolve streams (blocking call, ~1 second)
            if pred:
                streams = resolve_bypred(pred, minimum=0, timeout=self._RESOLVE_TIMEOUT)
            else:
                streams = resolve_streams(wait_time=self._RESOLVE_TIMEOUT)

            # Check cancellation after resolve
            if self._cancel_event.is_set():
                return

            # Build PortValues directly
            port_values: list[PortValue] = [
                np.array([s.name() for s in streams], dtype=np.dtypes.StringDType()),  # type: ignore
                np.array([s.type() for s in streams], dtype=np.dtypes.StringDType()),  # type: ignore
                np.array([s.channel_count() for s in streams], dtype=np.int32),
                np.array([s.channel_format() for s in streams], dtype=np.int32),
                np.array([s.nominal_srate() for s in streams], dtype=np.float64),
                np.array([s.source_id() for s in streams], dtype=np.dtypes.StringDType()),  # type: ignore
                np.array([s.hostname() for s in streams], dtype=np.dtypes.StringDType()),  # type: ignore
            ]

            # Dispatch on main thread
            if self._main_loop is None:
                return
            self._main_loop.call_soon_threadsafe(self._dispatch_streams, port_values)

            logger.info(f"{self.name}: Found {len(streams)} LSL streams")

        except Exception as e:
            logger.error(f"{self.name}: Failed to resolve streams: {e}")
        finally:
            self._resolving = False

    def _dispatch_streams(self, values: list[PortValue]):
        self._output_streams.dispatch(values)

        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()
