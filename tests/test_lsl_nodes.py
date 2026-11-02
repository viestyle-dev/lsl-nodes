import asyncio
import time

import numpy as np
import pytest

from nonoha_core.data_type import DataType
from lsl_nodes.lsl_nodes import LslOutletNode, LslInletNode, LslStreamResolverNode


def test_lsl_outlet_node_creation():
    """Test LslOutletNode creation."""
    node = LslOutletNode("test_outlet")
    assert node.type == "LslOutlet"
    assert "in" in node.inputs


def test_lsl_outlet_node_default_properties():
    """Test LslOutletNode default properties."""
    node = LslOutletNode("test_outlet")

    assert node.properties["stream_name"] == ""
    assert node.properties["stream_type"] == ""
    assert node.properties["channel_count"] == 1
    assert node.properties["sample_rate"] == 0.0
    assert node.properties["channel_format"] == "f32"
    assert node.properties["source_id"] == ""


def test_lsl_outlet_node_input_port_type():
    """Test LslOutletNode input port type matches channel_format and channel_count."""
    node = LslOutletNode("test_outlet")

    # Default: f32, channel_count=1 -> shape (1,)
    input_port = node.inputs["in"]
    assert len(input_port.value_type_list) == 1
    value_type = input_port.value_type_list[0]
    assert value_type.data_type == DataType.Float32
    assert value_type.shape == (1,)

    # Change to f64, channel_count=4 -> shape (4,)
    # Note: stream_name is required to create a valid LSL outlet
    node.set_properties({
        "stream_name": "test_port_type",
        "channel_format": "f64",
        "channel_count": 4,
    })
    input_port = node.inputs["in"]
    value_type = input_port.value_type_list[0]
    assert value_type.data_type == DataType.Float64
    assert value_type.shape == (4,)

    node.finalize()


def test_lsl_outlet_node_set_properties():
    """Test LslOutletNode property setting."""
    node = LslOutletNode("test_outlet")

    node.set_properties(
        {
            "stream_name": "test_stream",
            "stream_type": "EEG",
            "channel_count": 4,
            "sample_rate": 256.0,
            "channel_format": "f32",
            "source_id": "test_source",
        }
    )

    assert node.properties["stream_name"] == "test_stream"
    assert node.properties["stream_type"] == "EEG"
    assert node.properties["channel_count"] == 4
    assert node.properties["sample_rate"] == 256.0
    assert node.properties["channel_format"] == "f32"
    assert node.properties["source_id"] == "test_source"


def test_lsl_outlet_node_dispatch():
    """Test LslOutletNode data dispatch."""
    node = LslOutletNode("test_outlet")
    node.set_properties(
        {
            "stream_name": "test_stream",
            "channel_count": 2,
        }
    )

    # Create test data: 3 samples x 2 channels -> shape (3, 2)
    test_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    # This should not raise an exception
    node.inputs["in"].dispatch([test_data])

    # Cleanup
    node.finalize()


def test_lsl_outlet_node_dispatch_single_channel():
    """Test LslOutletNode data dispatch with single channel."""
    node = LslOutletNode("test_outlet")
    node.set_properties(
        {
            "stream_name": "test_stream",
            "channel_count": 1,
        }
    )

    # Create test data: 5 samples x 1 channel -> shape (5, 1)
    test_data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)

    # This should not raise an exception
    node.inputs["in"].dispatch([test_data])

    # Cleanup
    node.finalize()


def test_lsl_inlet_node_creation():
    """Test LslInletNode creation."""

    async def _test():
        node = LslInletNode("test_inlet")
        assert node.type == "LslInlet"
        assert "out" in node.outputs
        # Now single 'out' port with both timestamp and sample values
        assert len(node.outputs) == 1

    asyncio.run(_test())


def test_lsl_inlet_node_default_properties():
    """Test LslInletNode default properties."""

    async def _test():
        node = LslInletNode("test_inlet")

        assert node.properties["stream_name"] == ""
        assert node.properties["stream_type"] == ""
        assert node.properties["channel_format"] == "f32"
        assert node.properties["channel_count"] == 1
        assert node.properties["pull_interval_ms"] == 10

    asyncio.run(_test())


def test_lsl_inlet_node_output_port_type():
    """Test LslInletNode output port type matches channel_format and channel_count."""

    async def _test():
        node = LslInletNode("test_inlet")

        # Default: f32, channel_count=1
        # Output port has 2 value types: [timestamp (f64), sample (f32, shape=(1,))]
        output_port = node.outputs["out"]
        assert len(output_port.value_type_list) == 2

        # First value type: timestamp (f64)
        timestamp_type = output_port.value_type_list[0]
        assert timestamp_type.data_type == DataType.Float64

        # Second value type: sample (f32, shape=(1,))
        value_type = output_port.value_type_list[1]
        assert value_type.data_type == DataType.Float32
        assert value_type.shape == (1,)

        # Change to f64, channel_count=8 -> shape (8,)
        node.set_properties({"channel_format": "f64", "channel_count": 8})
        output_port = node.outputs["out"]

        # Still 2 value types
        assert len(output_port.value_type_list) == 2

        # Timestamp type unchanged
        timestamp_type = output_port.value_type_list[0]
        assert timestamp_type.data_type == DataType.Float64

        # Sample type changed to f64, shape=(8,)
        value_type = output_port.value_type_list[1]
        assert value_type.data_type == DataType.Float64
        assert value_type.shape == (8,)

        node.finalize()

    asyncio.run(_test())


def test_lsl_inlet_node_set_properties():
    """Test LslInletNode property setting."""

    async def _test():
        node = LslInletNode("test_inlet")

        node.set_properties(
            {
                "stream_name": "test_stream",
                "stream_type": "EEG",
                "channel_format": "f64",
                "channel_count": 8,
                "pull_interval_ms": 20,
            }
        )

        assert node.properties["stream_name"] == "test_stream"
        assert node.properties["stream_type"] == "EEG"
        assert node.properties["channel_format"] == "f64"
        assert node.properties["channel_count"] == 8
        assert node.properties["pull_interval_ms"] == 20

        # Cleanup
        node.finalize()

    asyncio.run(_test())


def test_lsl_stream_resolver_node_creation():
    """Test LslStreamResolverNode creation."""

    async def _test():
        node = LslStreamResolverNode("test_resolver")
        assert node.type == "LslStreamResolver"
        assert "resolve" in node.inputs
        assert "streams" in node.outputs
        node.finalize()

    asyncio.run(_test())


def test_lsl_stream_resolver_node_default_properties():
    """Test LslStreamResolverNode default properties."""

    async def _test():
        node = LslStreamResolverNode("test_resolver")

        # timeout property removed (fixed at 1.0 seconds)
        assert "timeout" not in node.properties
        assert node.properties["stream_name"] == ""
        assert node.properties["stream_type"] == ""
        node.finalize()

    asyncio.run(_test())


def test_lsl_stream_resolver_node_output_port():
    """Test LslStreamResolverNode output port has 7 fields."""

    async def _test():
        node = LslStreamResolverNode("test_resolver")

        output_port = node.outputs["streams"]
        assert len(output_port.value_type_list) == 7

        # Verify field types
        assert output_port.value_type_list[0].data_type == DataType.String  # name
        assert output_port.value_type_list[1].data_type == DataType.String  # type
        assert output_port.value_type_list[2].data_type == DataType.Int32  # channel_count
        assert output_port.value_type_list[3].data_type == DataType.Int32  # channel_format
        assert output_port.value_type_list[4].data_type == DataType.Float64  # nominal_srate
        assert output_port.value_type_list[5].data_type == DataType.String  # source_id
        assert output_port.value_type_list[6].data_type == DataType.String  # hostname

        node.finalize()

    asyncio.run(_test())


def test_lsl_stream_resolver_node_set_properties():
    """Test LslStreamResolverNode property setting."""

    async def _test():
        node = LslStreamResolverNode("test_resolver")

        node.set_properties(
            {
                "stream_name": "test_stream",
                "stream_type": "EEG",
            }
        )

        assert node.properties["stream_name"] == "test_stream"
        assert node.properties["stream_type"] == "EEG"
        node.finalize()

    asyncio.run(_test())


# =============================================================================
# Integration Tests: Outlet <-> Inlet Communication
# =============================================================================


def test_outlet_inlet_connection():
    """Test data transfer from LslOutletNode to LslInletNode."""

    async def _test():
        stream_name = "test_integration_stream"
        channel_count = 2

        # Create outlet
        outlet = LslOutletNode("test_outlet")
        outlet.set_properties(
            {
                "stream_name": stream_name,
                "stream_type": "Test",
                "channel_count": channel_count,
                "sample_rate": 100.0,
            }
        )

        # Create inlet with matching configuration
        inlet = LslInletNode("test_inlet")
        inlet.set_properties(
            {
                "stream_name": stream_name,
                "channel_count": channel_count,
            }
        )

        # Collect received data (now single port with [timestamps, samples])
        received_timestamps: list[np.ndarray] = []
        received_samples: list[np.ndarray] = []

        def on_out(values):
            # values[0] = timestamps, values[1] = samples
            received_timestamps.append(values[0].copy())
            received_samples.append(values[1].copy())

        inlet.outputs["out"].subscribe(id(inlet), on_out)

        try:
            # Start inlet (will resolve stream in background thread)
            inlet.activate()

            # Wait for stream resolution
            await asyncio.sleep(0.5)

            # Send test data
            test_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
            outlet.inputs["in"].dispatch([test_data])

            # Wait for data to be received and process callbacks
            for _ in range(50):  # Max 500ms
                await asyncio.sleep(0.01)
                if len(received_samples) > 0:
                    break

            # Verify data was received
            assert len(received_samples) > 0, "No samples received from inlet"
            assert len(received_timestamps) > 0, "No timestamps received from inlet"

            # Check received data shape and values
            samples = received_samples[0]
            assert samples.shape[1] == channel_count
            assert samples.dtype == np.float32

        finally:
            inlet.finalize()
            outlet.finalize()

    asyncio.run(_test())


def test_outlet_inlet_channel_count_mismatch():
    """Test that inlet rejects connection when channel count mismatches.

    Channel count validation happens at connection time (not per-pull) for
    hot path performance. If mismatch, the inlet will not connect and will
    retry periodically.
    """

    async def _test():
        stream_name = "test_mismatch_stream"

        # Create outlet with 4 channels
        outlet = LslOutletNode("test_outlet")
        outlet.set_properties(
            {
                "stream_name": stream_name,
                "stream_type": "Test",
                "channel_count": 4,
            }
        )

        # Create inlet configured for 2 channels (mismatch!)
        inlet = LslInletNode("test_inlet")
        inlet.set_properties(
            {
                "stream_name": stream_name,
                "channel_count": 2,  # Mismatch with outlet's 4 channels
            }
        )

        # Collect received data
        received_samples: list[np.ndarray] = []

        def on_out(values):
            # values[0] = timestamps, values[1] = samples
            received_samples.append(values[1].copy())

        inlet.outputs["out"].subscribe(id(inlet), on_out)

        try:
            # Start inlet
            inlet.activate()

            # Wait for stream resolution attempt
            # Inlet will find the stream but reject connection due to channel mismatch
            await asyncio.sleep(0.5)

            # Send test data
            test_data = np.array(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
            )
            outlet.inputs["in"].dispatch([test_data])

            # Wait and process callbacks
            for _ in range(30):
                await asyncio.sleep(0.01)

            # Data should NOT be dispatched because inlet rejected connection
            assert len(received_samples) == 0, "Data should not be dispatched on channel mismatch"

        finally:
            inlet.finalize()
            outlet.finalize()

    asyncio.run(_test())


@pytest.mark.parametrize(
    "channel_format,np_dtype",
    [
        ("f32", np.float32),
        ("f64", np.float64),
        ("i32", np.int32),
        ("i16", np.int16),
    ],
)
def test_outlet_inlet_data_types(channel_format: str, np_dtype):
    """Test data transfer with different channel formats."""

    async def _test():
        stream_name = f"test_dtype_{channel_format}_stream"
        channel_count = 2

        # Create outlet
        outlet = LslOutletNode("test_outlet")
        outlet.set_properties(
            {
                "stream_name": stream_name,
                "channel_format": channel_format,
                "channel_count": channel_count,
            }
        )

        # Create inlet with matching configuration
        inlet = LslInletNode("test_inlet")
        inlet.set_properties(
            {
                "stream_name": stream_name,
                "channel_format": channel_format,
                "channel_count": channel_count,
            }
        )

        # Collect received samples
        received_samples: list[np.ndarray] = []

        def on_out(values):
            # values[0] = timestamps, values[1] = samples
            received_samples.append(values[1].copy())

        inlet.outputs["out"].subscribe(id(inlet), on_out)

        try:
            inlet.activate()
            await asyncio.sleep(0.5)

            # Send test data with appropriate dtype
            test_data = np.array([[1, 2], [3, 4]], dtype=np_dtype)
            outlet.inputs["in"].dispatch([test_data])

            # Wait for data
            for _ in range(50):
                await asyncio.sleep(0.01)
                if len(received_samples) > 0:
                    break

            # Verify
            assert len(received_samples) > 0, f"No data received for {channel_format}"
            assert received_samples[0].dtype == np_dtype

        finally:
            inlet.finalize()
            outlet.finalize()

    asyncio.run(_test())


def test_outlet_inlet_value_integrity():
    """Test that sent values exactly match received values.

    This test verifies data integrity through the LSL Outlet -> Inlet pipeline,
    ensuring no data corruption or loss occurs during transmission.
    """

    async def _test():
        stream_name = "test_value_integrity_stream"
        channel_count = 3

        # Create outlet
        outlet = LslOutletNode("test_outlet")
        outlet.set_properties(
            {
                "stream_name": stream_name,
                "stream_type": "Test",
                "channel_count": channel_count,
                "sample_rate": 100.0,
            }
        )

        # Create inlet with matching configuration
        inlet = LslInletNode("test_inlet")
        inlet.set_properties(
            {
                "stream_name": stream_name,
                "channel_count": channel_count,
            }
        )

        # Collect received data
        received_samples: list[np.ndarray] = []

        def on_out(values):
            # values[0] = timestamps, values[1] = samples
            received_samples.append(values[1].copy())

        inlet.outputs["out"].subscribe(id(inlet), on_out)

        # Test data with distinctive values for easy verification
        test_data = np.array(
            [
                [1.5, 2.5, 3.5],
                [10.0, 20.0, 30.0],
                [100.0, 200.0, 300.0],
                [-1.0, -2.0, -3.0],
                [0.123, 0.456, 0.789],
            ],
            dtype=np.float32,
        )

        try:
            inlet.activate()
            await asyncio.sleep(0.5)

            # Send test data
            outlet.inputs["in"].dispatch([test_data])

            # Wait for data to be received
            for _ in range(100):  # Max 1 second
                await asyncio.sleep(0.01)
                if len(received_samples) > 0:
                    # Check if we received all samples
                    total_samples = sum(s.shape[0] for s in received_samples)
                    if total_samples >= test_data.shape[0]:
                        break

            # Verify data was received
            assert len(received_samples) > 0, "No samples received"

            # Concatenate all received chunks
            all_received = np.concatenate(received_samples, axis=0)

            # Verify we received all samples
            assert all_received.shape[0] >= test_data.shape[0], (
                f"Expected at least {test_data.shape[0]} samples, "
                f"got {all_received.shape[0]}"
            )

            # Verify the values match (compare first N samples)
            received_subset = all_received[: test_data.shape[0]]
            np.testing.assert_array_almost_equal(
                received_subset,
                test_data,
                decimal=5,
                err_msg="Sent and received values do not match",
            )

        finally:
            inlet.finalize()
            outlet.finalize()

    asyncio.run(_test())


def test_outlet_inlet_multiple_chunks():
    """Test sending multiple chunks and verifying all values are received correctly."""

    async def _test():
        stream_name = "test_multi_chunk_stream"
        channel_count = 2

        # Create outlet
        outlet = LslOutletNode("test_outlet")
        outlet.set_properties(
            {
                "stream_name": stream_name,
                "stream_type": "Test",
                "channel_count": channel_count,
            }
        )

        # Create inlet with matching configuration
        inlet = LslInletNode("test_inlet")
        inlet.set_properties(
            {
                "stream_name": stream_name,
                "channel_count": channel_count,
                "pull_interval_ms": 5,  # Faster polling for test
            }
        )

        # Collect received data
        received_samples: list[np.ndarray] = []

        def on_out(values):
            received_samples.append(values[1].copy())

        inlet.outputs["out"].subscribe(id(inlet), on_out)

        # Multiple chunks with distinctive values
        chunks = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            np.array([[100.0, 200.0]], dtype=np.float32),
        ]
        expected_total = sum(c.shape[0] for c in chunks)  # 5 samples total

        try:
            inlet.activate()
            await asyncio.sleep(0.5)

            # Send all chunks at once (LSL buffers them)
            for chunk in chunks:
                outlet.inputs["in"].dispatch([chunk])

            # Wait for all data to be received (longer wait, more iterations)
            for _ in range(200):  # Max 2 seconds
                await asyncio.sleep(0.01)
                total_received = sum(s.shape[0] for s in received_samples)
                if total_received >= expected_total:
                    break

            # Verify all data received
            assert len(received_samples) > 0, "No samples received"
            all_received = np.concatenate(received_samples, axis=0)
            expected_all = np.concatenate(chunks, axis=0)

            assert all_received.shape[0] >= expected_total, (
                f"Expected {expected_total} samples, got {all_received.shape[0]}"
            )

            # Verify values match
            np.testing.assert_array_almost_equal(
                all_received[:expected_total],
                expected_all,
                decimal=5,
                err_msg="Multi-chunk sent and received values do not match",
            )

        finally:
            inlet.finalize()
            outlet.finalize()

    asyncio.run(_test())


def test_outlet_dtype_mismatch_casting():
    """Test that LslOutletNode casts input data to configured dtype when mismatched.

    This verifies the safety feature that casts incoming data to the correct dtype
    for LSL transmission, preventing ctypes errors from dtype mismatches.
    """

    async def _test():
        stream_name = "test_dtype_cast_stream"
        channel_count = 2

        # Create outlet configured for f32
        outlet = LslOutletNode("test_outlet")
        outlet.set_properties(
            {
                "stream_name": stream_name,
                "stream_type": "Test",
                "channel_count": channel_count,
                "channel_format": "f32",  # Configured for float32
            }
        )

        # Create inlet also configured for f32
        inlet = LslInletNode("test_inlet")
        inlet.set_properties(
            {
                "stream_name": stream_name,
                "channel_count": channel_count,
                "channel_format": "f32",
            }
        )

        # Collect received data
        received_samples: list[np.ndarray] = []

        def on_out(values):
            received_samples.append(values[1].copy())

        inlet.outputs["out"].subscribe(id(inlet), on_out)

        # Test data with MISMATCHED dtype (f64 instead of f32)
        # The outlet should cast this to f32 before sending
        test_data = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)

        try:
            inlet.activate()
            await asyncio.sleep(0.5)

            # Send f64 data to f32 outlet (should be cast automatically)
            outlet.inputs["in"].dispatch([test_data])

            # Wait for data
            for _ in range(100):
                await asyncio.sleep(0.01)
                if len(received_samples) > 0:
                    break

            # Verify data was received
            assert len(received_samples) > 0, "No samples received"

            # Verify received dtype is f32 (the configured type)
            assert received_samples[0].dtype == np.float32, (
                f"Expected float32, got {received_samples[0].dtype}"
            )

            # Verify values are approximately correct (may have f64->f32 precision loss)
            all_received = np.concatenate(received_samples, axis=0)
            np.testing.assert_array_almost_equal(
                all_received[:test_data.shape[0]],
                test_data.astype(np.float32),  # Compare with expected cast
                decimal=5,
                err_msg="Dtype-cast values do not match expected",
            )

        finally:
            inlet.finalize()
            outlet.finalize()

    asyncio.run(_test())
