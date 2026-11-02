# lsl-nodes

A nonoha-engine plugin for Lab Streaming Layer (LSL) communication, enabling bidirectional data streaming between nonoha-engine and LSL-compatible applications (EEG devices, BCI systems, physiological sensors, etc.).

## Overview

This plugin provides nodes for sending and receiving LSL streams within the nonoha-engine visual programming environment. It enables real-time communication with neuroscience research tools, brain-computer interfaces, and other LSL-compatible software.

## Features

### Nodes Provided

- **LslOutlet**: Sends data from nonoha to external LSL consumers
  - Configurable stream name, type, channel count, and sample rate
  - Supports various data formats (f32, f64, i32, i16, i8, i64)
  - Efficient chunk-based transmission

- **LslInlet**: Receives data from external LSL streams
  - Automatic stream discovery and connection
  - Background thread for continuous data pulling
  - Configurable polling interval

- **LslStreamResolver**: Discovers available LSL streams on the network
  - Filter by stream name and type
  - Outputs stream information as parallel arrays (name, type, channel_count, etc.)

## Installation

### From the Repository

```sh
pip install git+https://github.com/viestyle-dev/lsl-nodes.git
```

Requires Python 3.11+. `nonoha-core` is provided by the host runtime (the
nonoha engine), so it is not a runtime dependency of this package.

### As a Dependency in nonoha-engine Workspace

1. Add `lsl-nodes` to your workspace's `pyproject.toml`:

   ```toml
   [project]
   dependencies = [
       "lsl-nodes",
       # other dependencies...
   ]

   [tool.uv.sources]
   lsl-nodes = { path = "./plugins/lsl-nodes", editable = false }
   ```

2. Install dependencies using uv:

   ```bash
   uv sync
   ```

### As a Built-in Plugin

To include lsl-nodes as a built-in plugin in your nonoha-engine instance:

1. Import the plugin in your `src/main.py`:

   ```python
   import lsl_nodes
   ```

2. Add to RuntimeConfig:

   ```python
   from nonoha_engine.runtime_config import RuntimeConfig

   config = RuntimeConfig(
       builtin_plugins=[
           lsl_nodes,
           # other built-in plugins...
       ]
   )
   ```

## Usage

Once installed, the LSL nodes will appear in the nonoha-engine node palette. You can:

1. Drag and drop LSL nodes into your visual programming workspace
2. Configure node properties (stream name, type, channel count, data format)
3. Connect data flows between LSL nodes and other processing nodes
4. Run the graph to enable real-time LSL communication

### LslOutlet Properties

| Property | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| stream_name | string | "nonoha" | LSL stream name |
| stream_type | string | "Data" | Content type (EEG, Audio, etc.) |
| channel_count | int | 1 | Number of channels |
| sample_rate | float | 0.0 | Sampling rate (0 = irregular) |
| channel_format | string | "f32" | Data format (f32, f64, i32, i16, i8, i64) |
| source_id | string | "" | Unique source identifier |

### LslInlet Properties

| Property | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| stream_name | string | "" | Filter by name (empty = any) |
| stream_type | string | "" | Filter by type (empty = any) |
| channel_format | string | "f32" | Output data format (f32, f64, i32, i16, i8, i64) |
| channel_count | int | 1 | Number of channels (must match stream) |
| pull_interval_ms | int | 10 | Polling interval in milliseconds |

**Output port `out`:**

- `timestamp`: Float64 array of LSL timestamps
- `sample`: Array of samples (shape: [num_samples, channel_count])

### LslStreamResolver Properties

| Property | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| stream_name | string | "" | Filter by name (empty = any) |
| stream_type | string | "" | Filter by type (empty = any) |

**Note:** Resolution timeout is fixed at 1.0 second.

**Output port `streams`:**

- `name`: String array of stream names
- `type`: String array of stream types
- `channel_count`: Int32 array of channel counts
- `channel_format`: Int32 array of LSL format constants
- `nominal_srate`: Float64 array of nominal sample rates
- `source_id`: String array of source IDs
- `hostname`: String array of hostnames

## LSL Specification Notes

- **Single data type per stream**: LSL requires all data within a stream to have the same type. Use multiple nodes for different data types.
- **Timestamp synchronization**: LSL provides sub-millisecond timestamp accuracy for synchronizing multiple streams.

## Requirements

- Python >= 3.11
- nonoha-core (nonoha-engine core library)
- pylsl (Lab Streaming Layer Python bindings)
- numpy (array processing)

## Development

To contribute or modify the plugin:

```bash
uv sync        # also installs the dev dependency group (pytest, nonoha-core)
uv run pytest
```

## References

- [Lab Streaming Layer Documentation](https://labstreaminglayer.readthedocs.io/)
- [pylsl GitHub Repository](https://github.com/chkothe/pylsl)

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
Pull requests require signing the [nonoha CLA](CLA.md) (a one-time
comment on your first PR, shared across the nonoha repositories).

## Security

See [SECURITY.md](SECURITY.md) for how to report vulnerabilities.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
