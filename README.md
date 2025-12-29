# Sortformer Python Implementation

Python implementation of NVIDIA's Sortformer v2 streaming speaker diarization model.

## Overview

This repository contains a Python implementation of NVIDIA's Sortformer v2 streaming speaker diarization model. It's a python adaptation of sortformer realization from [parakeet-rs](https://github.com/altunenes/parakeet-rs).

## Features

- Lightweight.
- Supports up to 4 speakers
- Automatic provider detection (CUDA -> TensorRT -> CPU)
- Model downloading from Huggingface
- Model versions supported: v2, v2.1
- Customizable execution provider and device ID
- Configurable log severity
- Command-line interface for easy usage

## Installation

### Package Installation

Install the core package:
CPU-only:
```bash
pip install .[cpu]
```

GPU:
```bash
pip install .[gpu]
```
### CLI Tool Installation

Install with CLI support (includes librosa dependency):

```bash
pip install .[cpu,cli]
```
or
```bash
pip install .[gpu,cli]
```
## Usage

### Python API

```python
from sortformer import Sortformer, DiarizationConfig

# Initialize with model version (automatically downloads if needed)
sortformer = Sortformer(model_version='v2.1')

# Or specify a model directory
sortformer = Sortformer(model_version='v2', model_dir='/path/to/models')

# Customize execution provider and device
sortformer = Sortformer(
    model_version='v2.1',
    provider='cuda',  # or 'tensorrt', 'cpu'
    provider_options={'device_id': 0},
    log_severity=3  # 0=Verbose, 4=Error
)

# Load audio (16kHz mono)
audio = [...]  # List of float32 samples
sample_rate = 16000
channels = 1

# Run diarization
segments = sortformer.diarize(audio, sample_rate, channels)

# Print results
for segment in segments:
    print(f"Speaker {segment.speaker_id}: {segment.start:.2f}s - {segment.end:.2f}s")
```

### Command Line Interface

```bash
# Basic usage
sortformer-cli audio.mp3

# With model version and directory
sortformer-cli audio.mp3 --model v2 --model-dir /path/to/models

# With execution provider
sortformer-cli audio.mp3 --provider cuda --device-id 0

# With log severity
sortformer-cli audio.mp3 --log-severity 3

# JSON output
sortformer-cli audio.mp3 --output json
```

## Configuration

The `DiarizationConfig` class provides pre-tuned configurations:

```python
# CallHome dataset config (default)
config = DiarizationConfig.callhome()

# DIHARD3 dataset config
config = DiarizationConfig.dihard3()

# Custom config
config = DiarizationConfig.custom(onset=0.5, offset=0.4)
```

## License

This implementation is licensed under the same terms as [parakeet-rs](https://github.com/altunenes/parakeet-rs). 
See the LICENSE file for details.

## References

- Original Rust implementation: https://github.com/altunenes/parakeet-rs
- Original parakeet-rs HF repo: https://huggingface.co/altunenes/parakeet-rs
- ONNX model v2: https://huggingface.co/cgus/diar_streaming_sortformer_4spk-v2-onnx
- ONNX model v2.1: https://huggingface.co/cgus/diar_streaming_sortformer_4spk-v2.1-onnx
