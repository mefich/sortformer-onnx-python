#!/usr/bin/env python3
"""Command-line interface for Sortformer speaker diarization."""

import argparse
import librosa
import numpy as np
from sortformer import Sortformer, DiarizationConfig
import os
import sys

def format_time_seconds(seconds: float) -> str:
    """Format time in seconds as hh:mm:ss.ss"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Sortformer Speaker Diarization CLI')
    parser.add_argument('audio_file', help='Path to input audio file', nargs='?')
    parser.add_argument('--model', choices=['v2', 'v2.1'], default='v2.1',
                       help='Model version to use (default: v2.1)')
    parser.add_argument('--model-dir', help='Directory to save/download model to')
    parser.add_argument('--provider', help='Execution provider to use (e.g., cuda, tensorrt, cpu, or full provider name)')
    parser.add_argument('--device-id', type=int, default=0, help='Device ID to use (default: 0)')
    parser.add_argument('--log-severity', type=int, choices=[0, 1, 2, 3, 4],
                       help='Log severity level (0=Verbose, 4=Error)')
    parser.add_argument('--output', choices=['text', 'json'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--list-providers', action='store_true',
                       help='List available ONNX execution providers and exit')

    args = parser.parse_args()
    
    # Handle listing available providers
    if args.list_providers:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        print("Available ONNX execution providers:")
        for provider in available_providers:
            print(f"- {provider}")
        sys.exit(0)
    
    # Set log severity if specified
    if args.log_severity is not None:
        import onnxruntime as ort
        ort.set_default_logger_severity(args.log_severity)

    try:
        # Translate provider name
        provider_map = {
            'cuda': 'CUDAExecutionProvider',
            'tensorrt': 'TensorrtExecutionProvider',
            'cpu': 'CPUExecutionProvider'
        }

        # Handle provider specification
        if args.provider:
            # Check if it's a predefined short name
            onnx_provider = provider_map.get(args.provider)
            # If not a predefined name, use as-is (could be full provider name)
            if onnx_provider is None and args.provider not in provider_map.values():
                onnx_provider = args.provider
        else:
            onnx_provider = None

        # Initialize Sortformer
        provider_options = {'device_id': args.device_id} if onnx_provider and 'ExecutionProvider' in onnx_provider else None

        sortformer = Sortformer(
            model_path_or_name=None,  # Not used when model_version is specified
            model_version=args.model,
            model_dir=args.model_dir,
            provider=onnx_provider,
            provider_options=provider_options,
            log_severity=args.log_severity
        )

        # Load audio file
        if not args.audio_file:
            raise ValueError("Audio file is required when not using --list-providers")
        
        audio, sample_rate = librosa.load(args.audio_file, sr=16000, mono=True)
        audio = audio.tolist()  # Convert to list of float32 samples
        channels = 1

        # Run diarization
        segments = sortformer.diarize(audio, sample_rate, channels)

        # Print results
        if args.output == 'text':
            for segment in segments:
                start_sec = segment.start
                end_sec = segment.end
                start_time = format_time_seconds(start_sec)
                end_time = format_time_seconds(end_sec)
                print(f"Speaker {segment.speaker_id}: {start_sec:.2f}s - {end_sec:.2f}s [{start_time} - {end_time}]")
        else:
            # JSON output
            import json
            result = []
            for segment in segments:
                result.append({
                    'speaker_id': segment.speaker_id,
                    'start': segment.start,
                    'end': segment.end,
                    'start_time': format_time_seconds(segment.start),
                    'end_time': format_time_seconds(segment.end)
                })
            print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()