"""
Python implementation of NVIDIA Sortformer v2 Streaming Speaker Diarization.

This module implements NVIDIA's Sortformer v2 streaming model for speaker diarization
in Python, using numpy and onnxruntime for efficient numerical operations and model
inference.

Key features:
- Streaming inference with ~10s chunks (124 frames at 80ms each)
- FIFO buffer for context management
- Smart speaker cache compression (keeps important frames, not just recent)
- Silence profile tracking
- Post-processing: median filtering, hysteresis thresholding
- Supports up to 4 speakers

Reference: https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2
"""
import numpy as np
import onnxruntime as ort
from collections import namedtuple
from typing import List, Optional, Tuple

ort.preload_dlls()

# Model constants
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 128
PREEMPH = 0.97
LOG_ZERO_GUARD = 5.960464478e-8
SAMPLE_RATE = 16000
FMIN = 0.0
FMAX = 8000.0

# Streaming constants
CHUNK_LEN = 124  # Frames per chunk (~10s at 80ms)
FIFO_LEN = 124   # FIFO buffer length
SPKCACHE_LEN = 188  # Speaker cache length
SPKCACHE_UPDATE_PERIOD = 124
SUBSAMPLING = 8  # Audio frames -> model frames
EMB_DIM = 512   # Embedding dimension
NUM_SPEAKERS = 4  # Model supports 4 speakers
FRAME_DURATION = 0.08  # 80ms per frame

# Cache compression params (from NeMo)
SPKCACHE_SIL_FRAMES_PER_SPK = 3
PRED_SCORE_THRESHOLD = 0.25
STRONG_BOOST_RATE = 0.75
WEAK_BOOST_RATE = 1.5
MIN_POS_SCORES_RATE = 0.5
SIL_THRESHOLD = 0.2
MAX_INDEX = 99999

# Speaker segment namedtuple
SpeakerSegment = namedtuple('SpeakerSegment', ['start', 'end', 'speaker_id'])

class DiarizationConfig:
    """Post-processing configuration for speaker diarization."""

    def __init__(self, onset: float, offset: float, pad_onset: float = 0.0,
                 pad_offset: float = 0.0, min_duration_on: float = 0.1,
                 min_duration_off: float = 0.1, median_window: int = 11):
        self.onset = onset
        self.offset = offset
        self.pad_onset = pad_onset
        self.pad_offset = pad_offset
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.median_window = median_window

    @classmethod
    def callhome(cls):
        """CallHome dataset config for v2 (default)."""
        return cls(
            onset=0.641,
            offset=0.561,
            pad_onset=0.229,
            pad_offset=0.079,
            min_duration_on=0.511,
            min_duration_off=0.296,
            median_window=11
        )

    @classmethod
    def dihard3(cls):
        """DIHARD3 dataset config for v2."""
        return cls(
            onset=0.56,
            offset=1.0,
            pad_onset=0.063,
            pad_offset=0.002,
            min_duration_on=0.007,
            min_duration_off=0.151,
            median_window=11
        )

    @classmethod
    def custom(cls, onset: float, offset: float):
        """Create a custom config for fine-tuning diarization behavior."""
        return cls(onset, offset)

class Sortformer:
    """Streaming Sortformer v2 speaker diarization engine."""

    # Supported model versions
    MODEL_VERSIONS = {
        'v2': 'cgus/diar_streaming_sortformer_4spk-v2-onnx',
        'v2.1': 'cgus/diar_streaming_sortformer_4spk-v2.1-onnx'
    }

    def __init__(self, model_path_or_name: str, config: Optional[DiarizationConfig] = None,
                 provider: Optional[str] = None, provider_options: Optional[dict] = None,
                 log_severity: Optional[int] = None, model_dir: Optional[str] = None,
                 model_version: Optional[str] = None):
        """Initialize Sortformer with ONNX model.

        Args:
            model_path_or_name: Path to ONNX model file or model name (e.g., 'v2' or 'v2.1')
            config: Diarization configuration (default: callhome)
            provider: Optional execution provider ('CUDAExecutionProvider', 'TensorrtExecutionProvider', or 'CPUExecutionProvider')
            provider_options: Optional dictionary of provider-specific options (e.g., {'device_id': 0})
            log_severity: Log severity level (default: None - uses ONNX Runtime default)
            model_dir: Optional directory to save/download model to
            model_version: Optional model version ('v2' or 'v2.1')
        """
        # Set log severity if provided
        if log_severity is not None:
            ort.set_default_logger_severity(log_severity)

        # Resolve model path if model name is provided
        if model_version is not None:
            if model_version not in self.MODEL_VERSIONS:
                raise ValueError(f"Unsupported model version: {model_version}. Supported versions: {list(self.MODEL_VERSIONS.keys())}")
            model_name = self.MODEL_VERSIONS[model_version]
            model_path = get_model_path(model_name, model_dir)
        elif model_path_or_name in self.MODEL_VERSIONS.values():
            # If direct model name is provided
            model_name = model_path_or_name
            model_path = get_model_path(model_name, model_dir)
        else:
            # Assume it's a path
            model_path = model_path_or_name

        # Create session options
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # Determine execution providers based on availability and user preference
        available_providers = ort.get_available_providers()

        # Build provider list with fallback strategy
        providers = []
        provider_options_list = []

        # If user specified a provider, use it first
        if provider is not None:
            providers.append(provider)
            if provider_options is not None:
                provider_options_list.append(provider_options)
            else:
                # Default options for GPU providers
                if provider in ['CUDAExecutionProvider', 'TensorrtExecutionProvider']:
                    provider_options_list.append({'device_id': 0})
                else:
                    provider_options_list.append({})
        else:
            # Autodetection: CUDA -> TensorRT -> CPU
            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
                provider_options_list.append({'device_id': 0} if provider_options is None else provider_options)
            if 'TensorrtExecutionProvider' in available_providers:
                providers.append('TensorrtExecutionProvider')
                provider_options_list.append({'device_id': 0} if provider_options is None else provider_options)
            providers.append('CPUExecutionProvider')
            provider_options_list.append({} if provider_options is None else provider_options)

        # Try each provider in order until one works
        last_error = None
        for i, prov in enumerate(providers):
            try:
                if len(providers) == 1:
                    # Only one provider, use as is
                    self.session = ort.InferenceSession(model_path, providers=[prov],
                                                     provider_options=[provider_options_list[i]],
                                                     session_options=options)
                else:
                    # Multiple providers, try this one first
                    self.session = ort.InferenceSession(model_path, providers=[prov],
                                                     provider_options=[provider_options_list[i]],
                                                     session_options=options)
                break
            except Exception as e:
                last_error = e
                continue
        else:
            # If all providers failed, raise the last error
            if last_error:
                raise last_error
            else:
                raise RuntimeError(f"Failed to create inference session with any provider for model: {model_path}")

        self.config = config or DiarizationConfig.callhome()

        # Streaming state
        self.spkcache = np.zeros((1, 0, EMB_DIM), dtype=np.float32)
        self.spkcache_preds = None
        self.fifo = np.zeros((1, 0, EMB_DIM), dtype=np.float32)
        self.fifo_preds = np.zeros((1, 0, NUM_SPEAKERS), dtype=np.float32)
        self.mean_sil_emb = np.zeros((1, EMB_DIM), dtype=np.float32)
        self.n_sil_frames = 0

        # Mel filterbank (cached)
        self.mel_basis = self._create_mel_filterbank()

        # Reset state
        self.reset_state()

    def reset_state(self):
        """Reset streaming state."""
        self.spkcache = np.zeros((1, 0, EMB_DIM), dtype=np.float32)
        self.spkcache_preds = None
        self.fifo = np.zeros((1, 0, EMB_DIM), dtype=np.float32)
        self.fifo_preds = np.zeros((1, 0, NUM_SPEAKERS), dtype=np.float32)
        self.mean_sil_emb = np.zeros((1, EMB_DIM), dtype=np.float32)
        self.n_sil_frames = 0

    def diarize(self, audio: List[float], sample_rate: int, channels: int) -> List[SpeakerSegment]:
        """Main diarization entry point."""
        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sample_rate} Hz")

        # Convert to mono
        if channels > 1:
            audio = [sum(chunk) / channels for chunk in chunks(audio, channels)]

        # Reset state for new audio
        self.reset_state()

        # Extract mel features (B, T, D)
        features = self._extract_mel_features(audio)
        total_frames = features.shape[1]

        # Process in chunks
        chunk_stride = CHUNK_LEN * SUBSAMPLING
        num_chunks = (total_frames + chunk_stride - 1) // chunk_stride

        all_chunk_preds = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_stride
            end = min(start + chunk_stride, total_frames)
            current_len = end - start

            # Extract chunk features
            chunk_feat = features[:, start:end, :]

            # Pad last chunk if needed
            if current_len < chunk_stride:
                padded = np.zeros((1, chunk_stride, N_MELS), dtype=np.float32)
                padded[:, :current_len, :] = chunk_feat
                chunk_feat = padded

            # Run streaming update
            chunk_preds = self._streaming_update(chunk_feat, current_len)
            all_chunk_preds.append(chunk_preds)

        # Concatenate all predictions
        full_preds = self._concat_predictions(all_chunk_preds)

        # Apply median filtering
        if self.config.median_window > 1:
            filtered_preds = self._median_filter(full_preds)
        else:
            filtered_preds = full_preds

        # Binarize to segments
        segments = self._binarize(filtered_preds)

        return segments

    def _streaming_update(self, chunk_feat: np.ndarray, current_len: int) -> np.ndarray:
        """NeMo's streaming_update with smart cache compression."""
        spkcache_len = self.spkcache.shape[1]
        fifo_len = self.fifo.shape[1]

        # Prepare inputs
        chunk_lengths = np.array([current_len], dtype=np.int64)
        spkcache_lengths = np.array([spkcache_len], dtype=np.int64)
        fifo_lengths = np.array([fifo_len], dtype=np.int64)

        # Prepare FIFO input
        fifo_input = self.fifo if fifo_len > 0 else np.zeros((1, 0, EMB_DIM), dtype=np.float32)

        # Prepare spkcache input (may be empty)
        spkcache_input = self.spkcache if spkcache_len > 0 else np.zeros((1, 0, EMB_DIM), dtype=np.float32)

        # Run ONNX inference
        outputs = self.session.run(
            ["spkcache_fifo_chunk_preds", "chunk_pre_encode_embs", "chunk_pre_encode_lengths"],
            {
                "chunk": chunk_feat,
                "chunk_lengths": chunk_lengths,
                "spkcache": spkcache_input,
                "spkcache_lengths": spkcache_lengths,
                "fifo": fifo_input,
                "fifo_lengths": fifo_lengths
            }
        )

        # Extract outputs
        preds = outputs[0]  # spkcache_fifo_chunk_preds
        new_embs = outputs[1]  # chunk_pre_encode_embs

        # Calculate valid frames
        valid_frames = (current_len + SUBSAMPLING - 1) // SUBSAMPLING

        # Extract predictions for different parts
        if fifo_len > 0:
            fifo_preds = preds[:, spkcache_len:spkcache_len + fifo_len, :]
        else:
            fifo_preds = np.zeros((1, 0, NUM_SPEAKERS), dtype=np.float32)

        chunk_preds = preds[:, spkcache_len + fifo_len:spkcache_len + fifo_len + valid_frames, :]
        chunk_embs = new_embs[:, :valid_frames, :]

        # Append chunk embeddings to FIFO
        self.fifo = np.concatenate([self.fifo, chunk_embs], axis=1)

        # Update FIFO predictions
        if fifo_len > 0:
            combined = np.concatenate([fifo_preds, chunk_preds], axis=1)
            self.fifo_preds = combined
        else:
            self.fifo_preds = chunk_preds

        fifo_len_after = self.fifo.shape[1]

        # Move from FIFO to cache when FIFO exceeds limit
        if fifo_len_after > FIFO_LEN:
            pop_out_len = max(SPKCACHE_UPDATE_PERIOD, valid_frames - (FIFO_LEN - fifo_len) + fifo_len)
            pop_out_len = min(pop_out_len, fifo_len_after)

            pop_out_embs = self.fifo[:, :pop_out_len, :]
            pop_out_preds = self.fifo_preds[:, :pop_out_len, :]

            # Update silence profile
            self._update_silence_profile(pop_out_embs, pop_out_preds)

            # Remove from FIFO
            self.fifo = self.fifo[:, pop_out_len:, :]
            self.fifo_preds = self.fifo_preds[:, pop_out_len:, :]

            # Append to cache
            self.spkcache = np.concatenate([self.spkcache, pop_out_embs], axis=1)

            if self.spkcache_preds is not None:
                self.spkcache_preds = np.concatenate([self.spkcache_preds, pop_out_preds], axis=1)

            # Smart compression when cache exceeds limit
            if self.spkcache.shape[1] > SPKCACHE_LEN:
                if self.spkcache_preds is None:
                    # Initialize cache predictions from initial output
                    initial_cache_preds = preds[:, :spkcache_len, :]
                    combined = np.concatenate([initial_cache_preds, pop_out_preds], axis=1)
                    self.spkcache_preds = combined

                # Use smart compression
                self._compress_spkcache()

        return chunk_preds

    def _update_silence_profile(self, embs: np.ndarray, preds: np.ndarray):
        """Update mean silence embedding."""
        preds_2d = preds[0]

        for t in range(preds_2d.shape[0]):
            sum_scores = preds_2d[t].sum()
            if sum_scores < SIL_THRESHOLD:
                # This is a silence frame
                emb = embs[0, t]

                # Update running mean
                old_sum = self.mean_sil_emb[0] * self.n_sil_frames
                self.n_sil_frames += 1
                self.mean_sil_emb[0] = (old_sum + emb) / self.n_sil_frames

    def _compress_spkcache(self):
        """Smart cache compression."""
        if self.spkcache_preds is None:
            return

        cache_preds = self.spkcache_preds
        n_frames = self.spkcache.shape[1]
        spkcache_len_per_spk = SPKCACHE_LEN // NUM_SPEAKERS - SPKCACHE_SIL_FRAMES_PER_SPK
        strong_boost_per_spk = int(spkcache_len_per_spk * STRONG_BOOST_RATE)
        weak_boost_per_spk = int(spkcache_len_per_spk * WEAK_BOOST_RATE)
        min_pos_scores_per_spk = int(spkcache_len_per_spk * MIN_POS_SCORES_RATE)

        # Calculate quality scores
        preds_2d = cache_preds[0]
        scores = self._get_log_pred_scores(preds_2d)

        # Disable low scores
        scores = self._disable_low_scores(preds_2d, scores, min_pos_scores_per_spk)

        # Boost important frames
        scores = self._boost_topk_scores(scores, strong_boost_per_spk, 2.0)
        scores = self._boost_topk_scores(scores, weak_boost_per_spk, 1.0)

        # Add silence frames placeholder
        if SPKCACHE_SIL_FRAMES_PER_SPK > 0:
            padded = np.full((n_frames + SPKCACHE_SIL_FRAMES_PER_SPK, NUM_SPEAKERS), -np.inf)
            padded[:n_frames, :] = scores
            for i in range(n_frames, n_frames + SPKCACHE_SIL_FRAMES_PER_SPK):
                for j in range(NUM_SPEAKERS):
                    padded[i, j] = np.inf
            scores = padded

        # Select top frames
        topk_indices, is_disabled = self._get_topk_indices(scores, n_frames)

        # Gather embeddings
        new_embs, new_preds = self._gather_spkcache(topk_indices, is_disabled)

        self.spkcache = new_embs
        self.spkcache_preds = new_preds

    def _get_log_pred_scores(self, preds: np.ndarray) -> np.ndarray:
        """Calculate quality scores."""
        scores = np.zeros_like(preds)

        for t in range(preds.shape[0]):
            log_1_probs_sum = 0.0
            for s in range(NUM_SPEAKERS):
                p = max(preds[t, s], PRED_SCORE_THRESHOLD)
                log_1_p = np.log(max(1.0 - p, PRED_SCORE_THRESHOLD))
                log_1_probs_sum += log_1_p

            for s in range(NUM_SPEAKERS):
                p = max(preds[t, s], PRED_SCORE_THRESHOLD)
                log_p = np.log(p)
                log_1_p = np.log(max(1.0 - p, PRED_SCORE_THRESHOLD))
                scores[t, s] = log_p - log_1_p + log_1_probs_sum - 0.5 * np.log(2)

        return scores

    def _disable_low_scores(self, preds: np.ndarray, scores: np.ndarray, min_pos_scores_per_spk: int) -> np.ndarray:
        """Disable non-speech and overlapped speech."""
        # Count positive scores per speaker
        pos_count = np.zeros(NUM_SPEAKERS, dtype=int)
        for t in range(scores.shape[0]):
            for s in range(NUM_SPEAKERS):
                if scores[t, s] > 0.0:
                    pos_count[s] += 1

        for t in range(preds.shape[0]):
            for s in range(NUM_SPEAKERS):
                is_speech = preds[t, s] > 0.5

                if not is_speech:
                    scores[t, s] = -np.inf
                else:
                    is_pos = scores[t, s] > 0.0
                    if not is_pos and pos_count[s] >= min_pos_scores_per_spk:
                        scores[t, s] = -np.inf

        return scores

    def _boost_topk_scores(self, scores: np.ndarray, n_boost_per_spk: int, scale_factor: float) -> np.ndarray:
        """Boost top K frames per speaker."""
        for s in range(NUM_SPEAKERS):
            # Get column for this speaker
            col_scores = scores[:, s].copy()

            # Get indices sorted by score descending
            sorted_indices = np.argsort(col_scores)[::-1]

            # Boost top K
            for i in range(min(n_boost_per_spk, len(sorted_indices))):
                t = sorted_indices[i]
                if scores[t, s] != -np.inf:
                    scores[t, s] -= scale_factor * 0.5 * np.log(2)

        return scores

    def _get_topk_indices(self, scores: np.ndarray, n_frames_no_sil: int) -> Tuple[List[int], List[bool]]:
        """Get indices of top frames."""
        n_frames = scores.shape[0]

        # Flatten scores as (S, T) then reshape to (S*T,)
        flat_scores = []
        for s in range(NUM_SPEAKERS):
            for t in range(n_frames):
                flat_scores.append((s * n_frames + t, scores[t, s]))

        # Sort by score descending to get top-K
        flat_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top SPKCACHE_LEN and replace invalid scores with MAX_INDEX
        topk_flat = []
        for idx, score in flat_scores[:SPKCACHE_LEN]:
            if score == -np.inf:
                topk_flat.append(MAX_INDEX)
            else:
                topk_flat.append(idx)

        # Sort flat indices ascending (this puts MAX_INDEX at the end)
        topk_flat.sort()

        # Compute is_disabled and convert to frame indices
        is_disabled = []
        frame_indices = []
        for flat_idx in topk_flat:
            if flat_idx == MAX_INDEX:
                # Invalid entries are disabled
                is_disabled.append(True)
                frame_indices.append(0)  # We set disabled to 0
            else:
                # convert to frame index
                frame_idx = flat_idx % n_frames

                # check if frame is beyond valid range
                if frame_idx >= n_frames_no_sil:
                    is_disabled.append(True)
                    frame_indices.append(0)  # same as above: set disabled to 0
                else:
                    is_disabled.append(False)
                    frame_indices.append(frame_idx)

        return frame_indices, is_disabled

    def _gather_spkcache(self, indices: List[int], is_disabled: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """Gather selected frames."""
        new_embs = np.zeros((1, SPKCACHE_LEN, EMB_DIM), dtype=np.float32)
        new_preds = np.zeros((1, SPKCACHE_LEN, NUM_SPEAKERS), dtype=np.float32)

        cache_preds = self.spkcache_preds[0]

        for i, (idx, disabled) in enumerate(zip(indices, is_disabled)):
            if i >= SPKCACHE_LEN:
                break

            if disabled:
                # Use silence embedding
                new_embs[0, i] = self.mean_sil_emb[0]
                # Predictions stay zero
            elif idx < self.spkcache.shape[1]:
                new_embs[0, i] = self.spkcache[0, idx]
                new_preds[0, i] = cache_preds[idx]

        return new_embs, new_preds

    def _concat_predictions(self, preds_list: List[np.ndarray]) -> np.ndarray:
        """Concatenate predictions along time axis."""
        if not preds_list:
            return np.zeros((0, NUM_SPEAKERS), dtype=np.float32)
        if len(preds_list) == 1:
            return preds_list[0]

        # Concatenate predictions along time axis (axis=1)
        # Each chunk_preds has shape (1, chunk_frames, NUM_SPEAKERS)
        return np.concatenate(preds_list, axis=1)

    def _median_filter(self, preds: np.ndarray) -> np.ndarray:
        """Apply median filter to predictions."""
        window = self.config.median_window
        half = window // 2
        filtered = preds.copy()

        for spk in range(NUM_SPEAKERS):
            for t in range(preds.shape[0]):
                start = max(0, t - half)
                end = min(preds.shape[0], t + half + 1)

                values = preds[start:end, spk]
                values.sort()
                filtered[t, spk] = values[len(values) // 2]

        return filtered

    def _binarize(self, preds: np.ndarray) -> List[SpeakerSegment]:
        """Binarize predictions to segments (padding applied during thresholding)."""
        segments = []

        # preds shape is (1, time, speakers), remove batch dimension
        preds = preds[0]  # Now shape is (time, speakers)
        num_frames = preds.shape[0]

        for spk in range(NUM_SPEAKERS):
            in_seg = False
            seg_start = 0
            temp_segments = []

            for t in range(num_frames):
                # Extract the probability for the current speaker at time t
                p = preds[t, spk]

                if p >= self.config.onset and not in_seg:
                    in_seg = True
                    seg_start = t
                elif p < self.config.offset and in_seg:
                    in_seg = False

                    # Apply padding during conversion
                    start_t = max(0.0, (seg_start * FRAME_DURATION) - self.config.pad_onset)
                    end_t = t * FRAME_DURATION + self.config.pad_offset

                    if end_t - start_t >= self.config.min_duration_on:
                        temp_segments.append(SpeakerSegment(
                            start=start_t,
                            end=end_t,
                            speaker_id=spk
                        ))

            # Handle segment at end
            if in_seg:
                start_t = max(0.0, (seg_start * FRAME_DURATION) - self.config.pad_onset)
                end_t = num_frames * FRAME_DURATION + self.config.pad_offset

                if end_t - start_t >= self.config.min_duration_on:
                    temp_segments.append(SpeakerSegment(
                        start=start_t,
                        end=end_t,
                        speaker_id=spk
                    ))

            # Merge close segments (min_duration_off)
            if len(temp_segments) > 1:
                filtered = [temp_segments[0]]
                for seg in temp_segments[1:]:
                    last = filtered[-1]
                    gap = seg.start - last.end
                    if gap < self.config.min_duration_off:
                        last = SpeakerSegment(
                            start=last.start,
                            end=seg.end,
                            speaker_id=last.speaker_id
                        )
                        filtered[-1] = last
                    else:
                        filtered.append(seg)
                segments.extend(filtered)
            else:
                segments.extend(temp_segments)

        # Sort by start time
        segments.sort(key=lambda x: x.start)
        return segments

    def _apply_preemphasis(self, audio: List[float]) -> List[float]:
        """Apply preemphasis filter."""
        result = [audio[0]]
        for i in range(1, len(audio)):
            result.append(audio[i] - PREEMPH * audio[i - 1])
        return result

    def _hann_window(self, window_length: int) -> List[float]:
        """Create Hann window."""
        import math
        return [0.5 - 0.5 * math.cos(2 * math.pi * i / window_length) for i in range(window_length)]

    def _stft(self, audio: List[float]) -> np.ndarray:
        """Short-time Fourier Transform."""
        import numpy.fft as fft

        # Create Hann window of length win_length, then zero-pad to n_fft (centered)
        hann = self._hann_window(WIN_LENGTH)
        win_offset = (N_FFT - WIN_LENGTH) // 2
        fft_window = np.zeros(N_FFT)
        fft_window[win_offset:win_offset + WIN_LENGTH] = hann

        # Pad signal for center=True (like librosa/torch.stft)
        pad_amount = N_FFT // 2
        padded_audio = [0.0] * pad_amount + audio + [0.0] * pad_amount

        num_frames = (len(padded_audio) - N_FFT) // HOP_LENGTH + 1
        freq_bins = N_FFT // 2 + 1
        spectrogram = np.zeros((freq_bins, num_frames), dtype=np.float32)

        for frame_idx in range(num_frames):
            start = frame_idx * HOP_LENGTH
            frame = np.array([padded_audio[start + i] * fft_window[i] for i in range(N_FFT)], dtype=np.complex64)

            # Compute FFT
            fft_result = fft.fft(frame)
            for k in range(freq_bins):
                # Power spectrum (magnitude^2)
                magnitude = abs(fft_result[k])
                spectrogram[k, frame_idx] = magnitude * magnitude

        return spectrogram

    def _hz_to_mel_slaney(self, hz: float) -> float:
        """Librosa's Slaney mel scale (htk=False, which is the default)."""
        f_min = 0.0
        f_sp = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0

        if hz >= min_log_hz:
            return min_log_mel + np.log(hz / min_log_hz) / logstep
        else:
            return (hz - f_min) / f_sp

    def _mel_to_hz_slaney(self, mel: float) -> float:
        """Convert mel to hz using Slaney scale."""
        f_min = 0.0
        f_sp = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0

        if mel >= min_log_mel:
            return min_log_hz * np.exp(logstep * (mel - min_log_mel))
        else:
            return f_min + f_sp * mel

    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel filterbank."""
        freq_bins = N_FFT // 2 + 1
        filterbank = np.zeros((N_MELS, freq_bins), dtype=np.float32)

        # FFT frequencies
        fftfreqs = [k * SAMPLE_RATE / N_FFT for k in range(freq_bins)]

        # Mel center frequencies using Slaney scale (librosa default, htk=False)
        fmin_mel = self._hz_to_mel_slaney(FMIN)
        fmax_mel = self._hz_to_mel_slaney(FMAX)
        mel_f = [fmin_mel + (fmax_mel - fmin_mel) * i / (N_MELS + 1) for i in range(N_MELS + 2)]
        mel_f = [self._mel_to_hz_slaney(m) for m in mel_f]

        # Differences between consecutive mel frequencies
        fdiff = [mel_f[i + 1] - mel_f[i] for i in range(len(mel_f) - 1)]

        # Compute filterbank weights
        for i in range(N_MELS):
            for k in range(freq_bins):
                # Lower slope
                lower = (fftfreqs[k] - mel_f[i]) / fdiff[i]
                # Upper slope
                upper = (mel_f[i + 2] - fftfreqs[k]) / fdiff[i + 1]
                # Weight is max(0, min(lower, upper))
                filterbank[i, k] = max(0.0, min(lower, upper))

        # Apply Slaney normalization
        for i in range(N_MELS):
            enorm = 2.0 / (mel_f[i + 2] - mel_f[i])
            for k in range(freq_bins):
                filterbank[i, k] *= enorm

        return filterbank

    def _extract_mel_features(self, audio: List[float]) -> np.ndarray:
        """Extract mel features from audio."""
        # 1. Add dither (small random noise to prevent log(0))
        # NeMo uses dither=1e-5, but for determinism we skip random noise
        # The log_zero_guard handles zero values

        # 2. Apply preemphasis (NeMo uses preemph=0.97)
        preemphasized = self._apply_preemphasis(audio)

        # 3. STFT
        spectrogram = self._stft(preemphasized)

        # 4. Apply mel filterbank (with Slaney normalization)
        mel_spec = np.dot(self.mel_basis, spectrogram)

        # 5. Log with guard value (NeMo uses log_zero_guard_value = 2^-24)
        # NeMo uses normalize='NA' which means NO normalization
        log_mel_spec = np.log(mel_spec + LOG_ZERO_GUARD)

        # Transpose to (batch, time, features) - NeMo outputs (B, D, T), model expects (B, T, D)
        num_frames = log_mel_spec.shape[1]
        features = np.zeros((1, num_frames, N_MELS), dtype=np.float32)

        for t in range(num_frames):
            for m in range(N_MELS):
                features[0, t, m] = log_mel_spec[m, t]

        return features

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def download_model(model_name: str, model_dir: Optional[str] = None) -> str:
    """Download a model from Huggingface if it doesn't exist locally.

    Args:
        model_name: Name of the model to download (e.g., 'cgus/diar_streaming_sortformer_4spk-v2-onnx')
        model_dir: Optional directory to save the model to. If None, uses Huggingface cache.

    Returns:
        Path to the downloaded model file.
    """
    import os
    from huggingface_hub import hf_hub_download, try_to_load_from_cache, _CACHED_NO_EXIST

    # Model file name based on model version
    if 'v2.1' in model_name:
        model_file = 'diar_streaming_sortformer_4spk-v2.1.onnx'
    else:
        model_file = 'diar_streaming_sortformer_4spk-v2.onnx'

    # If model_dir is specified, check there first
    if model_dir:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            return model_path

    # Try to load from cache first
    cache_dir = model_dir if model_dir else None
    cached_path = try_to_load_from_cache(
        repo_id=model_name,
        filename=model_file,
        cache_dir=cache_dir
    )

    if isinstance(cached_path, str):
        # File exists in cache
        return cached_path
    elif cached_path is _CACHED_NO_EXIST:
        # File doesn't exist at the given commit hash
        raise RuntimeError(f"Model file {model_file} does not exist in repo {model_name}")
    else:
        # File is not cached, need to download
        try:
            downloaded_path = hf_hub_download(
                repo_id=model_name,
                filename=model_file,
                cache_dir=cache_dir
            )
            return downloaded_path
        except Exception as e:
            raise RuntimeError(f"Failed to download model {model_name}: {e}")

def get_model_path(model_name: str, model_dir: Optional[str] = None) -> str:
    """Get the path to a model, downloading it if necessary.

    Args:
        model_name: Name of the model (e.g., 'cgus/diar_streaming_sortformer_4spk-v2-onnx')
        model_dir: Optional directory to save/download the model to

    Returns:
        Path to the model file
    """
    import os
    from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

    # Model file name based on model version
    if 'v2.1' in model_name:
        model_file = 'diar_streaming_sortformer_4spk-v2.1.onnx'
    else:
        model_file = 'diar_streaming_sortformer_4spk-v2.onnx'

    # If model_dir is specified, check there first
    if model_dir:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            return model_path

    # Try to load from cache first
    cache_dir = model_dir if model_dir else None
    cached_path = try_to_load_from_cache(
        repo_id=model_name,
        filename=model_file,
        cache_dir=cache_dir
    )

    if isinstance(cached_path, str):
        # File exists in cache
        if model_dir and cached_path != os.path.join(model_dir, model_file):
            # If we're using a custom model_dir, copy to that location
            os.makedirs(model_dir, exist_ok=True)
            import shutil
            shutil.copy2(cached_path, os.path.join(model_dir, model_file))
            return os.path.join(model_dir, model_file)
        return cached_path
    elif cached_path is _CACHED_NO_EXIST:
        # File doesn't exist at the given commit hash
        raise RuntimeError(f"Model file {model_file} does not exist in repo {model_name}")
    else:
        # File is not cached, need to download
        return download_model(model_name, model_dir)
