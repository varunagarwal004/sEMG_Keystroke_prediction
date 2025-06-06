# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torchaudio


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )


@dataclass
class Resample:

    """Resamples the input EMG signal from the original frequency to a new frequency.

    This transform applies an anti-aliasing filter internally via torchaudio.transforms.Resample.
    
    Args:
        orig_freq (int): The original sampling frequency of the signal. (default: 2000 Hz)
        new_freq (int): The desired sampling frequency after resampling. (default: 1000 Hz)
    """
    orig_freq: int = 2000
    new_freq: int = 1000


    def __call__(self, tensor_data: torch.Tensor) -> torch.Tensor:
        # Check if the input has more than one dimension.
       
        #print(tensor_data.shape)
        tensor_data = tensor_data.permute(1,2,0)       
        # If the data is one-dimensional, apply resampling directly.
        resampler = torchaudio.transforms.Resample(orig_freq=self.orig_freq, new_freq=self.new_freq)
        resampled = resampler(tensor_data)
        resampled = resampled.permute(2,0,1)
        #print(resampled.shape)
        return resampled


@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)
'''

@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
    """

    n_fft: int = 64
    hop_length: int = 16

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)

'''


import torch.nn.functional as F





@dataclass
class LogSpectrogram:
    """
    A transform that computes a single 64-point FFT spectrogram, splits it into
    low-frequency (0–500 Hz) and high-frequency (500–1000 Hz) parts, and
    adaptively pools each part to produce 80% (low) and 20% (high) of the final
    frequency bins, respectively. The result is then log-scaled.

    Assumes a sample rate of 2000 Hz.

    Input shape:  (T, ..., C)
    Output shape: (T, ..., C, output_bins)

    Args:
        n_fft (int): Size of the FFT. Defaults to 64.
        hop_length (int): Stride between STFT windows. Defaults to 16.
        output_bins (int): Total number of frequency bins in the final output,
            with 80% coming from 0–500 Hz and 20% from 500–1000 Hz. Defaults to 33.
        sample_rate (int): Sampling rate in Hz. Defaults to 2000.
    """

    n_fft: int = 128
    hop_length: int = 16
    output_bins: int = 65
    sample_rate: int = 2000

    def __post_init__(self) -> None:
        # Create a spectrogram transform with the specified FFT size.
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            center=False,
        )

        # Frequency resolution is about (sample_rate/2) / (n_fft//2).
        # For n_fft=64 and sample_rate=2000, each bin is ~31.25 Hz.
        # So 500 Hz corresponds to bin ~16, we slice up to bin 16 or 17.
        freq_res = (self.sample_rate / 2) / (self.n_fft // 2)  # e.g. 1000 / 32 = 31.25
        self.cutoff_idx = int(300 // freq_res) + 1  # e.g. ~17

        # Decide how many output bins for low and high frequencies.
        self.n_low_out = int(round(0.7 * self.output_bins))
        self.n_high_out = self.output_bins - self.n_low_out

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Expect input shape: (T, ..., C). Move time dimension to the end.
        x = tensor.movedim(0, -1)  # shape: (..., C, T)

        # Compute a single spectrogram with ~33 freq bins for n_fft=64.
        spec = self.spectrogram(x)  # shape: (..., C, 33, T)

        # Split at the cutoff index corresponding to ~500 Hz.
        low_spec = spec[..., :self.cutoff_idx, :]   # 0–500 Hz
        high_spec = spec[..., self.cutoff_idx:, :]  # 500–1000 Hz

        # Adaptive pooling to get n_low_out bins from the low part.
        B, C, F_low, T_ = low_spec.shape
        low_reshaped = low_spec.permute(0, 3, 1, 2).reshape(B * T_, C, F_low)
        low_pooled = F.adaptive_avg_pool1d(low_reshaped, self.n_low_out)
        low_pooled = low_pooled.reshape(B, T_, C, self.n_low_out).permute(0, 2, 3, 1)

        # Adaptive pooling to get n_high_out bins from the high part.
        B2, C2, F_high, T2 = high_spec.shape
        high_reshaped = high_spec.permute(0, 3, 1, 2).reshape(B2 * T2, C2, F_high)
        high_pooled = F.adaptive_avg_pool1d(high_reshaped, self.n_high_out)
        high_pooled = high_pooled.reshape(B2, T2, C2, self.n_high_out).permute(0, 2, 3, 1)

        # Concatenate the low- and high-frequency parts.
        spec_combined = torch.cat([low_pooled, high_pooled], dim=2)  # (..., C, output_bins, T)

        # Apply log scaling (add small constant to avoid log(0)).
        logspec = torch.log10(spec_combined + 1e-6)

        # Move time dimension back to the front: (B, C, output_bins, T) -> (T, B, C, output_bins).
        logspec = logspec.movedim(-1, 0)
        return logspec




@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T, ..., C, freq) -> (..., C, freq, T)
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)
