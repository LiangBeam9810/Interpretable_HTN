
"""
add baseline wander composed of sinusoidal and Gaussian noise to the ECGs
"""

import multiprocessing as mp
from itertools import repeat
from numbers import Real
from random import randint
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

RNG = np.random.default_rng()

def _gen_gaussian_noise(siglen: int, mean: Real = 0, std: Real = 0) -> np.ndarray:
    """
    generate 1d Gaussian noise of given length, mean, and standard deviation
    Parameters
    ----------
    siglen: int,
        length of the noise signal
    mean: real number, default 0,
        mean of the noise
    std: real number, default 0,
        standard deviation of the noise
    Returns
    -------
    gn: ndarray,
        the gaussian noise of given length, mean, and standard deviation
    """
    gn = RNG.normal(mean, std, siglen)# type: ignore
    return gn


def _gen_sinusoidal_noise(
    siglen: int,
    start_phase: Real,
    end_phase: Real,
    amplitude: Real,
    amplitude_mean: Real = 0, # type: ignore
    amplitude_std: Real = 0, # type: ignore
) -> np.ndarray:
    """
    generate 1d sinusoidal noise of given length, amplitude, start phase, and end phase
    Parameters
    ----------
    siglen: int,
        length of the (noise) signal
    start_phase: real number,
        start phase, with units in degrees
    end_phase: real number,
        end phase, with units in degrees
    amplitude: real number,
        amplitude of the sinusoidal curve
    amplitude_mean: real number,
        mean amplitude of an extra Gaussian noise
    amplitude_std: real number, default 0,
        standard deviation of an extra Gaussian noise
    Returns
    -------
    sn: ndarray,
        the sinusoidal noise of given length, amplitude, start phase, and end phase
    """
    sn = np.linspace(start_phase, end_phase, siglen) # type: ignore
    sn = amplitude * np.sin(np.pi * sn / 180)
    sn += _gen_gaussian_noise(siglen, amplitude_mean, amplitude_std)
    return sn



def _gen_baseline_wander(
    siglen: int,
    fs: Real,
    bw_fs: Union[Real, Sequence[Real]],
    amplitude: Union[Real, Sequence[Real]],
    amplitude_gaussian: Sequence[Real] = [0, 0], # type: ignore
) -> np.ndarray:
    """
    generate 1d baseline wander of given length, amplitude, and frequency
    Parameters
    ----------
    siglen: int,
        length of the (noise) signal
    fs: real number,
        sampling frequency of the original signal
    bw_fs: real number, or list of real numbers,
        frequency (frequencies) of the baseline wander
    amplitude: real number, or list of real numbers,
        amplitude of the baseline wander (corr. to each frequency band)
    amplitude_gaussian: 2-tuple of real number, default [0,0],
        mean and std of amplitude of an extra Gaussian noise
    Returns
    -------
    bw: ndarray,
        the baseline wander of given length, amplitude, frequency
    Example
    -------
    >>> _gen_baseline_wander(4000, 400, [0.4,0.1,0.05], [0.1,0.2,0.4])
    """
    bw = _gen_gaussian_noise(siglen, amplitude_gaussian[0], amplitude_gaussian[1])
    if isinstance(bw_fs, Real):
        _bw_fs = [bw_fs]
    else:
        _bw_fs = bw_fs
    if isinstance(amplitude, Real):
        _amplitude = list(repeat(amplitude, len(_bw_fs)))
    else:
        _amplitude = amplitude
    assert len(_bw_fs) == len(_amplitude)
    duration = siglen / fs
    for bf, a in zip(_bw_fs, _amplitude):
        start_phase =  RNG.integers(0, 360) # type: ignore
        end_phase = duration * bf * 360 + start_phase
        bw += _gen_sinusoidal_noise(siglen, start_phase, end_phase, a, 0, 0)# type: ignore
    return bw


def gen_baseline_wander(
    sig: Tensor,
    fs: Real,
    prob = 0.5,
    bw_fs: Union[Real, Sequence[Real]] = None, # type: ignore
    ampl_ratio: np.ndarray = None, # type: ignore
    gaussian: np.ndarray = None, # type: ignore
) -> np.ndarray:
    """
    generate 1d baseline wander of given length, amplitude, and frequency
    Parameters
    ----------
    sig: Tensor,
        the ECGs to be augmented, of shape (batch, lead, siglen)
    prob: float, default 0.5,
        probability of performing the augmentation
    fs: real number,
        sampling frequency of the original signal
    bw_fs: real number, or list of real numbers,
        frequency (frequencies) of the baseline wander
    ampl_ratio: ndarray, optional,
        candidate ratios of noise amplitdes compared to the original ECGs for each `fs`,
        of shape (m,n)
    gaussian: ndarray, optional,
        candidate mean and std of the Gaussian noises,
        of shape (k, 2)
    Returns
    -------
    bw: ndarray,
        the baseline wander of given length, amplitude, frequency,
        of shape (batch, lead, siglen)
    """
    if(not bw_fs):
        bw_fs = np.array([0.33, 0.1, 0.05, 0.01])# type: ignore    
    if(not ampl_ratio):
        ampl_ratio = np.array(
                        [  # default ampl_ratio
                            [0.01, 0.01, 0.02, 0.03],  # low
                            [0.01, 0.02, 0.04, 0.05],  # low
                            [0.1, 0.06, 0.04, 0.02],  # low
                            [0.02, 0.04, 0.07, 0.1],  # low
                            [0.05, 0.1, 0.16, 0.25],  # medium
                            [0.1, 0.15, 0.25, 0.3],  # high
                            [0.25, 0.25, 0.3, 0.35],  # extremely high
                        ]
                    )
    if(not gaussian):
        gaussian = np.array(
                        [  # default gaussian, mean and std, in terms of ratio
                            [0.0, 0.001],
                            [0.0, 0.003],
                            [0.0, 0.01],
                        ]
                    )

    gaussian = np.concatenate(
                    (
                        np.zeros(
                            (
                                int((1 - prob) * gaussian.shape[0] / prob),
                                gaussian.shape[1],
                            )
                        ),
                        gaussian,
                    )
                )
    ampl_ratio = np.concatenate(
                    (
                        np.zeros(
                            (
                                int((1 - prob) * ampl_ratio.shape[0] / prob),
                                ampl_ratio.shape[1],
                            )
                        ),
                        ampl_ratio,
                    )
                )
    
    batch, lead, siglen = sig.shape
    _n_bw_choices = len(ampl_ratio)
    _n_gn_choices = len(gaussian)

    with mp.Pool(processes=max(1, mp.cpu_count() - 2)) as pool:
        bw = pool.starmap(
            _gen_baseline_wander,
            iterable=[
                (
                    siglen,
                    fs,
                    bw_fs,
                    ampl_ratio[randint(0, _n_bw_choices - 1)],
                    gaussian[randint(0, _n_gn_choices - 1)],
                )
                for i in range(sig.shape[0])
                for j in range(sig.shape[1])
            ],
        )
    bw = torch.as_tensor(np.array(bw), dtype=sig.dtype, device=sig.device).reshape(
        batch, lead, siglen
    )
    return bw # type: ignore

def mark_input(input,mark_lenth=500):
    batchsize,channelsize,sqenlenth = input.shape
    mark = torch.zeros([mark_lenth]).to(input.device)
    for i in range(batchsize):
        mark_index = torch.randint(mark_lenth,sqenlenth-mark_lenth,[1])
        #print(mark_index)
        for j in range(channelsize):
            input[i,j,mark_index:mark_index+mark_lenth]=mark
    return input