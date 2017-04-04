# DT2118, Lab 1 Feature Extraction
import numpy as np
from scipy import signal as s
from scipy import fftpack as fft
import matplotlib.pyplot as plt
import tools as t


# Function given by the exercise ----------------------------------

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)
    ceps = cepstrum(mspec, nceps)
    return t.lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
	"""
Slices the input samples into overlapping windows.

Args:
    winlen: window length in samples.
    winshift: shift of consecutive windows in samples
Returns:
    numpy array [N x winlen], where N is the number of windows that fit
    in the input signal
	"""
	N = int(np.floor(np.size(samples)/(winshift)))
	winlen = int(winlen)
	array_ret = np.zeros((N-1, winlen))
	i = 1
	array_ret[0] = samples[0:winlen]
	for i in range(N-1):
		high = winlen + int(winshift*i)
		low = high-winlen
		array_ret[i,:] = np.array(samples[low:high])
	return array_ret



def preemp(in_sig, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    return s.lfilter([1., -p],1, in_sig) # 1*y = 1*x(t) -p*x(t-1)

def windowing(in_sig):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windowed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    out = np.zeros(in_sig.shape)
    windows = np.zeros(in_sig.shape)

    for index,val in enumerate(in_sig):
    	window = s.hamming(np.size(val), sym=0)
    	diag_window = np.diag(window)
    	out[index] = np.dot(val.T, diag_window)
    	windows[index] = window
    #plot(window)
    return out

def plot(array_plot):
	plt.figure(1)
	plt.pcolormesh(array_plot)
	plt.show()

	return None

def powerSpectrum(in_sig, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    out = np.zeros((in_sig.shape[0],nfft))
    for index,v in enumerate(in_sig):
    	out[index] = np.power(np.abs(fft.fft(v, n=nfft)),2)
    return out


def logMelSpectrum(in_sig, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    mel_filt = t.trfbank(samplingrate, in_sig.shape[1])

    out = np.log(np.dot(in_sig,mel_filt.T))
    #plot(out)

    return out



def cepstrum(in_sig, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    d = fft.realtransforms.dct(in_sig, norm='ortho')[:, :nceps]
    return d

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
