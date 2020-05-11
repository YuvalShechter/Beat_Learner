import ffmpeg
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
from sklearn.preprocessing import normalize

SAMPLINGRATE = 44100

# Algorithm from: https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520
# Samples is all song data
# Sample Rate = Song Sample Rate
# Window S = ((BPM / 60)^-1)
# Stride Frac = Fraction of Window Size That Composes A Stride (Overlap %)
def spectrogramize(samples, sample_rate, stride_frac = 0.5, 
                          window_s = 0.05, max_freq = 10000, eps = 1e-14):

    window_size = int(sample_rate * window_s)
    stride_size = int(window_size * stride_frac)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, 
                                          shape = nshape, strides = nstrides)
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    
    # TODO: Understand what this does
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    # Cuts off maximum frequency then computes log (eps for no error)
    # Original line: specgram = np.log(fft[:ind, :] + eps)
    # Cutoff is probably not necessary for network?
    specgram = np.log(fft[:, :] + eps)
    return specgram

# Path to song, Song length in seconds
def ffmpegProcessing(songPath, songlengthcutoff):
    # mpl.rcParams['agg.path.chunksize'] = 10000
    print(songPath+"\n")
    out, _ = (ffmpeg
        .input(songPath)
        .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='44.1k')
        .overwrite_output()
        .run(capture_stdout=True)
    )
    # Read in pcm data points
    amplitudes = np.frombuffer(out, np.int16)
    # Calculates song length in seconds
    songLengthInSeconds = len(amplitudes)/SAMPLINGRATE
    # Checks song length is under 8 minutes (arbitrary cutoff) (480 seconds)
    if songLengthInSeconds > songlengthcutoff:
        return []
    # Currently memory stable (actually means 1 - overlap)
    overlap = 0.5
    # Required to be this low for some high freq songs
    windowSeconds = 0.01
    # Log data points in y 
    maxFreqCutoff = 10000
    isError = True
    # Samples, Sample Rate, Stride Factor, Window Size, Maximum Frequency, Epsilon (don't change)
    while(isError):
        try:
            spectrogram = spectrogramize(amplitudes, SAMPLINGRATE, overlap, windowSeconds, maxFreqCutoff)
            secondsPerWindowActual = round(songLengthInSeconds/np.shape(spectrogram)[1], 3)
            if secondsPerWindowActual != 0.005:
                with open("Logs/incorrect_step.log","a+") as incorrectStepF:
                    incorrectStepF.write(songPath+" ["+str(secondsPerWindowActual)+"]")
                    incorrectStepF.write("\n")
            isError = False
        except MemoryError:
            if overlap == 1.0:
                raise MemoryError("Not enough RAM boi")
            overlap+=((1 - overlap)/2)
    
    return spectrogram

[os.remove(os.path.join("Logs",f)) for f in os.listdir("Logs/")]

# converts song to spectrogram to pickle file
# pickle file contains 2d spectrogram array
# "All Songs/Beat_Saber_Dataset/" for actual
# "All Songs/" for testing
songDirectory = sys.argv[1]
# Iterate over each item in directory
for folder in os.listdir(songDirectory):
    # Check if item is directory
    if os.path.isdir(os.path.join(songDirectory,folder)):
        # Iterate over items in directory
        for currFile in os.listdir(os.path.join(songDirectory,folder)):
            # Check if song exists
            if ".egg" in currFile:
                try:
                    postProcess = ffmpegProcessing(os.path.join(songDirectory,folder,currFile),480)
                    if len(postProcess):
                        with open(os.path.join(songDirectory,folder,"spectrogram"), 'wb') as fp:
                            pickle.dump(postProcess, fp)
                except Exception as e:
                    print(e)
                    with open("Logs/error.log","a") as logfile:
                        logfile.write("Error Processing: "+os.path.join(songDirectory,folder,currFile))
                        logfile.write("\n")

# current PID = 22311