import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import time

def main():

    FORMAT = pyaudio.paInt16                # We use 16bit format per sample
    CHANNELS = 1
    RATE = 50000
    CHUNK = 1024                            # 1024bytes of data read from a buffer
    RECORD_SECONDS = 5
    peak = 0

    audio = pyaudio.PyAudio()

    time.sleep(3)
    print("Begin!")

    # start Recording
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    frames = b''.join(frames)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    print("Done!")
    
    fig = plt.figure()
    s = fig.add_subplot(111)
    amplitude = np.fromstring(frames, np.int16)

    s.plot(amplitude[0:20000])

    for i in range(0,len(amplitude)):
        if amplitude[i] < 800:
            amplitude[i] = 0
    
    waitTuple = (False,0)
    for i in range(1,len(amplitude)):
        if i == waitTuple[1]:
            waitTuple = (False,waitTuple[1])
        if not waitTuple[0]:
            if amplitude[i-1] == 0 and amplitude[i] > 0:
                print("N: {0}, y(i-1): {1}, y(i): {2}",i,amplitude[i-1],amplitude[i])
                peak+=1
                waitTuple = (True,i+900)

    print(peak)
    s.plot(amplitude[0:20000])
    fig.savefig('t.png')

main()