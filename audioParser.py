import ffmpeg
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# import scipy.signal as signal
# import scipy.stats as stats

mpl.rcParams['agg.path.chunksize'] = 10000

out, _ = (ffmpeg
    .input("./All Songs/centipede/centipede.egg")
    .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k')
    .overwrite_output()
    .run(capture_stdout=True)
)

amplitudes = np.frombuffer(out, np.int16)

fig = plt.figure()
s = fig.add_subplot(111)
s.plot(amplitudes)
fig.set_figwidth(100)
fig.savefig('amplitude.png')