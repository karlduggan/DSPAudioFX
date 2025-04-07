import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Load the input WAV file
signal, Fs = sf.read('/Users/karlduggan/Desktop/Hihat_open_1.wav')

# Check if the signal is stereo or mono
is_stereo = signal.ndim > 1

# Convert to float32 for processing
signal = signal.astype(np.float32)
original_signal = signal.copy()  # Save original for plotting

# Parameters
Sa = 256              # Analysis hop size
N = 2048              # Block length
alpha = 1.0           # Time scaling factor
Ss = int(round(Sa * alpha))  # Synthesis hop size
L = int(256 * alpha / 2)      # Overlap interval

# Zero-pad input signal
M = int(np.ceil(len(signal) / Sa))
pad_length = M * Sa + N - len(signal)
if is_stereo:
    signal = np.pad(signal, ((0, pad_length), (0, 0)), 'constant')
else:
    signal = np.pad(signal, (0, pad_length), 'constant')

# Initialize output buffer
Overlap = signal[:N].copy()

# Main SOLA loop
for ni in range(1, M):
    grain_start = ni * Sa
    grain = signal[grain_start:grain_start + N]

    if len(grain) < N:
        if is_stereo:
            grain = np.pad(grain, ((0, N - len(grain)), (0, 0)), 'constant')
        else:
            grain = np.pad(grain, (0, N - len(grain)), 'constant')

    ref = grain[:L]
    comp_start = ni * Ss
    comp_end = comp_start + L
    if comp_end > len(Overlap):
        break
    comp = Overlap[comp_start:comp_end]

    # Cross-correlation
    if is_stereo:
        xcorr = np.sum([correlate(ref[:, ch], comp[:, ch]) for ch in range(2)], axis=0)
    else:
        xcorr = correlate(ref, comp)

    km = np.argmax(xcorr)
    shift = km - (L - 1)

    overlap_start = comp_start + shift
    if overlap_start + L > len(Overlap):
        break

    Tail = Overlap[overlap_start:overlap_start + L]
    Begin = grain[:L]

    # Crossfade envelopes
    fadeout = np.linspace(1, 0, L)
    fadein = np.linspace(0, 1, L)

    if is_stereo:
        fadeout = np.tile(fadeout[:, None], (1, 2))
        fadein = np.tile(fadein[:, None], (1, 2))
    else:
        fadeout = fadeout[:, None]
        fadein = fadein[:, None]

    Add = Tail * fadeout + Begin * fadein

    before = Overlap[:overlap_start]
    after = grain[L:]

    Overlap = np.concatenate([before, Add, after])

# Clip to [-1.0, 1.0]
Overlap = np.clip(Overlap, -1.0, 1.0)




# Plotting
plt.figure(figsize=(12, 6))

# Original waveform
plt.subplot(2, 1, 1)
if is_stereo:
    plt.plot(original_signal[:, 0], label='Original - Left')
    plt.plot(original_signal[:, 1], label='Original - Right', alpha=0.7)
else:
    plt.plot(original_signal, label='Original')
plt.title("Original Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Time-stretched waveform
plt.subplot(2, 1, 2)
if is_stereo:
    plt.plot(Overlap[:, 0], label='Stretched - Left')
    plt.plot(Overlap[:, 1], label='Stretched - Right', alpha=0.7)
else:
    plt.plot(Overlap, label='Stretched')
plt.title(f"Time-Stretched Waveform (α = {alpha})")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Play the result using PortAudio
print("Playing time-stretched audio...")
sd.play(Overlap, samplerate=Fs)
sd.wait()
print("Done.")

# Optional: Save to file
sf.write('x1_time_stretch.wav', Overlap, Fs)

