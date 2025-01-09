# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import firwin, lfilter

# Step 1: Generate a Simple Sinusoidal Signal
# We're going to create a signal sampled at 1000 Hz and at a frequency of 5 Hz.
fs = 1000  # Sampling frequency (1000 samples per second)
t = np.linspace(0, 1, fs, endpoint=False)  # Create a time vector for 1 second
f = 5  # Frequency of the signal (5 Hz)
signal = np.sin(2 * np.pi * f * t)  # Generate the sine wave signal

# Plotting the original signal
plt.figure(figsize=(10, 4))
plt.plot(t, signal)
plt.title('Original Sinusoidal Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Step 2: Analyze the Frequency Content with Fourier Transform
# Now we use FFT to transform the signal from time domain to frequency domain.
fft_result = fft(signal)  # Apply FFT to the signal
fft_freq = fftfreq(len(t), 1/fs)  # Generate corresponding frequencies

# Plotting the frequency spectrum (only positive frequencies)
plt.figure(figsize=(10, 4))
plt.plot(fft_freq[:fs//2], np.abs(fft_result)[:fs//2])  # We take the positive half of the frequencies
plt.title('Frequency Spectrum of the Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

# Step 3: Apply a Low-Pass Filter to the Signal
# We'll design a simple low-pass filter to remove high-frequency noise.
cutoff = 10  # Setting cutoff frequency at 10 Hz
numtaps = 101  # Number of taps for the FIR filter
fir_coeff = firwin(numtaps, cutoff, fs=fs)  # Create the FIR filter

# Applying the filter to the original signal
filtered_signal = lfilter(fir_coeff, 1.0, signal)

# Plot the result of the filtered signal
plt.figure(figsize=(10, 4))
plt.plot(t, filtered_signal)
plt.title('Filtered Signal (Low-Pass Filter Applied)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Step 4: Add Noise to the Signal and Filter It Again
# Now, let's simulate a noisy signal and then filter it to remove the noise.
noise = np.random.normal(0, 0.5, signal.shape)  # Adding random Gaussian noise
noisy_signal = signal + noise  # Combining the original signal with noise

# Apply the filter to the noisy signal
filtered_noisy_signal = lfilter(fir_coeff, 1.0, noisy_signal)

# Plotting both noisy and filtered signals
plt.figure(figsize=(10, 4))
plt.plot(t, noisy_signal, label='Noisy Signal')
plt.plot(t, filtered_noisy_signal, label='Filtered Noisy Signal', linestyle='--')
plt.title('Noisy Signal vs Filtered Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
