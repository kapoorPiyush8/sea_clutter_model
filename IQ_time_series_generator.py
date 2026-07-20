# ---------------------------------------------------------------------------
# Layer 3: Complex IQ time-series generator
# ---------------------------------------------------------------------------

def _generate_iq(psd_power: np.ndarray, n_pulses: int, prf_hz: float,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Generate complex Gaussian clutter IQ samples with spectrum shaped by psd_power.
    Uses the method: colour the spectrum, then IFFT.
    Returns n_pulses complex samples at the given PRF.
    """
    n = n_pulses
    # Interpolate PSD to a uniform [−PRF/2, +PRF/2] grid of length n
    f_uniform = np.fft.fftfreq(n, d=1.0 / prf_hz)
    f_uniform = np.fft.fftshift(f_uniform)

    # PSD is defined on a dense axis; resample via nearest (or interp)
    # Here psd_power is already on f_uniform if caller set up freq axis correctly
    amplitude = np.sqrt(np.maximum(psd_power, 0.0))

    # White complex Gaussian noise
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)

    # Colour in frequency domain
    spectrum = np.fft.fftshift(np.fft.fft(noise)) * amplitude
    iq = np.fft.ifft(np.fft.ifftshift(spectrum))
    return iq
