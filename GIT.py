import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data container for simulation output
# ---------------------------------------------------------------------------
@dataclass
class ClutterResult:
    sigma0_hh_db: float
    sigma0_vv_db: float
    psd_freq_hz: np.ndarray            # Doppler frequency axis
    psd_hh: np.ndarray                 # HH power spectral density (m²/m²/Hz, linear)
    psd_vv: np.ndarray                 # VV power spectral density
    bragg_hh: float                    # Bragg amplitude HH  (m²/m²/Hz)
    bragg_vv: float                    # Bragg amplitude VV
    whitecap: float                    # Whitecap amplitude  (same both pols)
    spike: float                       # Spike amplitude HH  (0 for VV)
    f_bragg_hz: float                  # Bragg Doppler frequency (Hz)
    f_whitecap_hz: float               # Whitecap Doppler frequency (Hz)
    w_bragg_hz: float                  # Bragg bandwidth (Hz)
    w_whitecap_hz: float               # Whitecap bandwidth (Hz)
    w_spike_hz: float                  # Spike bandwidth (Hz)
    iq_hh: np.ndarray                  # Complex IQ time series, HH
    iq_vv: np.ndarray                  # Complex IQ time series, VV
    wind_speed_ms: float
    sea_state: float
    grazing_deg: float
    wind_aspect_deg: float


# ---------------------------------------------------------------------------
# Layer 1: σ0 models
# ---------------------------------------------------------------------------

class _GITModel:
    """GIT sea clutter model."""

    def __init__(self, freq_ghz: float):
        self.freq_ghz = freq_ghz
        self.lam = 0.3 / freq_ghz  # wavelength (m)

    # --- auxiliary ---
    def _aw(self, Vw: float) -> float:
        lam = self.lam
        exp = 1.1 * (lam + 0.015) ** (-0.4)
        return ((1.94 * Vw) / (1.0 + Vw / 15.4)) ** exp

    def _ai(self, psi: float, hav: float) -> float:
        lam = self.lam
        sig = ((14.4 * lam + 5.5) * psi * hav) / lam   # Eq. 9
        s4 = sig ** 4
        return s4 / (1.0 + s4)                           # Eq. 10

    def _au(self, psi: float, phi_rad: float) -> float:
        lam = self.lam
        return np.exp(0.2 * np.cos(phi_rad) * (1.0 - 2.8 * psi) * (lam + 0.015) ** (-0.4))

    # --- public ---
    def sigma0(self, grazing_deg: float, sea_state: float,
               wind_speed_ms: float, wind_aspect_deg: float,
               pol: Literal['HH', 'VV']) -> float:
        """Returns σ0 in dB."""
        lam = self.lam
        psi = np.deg2rad(grazing_deg)
        phi = np.deg2rad(wind_aspect_deg)
        hav = 0.08 * sea_state ** 2            # Eq. 2
        Vw = wind_speed_ms

        Aw = self._aw(Vw)
        Ai = self._ai(psi, hav)
        Au = self._au(psi, phi)

        s0_hh = 3.9e-6 * lam * (psi ** 0.4) * Ai * Au * Aw    # Eq. 12
        s0_hh = max(s0_hh, 1e-20)

        if pol == 'HH':
            return 10.0 * np.log10(s0_hh)

        # VV  — Eq. 13 with Corrigenda (exponent -0.4, ψ+0.0001)
        if self.freq_ghz >= 3.0:
            factor = 9.33 * (hav + 0.015) ** (-0.24) * lam ** 0.25 * (psi + 0.0001) ** 0.29
        else:
            factor = 166.0 * (hav + 0.015) ** (-0.4) * lam ** 0.87 * (psi + 0.0001) ** 0.57
        s0_vv = max(factor * s0_hh, 1e-20)
        return 10.0 * np.log10(s0_vv)
