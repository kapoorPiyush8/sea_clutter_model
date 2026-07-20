# ---------------------------------------------------------------------------
# Layer 2: Doppler PSD (Walker 3-component + Rozenburg scaling)
# ---------------------------------------------------------------------------

class _DopplerModel:
    """
    Walker 3-component Doppler model (DSTO-TR-2864 Sections 3-4).
    Frequency/bandwidth from Rozenburg's wavetank data (Section 3.2),
    scaled to radar wavelength and to open-ocean via Section 4.3 factors.
    Amplitude extraction: Method 1 (HH-first) or Method 2 (VV-first).
    """

    # Ocean scaling factors (Section 4.3)
    SCALE_BRAGG_F   = 1.3
    SCALE_WHITECAP_F_UP   = 2.7
    SCALE_WHITECAP_F_DOWN = 2.3
    SCALE_BRAGG_W   = 1.3
    SCALE_WHITECAP_W = 2.7
    WAVETANK_LAM = 0.021   # Rozenburg: 14 GHz → λ = 0.021 m

    def __init__(self, lam: float, method: Literal[1, 2] = 1):
        self.lam = lam
        self.method = method
        self._scale = self.WAVETANK_LAM / lam   # wavelength scaling

    # ---- Rozenburg frequency formulas (Section 3.2, Eqs 41-46) -------------

    def _fB_upwind(self, U: float) -> float:
        f = (17.36 + 10.59 * U**0.29 + 0.0153 * U**3.05) * self._scale
        return f * self.SCALE_BRAGG_F

    def _fB_downwind(self, U: float) -> float:
        f = (22.83 + 2.84 * U) * self._scale
        return f * self.SCALE_BRAGG_F

    def _fW_upwind(self, U: float) -> float:
        f = (-39.43 + 57.48 * np.sqrt(U) - 5.69 * U) * self._scale
        f = max(f, 0.0)
        return f * self.SCALE_WHITECAP_F_UP

    def _fW_downwind(self, U: float) -> float:
        f = (22.83 + 2.84 * U) * self._scale
        return f * self.SCALE_WHITECAP_F_DOWN

    # ---- Rozenburg bandwidth formulas (Section 3.2 + Eqs 47-52) ------------

    def _WB(self, U: float, phi_rad: float) -> float:
        Wu = 5.28 * U * self._scale * self.SCALE_BRAGG_W
        Wd = 3.92 * U * self._scale * self.SCALE_BRAGG_W
        return (Wu + Wd) / 2.0 + (Wu - Wd) / 2.0 * np.cos(phi_rad)

    def _WW(self, U: float, phi_rad: float) -> float:
        if U < 5.97:
            Wu = 6.15 * U * self._scale * self.SCALE_WHITECAP_W
        else:
            Wu = 36.7 * self._scale * self.SCALE_WHITECAP_W
        Wd = 3.92 * U * self._scale * self.SCALE_WHITECAP_W
        return (Wu + Wd) / 2.0 + (Wu - Wd) / 2.0 * np.cos(phi_rad)

    # ---- Angular interpolation (Eqs 43, 46) ---------------------------------

    def _interp_doppler(self, fu: float, fd: float, phi_rad: float) -> float:
        """cos(φ) transition between upwind and downwind Doppler peaks."""
        if phi_rad <= np.pi / 2:
            return fu * np.cos(phi_rad)
        else:
            return fd * np.cos(phi_rad)   # sign naturally goes negative → downwind

    # ---- Amplitude extraction (Sections 4.1 / 4.2) -------------------------

    def spectral_components(
        self,
        sigma0_hh_lin: float,
        sigma0_vv_lin: float,
        U: float,
        phi_rad: float,
    ) -> dict:
        """Return Bragg, whitecap, spike amplitudes (linear, m²/m²/Hz) and
        peak frequencies + bandwidths."""

        WB = max(self._WB(U, phi_rad), 1e-6)
        WW = max(self._WW(U, phi_rad), 1e-6)
        WS = 1.13 * WW          # Eq. 53

        fB = self._interp_doppler(
            self._fB_upwind(U), self._fB_downwind(U), phi_rad)
        fW = self._interp_doppler(
            self._fW_upwind(U), self._fW_downwind(U), phi_rad)

        # Spike/whitecap ratios (Eq. 56/57 for Method 1; Eq. 61 for Method 2)
        cos_phi = np.cos(phi_rad)

        # PW/PBH ratio (Eq. 57) — used in both methods
        ratio_PW_PBH = (14.1 * (U / 10.0) * (1 + cos_phi) / 2.0
                        + 1.05 * (U / 7.0) * (1 - cos_phi) / 2.0)

        # Spike ratio (Eq. 60 — Method 1 modified version)
        spike_ratio = max(0.0, 19.1 * ((U - 4.0) / 6.0)) * (1.0 + cos_phi) / 2.0

        if self.method == 1:
            # Method 1: start from σHH (Section 4.1)
            denom = np.sqrt(np.pi) * (WB + WW * ratio_PW_PBH + WS * spike_ratio)
            PBH = sigma0_hh_lin / max(denom, 1e-30)
            PW  = ratio_PW_PBH * PBH
            PS  = spike_ratio * PBH
            # VV Bragg from Eq. 59
            PBV = (sigma0_vv_lin - np.sqrt(np.pi) * WW * PW) / (np.sqrt(np.pi) * WB)
            PBV = max(PBV, 1e-30)

        else:
            # Method 2: start from σVV (Section 4.2)
            ratio_PW_PBV = (0.437 * (U / 10.0) * (1 + cos_phi) / 2.0
                            + 0.0417 * (U / 7.0) * (1 - cos_phi) / 2.0)   # Eq. 61
            denom = np.sqrt(np.pi) * (WB + WW * ratio_PW_PBV)
            PBV = sigma0_vv_lin / max(denom, 1e-30)
            PW  = ratio_PW_PBV * PBV
            # PBH from PW/PBH ratio (Eq. 57 rearranged)
            PBH = PW / max(ratio_PW_PBH, 1e-10)
            # Spike from remainder of σHH (Eq. 63)
            PS  = (sigma0_hh_lin - np.sqrt(np.pi) * WB * PBH
                   - np.sqrt(np.pi) * WW * PW) / (np.sqrt(np.pi) * WS)
            PS  = max(PS, 0.0)

        return dict(
            PBH=PBH, PBV=PBV, PW=PW, PS=PS,
            fB=fB, fW=fW, WB=WB, WW=WW, WS=WS
        )

    def psd(
        self, freq_axis: np.ndarray, comps: dict, pol: Literal['HH', 'VV']
    ) -> np.ndarray:
        """Evaluate PSD over freq_axis (Hz) — Eqs 39/40."""
        f = freq_axis
        fB = comps['fB']; fW = comps['fW']
        WB = comps['WB']; WW = comps['WW']; WS = comps['WS']

        bragg_amp = comps['PBH'] if pol == 'HH' else comps['PBV']

        G_bragg   = bragg_amp  * np.exp(-((f - fB) / WB) ** 2)
        G_whitcap = comps['PW'] * np.exp(-((f - fW) / WW) ** 2)

        if pol == 'HH':
            G_spike = comps['PS'] * np.exp(-((f - fW) / WS) ** 2)
            return G_bragg + G_whitcap + G_spike
        else:
            return G_bragg + G_whitcap
