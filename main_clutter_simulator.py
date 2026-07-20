class SeaClutterSimulator:
    """
    Sea clutter simulator following DSTO-TR-2864.

    Parameters
    ----------
    freq_ghz : radar carrier frequency in GHz (recommended 1–35 GHz)
    model    : 'GIT' | 'Hybrid' | 'NRL'
    pol      : 'HH' | 'VV' — primary polarisation for IQ output
    method   : 1 or 2 — spectral amplitude extraction method (Sections 4.1/4.2)
    seed     : random seed for reproducible IQ samples
    """

    def __init__(
        self,
        freq_ghz: float = 9.3,
        model: Literal['GIT', 'Hybrid', 'NRL'] = 'NRL',
        pol: Literal['HH', 'VV'] = 'HH',
        method: Literal[1, 2] = 1,
        seed: Optional[int] = None,
    ):
        self.freq_ghz = freq_ghz
        self.lam = 0.3 / freq_ghz
        self.model_name = model
        self.pol = pol
        self.method = method
        self._rng = np.random.default_rng(seed)

        # Instantiate amplitude model
        _model_map = {'GIT': _GITModel, 'Hybrid': _HybridModel, 'NRL': _NRLModel}
        if model not in _model_map:
            raise ValueError(f"model must be one of {list(_model_map.keys())}")
        self._amp_model = _model_map[model](freq_ghz)
        self._doppler   = _DopplerModel(self.lam, method=method)

    # ---- convenience: Douglas sea state → wind speed ----------------------
    @staticmethod
    def sea_state_to_wind(sea_state: float) -> float:
        """Nominal fully-developed wind speed (m/s) from Eq. 1: Vw = 3.16·SS^0.8."""
        return 3.16 * sea_state ** 0.8

    # ---- main entry point --------------------------------------------------
    def run(
        self,
        sea_state: float,
        grazing_deg: float,
        wind_aspect_deg: float = 0.0,
        wind_speed_ms: Optional[float] = None,
        n_pulses: int = 256,
        prf_hz: float = 3000.0,
        n_freq_bins: int = 4096,
    ) -> ClutterResult:
        """
        Simulate sea clutter.

        Parameters
        ----------
        sea_state        : Douglas sea state 1–5 (can be non-integer)
        grazing_deg      : radar beam grazing angle (degrees)
        wind_aspect_deg  : angle between radar look direction and wind (0 = upwind)
        wind_speed_ms    : override wind speed; if None, derived from sea_state
        n_pulses         : number of coherent pulses (IQ samples)
        prf_hz           : pulse repetition frequency (Hz) — sets Doppler unambiguous range
        n_freq_bins      : resolution of PSD frequency axis

        Returns
        -------
        ClutterResult dataclass
        """
        U = wind_speed_ms if wind_speed_ms is not None else self.sea_state_to_wind(sea_state)
        phi = np.deg2rad(wind_aspect_deg)

        # --- Layer 1: σ0 ---
        s0_hh_db = self._amp_model.sigma0(grazing_deg, sea_state, U, wind_aspect_deg, 'HH')
        s0_vv_db = self._amp_model.sigma0(grazing_deg, sea_state, U, wind_aspect_deg, 'VV')
        s0_hh_lin = 10.0 ** (s0_hh_db / 10.0)
        s0_vv_lin = 10.0 ** (s0_vv_db / 10.0)

        # --- Layer 2: Doppler PSD ---
        comps = self._doppler.spectral_components(s0_hh_lin, s0_vv_lin, U, phi)

        # Frequency axis: cover ±PRF/2 with n_freq_bins points
        f_axis = np.linspace(-prf_hz / 2.0, prf_hz / 2.0, n_freq_bins)
        psd_hh = self._doppler.psd(f_axis, comps, 'HH')
        psd_vv = self._doppler.psd(f_axis, comps, 'VV')

        # Resample PSD to uniform FFT grid for IQ generation
        f_iq = np.fft.fftshift(np.fft.fftfreq(n_pulses, d=1.0 / prf_hz))
        psd_hh_iq = np.interp(f_iq, f_axis, psd_hh)
        psd_vv_iq = np.interp(f_iq, f_axis, psd_vv)

        # --- Layer 3: IQ samples ---
        iq_hh = _generate_iq(psd_hh_iq, n_pulses, prf_hz, self._rng)
        iq_vv = _generate_iq(psd_vv_iq, n_pulses, prf_hz, self._rng)

        return ClutterResult(
            sigma0_hh_db=s0_hh_db,
            sigma0_vv_db=s0_vv_db,
            psd_freq_hz=f_axis,
            psd_hh=psd_hh,
            psd_vv=psd_vv,
            bragg_hh=comps['PBH'],
            bragg_vv=comps['PBV'],
            whitecap=comps['PW'],
            spike=comps['PS'],
            f_bragg_hz=comps['fB'],
            f_whitecap_hz=comps['fW'],
            w_bragg_hz=comps['WB'],
            w_whitecap_hz=comps['WW'],
            w_spike_hz=comps['WS'],
            iq_hh=iq_hh,
            iq_vv=iq_vv,
            wind_speed_ms=U,
            sea_state=sea_state,
            grazing_deg=grazing_deg,
            wind_aspect_deg=wind_aspect_deg,
        )

    # ---- quick summary print -----------------------------------------------
    def summary(self, result: ClutterResult) -> None:
        print(f"\n{'='*60}")
        print(f"Sea Clutter Simulator — {self.model_name} model, Method {self.method}")
        print(f"  Freq: {self.freq_ghz} GHz  λ={self.lam*100:.1f} cm")
        print(f"  Sea state: {result.sea_state}  Wind: {result.wind_speed_ms:.1f} m/s")
        print(f"  Grazing: {result.grazing_deg}°  Wind aspect: {result.wind_aspect_deg}°")
        print(f"{'─'*60}")
        print(f"  σ0 HH = {result.sigma0_hh_db:+.1f} dB")
        print(f"  σ0 VV = {result.sigma0_vv_db:+.1f} dB")
        print(f"{'─'*60}")
        print(f"  Bragg freq    = {result.f_bragg_hz:+.1f} Hz  (BW {result.w_bragg_hz:.1f} Hz)")
        print(f"  Whitecap freq = {result.f_whitecap_hz:+.1f} Hz  (BW {result.w_whitecap_hz:.1f} Hz)")
        print(f"  Spike BW      = {result.w_spike_hz:.1f} Hz")
        print(f"{'─'*60}")
        print(f"  P_Bragg  HH = {10*np.log10(max(result.bragg_hh,1e-30)):+.1f} dBm²/m²/Hz")
        print(f"  P_Bragg  VV = {10*np.log10(max(result.bragg_vv,1e-30)):+.1f} dBm²/m²/Hz")
        print(f"  P_Whitecap  = {10*np.log10(max(result.whitecap,1e-30)):+.1f} dBm²/m²/Hz")
        print(f"  P_Spike     = {10*np.log10(max(result.spike,  1e-30)):+.1f} dBm²/m²/Hz")
        print(f"{'─'*60}")
        print(f"  IQ samples  = {len(result.iq_hh)} (HH) / {len(result.iq_vv)} (VV)")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Quick validation / demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n--- Validation: σ0 vs sea state, upwind, 2° grazing, 9.3 GHz ---")
    print(f"{'Model':<8} {'SS':<4} {'σ0_HH (dB)':<14} {'σ0_VV (dB)':<14}")
    print("-" * 44)
    for model_name in ['GIT', 'Hybrid', 'NRL']:
        sim = SeaClutterSimulator(freq_ghz=9.3, model=model_name)
        for ss in [1, 3, 5]:
            U = sim.sea_state_to_wind(ss)
            r = sim.run(sea_state=ss, grazing_deg=2.0, wind_aspect_deg=0.0,
                        wind_speed_ms=U, n_pulses=128, prf_hz=3000.0)
            print(f"{model_name:<8} {ss:<4} {r.sigma0_hh_db:>+10.1f} dB   {r.sigma0_vv_db:>+10.1f} dB")
    print()

    # --- Full run with plots ---
    sim = SeaClutterSimulator(freq_ghz=9.3, model='NRL', pol='HH', method=1, seed=42)
    result = sim.run(
        sea_state=3,
        wind_speed_ms=7.6,
        grazing_deg=2.0,
        wind_aspect_deg=0.0,
        n_pulses=256,
        prf_hz=3000.0,
    )
    sim.summary(result)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Sea Clutter Simulator — NRL Model, SS=3, 9.3 GHz, 2° grazing, upwind",
                 fontsize=11)

    # PSD HH
    ax = axes[0, 0]
    psd_db_hh = 10 * np.log10(np.maximum(result.psd_hh, 1e-30))
    ax.plot(result.psd_freq_hz, psd_db_hh, 'b', lw=1.2)
    ax.set_xlabel("Doppler frequency (Hz)"); ax.set_ylabel("PSD (dB m²/m²/Hz)")
    ax.set_title("Doppler PSD — HH"); ax.set_xlim(-1500, 1500); ax.grid(True, alpha=0.3)

    # PSD VV
    ax = axes[0, 1]
    psd_db_vv = 10 * np.log10(np.maximum(result.psd_vv, 1e-30))
    ax.plot(result.psd_freq_hz, psd_db_vv, 'r', lw=1.2)
    ax.set_xlabel("Doppler frequency (Hz)"); ax.set_ylabel("PSD (dB m²/m²/Hz)")
    ax.set_title("Doppler PSD — VV"); ax.set_xlim(-1500, 1500); ax.grid(True, alpha=0.3)

    # IQ time series HH (magnitude)
    ax = axes[1, 0]
    t = np.arange(len(result.iq_hh)) / 3000.0 * 1e3
    ax.plot(t, np.abs(result.iq_hh), 'b', lw=0.8, alpha=0.8)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("|IQ| (arb)")
    ax.set_title("IQ envelope — HH"); ax.grid(True, alpha=0.3)

    # σ0 vs sea state comparison across models
    ax = axes[1, 1]
    ss_arr = np.linspace(1, 5, 50)
    colors = {'GIT': 'g', 'Hybrid': 'm', 'NRL': 'b'}
    for mn, col in colors.items():
        s_mn = SeaClutterSimulator(freq_ghz=9.3, model=mn)
        s0s = [s_mn._amp_model.sigma0(2.0, ss, s_mn.sea_state_to_wind(ss), 0.0, 'HH')
               for ss in ss_arr]
        ax.plot(ss_arr, s0s, color=col, label=mn, lw=1.5)
    ax.set_xlabel("Sea state"); ax.set_ylabel("σ0 HH (dB)")
    ax.set_title("σ0 HH vs sea state (2° grazing, upwind)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.show()

    print("Validation plot saved.")
