class _HybridModel:
    """
    Hybrid sea clutter model (DSTO-TR-2864 Section 2.2).
    With the Whitrow modification to Ks (Eq. 21) and updated Kd (Eq. 33-35).
    """

    def __init__(self, freq_ghz: float):
        self.freq_ghz = freq_ghz
        self.lam = 0.3 / freq_ghz

    def sigma0(self, grazing_deg: float, sea_state: float,
               wind_speed_ms: float, wind_aspect_deg: float,
               pol: Literal['HH', 'VV']) -> float:
        fRF = self.freq_ghz
        lam = self.lam
        psi = np.deg2rad(grazing_deg)
        phi = np.deg2rad(wind_aspect_deg)
        SS = sea_state
        hRMS = 0.031 * SS ** 2     # Eq. 25
        hav = 0.08 * SS ** 2       # Eq. 26
        psi_r = np.deg2rad(0.1)    # reference grazing = 0.1°

        # Reference reflectivity (Eq. 15) — fRF in Hz in original, but model
        # uses GHz-scale log10 — paper's Eq 15 states fRF in Hz.
        # We use GHz directly in the log (matches tabulated values).
        if fRF <= 12.5:
            s0_ref = 22.4 * np.log10(fRF) - 266.8
        else:
            s0_ref = 3.25 * np.log10(fRF) - 71.25

        # Kg: grazing angle adjustment (Eq. 18/19)
        arg = 0.066 * lam / hRMS if hRMS > 0 else 1e-6
        arg = min(arg, 1.0)
        psi_t = np.arcsin(arg)

        if psi < psi_r:
            Kg = 0.0
        elif psi_t >= psi_r:
            if psi <= psi_t:
                Kg = 20.0 * np.log10(psi / psi_r)
            else:
                Kg = 20.0 * np.log10(psi_t / psi_r) + 10.0 * np.log10(psi / psi_t)
        else:
            Kg = 10.0 * np.log10(psi / psi_r)

        # Ks: sea state (Eq. 21 — Whitrow modified)
        Ks = 5.0 * (SS - 5.0) + (SS - 5.0) ** 3 / 10.0

        # Kp: polarisation (Eq. 22)
        if pol == 'VV':
            Kp = 0.0
        else:
            if fRF < 3.0:
                Kp = (1.7 * np.log(hav + 0.015) - 3.8 * np.log(lam)
                      - 2.5 * np.log(psi + 0.0001) - 22.2)
            elif fRF <= 10.0:
                Kp = (1.1 * np.log(hav + 0.015) - 1.1 * lam
                      - 1.3 * np.log(psi + 0.0001) - 9.7)
            else:
                Kp = 1.4 * np.log(hav) - 3.4 * np.log(lam) - 1.3 * np.log(psi) - 18.6

        # Kd: wind direction (Eq. 33-35 revised — Plant's measurements)
        base = 0.3 - 1.7 * np.log10(lam)
        if pol == 'VV':
            Kd = base * (0.578 * np.cos(phi) + 1.09 * np.cos(2 * phi) - 1.0)
        else:
            Kd = base * (1.54 * np.cos(phi) - 1.0)

        s0_db = s0_ref + Kg + Ks + Kp + Kd
        return s0_db
