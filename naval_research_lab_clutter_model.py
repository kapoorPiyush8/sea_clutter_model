class _NRLModel:
    """NRL(naval research lab) sea clutter model (DSTO-TR-2864 Section 2.3, Table 2).
    Provides upwind σ0; aspect correction via Hybrid Kd (Section 2.4).
    """

    COEFFS = {
        'HH': (-72.76, 21.11, 24.78, 281.7,  35.62, -0.02949, 26.19, 5.354,  0.05031),
        'VV': (-48.56, 26.30, 29.05, -29.70, 60.56,  0.04839, 21.37, 4.278,  0.04623),
    }

    def __init__(self, freq_ghz: float):
        self.freq_ghz = freq_ghz
        self.lam = 0.3 / freq_ghz

    def sigma0(self, grazing_deg: float, sea_state: float,
               wind_speed_ms: float, wind_aspect_deg: float,
               pol: Literal['HH', 'VV']) -> float:
        c = self.COEFFS[pol]
        psi = np.deg2rad(grazing_deg)   # radians — Eq. 27 footnote
        SS = sea_state
        fRF = self.freq_ghz
        lam = self.lam

        # Eq. 27 — NRL upwind σ0
        denom = 1.0 + c[4] * psi + c[5] * SS
        freq_term = (c[2] + c[3] * psi) * np.log10(fRF) / denom
        exp_pow = 1.0 / (2.0 + c[7] * c[8] + c[8] * SS)  # exponent in Eq 27
        # Careful reconstruction of Eq 27:
        # σ0 = c1 + c2·log(sin ψ) + (c3 + c4·ψ)·log(fRF)/(1 + c5·ψ + c6·SS)
        #        + c7·(1+SS)^{1/(2 + c8·α + c9·SS)}
        # α is wind aspect — not listed separately; paper says upwind only.
        # Use α = 0 for base NRL, apply Hybrid Kd for aspect variation.
        alpha = 0.0   # upwind baseline
        power_exp = 1.0 / (2.0 + c[7] * alpha + c[8] * SS)
        s0_db = (c[0]
                 + c[1] * np.log10(np.sin(max(psi, 1e-6)))
                 + (c[2] + c[3] * psi) * np.log10(fRF) / (1.0 + c[4] * psi + c[5] * SS)
                 + c[6] * (1.0 + SS) ** power_exp)

        # Apply aspect correction via Hybrid Kd (Section 2.4)
        phi = np.deg2rad(wind_aspect_deg)
        base = 0.3 - 1.7 * np.log10(lam)
        if pol == 'VV':
            Kd = base * (0.578 * np.cos(phi) + 1.09 * np.cos(2 * phi) - 1.0)
        else:
            Kd = base * (1.54 * np.cos(phi) - 1.0)

        return s0_db + Kd
