# sea_clutter_model
Sea Clutter Simulator based on DSTO-TR-2864

A Model of Low Grazing Angle Sea Clutter for Coherent Radar Performance Analysis"

J L Whitrow, Defence Science and Technology Organisation, 2013 (Corrected 2017)

Implements:
  Layer 1 - Amplitude σ0: GIT, Hybrid, NRL models
  
  Layer 2 - Doppler PSD:  Walker 3-component model (Bragg + whitecap + spike)
                          with Rozenburg wavetank data + ocean scale factors (Sec 4.3)
                          Method 1 and Method 2 amplitude extraction (Sec 4.1/4.2)
  Layer 3 - IQ samples:   Complex Gaussian time-series shaped by PSD via IFFT
