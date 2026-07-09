#!/usr/bin/env python3
"""Etapa 2: pre-limpieza.

- High-pass Butterworth 4º orden @120 Hz (rumble / manipulación del teléfono).
- Atenuación SOLO de impulsos de banda ancha detectados en la etapa 1 que
  estén FUERA de regiones VAD-positivas (umbral 0.30, el conservador) o que
  saturen. Los transitorios dentro de habla no se tocan (oclusivas).
- La atenuación es un "ducking" suave con rampas de 30 ms, -18 dB en el
  núcleo del impulso, ventana total ~120 ms centrada en el onset.

Salida: 02_prelimpieza/prelimpio_v1.wav + prelimpieza_log_v1.csv + plot.
"""
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/user/Month-age/audio-work"
y, sr = sf.read(f"{BASE}/00_input/original_48k.wav")
trans = pd.read_csv(f"{BASE}/01_diagnostico/transitorios_v2.csv")

# --- High-pass 120 Hz ---
sos = butter(4, 120, btype="highpass", fs=sr, output="sos")
y_hp = sosfiltfilt(sos, y)

# --- Selección de impulsos a atenuar ---
cand = trans[(trans["clase"] == "impulso_aislado") & (trans["fuerte_p75"]) &
             ((~trans["en_habla_vad030"]) | (trans["satura"]))].copy()
print(f"Impulsos candidatos a atenuar: {len(cand)} de {len(trans)} transitorios")

# --- Ducking suave por impulso ---
gain = np.ones(len(y_hp))
core_db = -18.0
core_lin = 10 ** (core_db / 20)
half_core = int(0.030 * sr)   # ±30 ms núcleo
ramp = int(0.030 * sr)        # 30 ms rampas
for _, r in cand.iterrows():
    c = int(r["t"] * sr)
    a0, a1 = c - half_core - ramp, c - half_core
    b0, b1 = c + half_core, c + half_core + ramp
    a0, a1, b0, b1 = [max(0, min(len(gain), v)) for v in (a0, a1, b0, b1)]
    if a1 > a0:
        gain[a0:a1] = np.minimum(gain[a0:a1], np.linspace(1, core_lin, a1 - a0))
    gain[a1:b0] = np.minimum(gain[a1:b0], core_lin)
    if b1 > b0:
        gain[b0:b1] = np.minimum(gain[b0:b1], np.linspace(core_lin, 1, b1 - b0))

y_out = y_hp * gain
sf.write(f"{BASE}/02_prelimpieza/prelimpio_v1.wav", y_out.astype(np.float32), sr, subtype="PCM_16")
cand.to_csv(f"{BASE}/02_prelimpieza/prelimpieza_log_v1.csv", index=False)

fig, axes = plt.subplots(2, 1, figsize=(20, 6), sharex=True)
t_ax = np.arange(len(y)) / sr
axes[0].plot(t_ax[::20], y[::20], lw=0.4, color="#888")
axes[0].plot(t_ax[::20], y_out[::20], lw=0.4, color="#2266cc", alpha=0.75)
axes[0].set_title("gris=original · azul=pre-limpio (HPF 120 Hz + ducking impulsos)")
axes[1].plot(t_ax[::20], gain[::20], lw=0.6, color="#cc3322")
axes[1].set_ylabel("ganancia ducking")
axes[1].set_xlabel("tiempo (s)")
for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{BASE}/02_prelimpieza/prelimpieza_v1.png", dpi=110)
print("Pre-limpieza completa -> prelimpio_v1.wav")
