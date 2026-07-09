#!/usr/bin/env python3
"""Etapa 3c: noisereduce (puerta espectral no estacionaria) sobre
prelimpio_v1.wav.

Perfil de ruido: el VAD del paso 1 es negativo en casi todo el archivo (la
conversación es demasiado débil para Silero), así que dentro de los segmentos
VAD-negativos elegimos las ventanas de 0.5 s del decil inferior de RMS y sin
transitorios — ruido de sala puro, sin riesgo de incluir habla enmascarada.

Dos variantes: prop_decrease 0.75 (suave) y 1.0 (agresiva).
"""
import numpy as np
import pandas as pd
import soundfile as sf
import noisereduce as nr

BASE = "/home/user/Month-age/audio-work"
y, sr = sf.read(f"{BASE}/02_prelimpieza/prelimpio_v1.wav")
grid = pd.read_csv(f"{BASE}/01_diagnostico/mapa_temporal_v2.csv")

neg = grid[(~grid["vad030"]) & (grid["n_transitorios"] == 0)]
cut = neg["rms_db"].quantile(0.10)
noise_win = neg[neg["rms_db"] <= cut]
print(f"ventanas de perfil de ruido: {len(noise_win)} (rms_db <= {cut:.1f})")
noise = np.concatenate([y[int(a * sr):int((a + 0.5) * sr)] for a in noise_win["t_ini"]])
print(f"perfil de ruido: {len(noise)/sr:.1f}s")

for prop, tag in [(0.75, "suave075"), (1.0, "agresivo100")]:
    out = nr.reduce_noise(y=y, sr=sr, y_noise=noise, stationary=False,
                          prop_decrease=prop, n_fft=2048, time_mask_smooth_ms=64,
                          freq_mask_smooth_hz=250, n_jobs=4)
    path = f"{BASE}/03_enhancement/noisereduce_{tag}_v1.wav"
    sf.write(path, out.astype(np.float32), sr, subtype="PCM_16")
    print(f"OK -> {path}")
