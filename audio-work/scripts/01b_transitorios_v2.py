#!/usr/bin/env python3
"""Etapa 1b: re-detección de transitorios (v2) con umbral por defecto de
librosa (la señal es tan débil que el delta absoluto de v1 dejaba fuera casi
todo). Clasifica cada onset y regenera el mapa temporal.
"""
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/user/Month-age/audio-work"
OUT = f"{BASE}/01_diagnostico"

y, sr = librosa.load(f"{BASE}/00_input/original_48k.wav", sr=48000, mono=True)
dur = len(y) / sr
hop = 512
o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
times_o = librosa.times_like(o_env, sr=sr, hop_length=hop)
onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop, backtrack=False, units="frames")

vad = pd.read_csv(f"{OUT}/vad_regiones_v1.csv")
vad30 = vad[vad["umbral"] == 0.30]

def in_speech(t):
    return bool(((vad30["t_ini"] <= t) & (vad30["t_fin"] >= t)).any())

Sm = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop))
freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
frame_dur = hop / sr
rows = []
strengths = o_env[onsets]
p75 = np.percentile(strengths, 75)
for f in onsets:
    t = float(times_o[f])
    a, b = max(0, f - 2), min(Sm.shape[1], f + 3)
    spec = Sm[:, a:b].mean(axis=1)
    e_low = spec[freqs < 1000].sum(); e_high = spec[freqs >= 4000].sum()
    e_mid = spec[(freqs >= 1000) & (freqs < 4000)].sum()
    e_tot = spec.sum() + 1e-10
    broadband = (e_high / e_tot > 0.15) and (e_low / e_tot > 0.15)
    w = int(0.15 / frame_dur)
    local = o_env[max(0, f - w):f + w + 1]
    isolated = o_env[f] > 2.5 * np.median(local + 1e-10)
    s0, s1 = int(max(0, (t - 0.05) * sr)), int(min(len(y), (t + 0.05) * sr))
    peak = float(np.max(np.abs(y[s0:s1])))
    fuerte = o_env[f] >= p75
    rows.append({
        "t": round(t, 3), "fuerza_onset": round(float(o_env[f]), 3), "fuerte_p75": fuerte,
        "frac_baja": round(e_low / e_tot, 3), "frac_media": round(e_mid / e_tot, 3),
        "frac_alta": round(e_high / e_tot, 3), "banda_ancha": broadband,
        "aislado": isolated, "pico_local": round(peak, 4), "satura": peak > 0.98,
        "en_habla_vad030": in_speech(t),
        "clase": ("impulso_aislado" if (broadband and isolated) else
                  ("ruido_continuo" if broadband else "transitorio_estrecho")),
    })
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/transitorios_v2.csv", index=False)
print(df["clase"].value_counts().to_string())
print(f"fuertes(>=p75): {df['fuerte_p75'].sum()} | saturan: {df['satura'].sum()} | en VAD: {df['en_habla_vad030'].sum()}")
print(f"impulsos aislados fuertes fuera de VAD: {((df['clase']=='impulso_aislado') & df['fuerte_p75'] & ~df['en_habla_vad030']).sum()}")

# mapa temporal v2
rms = pd.read_csv(f"{OUT}/rms_050s_v1.csv")
grid = rms.copy()
grid["n_transitorios"] = [((df["t"] >= a) & (df["t"] < a + 0.5)).sum() for a in grid["t_ini"]]
grid["n_impulsos_aislados"] = [(((df["t"] >= a) & (df["t"] < a + 0.5)) & (df["clase"] == "impulso_aislado")).sum() for a in grid["t_ini"]]
grid["vad030"] = [in_speech(t + 0.25) for t in grid["t_ini"]]
grid.to_csv(f"{OUT}/mapa_temporal_v2.csv", index=False)

S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
librosa.display.specshow(S, sr=sr, hop_length=512, x_axis="time", y_axis="hz", ax=axes[0], cmap="magma")
axes[0].set_ylim(0, 8000)
axes[0].set_title("Mapa temporal v2 — espectrograma + RMS + VAD + transitorios (deteccion por defecto)")
axes[1].plot(grid["t_ini"] + 0.25, grid["rms_db"], color="#4477aa", lw=1.2)
axes[1].set_ylabel("RMS dBFS (0.5s)"); axes[1].grid(alpha=0.3)
for _, r in vad30.iterrows():
    axes[2].axvspan(r["t_ini"], r["t_fin"], color="#66bb66", alpha=0.6)
for _, r in df.iterrows():
    c = {"impulso_aislado": "red", "ruido_continuo": "orange", "transitorio_estrecho": "gray"}[r["clase"]]
    axes[2].axvline(r["t"], color=c, lw=1.0 if r["fuerte_p75"] else 0.4, alpha=0.9 if r["fuerte_p75"] else 0.4)
axes[2].set_yticks([]); axes[2].set_xlabel("tiempo (s)")
axes[2].set_title("verde=VAD@0.30 · rojo=impulso aislado · naranja=ruido continuo · gris=estrecho (trazo grueso = fuerte >=p75)", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/mapa_temporal_v2.png", dpi=110)
print("Mapa v2 listo.")
