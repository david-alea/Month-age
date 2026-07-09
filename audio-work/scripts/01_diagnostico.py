#!/usr/bin/env python3
"""Etapa 1: diagnóstico del audio.

Salidas en 01_diagnostico/ (versionadas v1):
  - espectrograma_global_v1.png
  - rms_050s_v1.csv
  - vad_regiones_v1.csv
  - transitorios_v1.csv
  - mapa_temporal_v1.csv  (rejilla 0.5s: rms, vad, transitorios)
  - mapa_temporal_v1.png  (plot combinado)
"""
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

BASE = "/home/user/Month-age/audio-work"
OUT = f"{BASE}/01_diagnostico"
WAV48 = f"{BASE}/00_input/original_48k.wav"
WAV16 = f"{BASE}/00_input/original_16k.wav"

y48, sr48 = librosa.load(WAV48, sr=48000, mono=True)
y16, sr16 = librosa.load(WAV16, sr=16000, mono=True)
dur = len(y48) / sr48
print(f"Duración: {dur:.2f}s  pico: {np.max(np.abs(y48)):.4f}")

# ---------- 1. Espectrograma global ----------
S = librosa.amplitude_to_db(np.abs(librosa.stft(y48, n_fft=2048, hop_length=512)), ref=np.max)
fig, ax = plt.subplots(figsize=(20, 6))
img = librosa.display.specshow(S, sr=sr48, hop_length=512, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
ax.set_ylim(0, 12000)
ax.set_title("Espectrograma global (0-12 kHz) — original_48k")
fig.colorbar(img, ax=ax, format="%+2.0f dB")
fig.tight_layout()
fig.savefig(f"{OUT}/espectrograma_global_v1.png", dpi=110)
plt.close(fig)

# ---------- 2. RMS por ventanas de 0.5 s ----------
win = int(0.5 * sr48)
n_win = int(np.ceil(len(y48) / win))
rows = []
for i in range(n_win):
    seg = y48[i * win:(i + 1) * win]
    rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    rows.append({"t_ini": round(i * 0.5, 2), "t_fin": round(min((i + 1) * 0.5, dur), 2),
                 "rms": rms, "rms_db": 20 * np.log10(rms + 1e-10),
                 "pico": float(np.max(np.abs(seg))) if len(seg) else 0.0})
rms_df = pd.DataFrame(rows)
rms_df.to_csv(f"{OUT}/rms_050s_v1.csv", index=False)

# ---------- 3. Silero VAD ----------
from silero_vad import load_silero_vad, get_speech_timestamps
model = load_silero_vad()
wav_t = torch.from_numpy(y16)
# Umbral bajo: la conversación de fondo es débil; preferimos falsos positivos aquí
speech_ts = get_speech_timestamps(wav_t, model, sampling_rate=16000,
                                  threshold=0.30, min_speech_duration_ms=200,
                                  min_silence_duration_ms=300, speech_pad_ms=100,
                                  return_seconds=True)
# Segunda pasada con umbral estándar para comparar confianza
speech_ts_std = get_speech_timestamps(torch.from_numpy(y16), model, sampling_rate=16000,
                                      threshold=0.50, min_speech_duration_ms=250,
                                      min_silence_duration_ms=300, speech_pad_ms=60,
                                      return_seconds=True)
vad_rows = [{"t_ini": s["start"], "t_fin": s["end"], "umbral": 0.30} for s in speech_ts]
vad_rows += [{"t_ini": s["start"], "t_fin": s["end"], "umbral": 0.50} for s in speech_ts_std]
vad_df = pd.DataFrame(vad_rows).sort_values(["umbral", "t_ini"])
vad_df.to_csv(f"{OUT}/vad_regiones_v1.csv", index=False)
print(f"VAD@0.30: {len(speech_ts)} regiones, {sum(s['end']-s['start'] for s in speech_ts):.1f}s habla")
print(f"VAD@0.50: {len(speech_ts_std)} regiones, {sum(s['end']-s['start'] for s in speech_ts_std):.1f}s habla")

def in_speech(t, ts_list):
    return any(s["start"] <= t <= s["end"] for s in ts_list)

# ---------- 4. Transitorios: onset strength / spectral flux ----------
hop = 512
o_env = librosa.onset.onset_strength(y=y48, sr=sr48, hop_length=hop)
times_o = librosa.times_like(o_env, sr=sr48, hop_length=hop)
onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr48, hop_length=hop,
                                    backtrack=False, delta=np.percentile(o_env, 95) * 0.5,
                                    units="frames")
# Clasificación de cada onset: impulso aislado vs. parte de ruido continuo
Sm = np.abs(librosa.stft(y48, n_fft=1024, hop_length=hop))
freqs = librosa.fft_frequencies(sr=sr48, n_fft=1024)
trans_rows = []
frame_dur = hop / sr48
for f in onsets:
    t = float(times_o[f])
    a, b = max(0, f - 2), min(Sm.shape[1], f + 3)
    spec = Sm[:, a:b].mean(axis=1)
    # banda ancha: energía significativa tanto <1kHz como >4kHz
    e_low = spec[freqs < 1000].sum()
    e_mid = spec[(freqs >= 1000) & (freqs < 4000)].sum()
    e_high = spec[freqs >= 4000].sum()
    e_tot = spec.sum() + 1e-10
    broadband = (e_high / e_tot > 0.15) and (e_low / e_tot > 0.15)
    # aislado: envolvente cae >60% dentro de ±150ms alrededor del pico
    w = int(0.15 / frame_dur)
    local = o_env[max(0, f - w):f + w + 1]
    isolated = o_env[f] > 2.5 * np.median(local + 1e-10)
    # saturación cercana
    s0, s1 = int(max(0, (t - 0.05) * sr48)), int(min(len(y48), (t + 0.05) * sr48))
    peak = float(np.max(np.abs(y48[s0:s1])))
    trans_rows.append({
        "t": round(t, 3), "fuerza_onset": float(o_env[f]),
        "frac_baja": round(e_low / e_tot, 3), "frac_media": round(e_mid / e_tot, 3),
        "frac_alta": round(e_high / e_tot, 3),
        "banda_ancha": broadband, "aislado": isolated, "pico_local": round(peak, 4),
        "satura": peak > 0.98,
        "en_habla_vad030": in_speech(t, speech_ts),
        "clase": ("impulso_aislado" if (broadband and isolated) else
                  ("ruido_continuo" if broadband else "transitorio_estrecho")),
    })
trans_df = pd.DataFrame(trans_rows)
trans_df.to_csv(f"{OUT}/transitorios_v1.csv", index=False)
print(f"Transitorios: {len(trans_df)} | impulsos aislados: {(trans_df['clase']=='impulso_aislado').sum()} | "
      f"dentro de VAD: {trans_df['en_habla_vad030'].sum()}")

# ---------- 5. Mapa temporal combinado (rejilla 0.5s) ----------
grid = rms_df.copy()
grid["vad030"] = [in_speech(t + 0.25, speech_ts) for t in grid["t_ini"]]
grid["vad050"] = [in_speech(t + 0.25, speech_ts_std) for t in grid["t_ini"]]
grid["n_transitorios"] = [((trans_df["t"] >= a) & (trans_df["t"] < a + 0.5)).sum() for a in grid["t_ini"]]
grid["n_impulsos_aislados"] = [(((trans_df["t"] >= a) & (trans_df["t"] < a + 0.5)) &
                                (trans_df["clase"] == "impulso_aislado")).sum() for a in grid["t_ini"]]
grid.to_csv(f"{OUT}/mapa_temporal_v1.csv", index=False)

fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1, 1]})
librosa.display.specshow(S, sr=sr48, hop_length=512, x_axis="time", y_axis="hz",
                         ax=axes[0], cmap="magma")
axes[0].set_ylim(0, 8000)
axes[0].set_title("Mapa temporal v1 — espectrograma + RMS + VAD + transitorios")
axes[1].plot(grid["t_ini"] + 0.25, grid["rms_db"], color="#4477aa", lw=1.2)
axes[1].set_ylabel("RMS dBFS (0.5s)")
axes[1].grid(alpha=0.3)
for s in speech_ts:
    axes[2].axvspan(s["start"], s["end"], color="#66bb66", alpha=0.5)
for s in speech_ts_std:
    axes[2].axvspan(s["start"], s["end"], ymin=0.55, color="#227722", alpha=0.7)
for _, r in trans_df.iterrows():
    c = {"impulso_aislado": "red", "ruido_continuo": "orange", "transitorio_estrecho": "gray"}[r["clase"]]
    axes[2].axvline(r["t"], color=c, lw=0.8, alpha=0.8)
axes[2].set_yticks([])
axes[2].set_xlabel("tiempo (s)")
axes[2].set_title("verde claro=VAD@0.30 · verde oscuro (mitad sup.)=VAD@0.50 · rojo=impulso aislado · naranja=ruido continuo · gris=estrecho", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/mapa_temporal_v1.png", dpi=110)
plt.close(fig)
print("Diagnóstico completo.")
