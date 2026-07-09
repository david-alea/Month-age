#!/usr/bin/env python3
"""Evaluación proxy de variantes: Silero VAD (umbral 0.30 y 0.50) sobre cada
variante normalizada. Más segundos de habla detectada = el enhancement ha
sacado la conversación por encima del ruido para el VAD. También sirve para
mapear dónde buscar habla en la transcripción.
Salida: 05_ganancia/eval_vad_v1.csv + regiones por variante.
"""
import glob
import os
import numpy as np
import pandas as pd
import torch
import librosa
from silero_vad import load_silero_vad, get_speech_timestamps

BASE = "/home/user/Month-age/audio-work"
model = load_silero_vad()

rows, regions = [], []
files = [f"{BASE}/00_input/original_48k.wav"] + sorted(glob.glob(f"{BASE}/05_ganancia/*_norm_v1.wav"))
for f in files:
    name = os.path.basename(f).replace("_norm_v1.wav", "").replace("_48k.wav", "")
    y, _ = librosa.load(f, sr=16000, mono=True)
    for thr in (0.30, 0.50):
        ts = get_speech_timestamps(torch.from_numpy(y), model, sampling_rate=16000,
                                   threshold=thr, min_speech_duration_ms=200,
                                   min_silence_duration_ms=300, speech_pad_ms=100,
                                   return_seconds=True)
        sec = sum(t["end"] - t["start"] for t in ts)
        rows.append({"variante": name, "umbral": thr, "n_regiones": len(ts),
                     "seg_habla": round(sec, 1)})
        for t in ts:
            regions.append({"variante": name, "umbral": thr,
                            "t_ini": t["start"], "t_fin": t["end"]})
    model.reset_states()

df = pd.DataFrame(rows)
df.to_csv(f"{BASE}/05_ganancia/eval_vad_v1.csv", index=False)
pd.DataFrame(regions).to_csv(f"{BASE}/05_ganancia/eval_vad_regiones_v1.csv", index=False)
print(df.pivot(index="variante", columns="umbral", values="seg_habla").to_string())
