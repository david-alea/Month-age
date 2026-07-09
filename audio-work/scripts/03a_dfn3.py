#!/usr/bin/env python3
"""Etapa 3a: DeepFilterNet3 sobre prelimpio_v1.wav con tres límites de
atenuación (12 dB, 24 dB, sin límite). El límite evita que el modelo suprima
habla débil junto con el ruido. Variantes independientes, sin mezclar.
"""
import torch
import soundfile as sf
import numpy as np
from df.enhance import enhance, init_df
from df.io import load_audio, save_audio

BASE = "/home/user/Month-age/audio-work"
IN = f"{BASE}/02_prelimpieza/prelimpio_v1.wav"

model, df_state, _ = init_df()
sr = df_state.sr()
audio, _ = load_audio(IN, sr)
print(f"audio: {audio.shape}, sr={sr}")

for lim, tag in [(12.0, "att12"), (24.0, "att24"), (None, "attNone")]:
    with torch.no_grad():
        out = enhance(model, df_state, audio, atten_lim_db=lim)
    path = f"{BASE}/03_enhancement/dfn3_{tag}_v1.wav"
    save_audio(path, out, sr)
    print(f"OK -> {path}")
