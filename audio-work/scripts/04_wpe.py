#!/usr/bin/env python3
"""Etapa 4: dereverberación WPE (nara_wpe) sobre la mejor variante de la
etapa 3. Sin métrica de referencia limpia, la selección se hace por proxy:
mayor energía en la banda de habla (300-3400 Hz) relativa al total tras el
enhancement, y se aplicará también a la segunda candidata para comparar.

WPE mono: STFT 1024/256, taps=10, delay=3, 5 iteraciones.
"""
import sys
import numpy as np
import soundfile as sf
from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft

BASE = "/home/user/Month-age/audio-work"

def run_wpe(in_path, out_path):
    y, sr = sf.read(in_path)
    Y = stft(y[None, :], size=1024, shift=256)  # (1, T, F)
    Y = Y.transpose(2, 0, 1)                    # (F, D, T)
    Z = wpe(Y, taps=10, delay=3, iterations=5, statistics_mode="full")
    z = istft(Z.transpose(1, 2, 0), size=1024, shift=256)[0]
    z = z[:len(y)]
    peak = np.max(np.abs(z)) + 1e-9
    if peak > 0.999:
        z = z / peak * 0.999
    sf.write(out_path, z.astype(np.float32), sr, subtype="PCM_16")
    print(f"OK -> {out_path}")

if __name__ == "__main__":
    for name in sys.argv[1:]:
        stem = name.rsplit("/", 1)[-1].replace(".wav", "")
        run_wpe(f"{BASE}/03_enhancement/{name}", f"{BASE}/04_dereverb/{stem}_wpe_v1.wav")
