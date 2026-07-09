#!/usr/bin/env python3
"""Etapa AUDIBILIDAD: prioriza que se OIGA la conversacion de fondo por encima
de todo. El problema de la etapa 5 era que los impulsos fuertes topaban el
true-peak y el loudnorm dejaba la voz ~40 dB por debajo. Aqui:

  1. Se parte de las variantes con mejor VAD (noisereduce agresivo, +/- WPE)
     y tambien del original con gating fuerte.
  2. compand: compresion hacia ARRIBA agresiva -> los pasajes debiles (-55..-40
     dB) se levantan a ~-16 dB; los picos se dejan cerca del techo. Esto aplasta
     el rango dinamico para que la voz suba sin que el ruido/impulsos exploten.
  3. alimiter de seguridad a -1 dB.

Genera varias intensidades para que el usuario elija. Mide mean/max volume.
"""
import os
import subprocess

BASE = "/home/user/Month-age/audio-work"
OUT = f"{BASE}/05_ganancia"

# curva compand: input_dB/output_dB. Levanta lo muy debil, mantiene picos.
# gap peak-voz del material ~37 dB -> lo cerramos mapeando -55->-20, -40->-14 ...
CURVES = {
    # moderada: sube la voz a nivel claramente audible con ruido controlado
    "aud_media": "attacks=0.02:decays=0.4:points=-80/-45|-55/-20|-40/-14|-28/-10|-15/-7|-3/-3|0/-1:soft-knee=8",
    # fuerte: maxima audibilidad, asume mas ruido de sala y bombeo
    "aud_fuerte": "attacks=0.015:decays=0.35:points=-80/-38|-55/-16|-42/-12|-30/-9|-18/-7|-6/-4|0/-1:soft-knee=6",
}

# fuentes: (archivo, etiqueta, filtros previos extra)
SOURCES = [
    ("03_enhancement/noisereduce_agresivo100_v1.wav", "nr", ""),
    ("04_dereverb/noisereduce_agresivo100_v1_wpe_v1.wav", "nrwpe", ""),
    # original con gating: HPF + reduccion ya viene del prelimpio; usamos nr suave
    ("03_enhancement/noisereduce_suave075_v1.wav", "nrsuave", ""),
]

def mean_max(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    mv = [l.split(":")[1].strip() for l in r.splitlines() if "mean_volume" in l]
    xv = [l.split(":")[1].strip() for l in r.splitlines() if "max_volume" in l]
    return (mv[0] if mv else "?"), (xv[0] if xv else "?")

for src, stag, pre in SOURCES:
    ipath = f"{BASE}/{src}"
    if not os.path.exists(ipath):
        print(f"(falta {src})"); continue
    for ctag, curve in CURVES.items():
        af = (pre + "," if pre else "") + f"compand={curve},alimiter=limit=-1dB:level=false"
        dst = f"{OUT}/{stag}_{ctag}_v1.wav"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", ipath, "-af", af, "-ar", "48000",
                        "-c:a", "pcm_s16le", dst], check=True)
        mv, xv = mean_max(dst)
        print(f"{os.path.basename(dst):40s} mean={mv}  max={xv}")
