#!/usr/bin/env python3
"""Etapa 5: ganancia. Para cada variante final:
  1. loudnorm EBU R128 en dos pasadas (I=-18 LUFS, TP=-1.5 dB, LRA=11)
  2. compresión suave acompressor ratio 3:1 (umbral -30 dB, ataque 20 ms,
     release 250 ms, makeup 4 dB) para levantar pasajes débiles
  3. alimiter a -1.5 dBTP de seguridad
Salida: 05_ganancia/<variante>_norm_v1.wav (48 kHz PCM16).
"""
import json
import subprocess
import sys
import glob
import os

BASE = "/home/user/Month-age/audio-work"

def loudnorm_two_pass(src, dst):
    p1 = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", src, "-af",
         "loudnorm=I=-18:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True)
    j = json.loads("{" + p1.stderr.rsplit("{", 1)[1])
    ln = (f"loudnorm=I=-18:TP=-1.5:LRA=11:measured_I={j['input_i']}:"
          f"measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:"
          f"measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")
    af = (f"{ln},acompressor=threshold=-30dB:ratio=3:attack=20:release=250:makeup=4dB,"
          f"alimiter=limit=-1.5dB:level=false")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                    "-af", af, "-ar", "48000", "-c:a", "pcm_s16le", dst], check=True)
    print(f"OK -> {dst}  (I_in={j['input_i']} LUFS)")

if __name__ == "__main__":
    srcs = sys.argv[1:]
    if not srcs:
        srcs = sorted(glob.glob(f"{BASE}/03_enhancement/*.wav") + glob.glob(f"{BASE}/04_dereverb/*.wav"))
    for src in srcs:
        stem = os.path.basename(src).replace("_v1.wav", "").replace(".wav", "")
        loudnorm_two_pass(src, f"{BASE}/05_ganancia/{stem}_norm_v1.wav")
