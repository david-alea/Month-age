#!/usr/bin/env python3
"""Etapa PRO-POST: aplica la cadena de audibilidad/inteligibilidad a cada
front-end DL generado en 10_pro y hace recortes para comparar. Misma cadena
validada (EQ voz + AGC dinamico + compand upward + limiter), sin denoise extra
porque los front-ends DL ya limpian (evita doble procesado).
"""
import os, glob, subprocess

BASE = "/home/user/Month-age/audio-work"
ENH = f"{BASE}/03_enhancement"
OUT = f"{BASE}/05_ganancia"
REC = f"{OUT}/recortes"
os.makedirs(REC, exist_ok=True)

COMPAND = ("compand=attacks=0.015:decays=0.35:"
           "points=-80/-30|-50/-12|-38/-8|-26/-6|-14/-4|-4/-2|0/-1:soft-knee=6")
EQ = ("highpass=f=110,lowpass=f=8500,"
      "equalizer=f=300:t=q:w=1.0:g=-3,"
      "equalizer=f=1800:t=q:w=1.2:g=3,"
      "equalizer=f=3300:t=q:w=1.4:g=4.5")
CHAIN = f"{EQ},dynaudnorm=f=250:g=15:p=0.9:m=25:s=8,{COMPAND},alimiter=limit=-1dB:level=false"

SOURCES = {
    "pro_wpe_se": f"{ENH}/pro_wpe_se_v1.wav",
    "pro_se_sr": f"{ENH}/pro_se_sr_v1.wav",
    "pro_wpe_se_sr": f"{ENH}/pro_wpe_se_sr_v1.wav",
    "pro_se_x2": f"{ENH}/pro_se_x2_v1.wav",
}

def mean_at(path, ss, t):
    r = subprocess.run(["ffmpeg","-hide_banner","-nostats","-ss",str(ss),"-t",str(t),
                        "-i",path,"-af","volumedetect","-f","null","-"],
                       capture_output=True, text=True).stderr
    for l in r.splitlines():
        if "mean_volume" in l: return l.split(":")[1].strip()
    return "?"

for tag, src in SOURCES.items():
    if not os.path.exists(src):
        print(f"(falta {tag})"); continue
    dst = f"{OUT}/{tag}_aud_v1.wav"
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",src,
                    "-af",CHAIN,"-ar","48000","-c:a","pcm_s16le",dst], check=True)
    for ss,dur,lab in [(215,20,"215-235"),(83,9,"83-92")]:
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-ss",str(ss),
                        "-t",str(dur),"-i",dst,"-c:a","pcm_s16le",
                        f"{REC}/{tag}_aud_{lab}.wav"], check=True)
    print(f"{tag:16s} voz(218-223)={mean_at(dst,218,5)}")
print("PRO-POST listo.")
