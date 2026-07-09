#!/usr/bin/env python3
"""Etapa AMPLIFICACION AVANZADA sobre base nrsuave (noisereduce suave, que
conserva mejor la voz). Objetivo: maxima inteligibilidad/volumen de la
conversacion de fondo. Se generan varias candidatas con el mejor arsenal:

  EQ de inteligibilidad: quita mud (250-400 Hz), realza formantes (1.5-2 kHz)
    y banda de consonantes (3-4 kHz); HPF 110 / LPF 7.5k para concentrar en voz.
  afftdn: denoise FFT con seguimiento, para compensar el ruido que el gating
    suave deja (el defecto de nrsuave era 'mas voz pero mas ruido').
  dynaudnorm: AGC por ventanas -> nivela y sube los pasajes debiles.
  compand: compresion hacia arriba agresiva (curva aud_max ya validada).
  acrossover: compresion MULTIBANDA -> levanta la banda de voz sin bombear el
    resto (evita que el ruido de sala suba con la voz).
  aexciter: armonicos de presencia para claridad percibida.
  alimiter -1 dB de seguridad.

Salida: 05_ganancia/nrsuave_<tecnica>_v1.wav + recortes 211-234 y 84-92.
"""
import os, subprocess

BASE = "/home/user/Month-age/audio-work"
SRC = f"{BASE}/03_enhancement/noisereduce_suave075_v1.wav"
OUT = f"{BASE}/05_ganancia"
REC = f"{OUT}/recortes"
os.makedirs(REC, exist_ok=True)

# curva compand upward validada (aud_max)
COMPAND = ("compand=attacks=0.015:decays=0.35:"
           "points=-80/-30|-50/-12|-38/-8|-26/-6|-14/-4|-4/-2|0/-1:soft-knee=6")

# EQ de inteligibilidad de voz
EQ = ("highpass=f=110,lowpass=f=7800,"
      "equalizer=f=300:t=q:w=1.0:g=-3,"      # menos mud
      "equalizer=f=900:t=q:w=1.2:g=1.5,"
      "equalizer=f=1800:t=q:w=1.2:g=3,"      # formantes
      "equalizer=f=3300:t=q:w=1.4:g=4.5")    # consonantes/presencia

# afftdn moderado (nr=12 dB, con perfil por ruido de banda)
DENOISE = "afftdn=nr=12:nf=-45:tn=1"

VARIANTS = {
    # A: EQ + AGC dinamico + compand + limiter
    "amp_eqagc": f"{EQ},{DENOISE},dynaudnorm=f=250:g=15:p=0.9:m=25:s=8,{COMPAND},alimiter=limit=-1dB:level=false",
    # B: EQ + exciter + AGC + compand
    "amp_exciter": f"{EQ},{DENOISE},aexciter=amount=2:drive=7:blend=2:freq=3500:ceil=9999,dynaudnorm=f=250:g=13:p=0.9:m=22,{COMPAND},alimiter=limit=-1dB:level=false",
    # C: solo AGC fuerte + compand (mas volumen bruto, menos coloreado)
    "amp_agcfuerte": f"highpass=f=110,{DENOISE},dynaudnorm=f=200:g=21:p=0.95:m=40:s=5,{COMPAND},alimiter=limit=-1dB:level=false",
}

def multiband(src, dst):
    """Compresion multibanda: banda de voz (300-3800) muy realzada."""
    fg = (
        f"[0:a]highpass=f=110,{DENOISE}[d];"
        "[d]acrossover=split=300 3800:order=4th[low][mid][high];"
        "[low]volume=0.35[lowc];"
        "[mid]compand=attacks=0.01:decays=0.3:points=-70/-22|-45/-9|-30/-6|-15/-4|-3/-2|0/-1:soft-knee=6,"
        "equalizer=f=1800:t=q:w=1.2:g=2,equalizer=f=3200:t=q:w=1.4:g=3[midc];"
        "[high]compand=attacks=0.01:decays=0.3:points=-70/-30|-45/-16|-25/-10|-8/-6|0/-2,volume=1.1[highc];"
        "[lowc][midc][highc]amix=inputs=3:normalize=0,dynaudnorm=f=250:g=11:p=0.9:m=18,"
        "alimiter=limit=-1dB:level=false[out]"
    )
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                    "-filter_complex", fg, "-map", "[out]", "-ar", "48000",
                    "-c:a", "pcm_s16le", dst], check=True)

def mean_at(path, ss, t):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", str(ss), "-t", str(t),
                        "-i", path, "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    for l in r.splitlines():
        if "mean_volume" in l: return l.split(":")[1].strip()
    return "?"

def excerpts(tag, path):
    for ss, dur, lab in [(215, 20, "215-235"), (83, 9, "83-92")]:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(ss),
                        "-t", str(dur), "-i", path, "-c:a", "pcm_s16le",
                        f"{REC}/{tag}_{lab}.wav"], check=True)

for tag, af in VARIANTS.items():
    dst = f"{OUT}/{tag}_v1.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", SRC,
                    "-af", af, "-ar", "48000", "-c:a", "pcm_s16le", dst], check=True)
    excerpts(tag, dst)
    print(f"{tag:16s} voz(218-223)={mean_at(dst,218,5)}  global={mean_at(dst,0,246)}")

dst = f"{OUT}/amp_multibanda_v1.wav"
multiband(SRC, dst)
excerpts("amp_multibanda", dst)
print(f"{'amp_multibanda':16s} voz(218-223)={mean_at(dst,218,5)}  global={mean_at(dst,0,246)}")
print("Listo.")
