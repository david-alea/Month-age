#!/usr/bin/env python3
"""Etapa PRO: front-ends DL potentes para maxima claridad. El post-proceso ya
estaba saturado (todas sonaban igual); el margen esta en la senal de entrada.
Cascadas (cada paso guarda su salida en 03_enhancement/ o 04_dereverb/):

  A  prelimpio -> WPE(dereverb) -> MossFormer2_SE            (pro_wpe_se)
  B  MossFormer2_SE(ya hecho) -> MossFormer2_SR(super-res)   (pro_se_sr)
  C  prelimpio -> WPE -> MossFormer2_SE -> MossFormer2_SR    (pro_wpe_se_sr)  <- candidata fuerte
  D  MossFormer2_SE -> MossFormer2_SE (cascada denoise x2)   (pro_se_x2)

Robusto: cada paso en try/except, continua si uno falla. Los .wav intermedios
quedan para inspeccion. El post-proceso de audibilidad se aplica en 11_pro_post.
"""
import os, traceback
import numpy as np
import soundfile as sf

BASE = "/home/user/Month-age/audio-work"
ENH = f"{BASE}/03_enhancement"
DER = f"{BASE}/04_dereverb"
PRE = f"{BASE}/02_prelimpieza/prelimpio_v1.wav"
SE_DONE = f"{ENH}/mossformer2se48k_v1.wav"

def log(m): print(m, flush=True)

# ---------- WPE(prelimpio) ----------
def wpe_file(inp, outp):
    from nara_wpe.wpe import wpe
    from nara_wpe.utils import stft, istft
    y, sr = sf.read(inp)
    Y = stft(y[None, :], size=1024, shift=256).transpose(2, 0, 1)
    Z = wpe(Y, taps=10, delay=3, iterations=5, statistics_mode="full")
    z = istft(Z.transpose(1, 2, 0), size=1024, shift=256)[0][:len(y)]
    peak = np.max(np.abs(z)) + 1e-9
    if peak > 0.999: z = z / peak * 0.999
    sf.write(outp, z.astype(np.float32), sr, subtype="PCM_16")
    log(f"  WPE -> {outp}")

# ---------- ClearVoice helpers ----------
def cv_run(task, model, inp, outp):
    from clearvoice import ClearVoice
    cv = ClearVoice(task=task, model_names=[model])
    out = cv(input_path=inp, online_write=False)
    cv.write(out, output_path=outp)
    log(f"  {model} -> {outp}")

steps = []

# A: WPE del prelimpio, luego SE
def stepA():
    wpe_file(PRE, f"{DER}/prelimpio_wpe_v1.wav")
    cv_run("speech_enhancement", "MossFormer2_SE_48K", f"{DER}/prelimpio_wpe_v1.wav", f"{ENH}/pro_wpe_se_v1.wav")
steps.append(("A pro_wpe_se", stepA))

# B: SR sobre el SE ya hecho
def stepB():
    cv_run("speech_super_resolution", "MossFormer2_SR_48K", SE_DONE, f"{ENH}/pro_se_sr_v1.wav")
steps.append(("B pro_se_sr", stepB))

# C: SR sobre A (WPE+SE)
def stepC():
    cv_run("speech_super_resolution", "MossFormer2_SR_48K", f"{ENH}/pro_wpe_se_v1.wav", f"{ENH}/pro_wpe_se_sr_v1.wav")
steps.append(("C pro_wpe_se_sr", stepC))

# D: SE en cascada sobre el SE hecho
def stepD():
    cv_run("speech_enhancement", "MossFormer2_SE_48K", SE_DONE, f"{ENH}/pro_se_x2_v1.wav")
steps.append(("D pro_se_x2", stepD))

for name, fn in steps:
    try:
        log(f"== {name} ==")
        fn()
    except Exception:
        log(f"!! FALLO en {name}:\n{traceback.format_exc()}")

log("PRO pipeline terminado.")
