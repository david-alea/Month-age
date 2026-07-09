#!/usr/bin/env python3
"""Etapa 3b: ClearerVoice-Studio MossFormer2_SE_48K sobre prelimpio_v1.wav."""
from clearvoice import ClearVoice

BASE = "/home/user/Month-age/audio-work"
cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
out = cv(input_path=f"{BASE}/02_prelimpieza/prelimpio_v1.wav", online_write=False)
cv.write(out, output_path=f"{BASE}/03_enhancement/mossformer2se48k_v1.wav")
print("OK -> mossformer2se48k_v1.wav")
