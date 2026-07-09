#!/usr/bin/env python3
"""Etapa 7: transcripcion comparada con faster-whisper large-v3.

Transcribe el original + cada variante (incluida la ganadora nrsuave_aud_max),
en espanol, con word timestamps, avg_logprob y no_speech_prob. Genera:
  - 07_transcripcion/segmentos_<variante>_v1.csv  (por variante)
  - 07_transcripcion/comparativa_v1.csv           (segmentos alineados por tiempo)
  - 07_transcripcion/comparativa_v1.html          (tabla; en rojo los sospechosos)

Heuristica de "probable alucinacion" (marca roja):
  - avg_logprob < -0.9  (baja confianza), o
  - no_speech_prob > 0.6 (probable no-voz), o
  - texto largo (>4 palabras) sobre un segmento cuyo RMS medio esta en el
    decil inferior del archivo (energia muy baja pero Whisper "inventa" fluido), o
  - alta discrepancia entre variantes en el mismo instante.
"""
import os, glob, html
import numpy as np
import pandas as pd
import soundfile as sf
from faster_whisper import WhisperModel

BASE = "/home/user/Month-age/audio-work"
OUT = f"{BASE}/07_transcripcion"
MODEL_DIR = f"{BASE}/models/fw-large-v3"

# Variantes a transcribir (nombre_amigable -> ruta)
VARIANTS = {
    "original": f"{BASE}/00_input/original_48k.wav",
    "nrsuave_aud_max": f"{BASE}/05_ganancia/nrsuave_aud_max_v1.wav",
    "nr_aud_max": f"{BASE}/05_ganancia/nr_aud_max_v1.wav",
    "noisereduce_agresivo": f"{BASE}/03_enhancement/noisereduce_agresivo100_v1.wav",
    "dfn3_att12": f"{BASE}/05_ganancia/dfn3_att12_norm_v1.wav",
}
# se anade MossFormer2_SE si existe
mf = f"{BASE}/03_enhancement/mossformer2se48k_v1.wav"
if os.path.exists(mf):
    VARIANTS["mossformer2_se"] = mf

def rms_low_decile(path):
    y, sr = sf.read(path)
    if y.ndim > 1: y = y.mean(1)
    win = int(0.5 * sr)
    rms = np.array([np.sqrt(np.mean(y[i:i+win]**2)+1e-12) for i in range(0, len(y), win)])
    db = 20*np.log10(rms+1e-10)
    return db, np.percentile(db, 10), sr, win

def transcribe(model, path):
    segs, _ = model.transcribe(path, language="es", word_timestamps=True,
                               vad_filter=False, beam_size=5,
                               condition_on_previous_text=False)
    rows = []
    db, low10, sr, win = rms_low_decile(path)
    for s in segs:
        idx = int((s.start) / 0.5)
        seg_db = float(db[idx]) if idx < len(db) else -100.0
        rows.append({"t_ini": round(s.start,2), "t_fin": round(s.end,2),
                     "texto": s.text.strip(), "avg_logprob": round(s.avg_logprob,3),
                     "no_speech_prob": round(s.no_speech_prob,3),
                     "n_palabras": len(s.text.split()), "seg_rms_db": round(seg_db,1),
                     "energia_baja": seg_db <= low10})
    return pd.DataFrame(rows)

def main():
    model = WhisperModel(MODEL_DIR, device="cpu", compute_type="int8")
    all_segs = {}
    for name, path in VARIANTS.items():
        if not os.path.exists(path):
            print(f"(falta {name})"); continue
        print(f"transcribiendo {name} ...")
        df = transcribe(model, path)
        df["sospechoso"] = ((df.avg_logprob < -0.9) | (df.no_speech_prob > 0.6) |
                            ((df.n_palabras > 4) & df.energia_baja))
        df.to_csv(f"{OUT}/segmentos_{name}_v1.csv", index=False)
        all_segs[name] = df
        print(f"  {len(df)} segmentos, {df.sospechoso.sum()} sospechosos")

    # comparativa por rejilla temporal de 2s
    dur = 247
    grid = np.arange(0, dur, 2.0)
    comp_rows = []
    for t in grid:
        row = {"t": t}
        texts = []
        for name, df in all_segs.items():
            hit = df[(df.t_ini < t+2) & (df.t_fin > t)]
            txt = " ".join(hit.texto.tolist())
            row[name] = txt
            if txt: texts.append(txt)
        # discrepancia: nº de textos distintos no vacios
        uniq = len(set(t2.lower().strip() for t2 in texts if t2.strip()))
        row["_n_variantes_con_texto"] = len(texts)
        row["_discrepancia"] = uniq
        comp_rows.append(row)
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(f"{OUT}/comparativa_v1.csv", index=False)

    # HTML
    names = list(all_segs.keys())
    def susp_at(name, t):
        df = all_segs[name]
        hit = df[(df.t_ini < t+2) & (df.t_fin > t)]
        return bool(hit.sospechoso.any()) if len(hit) else False
    parts = ["<html><head><meta charset='utf-8'><style>",
             "body{font-family:sans-serif;font-size:13px}table{border-collapse:collapse}",
             "td,th{border:1px solid #ccc;padding:4px;vertical-align:top;max-width:260px}",
             ".susp{color:#c00;font-weight:bold}.disc{background:#fff3cd}",
             "th{background:#eee;position:sticky;top:0}</style></head><body>",
             "<h2>Transcripcion comparada large-v3 (rojo=posible alucinacion, amarillo=alta discrepancia)</h2>",
             "<table><tr><th>t (s)</th>"]
    for n in names: parts.append(f"<th>{html.escape(n)}</th>")
    parts.append("</tr>")
    for _, r in comp.iterrows():
        if r["_n_variantes_con_texto"] == 0: continue
        disc = " class='disc'" if r["_discrepancia"] >= 3 else ""
        parts.append(f"<tr{disc}><td>{r['t']:.0f}</td>")
        for n in names:
            cell = html.escape(str(r[n]))
            cls = " class='susp'" if susp_at(n, r["t"]) else ""
            parts.append(f"<td{cls}>{cell}</td>")
        parts.append("</tr>")
    parts.append("</table></body></html>")
    with open(f"{OUT}/comparativa_v1.html", "w") as f:
        f.write("".join(parts))
    print("comparativa_v1.csv + comparativa_v1.html listos")

if __name__ == "__main__":
    main()
