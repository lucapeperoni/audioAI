#!/usr/bin/env python3
# coding: utf-8
"""
demo.py – Live-Mel-CAM (versione stabile pre-eval.py)

• Slider sincronizzato con l’audio (Play / Pause / Stop)
• Spettrogramma Mel 256×128, palette inferno
• Grad-CAM overlay in rosso
• Classificazione istantanea + top-2 globale (live e finale)
• Analisi offline opzionale (senza riprodurre l’audio)
"""

from __future__ import annotations
import io, time, queue, threading, pathlib

import streamlit as st
import numpy as np
import cv2, torch, librosa, sounddevice as sd
from torchvision.models import resnet18
from torchcam.methods import GradCAM
import torch.nn.functional as F

# ─────────── costanti ────────────────────────────────────────────────────
SR          = 22_050
N_MELS      = 128      # deve combaciare con il training
HOP         = 512
FRAME_DUR   = HOP / SR
GENRES      = ['blues','classical','country','disco','hiphop',
               'jazz','metal','pop','reggae','rock']

sd.default.blocksize = 4096
sd.default.latency   = ('high', 'high')

# ─────────── modello + CAM ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Carica ResNet-18; avvisa se mancano i pesi."""
    m = resnet18(weights=None)
    m.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    m.fc    = torch.nn.Linear(512, 10)
    # cerca prima in src/models/, poi in <repo>/models/
    ckpt_src  = pathlib.Path(__file__).parent / "models" / "best_resnet18.pt"
    ckpt_root = pathlib.Path(__file__).parent.parent / "models" / "best_resnet18.pt"
    ckpt = ckpt_src if ckpt_src.exists() else ckpt_root
    if ckpt.exists():
        m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    else:
        st.warning("⚠️ Pesi non trovati: modello casuale.")
    m.eval()
    cam = GradCAM(m, target_layer="layer3")
    return m, cam

model, cam_extr = load_model()

# ─────────── session state init ──────────────────────────────────────────
S = st.session_state
for k, v in dict(play=False, start=0.0, idx=0,
                 y=None, S_db=None, max_idx=0,
                 cache={}, logits_dict={}, todo=None).items():
    S.setdefault(k, v)

# ─────────── helper audio → Mel ──────────────────────────────────────────
def audio_to_mel(y: np.ndarray) -> np.ndarray:
    """Ritorna log‑Mel spectrogram (compatibile con Librosa ≥0.11)."""
    S = librosa.feature.melspectrogram(
        y=y, sr=SR,
        n_fft=2048, hop_length=HOP,
        n_mels=N_MELS, fmin=20, fmax=11_000
    )
    return librosa.power_to_db(S, ref=np.max)

# ─────────── CAM worker thread ───────────────────────────────────────────
def cam_worker(q: queue.Queue, cache: dict, logits_d: dict):
    while True:
        idx, spec = q.get()
        z = (spec - spec.mean()) / (spec.std() + 1e-6)
        t = torch.tensor(z).unsqueeze(0).unsqueeze(0)
        t.requires_grad_()
        log = model(t)
        logits_d[idx] = log.softmax(1).squeeze().detach()
        heat = cam_extr(log.argmax(1).item(), log)[0].squeeze()
        heat = F.interpolate(heat[None,None], size=spec.shape,
                             mode='bilinear', align_corners=False)[0,0]
        heat_u8 = cv2.normalize(heat.numpy(), None, 0, 255,
                                cv2.NORM_MINMAX).astype('uint8')
        mask = heat_u8 > 180
        spec_u8 = cv2.normalize(spec, None, 0, 255,
                                cv2.NORM_MINMAX).astype('uint8')
        spec_c  = cv2.applyColorMap(spec_u8, cv2.COLORMAP_INFERNO)
        red     = spec_c.copy(); red[mask] = (0,0,255)
        blend   = cv2.addWeighted(spec_c, 0.7, red, 0.3, 0)
        blend   = cv2.resize(blend, (512, 512), cv2.INTER_CUBIC)
        cache[idx] = blend
        q.task_done()

# ─────────── UI Streamlit ────────────────────────────────────────────────
st.set_page_config(page_title="Live-Mel-CAM", layout="centered")
st.title("🎧 Live-Mel-CAM (stabile)")

up = st.file_uploader("Carica MP3 o WAV", type=["mp3", "wav"])
if up and (("last_name" not in S) or (up.name != S.last_name)):
    # reset stato
    for k in ["y","S_db","cache","logits_dict","todo",
              "play","idx","start"]:
        if k in S:
            del S[k]
    S.last_name = up.name
    S.play = False
    S.idx = 0
    y, _ = librosa.load(up, sr=SR, mono=True)
    S.y = y
    S.S_db = audio_to_mel(y)
    if S.S_db.shape[1] < 128:
        S.S_db = np.pad(S.S_db, ((0,0), (0, 128 - S.S_db.shape[1])))
    S.max_idx = S.S_db.shape[1] - 128
    S.cache = {}
    S.logits_dict = {}
    S.todo = queue.Queue(maxsize=16)
    threading.Thread(target=cam_worker,
                     args=(S.todo, S.cache, S.logits_dict),
                     daemon=True).start()
    S.todo.put((0, S.S_db[:,0:128]))
    S.todo.put((1, S.S_db[:,1:129]))

# ─── controlli ───────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
if c1.button("▶ Play", disabled=not up):
    if not S.play and S.y is not None:
        sd.stop(); sd.play(S.y[S.idx*HOP:], SR)
        S.start = time.perf_counter() - (S.idx * FRAME_DUR)
        S.play = True
if c2.button("⏸ Pause", disabled=not up):
    if S.play:
        sd.stop(); S.play = False
if c3.button("⏹ Stop", disabled=not up):
    sd.stop(); S.play = False; S.idx = 0

# ─── aggiorna indice durante la riproduzione ─────────────────────────────
if S.play:
    S.idx = int((time.perf_counter() - S.start) / FRAME_DUR)
    if S.idx > S.max_idx:
        S.play = False

# ─── slider posizione ───────────────────────────────────────────────────
if up:
    S.idx = st.slider("Posizione", 0, S.max_idx, S.idx, disabled=S.play)

# ─── analisi offline (senza riproduzione audio) ──────────────────────────
if up and S.S_db is not None:
    if st.button("🔍 Analizza offline (genere completo)", key="offline"):
        with st.spinner("Analisi offline in corso…"):
            # disattiva i hook della Grad‑CAM: altrimenti richiedono gradienti
            cam_extr.remove_hooks()    # disattiva i hook CAM
            logits_buf = []
            # stride 1: analizza ogni finestra da 128 frame
            for i in range(0, S.S_db.shape[1] - 128 + 1, 1):
                spec = S.S_db[:, i:i+128]
                z = (spec - spec.mean()) / (spec.std() + 1e-6)
                t = torch.tensor(z, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():            # CAM non necessaria offline
                    log = model(t)
                logits_buf.append(log.squeeze())
            g_soft = torch.stack(logits_buf).mean(0).softmax(0)
            g1 = int(g_soft.argmax()); p1 = float(g_soft[g1]) * 100
            g_soft[g1] = -1.0
            g2 = int(g_soft.argmax()); p2 = float(g_soft[g2]) * 100
            st.success(f"**Genere offline:** {GENRES[g1]} ({p1:.1f} %) – "
                       f"secondo: {GENRES[g2]} ({p2:.1f} %)")
        st.stop()

# ─── mostra frame corrente ──────────────────────────────────────────────
if up:
    spec = S.S_db[:, S.idx:S.idx+128]
    if S.idx not in S.cache and not S.todo.full():
        S.todo.put((S.idx, spec))
    frame = S.cache.get(S.idx)
    if frame is None:
        spec_u8 = cv2.normalize(spec, None, 0, 255, cv2.NORM_MINMAX
                                ).astype('uint8')
        frame = cv2.applyColorMap(spec_u8, cv2.COLORMAP_INFERNO)
        frame = cv2.resize(frame, (512, 512), cv2.INTER_CUBIC)
    st.image(frame, channels="BGR", use_container_width=True)

    # top-2 live
    if S.logits_dict:
        g_soft = torch.stack(list(S.logits_dict.values())).mean(0).softmax(0)
        g1, g2 = torch.topk(g_soft, 2).indices
        p1, p2 = g_soft[g1] * 100, g_soft[g2] * 100
        st.markdown(f"**Genere globale provvisorio**  \n"
                    f"1️⃣ {GENRES[g1]} {p1:5.1f}%  \n"
                    f"2️⃣ {GENRES[g2]} {p2:5.1f}%")

# ─── auto-refresh mentre suona ───────────────────────────────────────────
if S.play:
    time.sleep(0.12)
    # Streamlit ≥ 1.25 usa `st.rerun`; sulle versioni precedenti rimane `experimental_rerun`
    (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()