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
from streamlit.runtime.scriptrunner import add_script_run_ctx
import numpy as np
import cv2, torch, librosa, sounddevice as sd
from torchvision.models import resnet18
from torchcam.methods import GradCAM
import torch.nn.functional as F

SR          = 22_050
N_MELS      = 128      # deve combaciare con il training
VIS_MELS    = 512      # per visualizzare lo spettro in alta definizione
VIEW_FRAMES   = 128                 # n. di frame mostrati
CENTER_FRAME  = VIEW_FRAMES // 2    # frame corrispondente alla barra
BAR_X         = 640 // 2            # barra verde al centro (320 px)
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
    # carica il checkpoint dal path assoluto: <repo>/models/best_resnet18_fold4.pt
    ckpt = pathlib.Path(__file__).resolve().parent.parent / "models" / "best_resnet18_fold4.pt"
    if ckpt.exists():
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    else:
        st.error(f"⚠️ Checkpoint non trovato: {ckpt}")
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

# ─────────── helper slice → HD frame ────────────────────────────────────
def slice_to_hd(spec_low: np.ndarray) -> np.ndarray:
    """
    Converte la finestra 128×128 (spec_low) in un frame HD 640×360:
    * ricompone la parte di S_db corrispondente con 512 Mel
    * applica colormap inferno
    """
    # upsample verticale ricombinando il log‐Mel con più bande
    # spec_low shape: (128, 128)
    # usiamo ridimensionamento Lanczos verticale + orizzontale
    spec_hd = cv2.resize(spec_low, (128, VIS_MELS), interpolation=cv2.INTER_LANCZOS4)
    spec_hd = cv2.resize(spec_hd, (640, 360), interpolation=cv2.INTER_LANCZOS4)
    spec_u8 = cv2.normalize(spec_hd, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    spec_u8_col = cv2.applyColorMap(spec_u8, cv2.COLORMAP_INFERNO)
    return spec_u8_col

def get_window(idx: int) -> np.ndarray:
    """
    Restituisce una finestra 128×128 centrata sul frame corrente. Se
    siamo all’inizio o alla fine, la parte mancante viene riempita di zeri.
    """
    start = idx - CENTER_FRAME
    if start < 0:
        spec = S.S_db[:, 0 : start + VIEW_FRAMES]          # fino a 128
        pad_left = -start
        spec = np.pad(spec, ((0,0), (pad_left,0)), mode="constant")
    else:
        spec = S.S_db[:, start : start + VIEW_FRAMES]
        if spec.shape[1] < VIEW_FRAMES:                    # fine brano
            pad_right = VIEW_FRAMES - spec.shape[1]
            spec = np.pad(spec, ((0,0), (0,pad_right)), mode="constant")
    return spec

# ─────────── CAM worker thread ───────────────────────────────────────────
def cam_worker(q: queue.Queue, cache: dict, logits_d: dict):
    cam_local = GradCAM(model, target_layer="layer3")   # thread‑local
    while True:
        idx = q.get()
        if idx is None:                # sentinel to terminate old thread
            q.task_done()
            break
        # wait until S_db is available
        if "S_db" not in S:
            q.task_done()
            time.sleep(0.05)
            continue
        spec = get_window(idx)
        z = (spec - spec.mean()) / (spec.std() + 1e-6)
        t = torch.tensor(z).unsqueeze(0).unsqueeze(0)
        t.requires_grad_()
        log = model(t)
        logits_d[idx] = log.softmax(1).squeeze().detach()
        heat = cam_local(log.argmax(1).item(), log)[0].squeeze()
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
        blend   = cv2.resize(blend, (640, 360), interpolation=cv2.INTER_LANCZOS4)
        cache[idx] = blend
        q.task_done()

# ─────────── UI Streamlit ────────────────────────────────────────────────
st.set_page_config(page_title="Live-Mel-CAM", layout="centered")
st.markdown("## 🎧 Live‑Mel‑CAM")
st.markdown(
    """
    <style>
    .transport-btn button {
        border-radius: 50% !important;
        height: 48px !important;
        width: 48px !important;
        font-size: 24px !important;
        padding: 0 !important;
    }
    .transport-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .transport-row .stSlider {
        flex: 1 1 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ─── CSS for rounded image + shadow ───────────────────────────────────────
st.markdown(
    "<style>img{border-radius:8px;box-shadow:0 0 6px #0003;}</style>",
    unsafe_allow_html=True
)

app = st.container()        # main dashboard block

with app:
    up = st.file_uploader("Carica MP3 o WAV", type=["mp3", "wav"])
    if up and (("last_name" not in S) or (up.name != S.last_name)):
        # termina eventuale vecchio worker
        if S.get("todo") is not None:
            try:
                S.todo.put_nowait(None)    # sentinel to shut down thread
            except queue.Full:
                pass
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
        worker = threading.Thread(
            target=cam_worker,
            args=(S.todo, S.cache, S.logits_dict),
            daemon=True
        )
        add_script_run_ctx(worker)   # collega il contesto Streamlit
        worker.start()
        S.todo.put(0)
        S.todo.put(CENTER_FRAME)

    # ─── controlli ───────────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="transport-row">', unsafe_allow_html=True)

        # ── Play / Pause toggle + Stop ──────────────────────────────
        col_toggle, col_stop = st.columns([1, 1], gap="small")

        btn_label = "⏸" if S.play else "▶"
        btn_help  = "Pause" if S.play else "Play"

        with col_toggle:
            toggle_clicked = st.button(
                btn_label,
                key="toggle_btn",
                help=btn_help,
                type="secondary",
                disabled=not up,
            )
        with col_stop:
            stop_clicked = st.button(
                "■",
                key="stop_btn",
                help="Stop",
                type="secondary",
                disabled=not up,
            )

        # slider occupies the full remaining width because of flexbox
        max_val = max(S.max_idx, 1)
        S.idx = st.slider(
            "Posizione",
            0,
            max_val,
            S.idx,
            label_visibility="collapsed",
            disabled=S.play or S.max_idx == 0,
            key="pos_slider",
        )

    if toggle_clicked and up:
        if S.play:
            # was playing → now pause
            sd.stop()
            S.play = False
        else:
            # was paused/stopped → now play
            sd.stop()
            sd.play(S.y[S.idx * HOP :], SR)
            S.start = time.perf_counter() - (S.idx * FRAME_DUR)
            S.play = True

    if stop_clicked and up:
        sd.stop()
        S.play = False
        S.idx  = 0

    # ─── aggiorna indice durante la riproduzione ─────────────────────────────
    if S.play:
        S.idx = int((time.perf_counter() - S.start) / FRAME_DUR)
        if S.idx > S.max_idx:
            S.play = False

    # ─── analisi offline (senza riproduzione audio) ──────────────────────────
    if up and S.S_db is not None:
        if st.button("🔍 Analizza offline (genere completo)", key="offline"):
            with st.spinner("Analisi offline in corso…"):
                # Disattiva gli hook di Grad‑CAM per l'analisi offline
                if hasattr(cam_extr, "remove_hooks"):
                    cam_extr.remove_hooks()

                logits_sum = torch.zeros(10)
                n_frames   = 0
                for i in range(0, S.S_db.shape[1] - 128 + 1, 4):  # stride 4 ≈ 90 ms
                    spec = S.S_db[:, i:i+128]
                    z = (spec - spec.mean()) / (spec.std() + 1e-6)
                    t = torch.tensor(z, dtype=torch.float32).unsqueeze(0).unsqueeze(0).requires_grad_(True)
                    log = model(t)                 # forward con hook
                    logits_sum += log.detach().squeeze()   # stacca grad
                    n_frames   += 1
                g_soft = (logits_sum / n_frames).softmax(0)
                g1 = int(g_soft.argmax()); p1 = float(g_soft[g1])*100
                g_soft[g1] = -1
                g2 = int(g_soft.argmax()); p2 = float(g_soft[g2])*100
                st.success(f"**Genere offline:** {GENRES[g1]} ({p1:.1f} %) – "
                           f"secondo: {GENRES[g2]} ({p2:.1f} %)")
            st.stop()

    # ─── mostra frame corrente ──────────────────────────────────────────────
    if up:
        spec = get_window(S.idx)
        if S.idx not in S.cache and not S.todo.full():
            S.todo.put(S.idx)
        frame = S.cache.get(S.idx)
        if frame is None:
            frame = slice_to_hd(spec)

        # barra verde al centro (istante corrente)
        frame_disp = frame.copy()
        h = frame_disp.shape[0]                      # 360
        cv2.rectangle(frame_disp, (BAR_X-3, 0), (BAR_X+3, h-1), (0, 255, 0), -1)  # verde pieno

        st.image(frame_disp, channels="BGR", width=640)

        # top-2 live
        if S.logits_dict:
            g_soft = torch.stack(list(S.logits_dict.values())).mean(0).softmax(0)
            g1, g2 = torch.topk(g_soft, 2).indices
            p1, p2 = g_soft[g1] * 100, g_soft[g2] * 100
            st.markdown(
                f":musical_note: **{GENRES[g1]}** {p1:.1f}% • "
                f"{GENRES[g2]} {p2:.1f}%"
            )


# ─── auto-refresh mentre suona ───────────────────────────────────────────
if S.play:
    time.sleep(0.12)
    # Streamlit ≥ 1.25 usa `st.rerun`; sulle versioni precedenti rimane `experimental_rerun`
    (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()