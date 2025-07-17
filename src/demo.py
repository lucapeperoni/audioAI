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
def load_models():
    """Carica 4 modelli ResNet-18 per le rappresentazioni: mel, lin, mfcc, dmfcc."""
    models = {}
    cams = {}
    reps = ['mel', 'lin', 'mfcc', 'dmfcc']
    ckpt_by_rep = {
        "mel": "best_resnet18_mel_fold4.pt",
        "lin": "best_resnet18_lin_fold3.pt",
        "mfcc": "best_resnet18_mfcc_fold5.pt",
        "dmfcc": "best_resnet18_dmfcc_fold4.pt",
    }
    for rep in reps:
        # carica il checkpoint dal path assoluto: <repo>/models/<specific_ckpt>.pt
        ckpt_path = pathlib.Path(__file__).resolve().parent.parent / "models" / ckpt_by_rep[rep]
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            in_channels = ckpt["conv1.weight"].shape[1]
            m = resnet18(weights=None)
            m.conv1 = torch.nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
            m.fc    = torch.nn.Linear(512, 10)
            m.load_state_dict(ckpt)
        else:
            st.error(f"⚠️ Checkpoint non trovato: {ckpt_path}")
            # fallback: assume 1 channel for mel/lin, 2 for mfcc/dmfcc
            m = resnet18(weights=None)
            in_channels = 1 if rep in ['mel', 'lin'] else 2
            m.conv1 = torch.nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
            m.fc    = torch.nn.Linear(512, 10)
        m.eval()
        cam = GradCAM(m, target_layer="layer3")
        models[rep] = m
        cams[rep] = cam
    return models, cams

models, cams = load_models()

# ─────────── session state init ──────────────────────────────────────────
S = st.session_state
for k, v in dict(play=False, start=0.0, idx=0,
                 y=None, S_db=None, max_idx=0,
                 cache={}, logits_dict={}, todo=None,
                 rep='mel').items():
    S.setdefault(k, v)

# ─────────── helper audio → spettrogrammi ──────────────────────────────────────────
def compute_all_specs(y: np.ndarray) -> dict[str, np.ndarray]:
    """Calcola e ritorna tutti e 4 gli spettrogrammi richiesti."""
    specs = {}
    # Mel spectrogram
    S_mel = librosa.feature.melspectrogram(
        y=y, sr=SR,
        n_fft=2048, hop_length=HOP,
        n_mels=N_MELS, fmin=20, fmax=11_000
    )
    specs['mel'] = librosa.power_to_db(S_mel, ref=np.max)
    # Linear spectrogram (log power)
    S_lin = librosa.stft(y, n_fft=2048, hop_length=HOP, win_length=2048)
    S_lin_power = np.abs(S_lin)**2
    S_lin_db = librosa.power_to_db(S_lin_power, ref=np.max)
    # Limit freq to 11kHz (SR=22050, freq bins = n_fft/2+1=1025)
    freq_bin_limit = int(11000 / (SR / 2) * (S_lin_db.shape[0]-1))
    specs['lin'] = S_lin_db[:freq_bin_limit+1, :]
    # MFCC (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, hop_length=HOP)
    specs['mfcc'] = mfcc
    # Delta MFCC (first order difference)
    dmfcc = librosa.feature.delta(mfcc)
    specs['dmfcc'] = np.vstack([mfcc, dmfcc])  # shape: (26, frames)
    return specs

# ─────────── helper slice → HD frame ────────────────────────────────────
def slice_to_hd(spec_low: np.ndarray) -> np.ndarray:
    """
    Converte una finestra di spettrogramma (spec_low) in un frame HD 640×360 BGR:
    * Ridimensionamento verticale personalizzato in base al tipo di spettrogramma.
    * Applica colormap per migliorare la leggibilità.
    """
    freq_bins, time_frames = spec_low.shape

    # upsampling verticale differenziato per migliorare leggibilità
    if freq_bins == 13 or freq_bins == 26:
        target_h = 256
        colormap = cv2.COLORMAP_JET
    elif freq_bins == 52:
        target_h = 300
        colormap = cv2.COLORMAP_JET
    elif freq_bins <= 128:
        target_h = 384
        colormap = cv2.COLORMAP_INFERNO
    else:
        target_h = 512
        colormap = cv2.COLORMAP_INFERNO

    spec_hd = cv2.resize(spec_low, (time_frames, target_h), interpolation=cv2.INTER_LANCZOS4)
    spec_hd = cv2.resize(spec_hd, (640, 360), interpolation=cv2.INTER_LANCZOS4)

    spec_u8 = cv2.normalize(spec_hd, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    spec_col = cv2.applyColorMap(spec_u8, colormap)

    return spec_col

def get_window(idx: int, rep: str) -> np.ndarray:
    """
    Restituisce una finestra centrata sul frame corrente per la rappresentazione rep.
    Se siamo all’inizio o alla fine, la parte mancante viene riempita di zeri.
    """
    start = idx - CENTER_FRAME
    spec_full = S.S_db[rep]
    if start < 0:
        spec = spec_full[:, 0 : start + VIEW_FRAMES]          # fino a 128
        pad_left = -start
        spec = np.pad(spec, ((0,0), (pad_left,0)), mode="constant")
    else:
        spec = spec_full[:, start : start + VIEW_FRAMES]
        if spec.shape[1] < VIEW_FRAMES:                    # fine brano
            pad_right = VIEW_FRAMES - spec.shape[1]
            spec = np.pad(spec, ((0,0), (0,pad_right)), mode="constant")
    return spec

# ─────────── CAM worker thread ───────────────────────────────────────────
def cam_worker(q: queue.Queue, cache: dict, logits_d: dict):
    while True:
        idx, rep = q.get()
        if idx is None:                # sentinel to terminate old thread
            q.task_done()
            break
        # wait until S_db is available
        if "S_db" not in S or rep not in S.S_db:
            q.task_done()
            time.sleep(0.05)
            continue
        spec = get_window(idx, rep)
        z = (spec - spec.mean()) / (spec.std() + 1e-6)
        if rep == 'dmfcc':
            if z.shape[0] == 26:
                z = z.reshape(2, 13, z.shape[1])  # (2, 13, 128)
            t = torch.tensor(z).unsqueeze(0).float()  # (1, 2, 13, 128)
        elif rep == 'mfcc':
            t = torch.tensor(z).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 13, 128)
        else:
            t = torch.tensor(z).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 128, 128)
        t.requires_grad_()
        model = models[rep]
        cam_local = GradCAM(model, target_layer="layer3")
        log = model(t)
        logits_d[(idx, rep)] = log.softmax(1).squeeze().detach()
        heat = cam_local(log.argmax(1).item(), log)[0].squeeze()
        # Garantisce che heat sia 2D
        if heat.dim() == 1:
            h_h, w_w = spec.shape[-2], spec.shape[-1]
            heat = heat.view(1, 1, 1, -1)
            heat = F.interpolate(heat, size=(h_h, w_w), mode='bilinear', align_corners=False)[0, 0]
        elif heat.dim() == 2:
            heat = F.interpolate(heat.unsqueeze(0).unsqueeze(0), size=spec.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
        else:
            # fallback: tenta comunque con squeeze
            heat = F.interpolate(heat.unsqueeze(0).unsqueeze(0), size=spec.shape[-2:], mode='bilinear', align_corners=False).squeeze()
        heat_u8 = cv2.normalize(heat.numpy(), None, 0, 255,
                                cv2.NORM_MINMAX).astype('uint8')
        mask = heat_u8 > 180
        spec_u8 = cv2.normalize(spec, None, 0, 255,
                                cv2.NORM_MINMAX).astype('uint8')
        spec_c  = cv2.applyColorMap(spec_u8, cv2.COLORMAP_INFERNO)
        red     = spec_c.copy(); red[mask] = (0,0,255)
        blend   = cv2.addWeighted(spec_c, 0.7, red, 0.3, 0)
        blend   = cv2.resize(blend, (640, 360), interpolation=cv2.INTER_LANCZOS4)
        cache[(idx, rep)] = blend
        q.task_done()

# ─────────── UI Streamlit ────────────────────────────────────────────────
st.set_page_config(page_title="Riconoscimento Genere Musicale", layout="centered")
st.markdown("## 🎵 Riconoscimento Genere Musicale – Multi‑rappresentazione")
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
                S.todo.put_nowait((None, None))    # sentinel to shut down thread
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
        # Calcola tutti gli spettrogrammi
        S.S_db = compute_all_specs(y)
        # Padding per ogni rappresentazione se necessario
        for rep in S.S_db:
            if S.S_db[rep].shape[1] < 128:
                S.S_db[rep] = np.pad(S.S_db[rep], ((0,0), (0, 128 - S.S_db[rep].shape[1])))
        # max_idx calcolato come il minimo tra tutte le rep
        max_idx_vals = []
        for rep in S.S_db:
            max_idx_vals.append(S.S_db[rep].shape[1] - 128)
        S.max_idx = min(max_idx_vals)
        S.cache = {}
        S.logits_dict = {}
        S.todo = queue.Queue(maxsize=64)
        worker = threading.Thread(
            target=cam_worker,
            args=(S.todo, S.cache, S.logits_dict),
            daemon=True
        )
        add_script_run_ctx(worker)   # collega il contesto Streamlit
        worker.start()
        S.todo.put((0, S.rep))
        S.todo.put((CENTER_FRAME, S.rep))

    # ─── selezione rappresentazione ───────────────────────────────────────────
    if up:
        rep_sel = st.radio("Seleziona rappresentazione", ['mel', 'lin', 'mfcc', 'dmfcc'], index=['mel', 'lin', 'mfcc', 'dmfcc'].index(S.rep))
        if rep_sel != S.rep:
            S.rep = rep_sel
            # reset cache e logits per la nuova rep
            S.cache = {}
            S.logits_dict = {}
            S.todo.queue.clear()
            S.todo.put((S.idx, S.rep))
            S.todo.put((CENTER_FRAME, S.rep))
            st.rerun()

    # ─── controlli ───────────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="transport-row">', unsafe_allow_html=True)

        # ── Play / Pause toggle + Restart + Offline ──────────────
        col_toggle, col_restart, col_offline = st.columns([1, 1, 2], gap="small")

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
        with col_restart:
            restart_clicked = st.button(
                "↺",
                key="restart_btn",
                help="Restart",
                type="secondary",
                disabled=not up,
            )
        with col_offline:
            offline_clicked = st.button(
                "🔍",
                key="offline_btn",
                help="Analisi offline",
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

    if restart_clicked and up:
        sd.stop()
        S.play = False
        S.idx = 0
        sd.play(S.y, SR)
        S.start = time.perf_counter()
        S.play = True

    # ─── aggiorna indice durante la riproduzione ─────────────────────────────
    if S.play:
        S.idx = int((time.perf_counter() - S.start) / FRAME_DUR)
        if S.idx > S.max_idx:
            S.play = False
        else:
            if not S.todo.full():
                S.todo.put((S.idx, S.rep))

    # ─── analisi offline (senza riproduzione audio) ──────────────────────────
    if up and S.S_db is not None:
        if offline_clicked:
            with st.spinner("Analisi offline in corso…"):
                results = {}
                # Disattiva gli hook di Grad‑CAM per l'analisi offline
                for rep in ['mel', 'lin', 'mfcc', 'dmfcc']:
                    cam = cams[rep]
                    if hasattr(cam, "remove_hooks"):
                        cam.remove_hooks()
                    logits_sum = torch.zeros(10)
                    n_frames   = 0
                    spec_full = S.S_db[rep]
                    max_i = spec_full.shape[1] - 128 + 1
                    for i in range(0, max_i, 4):  # stride 4 ≈ 90 ms
                        spec = spec_full[:, i:i+128]
                        # Nuovo blocco per normalizzazione e batching (compatibilità canali/modello)
                        z = (spec - spec.mean()) / (spec.std() + 1e-6)
                        if rep in ['mfcc', 'dmfcc'] and z.shape[0] == 26:
                            z = z.reshape(2, 13, z.shape[1])  # (2, 13, 128)
                        t = torch.tensor(z, dtype=torch.float32)
                        if t.dim() == 2:
                            t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                        elif t.dim() == 3:
                            t = t.unsqueeze(0)               # (1, C, H, W)
                        model = models[rep]
                        log = model(t)                 # forward con hook
                        logits_sum += log.detach().squeeze()   # stacca grad
                        n_frames   += 1
                    g_soft = (logits_sum / n_frames).softmax(0)
                    g1 = int(g_soft.argmax()); p1 = float(g_soft[g1])*100
                    g_soft[g1] = -1
                    g2 = int(g_soft.argmax()); p2 = float(g_soft[g2])*100
                    results[rep] = (g1, p1, g2, p2)
                for rep in results:
                    g1, p1, g2, p2 = results[rep]
                    st.success(f"**Genere offline [{rep}]:** {GENRES[g1]} ({p1:.1f} %) – "
                               f"secondo: {GENRES[g2]} ({p2:.1f} %)")
            st.stop()

    # ─── mostra frame corrente ──────────────────────────────────────────────
    if up:
        spec = get_window(S.idx, S.rep)
        if (S.idx, S.rep) not in S.cache and not S.todo.full():
            S.todo.put((S.idx, S.rep))
        frame = S.cache.get((S.idx, S.rep))
        if frame is None:
            frame = slice_to_hd(spec)

        # barra verde al centro (istante corrente)
        frame_disp = frame.copy()
        h = frame_disp.shape[0]                      # 360
        cv2.rectangle(frame_disp, (BAR_X-3, 0), (BAR_X+3, h-1), (0, 255, 0), -1)  # verde pieno

        st.image(frame_disp, channels="BGR", width=640)

        # top-2 live per la rappresentazione corrente
        if S.logits_dict:
            logits = [v for k,v in S.logits_dict.items() if k[1]==S.rep]
            if logits:
                g_soft = torch.stack(logits).mean(0).softmax(0)
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