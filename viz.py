"""画像・DVH 表示の共通ヘルパ + キャッシュ済みローダ。

各ページで重複していた window/colorize/overlay/slice 系をここに集約。
"""
from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image

from dose_io import (
    find_dicom_folder, load_ct, load_rtdose_aligned, list_rtdose_files,
    load_rtstruct_masks,
)
import pydicom

WL_PRESETS = {
    "軟部 (W400/L40)": (400, 40),
    "肺 (W1500/L-600)": (1500, -600),
    "骨 (W2000/L300)": (2000, 300),
    "脳 (W80/L40)": (80, 40),
    "縦隔 (W350/L50)": (350, 50),
    "広域 (W2000/L0)": (2000, 0),
}

ROI_PALETTE = ["#ff00ff", "#ff5555", "#55ff55", "#55ffff",
               "#ffaa55", "#aa55ff", "#ffff55", "#ffffff"]

# RTDOSE ファイル名 → 分かりやすい表示ラベル(水ファントムの合成サンプル)
_RTDOSE_LABELS = {
    "RD.prior.dcm": "サンプル: 過去プラン 50 Gy(通常分割の想定)",
    "RD.current.dcm": "サンプル: 今回プラン 30 Gy(SBRT の想定)",
    "RD.synthetic.dcm": "サンプル: デモ用 60 Gy",
}


def rtdose_label(name: str) -> str:
    """RTDOSE ファイル名を人が読める表示名に変換(未知ファイルは素名を返す)。"""
    if name in _RTDOSE_LABELS:
        return _RTDOSE_LABELS[name]
    return name.replace(".dcm", "")

# Eclipse 慣習のアイソドーズ配色 (高→低 %)
ISODOSE_LEVELS = [
    (100, "#ff2020"), (95, "#ff7700"), (90, "#ffbb00"), (80, "#ffff00"),
    (70, "#88ff00"), (50, "#00ffaa"), (30, "#00aaff"), (10, "#9966ff"),
]

# 詳細アイソドーズ: 高線量は 5% 刻み(TPS慣習の配色)、低線量は 10% 刻み(重ね合わせの裾野把握用)
# %は「基準線量(100%)」に対する割合。基準を処方線量にすると 105/110% のホットスポットも見える。
ISODOSE_FULL = [
    (110, "#ff2525"),  # 赤
    (105, "#ff62a8"),  # ピンク
    (100, "#ffd400"),  # 黄
    (95,  "#37c837"),  # 緑
    (90,  "#2b6cff"),  # 青
    (85,  "#25c4ff"),  # 水色
    (80,  "#5a4fd0"),  # 藍
    (70,  "#7d5bd0"),  # 以下は低線量(裾野)
    (60,  "#9a5fce"),
    (50,  "#b25fbf"),
    (40,  "#8f6fcb"),
    (30,  "#6f86cf"),
    (20,  "#5aa0c8"),
    (10,  "#4fb8b8"),
]


# ---------- cached loaders (全ページ共通) ----------
@st.cache_resource(show_spinner="CT 読み込み中…")
def get_ct():
    return load_ct(find_dicom_folder())


@st.cache_resource(show_spinner=False)
def get_rtdose_names() -> list[str]:
    return [f.name for f in list_rtdose_files(find_dicom_folder())]


@st.cache_resource(show_spinner="RTDOSE 読み込み中…")
def get_dose(name: str):
    return load_rtdose_aligned(find_dicom_folder(), get_ct(), filename=name)


@st.cache_resource(show_spinner="RTSTRUCT ラスタライズ中…")
def get_masks():
    return load_rtstruct_masks(find_dicom_folder(), get_ct())


@st.cache_resource(show_spinner=False)
def get_patient_info() -> dict:
    folder = find_dicom_folder()
    for f in folder.rglob("*.dcm"):
        try:
            d = pydicom.dcmread(f, stop_before_pixels=True)
            if d.Modality == "CT":
                return {
                    "name": str(getattr(d, "PatientName", "")),
                    "id": str(getattr(d, "PatientID", "")),
                    "study": str(getattr(d, "StudyDescription", "")),
                    "manufacturer": str(getattr(d, "Manufacturer", "")),
                }
        except Exception:
            continue
    return {"name": "", "id": "", "study": "", "manufacturer": ""}


# ---------- image helpers ----------
def window_ct(hu_slice, w, l):
    lo, hi = l - w / 2, l + w / 2
    return np.clip((hu_slice - lo) / (hi - lo), 0, 1)


def to_rgb(gray01):
    return (gray01[..., None] * 255).astype(np.uint8).repeat(3, axis=-1)


def colorize(values01, cmap_name):
    from matplotlib import colormaps
    rgba = colormaps[cmap_name](values01)
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay(ct_gray01, dose, vmax, threshold, cmap, alpha, divergent=False):
    base = to_rgb(ct_gray01)
    if vmax <= 0:
        return base
    if divergent:
        norm = np.clip((dose + vmax) / (2 * vmax), 0, 1)
        mask = np.abs(dose) >= threshold
    else:
        norm = np.clip(dose / vmax, 0, 1)
        mask = dose >= threshold
    rgb = colorize(norm, cmap)
    a = (mask * alpha)[..., None]
    return (base * (1 - a) + rgb * a).astype(np.uint8)


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def slice_outline_2d(mask_2d):
    if not mask_2d.any():
        return np.zeros_like(mask_2d)
    interior = (mask_2d
                & np.roll(mask_2d, 1, 0) & np.roll(mask_2d, -1, 0)
                & np.roll(mask_2d, 1, 1) & np.roll(mask_2d, -1, 1))
    return mask_2d & ~interior


def apply_roi_outlines(img_rgb, roi_slices):
    out = img_rgb.copy()
    for mask2d, color in roi_slices:
        outline = slice_outline_2d(mask2d)
        if outline.any():
            out[outline] = hex_to_rgb(color) if isinstance(color, str) else color
    return out


def apply_isodose_lines(img_rgb, dose, vmax, levels):
    out = img_rgb.copy()
    for pct, color in levels:
        mask = dose >= vmax * pct / 100.0
        outline = slice_outline_2d(mask)
        if outline.any():
            out[outline] = hex_to_rgb(color)
    return out


def take_slice(vol, axis, idx):
    """axis: 'Axial' / 'Coronal' / 'Sagittal'。Coronal/Sagittal は上下反転(上=頭側)。"""
    if axis == "Axial":
        return vol[idx]
    if axis == "Coronal":
        return vol[::-1, idx, :]
    return vol[::-1, :, idx]


def slice_count(shape, axis):
    return shape[{"Axial": 0, "Coronal": 1, "Sagittal": 2}[axis]]


def show_image(img_rgb, caption=""):
    st.image(Image.fromarray(img_rgb), caption=caption, use_container_width=True)


def colorbar(vmax, cmap, divergent=False, label=""):
    grad = np.tile(np.linspace(0, 1, 512), (18, 1))
    st.caption(label or (f"{-vmax:+.1f} … {vmax:+.1f} Gy" if divergent else f"0 … {vmax:.1f} Gy"))
    st.image(Image.fromarray(colorize(grad, cmap)), use_container_width=True)
