"""TPS スタイルの3直交断面ビューア (Varian Eclipse 風 UI 慣習)。

- Axial / Coronal / Sagittal の同時表示
- 世界座標 (X,Y,Z) スライダで全断面同期 + クロスヘア
- 線量カラーウォッシュ + Eclipse 流アイソドーズライン
- 構造体カラーオーバーレイ (輪郭/塗りつぶし、個別 ON/OFF)
- カーソル位置の HU / Physical / EQD2 表示
- DVH パネル
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydicom
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dose_io import (  # noqa: E402
    find_dicom_folder, load_ct, load_rtdose_aligned, list_rtdose_files,
    load_rtstruct_masks, compute_dvh, eqd2_map,
)
from models import model_picker, eqd2_volume  # noqa: E402
from viz import rtdose_label  # noqa: E402

ALPHA_BETA_OPTIONS = [1.0, 1.5, 2.0, 3.0, 10.0]

# Eclipse 流アイソドーズ配色 (高→低 の 8 準位)。
# 実機 (Varian Eclipse) は準位を **絶対線量 [Gy]** で保持し、凡例も Gy 表示。
ISODOSE_COLORS = ["#ff2525", "#ff62a8", "#ffd400", "#37c837",
                  "#2b6cff", "#25c4ff", "#5a4fd0", "#7d5bd0"]
_DEFAULT_FRACS = [1.00, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.10]


def default_levels_gy(peak: float) -> list[float]:
    """ピーク線量から既定のアイソドーズ準位 (絶対 Gy, 高→低) を作る。"""
    return [round(peak * f, 3) for f in _DEFAULT_FRACS]

# 構造体パレット (順に割り当て)
ROI_PALETTE = ["#ff00ff", "#ff5555", "#55ff55", "#55ffff",
               "#ffaa55", "#aa55ff", "#ffff55", "#ffffff"]

WL_PRESETS = {
    "Soft Tissue (W400/L40)":  (400, 40),
    "Lung (W1500/L-600)":      (1500, -600),
    "Bone (W2000/L300)":       (2000, 300),
    "Brain (W80/L40)":         (80, 40),
    "Mediastinum (W350/L50)":  (350, 50),
    "Wide (W2000/L0)":         (2000, 0),
}


# ---------- cached loaders ----------
@st.cache_resource(show_spinner="CT 読み込み中…")
def _load_ct():
    return load_ct(find_dicom_folder())


@st.cache_resource(show_spinner=False)
def _list_rtdose():
    return [f.name for f in list_rtdose_files(find_dicom_folder())]


@st.cache_resource(show_spinner="RTDOSE 読み込み中…")
def _load_dose(name: str):
    return load_rtdose_aligned(find_dicom_folder(), _load_ct(), filename=name)


@st.cache_resource(show_spinner="RTSTRUCT ラスタライズ中…")
def _load_masks():
    return load_rtstruct_masks(find_dicom_folder(), _load_ct())


@st.cache_resource(show_spinner=False)
def _patient_info() -> dict:
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
def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def window_ct(hu_slice, w, l):
    lo, hi = l - w / 2, l + w / 2
    return np.clip((hu_slice - lo) / (hi - lo), 0, 1)


def to_rgb(gray_01):
    return (gray_01[..., None] * 255).astype(np.uint8).repeat(3, axis=-1)


def colorize(values_01, cmap_name):
    from matplotlib import colormaps
    cmap = colormaps[cmap_name]
    rgba = cmap(values_01)
    return (rgba[..., :3] * 255).astype(np.uint8)


def slice_outline_2d(mask_2d):
    if not mask_2d.any():
        return np.zeros_like(mask_2d)
    interior = (
        mask_2d
        & np.roll(mask_2d, 1, axis=0) & np.roll(mask_2d, -1, axis=0)
        & np.roll(mask_2d, 1, axis=1) & np.roll(mask_2d, -1, axis=1)
    )
    return mask_2d & ~interior


def apply_dose_wash(rgb, dose, vmax, threshold, cmap_name, alpha):
    if vmax <= 0:
        return rgb
    norm = np.clip(dose / vmax, 0, 1)
    dose_rgb = colorize(norm, cmap_name)
    mask = dose >= threshold
    a = (mask * alpha)[..., None]
    return (rgb * (1 - a) + dose_rgb * a).astype(np.uint8)


def apply_isodose_lines(rgb, dose, levels_gy):
    """Isodose Levels: 各準位 (絶対 Gy) の等高線のみを描く (塗らない)。"""
    out = rgb.copy()
    for gy, color in levels_gy:
        outline = slice_outline_2d(dose >= gy)
        if outline.any():
            out[outline] = hex_to_rgb(color)
    return out


def apply_isodose_wash(rgb, dose, levels_gy, alpha):
    """Isodose Color Wash: 各準位 (絶対 Gy) を離散バンドで塗る。

    低線量から順に塗り高線量で上書きするので、各 voxel は「超えた最大準位」の色になる。
    Eclipse と同様、線表示 (Isodose Levels) とは **排他** で、同時には重ねない。
    """
    out = rgb.astype(np.float32)
    for gy, color in sorted(levels_gy, key=lambda t: t[0]):  # 低→高
        mask = dose >= gy
        if mask.any():
            c = np.array(hex_to_rgb(color), dtype=np.float32)
            a = (mask * alpha)[..., None]
            out = out * (1 - a) + c * a
    return out.astype(np.uint8)


def apply_roi_overlay(rgb, masks_2d_with_colors, mode: str, fill_alpha: float):
    out = rgb.astype(np.float32)
    for mask_2d, color in masks_2d_with_colors:
        c = np.array(hex_to_rgb(color), dtype=np.float32)
        if mode == "輪郭":
            outline = slice_outline_2d(mask_2d)
            if outline.any():
                out[outline] = c
        else:  # 塗りつぶし
            if mask_2d.any():
                a = (mask_2d * fill_alpha)[..., None]
                out = out * (1 - a) + c * a
    return out.astype(np.uint8)


def draw_crosshair(rgb, h_col: int, v_row: int, color=(0, 220, 255)):
    out = rgb.copy()
    if 0 <= v_row < out.shape[0]:
        out[v_row, :] = color
    if 0 <= h_col < out.shape[1]:
        out[:, h_col] = color
    return out


# ---------- world ↔ index ----------
def world_to_idx(ct, x_w, y_w, z_w):
    nx = ct.cols; ny = ct.rows; nz = len(ct.z_positions)
    x_idx = int(np.clip(round((x_w - ct.origin[0]) / ct.px_x), 0, nx - 1))
    y_idx = int(np.clip(round((y_w - ct.origin[1]) / ct.px_y), 0, ny - 1))
    zs = np.array(ct.z_positions)
    z_idx = int(np.argmin(np.abs(zs - z_w)))
    return x_idx, y_idx, z_idx


# ---------- main ----------
def main():
    from ui_theme import page_header, page_help
    st.markdown(
        "<style>div[data-testid='stMetricValue']{font-size:18px;}</style>",
        unsafe_allow_html=True,
    )
    page_header("画像ビューア — Eclipse 風 3 直交ビュー",
                "Axial / Coronal / Sagittal を世界座標で同期。アイソドーズ + 構造体 + カーソル線量表示。")
    page_help(
        "**何ができる:** 商用 TPS の慣習に倣った 3 直交同期ビューで、線量分布を空間的に確認します。\n\n"
        "**使い方:**\n"
        "1. サイドバーで RTDOSE・分割数 n・α/β・線量 (Physical / EQD2) を選ぶ。\n"
        "2. 「クロスヘア位置」の X / Y / Z スライダを動かすと、Axial / Coronal / Sagittal の3断面が世界座標で連動。\n"
        "3. CT窓・カラーウォッシュ・アイソドーズ線・構造体の表示を切り替え。\n\n"
        "**見方:** クロスヘア位置の HU・Physical・EQD2 が常時表示。アイソドーズは高線量から "
        "赤→橙→黄→緑→シアン→青→紫。")

    ct = _load_ct()
    rd_names = _list_rtdose()
    if not rd_names:
        st.error("RTDOSE がありません。")
        st.stop()
    structures = _load_masks()
    info = _patient_info()

    # ===== Sidebar =====
    g_model, g_params = model_picker()
    st.sidebar.markdown("### プラン")
    rd_name = st.sidebar.selectbox("線量分布 (RTDOSE)", rd_names,
                                   format_func=rtdose_label,
                                   help="評価する線量分布。同梱は水ファントム用の合成サンプルです。")
    dose = _load_dose(rd_name)
    physical = dose.dose_gy
    peak_phys = float(physical.max())

    st.sidebar.markdown("### EQD2")
    n_fx = st.sidebar.number_input("分割数 n", 1, 100, 28, 1)
    ab = st.sidebar.selectbox("α/β (Gy)", ALPHA_BETA_OPTIONS,
                               index=ALPHA_BETA_OPTIONS.index(3.0))
    show_eqd2 = st.sidebar.radio("線量表示", ["Physical", "EQD2"], horizontal=True)
    eqd2_vol = eqd2_volume(g_model, physical, int(n_fx), float(ab), g_params)
    display_vol = eqd2_vol if show_eqd2 == "EQD2" else physical
    peak_display = float(display_vol.max())

    st.sidebar.markdown("### CT ウィンドウ")
    wl = st.sidebar.selectbox("プリセット", list(WL_PRESETS.keys()))
    W, L = WL_PRESETS[wl]

    st.sidebar.markdown("### アイソドーズ")
    st.sidebar.caption("Eclipse と同じく **Color Wash(塗り) と Levels(線) は排他**です。"
                       "準位は絶対線量 [Gy] で、画像左上のパネルで編集できます。")
    wash_alpha = st.sidebar.slider("Color Wash 透明度", 0.0, 1.0, 0.45, 0.05)

    st.sidebar.markdown("### 構造体")
    roi_selections: dict[str, str] = {}
    if structures and structures.rois:
        for i, name in enumerate(structures.rois.keys()):
            color = ROI_PALETTE[i % len(ROI_PALETTE)]
            # default ON: PTV-like or first ROI
            default = (
                name.upper() == "ROI"
                or any(k in name.upper() for k in ("PTV", "GTV", "CTV", "TARGET"))
                or i == 0
            )
            on = st.sidebar.checkbox(name, value=default, key=f"roi_{name}")
            if on:
                roi_selections[name] = color
    roi_mode = st.sidebar.radio("構造体表示", ["輪郭", "塗りつぶし"], horizontal=True)
    roi_alpha = st.sidebar.slider("塗りつぶし透明度", 0.0, 0.5, 0.18, 0.02)

    # ===== Patient banner =====
    banner = st.columns([3, 2, 2, 2])
    banner[0].markdown(
        f"<div style='background:#15202b; padding:8px 14px; border-left:4px solid #00d4ff;'>"
        f"<b>Patient:</b> {info['name'] or '?'}  &nbsp;|&nbsp;  "
        f"<b>ID:</b> {info['id'] or '?'}<br>"
        f"<small>{info['study']} · {info['manufacturer']}</small></div>",
        unsafe_allow_html=True,
    )
    banner[1].markdown(
        f"<div style='background:#15202b; padding:8px 14px;'>"
        f"<b>Plan:</b> {rd_name}<br>"
        f"<small>Physical peak <b>{peak_phys:.2f} Gy</b></small></div>",
        unsafe_allow_html=True,
    )
    banner[2].markdown(
        f"<div style='background:#15202b; padding:8px 14px;'>"
        f"<b>Fractionation:</b> {n_fx} fx<br>"
        f"<small>d at peak <b>{peak_phys/int(n_fx):.2f} Gy/fx</b></small></div>",
        unsafe_allow_html=True,
    )
    banner[3].markdown(
        f"<div style='background:#15202b; padding:8px 14px;'>"
        f"<b>EQD2 peak:</b> {eqd2_vol.max():.2f} Gy<br>"
        f"<small>α/β = {ab}</small></div>",
        unsafe_allow_html=True,
    )

    # ===== World coord controls =====
    nz, ny, nx = ct.hu.shape
    x_min = ct.origin[0]; x_max = ct.origin[0] + (nx - 1) * ct.px_x
    y_min = ct.origin[1]; y_max = ct.origin[1] + (ny - 1) * ct.px_y
    z_min = float(min(ct.z_positions)); z_max = float(max(ct.z_positions))

    # Default crosshair = dose hotspot
    if "tps_xyz" not in st.session_state:
        hz, hy, hx = np.unravel_index(int(physical.argmax()), physical.shape)
        st.session_state.tps_xyz = (
            float(ct.origin[0] + hx * ct.px_x),
            float(ct.origin[1] + hy * ct.px_y),
            float(ct.z_positions[hz]),
        )

    st.markdown(
        "<div style='margin-top:12px;'><b>クロスヘア位置 (世界座標 mm)</b> &nbsp;"
        "<small>3 つのスライダで全断面同期</small></div>",
        unsafe_allow_html=True,
    )
    cc = st.columns(3)
    with cc[0]:
        x_w = st.slider("X", x_min, x_max, st.session_state.tps_xyz[0],
                         step=float(ct.px_x), key="tps_x")
    with cc[1]:
        y_w = st.slider("Y", y_min, y_max, st.session_state.tps_xyz[1],
                         step=float(ct.px_y), key="tps_y")
    with cc[2]:
        z_w = st.slider("Z", z_min, z_max, st.session_state.tps_xyz[2],
                         step=1.0, key="tps_z")
    st.session_state.tps_xyz = (x_w, y_w, z_w)

    x_idx, y_idx, z_idx = world_to_idx(ct, x_w, y_w, z_w)

    # ===== Isodose 制御パネル (左上・Eclipse 準拠) + 3 直交ビュー =====
    # 準位は絶対線量 [Gy]。表示中線量 (Physical/EQD2) ごとに既定値を持つ。
    lv_key = f"iso_levels_{show_eqd2}"
    default_lv = default_levels_gy(peak_display)

    panel_col, views_col = st.columns([1.7, 6])
    with panel_col:
        mode = st.radio("表示", ["Color Wash", "Levels (線)", "オフ"], key="iso_mode",
                        help="Eclipse と同じく塗りと線は排他。同時には重ねません。")
        title = ("Isodose Color Wash [Gy]" if mode == "Color Wash"
                 else "Isodose Levels [Gy]" if mode.startswith("Levels")
                 else "Isodose (オフ)")
        st.markdown(f"<div style='color:#ff5555;font-weight:700;font-size:12px;"
                    f"margin:6px 0 2px'>{title}</div>", unsafe_allow_html=True)

        ed = st.data_editor(
            pd.DataFrame({"✓": [True] * len(default_lv), "Gy": default_lv}),
            hide_index=True, use_container_width=True, key=lv_key,
            column_config={
                "✓": st.column_config.CheckboxColumn("✓", width="small"),
                "Gy": st.column_config.NumberColumn("線量 [Gy]", min_value=0.0,
                                                    step=0.1, format="%.3f"),
            })
        active = [(float(r["Gy"]), ISODOSE_COLORS[i % len(ISODOSE_COLORS)])
                  for i, r in ed.iterrows() if bool(r["✓"])]
        # Eclipse 風の色付き凡例 (チェック済みの準位のみ)
        if active and mode != "オフ":
            st.markdown("".join(
                f"<div style='color:{c};font-weight:700;font-size:12px;"
                f"font-family:monospace;line-height:1.35'>✓ {gy:.3f}</div>"
                for gy, c in active), unsafe_allow_html=True)
        st.caption(f"ピーク {peak_display:.2f} Gy")

    def render(ct_slice, dose_slice, masks_2d, xhair_h, xhair_v):
        rgb = to_rgb(window_ct(ct_slice, W, L))
        # Color Wash と Levels は排他 (Eclipse 準拠)
        if active:
            if mode == "Color Wash":
                rgb = apply_isodose_wash(rgb, dose_slice, active, wash_alpha)
            elif mode.startswith("Levels"):
                rgb = apply_isodose_lines(rgb, dose_slice, active)
        if masks_2d:
            rgb = apply_roi_overlay(rgb, masks_2d, roi_mode, roi_alpha)
        return draw_crosshair(rgb, xhair_h, xhair_v)

    # Axial: (Y, X), crosshair at (col=x_idx, row=y_idx)
    ax_img = render(
        ct.hu[z_idx], display_vol[z_idx],
        [(structures.rois[n][z_idx], c) for n, c in roi_selections.items()],
        x_idx, y_idx,
    )
    # Coronal: (Z flipped, X), crosshair at (col=x_idx, row=(nz-1)-z_idx)
    cor_img = render(
        ct.hu[::-1, y_idx, :], display_vol[::-1, y_idx, :],
        [(structures.rois[n][::-1, y_idx, :], c) for n, c in roi_selections.items()],
        x_idx, (nz - 1) - z_idx,
    )
    # Sagittal: (Z flipped, Y), crosshair at (col=y_idx, row=(nz-1)-z_idx)
    sag_img = render(
        ct.hu[::-1, :, x_idx], display_vol[::-1, :, x_idx],
        [(structures.rois[n][::-1, :, x_idx], c) for n, c in roi_selections.items()],
        y_idx, (nz - 1) - z_idx,
    )

    with views_col:
        vcols = st.columns(3)
        with vcols[0]:
            st.image(Image.fromarray(ax_img),
                     caption=f"Axial   Z = {z_w:+.1f} mm   ({z_idx+1}/{nz})",
                     use_container_width=True)
        with vcols[1]:
            st.image(Image.fromarray(cor_img),
                     caption=f"Coronal   Y = {y_w:+.1f} mm   ({y_idx+1}/{ny})",
                     use_container_width=True)
        with vcols[2]:
            st.image(Image.fromarray(sag_img),
                     caption=f"Sagittal   X = {x_w:+.1f} mm   ({x_idx+1}/{nx})",
                     use_container_width=True)

    # ===== Cursor info bar =====
    hu_at = float(ct.hu[z_idx, y_idx, x_idx])
    dose_at = float(physical[z_idx, y_idx, x_idx])
    eqd2_at = float(eqd2_vol[z_idx, y_idx, x_idx])

    ic = st.columns(4)
    ic[0].metric("HU @ crosshair", f"{hu_at:.0f}")
    ic[1].metric("Physical @ crosshair", f"{dose_at:.2f} Gy")
    ic[2].metric(f"EQD2 (n={n_fx}, α/β={ab})", f"{eqd2_at:.2f} Gy",
                  delta=f"{eqd2_at - dose_at:+.2f} Gy vs Physical")
    if peak_phys > 0:
        ic[3].metric("dose / Rx peak", f"{100*dose_at/peak_phys:.1f} %")

    # ===== DVH panel =====
    if structures and roi_selections:
        st.markdown("---")
        st.subheader("DVH")
        fig = go.Figure()
        for name, color in roi_selections.items():
            mask = structures.rois[name]
            d, v = compute_dvh(physical, mask)
            fig.add_trace(go.Scatter(
                x=d, y=v, mode="lines", name=f"{name} Phys",
                line=dict(color=color, width=2.5),
            ))
            d, v = compute_dvh(eqd2_vol, mask)
            fig.add_trace(go.Scatter(
                x=d, y=v, mode="lines", name=f"{name} EQD2",
                line=dict(color=color, width=2, dash="dash"),
            ))
        fig.update_layout(
            xaxis_title="Dose (Gy)", yaxis_title="Volume (%)",
            yaxis_range=[0, 105], height=380,
            template="plotly_dark", hovermode="x unified",
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        )
        st.plotly_chart(fig, use_container_width=True)


main()
