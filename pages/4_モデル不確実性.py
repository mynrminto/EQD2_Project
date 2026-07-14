"""モデル・不確実性 — モデル選択と α/β の感度を検討する QA / 教育ページ。

[QA・教育] 同じプランを 4 モデルで voxel-wise 比較し、α/β の文献的ばらつきが
EQD2 にどれだけ効くか (worst-case) を可視化する。臨床判断そのものではなく、
「モデル・パラメータの選択が結論をどれだけ動かすか」を確かめるための場所。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dose_io import (  # noqa: E402
    compute_dvh, dvh_metrics, eqd2_map, eqd2_lq_l, eqd2_usc, eqd2_ir,
    usc_transition_dose, eqd2_range_map, suggest_alpha_beta,
)
from models import ALPHA_BETA_OPTIONS, ALPHA_BETA_HINT, MODEL_COLORS  # noqa: E402
from ui_theme import apply_theme, page_header
import viz  # noqa: E402


def tab_model_compare(ct, dose, structures):
    st.caption("同じプランを 4 モデルで voxel-wise 計算。低線量=IR補正、高線量=LQ-L/USCがLQの過大評価を補正。")
    c = st.columns([1, 1, 1])
    n_fx = c[0].number_input("分割数 n", 1, 100, 3, 1, key="ml_n")
    ab = c[1].selectbox("α/β (Gy)", ALPHA_BETA_OPTIONS, index=ALPHA_BETA_OPTIONS.index(10.0),
                        format_func=lambda x: ALPHA_BETA_HINT[x], key="ml_ab")
    axis = c[2].selectbox("断面", ["Axial", "Coronal", "Sagittal"], key="ml_axis")

    with st.expander("モデルパラメータ"):
        pc = st.columns(4)
        d_T = pc[0].number_input("LQ-L d_T", 1.0, 20.0, 6.0, 0.5, key="ml_dT")
        usc_a = pc[1].number_input("USC α", 0.05, 1.0, 0.30, 0.01, key="ml_a")
        usc_d0 = pc[2].number_input("USC D0", 0.5, 3.0, 1.25, 0.05, key="ml_d0")
        usc_dq = pc[3].number_input("USC Dq", 0.0, 5.0, 1.8, 0.1, key="ml_dq")
        pc2 = st.columns(4)
        ir_ratio = pc2[0].number_input("IR α_s/α_r", 1.0, 15.0, 3.0, 0.5, key="ml_irr")
        d_c = pc2[1].number_input("IR d_c", 0.05, 1.0, 0.25, 0.05, key="ml_dc")

    physical = dose.dose_gy
    maps = {
        "古典LQ": eqd2_map(physical, int(n_fx), float(ab)),
        "LQ-L (Astrahan)": eqd2_lq_l(physical, int(n_fx), float(ab), d_T=d_T),
        "USC (Park/Timmerman)": eqd2_usc(physical, int(n_fx), float(ab),
                                         alpha=usc_a, D0=usc_d0, Dq=usc_dq),
        "IR (Induced Repair)": eqd2_ir(physical, int(n_fx), float(ab),
                                       ir_ratio=ir_ratio, d_c=d_c),
    }
    vmax = max(float(m.max()) for m in maps.values())
    n_slices = viz.slice_count(physical.shape, axis)
    hot = int(np.unravel_index(int(physical.argmax()), physical.shape)[
        {"Axial": 0, "Coronal": 1, "Sagittal": 2}[axis]])
    idx = st.slider("スライス", 0, n_slices - 1, hot, key="ml_idx")
    ct_gray = viz.window_ct(viz.take_slice(ct.hu, axis, idx), 2000, 0)

    cols = st.columns(4)
    for col, (name, m) in zip(cols, maps.items()):
        with col:
            img = viz.overlay(ct_gray, viz.take_slice(m, axis, idx), vmax, vmax * 0.05, "jet", 0.55)
            viz.show_image(img, f"{name.split()[0]} (peak {m.max():.1f})")

    # d/fx 応答曲線
    d_axis = np.linspace(0.1, 25, 200)
    n = int(n_fx)
    curves = {
        "古典LQ": eqd2_map(d_axis * n, n, ab),
        "LQ-L (Astrahan)": eqd2_lq_l(d_axis * n, n, ab, d_T=d_T),
        "USC (Park/Timmerman)": eqd2_usc(d_axis * n, n, ab, alpha=usc_a, D0=usc_d0, Dq=usc_dq),
        "IR (Induced Repair)": eqd2_ir(d_axis * n, n, ab, ir_ratio=ir_ratio, d_c=d_c),
    }
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure()
        for name, y in curves.items():
            fig.add_trace(go.Scatter(x=d_axis, y=y, name=name.split()[0],
                                     line=dict(color=MODEL_COLORS[name], width=2.5)))
        fig.add_vrect(x0=8, x1=25, fillcolor="rgba(239,68,68,0.08)", line_width=0)
        fig.update_layout(title=f"d/fx vs EQD2 (n={n_fx}, α/β={ab})", xaxis_title="d per fraction (Gy)",
                          yaxis_title="EQD2 (Gy)", height=380, template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=46, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if structures and structures.rois:
            roi_opts = list(structures.rois.keys())
            default = next((i for i, nm in enumerate(roi_opts)
                            if any(k in nm.upper() for k in ("PTV", "ROI", "GTV", "WATER"))), 0)
            roi = st.selectbox("DVH 対象 ROI", roi_opts, index=default, key="ml_roi")
            fig = go.Figure()
            dashes = {"古典LQ": None, "LQ-L (Astrahan)": "dash",
                      "USC (Park/Timmerman)": "dot", "IR (Induced Repair)": "dashdot"}
            for name, m in maps.items():
                d, v = compute_dvh(m, structures.rois[roi])
                fig.add_trace(go.Scatter(x=d, y=v, name=name.split()[0],
                                         line=dict(color=MODEL_COLORS[name], width=2.5, dash=dashes[name])))
            fig.update_layout(title=f"{roi} DVH (モデル別)", xaxis_title="EQD2 (Gy)",
                              yaxis_title="Volume (%)", yaxis_range=[0, 105], height=380,
                              template="plotly_dark", hovermode="x unified",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=46, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)


def tab_ab_uncertainty(ct, dose, structures):
    st.caption("α/β の文献的ばらつき (例: 脊髄 1.7–4.4) で EQD2 を範囲評価。OAR は低 α/β が worst-case。")
    c = st.columns([1, 1, 1, 1])
    n_fx = c[0].number_input("分割数 n", 1, 100, 5, 1, key="abu_n")
    ab_low = c[1].number_input("α/β low", 0.5, 15.0, 2.0, 0.5, key="abu_lo")
    ab_high = c[2].number_input("α/β high", 0.5, 15.0, 5.0, 0.5, key="abu_hi")
    mdl = c[3].selectbox("モデル", ["LQ", "LQ-L"], key="abu_model")
    if ab_low >= ab_high:
        st.warning("low < high にしてください。")
        return

    physical = dose.dose_gy
    r = eqd2_range_map(physical, int(n_fx), float(ab_low), float(ab_high), model=mdl)
    axis = st.selectbox("断面", ["Axial", "Coronal", "Sagittal"], key="abu_axis")
    n_slices = viz.slice_count(physical.shape, axis)
    idx = st.slider("スライス", 0, n_slices - 1, n_slices // 2, key="abu_idx")
    vmax = float(r["max"].max())
    ct_gray = viz.window_ct(viz.take_slice(ct.hu, axis, idx), 2000, 0)

    sub = st.columns(3)
    with sub[0]:
        viz.show_image(viz.overlay(ct_gray, viz.take_slice(r["low_ab"], axis, idx), vmax, vmax*0.05, "jet", 0.5),
                       f"α/β={ab_low} (peak {r['low_ab'].max():.1f})")
    with sub[1]:
        viz.show_image(viz.overlay(ct_gray, viz.take_slice(r["high_ab"], axis, idx), vmax, vmax*0.05, "jet", 0.5),
                       f"α/β={ab_high} (peak {r['high_ab'].max():.1f})")
    with sub[2]:
        rng = max(float(r["range"].max()), 1e-6)
        viz.show_image(viz.overlay(ct_gray, viz.take_slice(r["range"], axis, idx), rng, rng*0.05, "hot", 0.6),
                       f"不確実性幅 |low−high| (peak {r['range'].max():.1f})")

    if structures and structures.rois:
        roi_opts = list(structures.rois.keys())
        default = next((i for i, nm in enumerate(roi_opts)
                        if (sa := suggest_alpha_beta(nm)) is not None and sa <= 3.0), 0)
        roi = st.selectbox("DVH 対象 ROI", roi_opts, index=default, key="abu_roi")
        d_lo, v_lo = compute_dvh(r["low_ab"], structures.rois[roi])
        d_hi, v_hi = compute_dvh(r["high_ab"], structures.rois[roi])
        xc = np.linspace(0, max(d_lo[-1], d_hi[-1]), 250)
        vl, vh = np.interp(xc, d_lo, v_lo), np.interp(xc, d_hi, v_hi)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xc, y=np.maximum(vl, vh), line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xc, y=np.minimum(vl, vh), line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(245,158,11,0.25)", name=f"不確実性帯 [α/β {ab_low}..{ab_high}]"))
        fig.add_trace(go.Scatter(x=d_lo, y=v_lo, name=f"α/β={ab_low} (OAR worst)", line=dict(color="#ef4444", width=2.5)))
        fig.add_trace(go.Scatter(x=d_hi, y=v_hi, name=f"α/β={ab_high}", line=dict(color="#3b82f6", width=2.5, dash="dash")))
        fig.update_layout(title=f"{roi} DVH 不確実性帯 ({mdl}, n={n_fx})", xaxis_title="EQD2 (Gy)",
                          yaxis_title="Volume (%)", yaxis_range=[0, 105], height=420, template="plotly_dark",
                          hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=46, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="モデル・不確実性", page_icon="🔬", layout="wide")
    apply_theme()
    page_header("モデル・不確実性 — QA / 教育",
                "モデル選択と α/β の感度を検討。臨床判断ではなく「選択が結論をどれだけ動かすか」を確認。",
                badges=["4モデル voxel 比較", "α/β worst-case", "ReCOG 2024 課題"])

    ct = viz.get_ct()
    rd_names = viz.get_rtdose_names()
    if not rd_names:
        st.error("RTDOSE がありません。")
        st.stop()
    rd_name = st.sidebar.selectbox("RTDOSE", rd_names)
    dose = viz.get_dose(rd_name)
    structures = viz.get_masks()
    st.caption(f"RTDOSE: {rd_name} | peak {dose.dose_gy.max():.1f} Gy")

    t1, t2 = st.tabs(["4モデル voxel 比較", "α/β 不確実性 (worst-case)"])
    with t1:
        tab_model_compare(ct, dose, structures)
    with t2:
        tab_ab_uncertainty(ct, dose, structures)


if __name__ == "__main__":
    main()
