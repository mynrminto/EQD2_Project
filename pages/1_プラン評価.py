"""プラン評価 — 単一プランの画像 + DVH + 構造別 α/β を1ページに統合。

[計画レビュー] 1つの RTDOSE を読み込み、画像オーバーレイ・DVH・構造別評価を
同じページのタブで確認する。モデル/α/β はサイドバーの全ページ共通設定を使用。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dose_io import compute_dvh, dvh_metrics, suggest_alpha_beta  # noqa: E402
from models import (  # noqa: E402
    ALPHA_BETA_OPTIONS, ALPHA_BETA_HINT, model_picker, eqd2_volume,
)
from ui_theme import apply_theme, page_header
import viz  # noqa: E402

ROI_COLORS = ["#22d3ee", "#ef4444", "#22c55e", "#a855f7", "#f59e0b", "#3b82f6", "#ec4899", "#ffffff"]


def tab_image(ct, dose, structures, model, params, n_fx, ab):
    physical = dose.dose_gy
    eqd2 = eqd2_volume(model, physical, int(n_fx), float(ab), params)

    diff = eqd2 - physical

    c = st.columns([1, 1, 1])
    axis = c[0].selectbox("断面", ["Axial", "Coronal", "Sagittal"], key="pv_axis")
    wl = c[1].selectbox("CT窓", list(viz.WL_PRESETS.keys()), key="pv_wl")
    cmap = c[2].selectbox("カラーマップ (Phys/EQD2)", ["jet", "turbo", "viridis", "hot"], key="pv_cmap")
    W, L = viz.WL_PRESETS[wl]

    n_slices = viz.slice_count(physical.shape, axis)
    hot = int(np.unravel_index(int(physical.argmax()), physical.shape)[
        {"Axial": 0, "Coronal": 1, "Sagittal": 2}[axis]])
    idx = st.slider("スライス", 0, n_slices - 1, hot, key="pv_idx")

    cc = st.columns([1, 1, 1])
    alpha = cc[0].slider("透明度", 0.0, 1.0, 0.5, 0.05, key="pv_alpha")
    thr_pct = cc[1].slider("表示閾値 (%)", 0, 100, 10, 1, key="pv_thr")
    iso = cc[2].checkbox("アイソドーズ線 (Phys/EQD2)", True, key="pv_iso")

    rois = []
    if structures and structures.rois:
        sel = st.multiselect("ROI 輪郭", list(structures.rois.keys()),
                             default=[r for r in structures.rois if r.lower() in
                                      ("water", "roi") or "ptv" in r.lower()][:1],
                             key="pv_rois")
        rois = [(structures.rois[n], viz.ROI_PALETTE[i % len(viz.ROI_PALETTE)])
                for i, n in enumerate(sel)]

    ct_gray = viz.window_ct(viz.take_slice(ct.hu, axis, idx), W, L)
    roi_slices = [(viz.take_slice(m, axis, idx), col) for m, col in rois]

    def panel(vol, label, cmap_name, divergent):
        vmax = (float(np.max(np.abs(vol))) if divergent else float(vol.max())) or 1e-6
        dose_slice = viz.take_slice(vol, axis, idx)
        img = viz.overlay(ct_gray, dose_slice, vmax, vmax * thr_pct / 100.0,
                          cmap_name, alpha, divergent=divergent)
        if iso and not divergent:
            img = viz.apply_isodose_lines(img, dose_slice, vmax, viz.ISODOSE_LEVELS)
        if roi_slices:
            img = viz.apply_roi_outlines(img, roi_slices)
        viz.show_image(img, label)
        viz.colorbar(vmax, cmap_name, divergent=divergent)

    st.caption(f"{axis} slice={idx} — 同じ断面を Physical / EQD2 / 差 で並べて比較")
    p = st.columns(3)
    with p[0]:
        panel(physical, f"Physical (peak {physical.max():.1f} Gy)", cmap, False)
    with p[1]:
        panel(eqd2, f"EQD2 (peak {eqd2.max():.1f} Gy)", cmap, False)
    with p[2]:
        panel(diff, f"差 EQD2−Physical (±{np.abs(diff).max():.1f} Gy)", "RdBu_r", True)

    m = st.columns(5)
    m[0].metric("Physical peak", f"{physical.max():.2f} Gy")
    m[1].metric("d at peak", f"{physical.max()/int(n_fx):.2f} Gy/fx")
    m[2].metric("EQD2 peak", f"{eqd2.max():.2f} Gy")
    m[3].metric("差 最大 (増)", f"{float(diff.max()):+.2f} Gy")
    m[4].metric("差 最小 (減)", f"{float(diff.min()):+.2f} Gy")
    st.caption("差パネル: 赤=EQD2がPhysicalより高い(d>2Gy/fx、寡分割の生物学的増分)／"
               "青=低い(d<2Gy/fx)／d=2Gy/fxで0(白)。")


def tab_dvh(dose, structures, model, params, n_fx, ab):
    if not (structures and structures.rois):
        st.info("RTSTRUCT がありません。")
        return
    physical = dose.dose_gy
    eqd2 = eqd2_volume(model, physical, int(n_fx), float(ab), params)
    sel = st.multiselect("ROI 選択", list(structures.rois.keys()),
                         default=list(structures.rois.keys())[:2], key="dvh_rois")
    if not sel:
        st.warning("ROI を選択してください。")
        return
    fig = go.Figure()
    rows = []
    for i, name in enumerate(sel):
        mask = structures.rois[name]
        color = ROI_COLORS[i % len(ROI_COLORS)]
        d, v = compute_dvh(physical, mask)
        fig.add_trace(go.Scatter(x=d, y=v, name=f"{name} Phys",
                                 line=dict(color=color, width=2.5)))
        d, v = compute_dvh(eqd2, mask)
        fig.add_trace(go.Scatter(x=d, y=v, name=f"{name} EQD2",
                                 line=dict(color=color, width=2, dash="dash")))
        pm = dvh_metrics(physical, mask, structures.px_x, structures.px_y, structures.slice_thickness)
        em = dvh_metrics(eqd2, mask, structures.px_x, structures.px_y, structures.slice_thickness)
        rows.append({"ROI": name, "Vol(cc)": f"{pm['volume_cc']:.1f}",
                     "Phys Mean": f"{pm['mean_Gy']:.1f}", "EQD2 Mean": f"{em['mean_Gy']:.1f}",
                     "Phys D95": f"{pm['D95_Gy']:.1f}", "EQD2 D95": f"{em['D95_Gy']:.1f}",
                     "Phys Max": f"{pm['max_Gy']:.1f}", "EQD2 Max": f"{em['max_Gy']:.1f}"})
    fig.update_layout(title=f"累積 DVH (実線=Physical, 破線=EQD2 / n={n_fx}, α/β={ab})",
                      xaxis_title="Dose (Gy)", yaxis_title="Volume (%)", yaxis_range=[0, 105],
                      height=480, template="plotly_dark", hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def tab_roi_ab(dose, structures, model, params, n_fx):
    """構造別 α/β: ROI 名から推奨 α/β を自動提示し、ROI ごとに別の α/β で評価。"""
    if not (structures and structures.rois):
        st.info("RTSTRUCT がありません。")
        return
    st.caption("ROI 名から推奨 α/β を自動提示。組織ごとに異なる α/β で同時に EQD2 DVH を計算。")
    physical = dose.dose_gy
    rows = [{"ROI": n, "推奨 α/β": suggest_alpha_beta(n) or 3.0,
             "使用 α/β": suggest_alpha_beta(n) or 3.0} for n in structures.rois]
    edited = st.data_editor(
        pd.DataFrame(rows), hide_index=True, use_container_width=True,
        column_config={"使用 α/β": st.column_config.NumberColumn("使用 α/β (Gy)", min_value=0.5,
                                                                  max_value=15.0, step=0.1),
                       "推奨 α/β": st.column_config.NumberColumn("自動推奨", disabled=True)},
        key="roiab_editor")
    fig = go.Figure()
    out = []
    for i, r in edited.iterrows():
        name, ab = r["ROI"], float(r["使用 α/β"])
        mask = structures.rois[name]
        if not mask.any():
            continue
        e = eqd2_volume(model, physical, int(n_fx), ab, params)
        color = ROI_COLORS[i % len(ROI_COLORS)]
        d, v = compute_dvh(e, mask)
        fig.add_trace(go.Scatter(x=d, y=v, name=f"{name} (α/β={ab})",
                                 line=dict(color=color, width=2.5)))
        m = dvh_metrics(e, mask, structures.px_x, structures.px_y, structures.slice_thickness)
        out.append({"ROI": name, "α/β": ab, "Mean": f"{m['mean_Gy']:.1f}",
                    "Max": f"{m['max_Gy']:.1f}", "D95": f"{m['D95_Gy']:.1f}"})
    fig.update_layout(title=f"構造別 α/β を用いた EQD2 DVH (n={n_fx})", xaxis_title="EQD2 (Gy)",
                      yaxis_title="Volume (%)", yaxis_range=[0, 105], height=460,
                      template="plotly_dark", hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(out), use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="プラン評価", page_icon="🩻", layout="wide")
    apply_theme()
    page_header("プラン評価 — 画像 + DVH + 構造別α/β",
                "1つのプランを画像・DVH・構造別評価で総合レビュー。",
                badges=["単一プラン", "EQD2 オーバーレイ", "DVH", "構造別 α/β"])

    ct = viz.get_ct()
    rd_names = viz.get_rtdose_names()
    if not rd_names:
        st.error("RTDOSE がありません。synth_dose.py で生成するか TCIA データに切替えてください。")
        st.stop()

    model, params = model_picker()
    st.sidebar.markdown("### 📋 プラン")
    rd_name = st.sidebar.selectbox("RTDOSE", rd_names)
    n_fx = st.sidebar.number_input("分割数 n", 1, 100, 20, 1)
    ab = st.sidebar.selectbox("DVH 用 α/β (Gy)", ALPHA_BETA_OPTIONS,
                              index=ALPHA_BETA_OPTIONS.index(3.0),
                              format_func=lambda x: ALPHA_BETA_HINT[x])
    dose = viz.get_dose(rd_name)
    structures = viz.get_masks()
    st.caption(f"RTDOSE: {rd_name} | peak {dose.dose_gy.max():.1f} Gy | モデル: {model}")

    t1, t2, t3 = st.tabs(["🖼 画像オーバーレイ", "📈 DVH", "🧬 構造別 α/β"])
    with t1:
        tab_image(ct, dose, structures, model, params, n_fx, ab)
    with t2:
        tab_dvh(dose, structures, model, params, n_fx, ab)
    with t3:
        tab_roi_ab(dose, structures, model, params, n_fx)


if __name__ == "__main__":
    main()
