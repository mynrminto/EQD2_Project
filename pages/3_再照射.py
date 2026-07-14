"""再照射 — 複数プランの累積 EQD2 評価とコース寄与分解。

[再照射外来] Prior + Current の累積 EQD2 (recovery 込み) と、各照射コースが
どの voxel にどれだけ寄与しているかの explainable な分解 (course contribution)。
モデル/α/β はサイドバー共通設定を使用 (LQ-L/USC/IR でも加算可能)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dose_io import compute_dvh, dvh_metrics, course_contribution_maps  # noqa: E402
from models import ALPHA_BETA_OPTIONS, ALPHA_BETA_HINT, model_picker, eqd2_volume  # noqa: E402
from ui_theme import apply_theme, page_header
import viz  # noqa: E402

COURSE_COLORS = ["#3b82f6", "#ef4444", "#22c55e", "#a855f7", "#f59e0b", "#0891b2"]


def tab_cumulative(ct, structures, rd_names, model, params, ab):
    c = st.columns([1, 1, 1, 1])
    prior_name = c[0].selectbox("Prior RTDOSE", rd_names,
                                index=rd_names.index("RD.prior.dcm") if "RD.prior.dcm" in rd_names else 0)
    prior_n = c[1].number_input("Prior 分割数", 1, 100, 25)
    cur_idx = rd_names.index("RD.current.dcm") if "RD.current.dcm" in rd_names else min(1, len(rd_names) - 1)
    current_name = c[2].selectbox("Current RTDOSE", rd_names, index=cur_idx)
    current_n = c[3].number_input("Current 分割数", 1, 100, 5)

    cc = st.columns([1, 1, 1])
    recovery = cc[0].slider("Recovery factor", 0.0, 1.0, 0.5, 0.05,
                            help="0=完全蓄積, 1=完全回復")
    mode = cc[1].selectbox("マップ", ["累積 EQD2", "Prior EQD2", "Current EQD2", "Prior 寄与分"])
    axis = cc[2].selectbox("断面", ["Axial", "Coronal", "Sagittal"], key="reirr_axis")

    prior = viz.get_dose(prior_name).dose_gy
    current = viz.get_dose(current_name).dose_gy
    if prior.shape != current.shape:
        st.error(f"グリッド形状が異なります ({prior.shape} vs {current.shape})。"
                 "実臨床では deformable registration が必要です。")
        return

    e_prior = eqd2_volume(model, prior, int(prior_n), float(ab), params)
    e_current = eqd2_volume(model, current, int(current_n), float(ab), params)
    e_cum = e_prior * (1.0 - recovery) + e_current

    vol = {"累積 EQD2": e_cum, "Prior EQD2": e_prior, "Current EQD2": e_current,
           "Prior 寄与分": e_cum - e_current}[mode]
    vmax = float(vol.max())
    z, y, x = np.unravel_index(int(e_cum.argmax()), e_cum.shape)
    default_idx = {"Axial": int(z), "Coronal": int(y), "Sagittal": int(x)}[axis]
    n_slices = viz.slice_count(ct.hu.shape, axis)
    idx = st.slider("スライス", 0, n_slices - 1, default_idx, key="reirr_idx")

    ct_gray = viz.window_ct(viz.take_slice(ct.hu, axis, idx), 2000, 0)
    img = viz.overlay(ct_gray, viz.take_slice(vol, axis, idx), vmax, vmax * 0.05, "jet", 0.6)

    col_img, col_info = st.columns([3, 1])
    with col_img:
        viz.show_image(img, f"{axis} slice={idx} | {mode} (α/β={ab}, recovery={recovery:.2f})")
        viz.colorbar(vmax, "jet", label=f"{mode}: 0 … {vmax:.1f} Gy")
    with col_info:
        st.metric("Prior EQD2 max", f"{e_prior.max():.1f} Gy")
        st.metric("Current EQD2 max", f"{e_current.max():.1f} Gy")
        st.metric("累積 EQD2 max", f"{e_cum.max():.1f} Gy",
                  delta=f"+{e_cum.max()-e_current.max():.1f} vs Current")
        st.caption(f"Hotspot: ({ct.origin[0]+x*ct.px_x:+.0f}, "
                   f"{ct.origin[1]+y*ct.px_y:+.0f}, {ct.z_positions[z]:+.0f}) mm")

    if structures and "Water" in structures.rois or (structures and structures.rois):
        roi = "Water" if structures and "Water" in structures.rois else list(structures.rois.keys())[0]
        st.markdown(f"**DVH (ROI: {roi})**")
        fig = go.Figure()
        for arr, name, color, w in [(e_prior, "Prior", "#3b82f6", 2),
                                    (e_current, "Current", "#ef4444", 2),
                                    (e_cum, "累積", "#22c55e", 3)]:
            d, v = compute_dvh(arr, structures.rois[roi])
            fig.add_trace(go.Scatter(x=d, y=v, name=name, line=dict(color=color, width=w)))
        fig.update_layout(xaxis_title="EQD2 (Gy)", yaxis_title="Volume (%)", yaxis_range=[0, 105],
                          height=380, template="plotly_dark", hovermode="x unified",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)


def tab_contribution(ct, structures, rd_names, model, params, ab):
    st.caption("累積 EQD2 を「どの照射コースがどの voxel にどれだけ寄与したか」voxel 単位で分解 (explainable)。")
    n_courses = st.number_input("コース数", 2, min(6, len(rd_names)), 2, 1)
    cols = st.columns(int(n_courses))
    courses = []
    for i in range(int(n_courses)):
        with cols[i]:
            st.markdown(f"<div style='background:{COURSE_COLORS[i]};padding:4px 8px;color:#fff;"
                        f"border-radius:6px;font-weight:600;margin-bottom:4px'>Course {i+1}</div>",
                        unsafe_allow_html=True)
            f = st.selectbox("RTDOSE", rd_names, index=min(i, len(rd_names) - 1), key=f"cc_f_{i}")
            n_i = st.number_input("分割数", 1, 100, [25, 5, 5, 5, 5, 5][i], key=f"cc_n_{i}")
            rec = st.slider("Recovery", 0.0, 1.0, 0.5 if i == 0 else 0.0, 0.05, key=f"cc_r_{i}")
            courses.append({"file": f, "n": int(n_i), "rec": rec, "color": COURSE_COLORS[i]})

    axis = st.selectbox("断面", ["Axial", "Coronal", "Sagittal"], key="cc_axis")
    eqd2_list = []
    for c in courses:
        phys = viz.get_dose(c["file"]).dose_gy
        eqd2_list.append((eqd2_volume(model, phys, c["n"], float(ab), params), c["rec"]))
    if len({e.shape for e, _ in eqd2_list}) > 1:
        st.error("コース間でグリッド形状が異なります。registration が必要です。")
        return

    res = course_contribution_maps(eqd2_list)
    cum, contribs, dominant, top_pct = (res["cumulative"], res["contributions"],
                                        res["dominant"], res["top_contribution_pct"])
    z, y, x = np.unravel_index(int(cum.argmax()), cum.shape)
    default_idx = {"Axial": int(z), "Coronal": int(y), "Sagittal": int(x)}[axis]
    n_slices = viz.slice_count(cum.shape, axis)
    idx = st.slider("スライス", 0, n_slices - 1, default_idx, key="cc_idx")
    ct_gray = viz.window_ct(viz.take_slice(ct.hu, axis, idx), 2000, 0)
    vmax = float(cum.max())

    r1 = st.columns(2)
    with r1[0]:
        img = viz.overlay(ct_gray, viz.take_slice(cum, axis, idx), vmax, vmax * 0.05, "jet", 0.6)
        viz.show_image(img, f"累積 EQD2 (peak {vmax:.1f} Gy)")
    with r1[1]:
        dom = viz.take_slice(dominant, axis, idx).astype(int)
        valid = viz.take_slice(cum, axis, idx) > vmax * 0.05
        tp = viz.take_slice(top_pct, axis, idx)
        rgb = viz.to_rgb(ct_gray).astype(np.float32)
        for i, c in enumerate(courses):
            sel = (dom == i) & valid
            a = np.where(sel, np.clip(tp / 100.0, 0.3, 0.85), 0.0)[..., None]
            rgb = rgb * (1 - a) + np.array(viz.hex_to_rgb(c["color"]), float) * a
        viz.show_image(rgb.astype(np.uint8), "優勢コース map (色=コース, 不透明度=寄与%)")

    sub = st.columns(int(n_courses))
    cmaps = ["Blues", "Reds", "Greens", "Purples", "Oranges", "BuGn"]
    for i, c in enumerate(courses):
        with sub[i]:
            cmax = max(float(contribs[i].max()), 1e-6)
            img = viz.overlay(ct_gray, viz.take_slice(contribs[i], axis, idx), cmax,
                              cmax * 0.05, cmaps[i % len(cmaps)], 0.6)
            viz.show_image(img, f"Course {i+1} 寄与 (peak {contribs[i].max():.1f}, rec={c['rec']:.2f})")

    st.markdown(f"**Hotspot 解析**: 累積最大 voxel = "
                f"({ct.origin[0]+x*ct.px_x:+.0f}, {ct.origin[1]+y*ct.px_y:+.0f}, "
                f"{ct.z_positions[z]:+.0f}) mm  累積 EQD2 = **{cum[z,y,x]:.1f} Gy**")
    fig = go.Figure()
    for i, c in enumerate(courses):
        val = contribs[i][z, y, x]
        pct = res["fractions"][i][z, y, x] * 100
        fig.add_trace(go.Bar(x=["Hotspot"], y=[val], name=f"Course {i+1} ({c['file']})",
                             marker_color=c["color"], text=f"{val:.1f}Gy<br>{pct:.0f}%",
                             textposition="inside"))
    fig.update_layout(barmode="stack", template="plotly_dark", height=260,
                      yaxis_title="EQD2 寄与 (Gy)", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)


def main():
    page_header("再照射 — 累積 EQD2 + コース寄与",
                "複数プランを EQD2 空間で加算 (recovery 込み)。コース寄与で説明可能な評価。",
                badges=["複数プラン", "recovery factor", "Course contribution ★", "累積 DVH"])

    ct = viz.get_ct()
    rd_names = viz.get_rtdose_names()
    if len(rd_names) < 2:
        st.error("再照射評価には RTDOSE が 2 つ以上必要です。")
        st.code("python3 synth_dose.py --total 50 --fractions 25 --sigma 60 --out RD.prior.dcm\n"
                "python3 synth_dose.py --total 30 --fractions 5 --sigma 25 --offset-x 40 --out RD.current.dcm")
        st.stop()

    model, params = model_picker()
    ab = st.sidebar.selectbox("加算用 α/β (Gy)", ALPHA_BETA_OPTIONS,
                              index=ALPHA_BETA_OPTIONS.index(3.0),
                              format_func=lambda x: ALPHA_BETA_HINT[x])
    structures = viz.get_masks()
    st.caption(f"モデル: {model} | 加算 α/β: {ab}")

    t1, t2 = st.tabs(["累積評価", "コース寄与マップ"])
    with t1:
        tab_cumulative(ct, structures, rd_names, model, params, ab)
    with t2:
        tab_contribution(ct, structures, rd_names, model, params, ab)

    with st.expander("⚠️ 注意事項"):
        st.markdown("""
- 加算は線形 recovery モデル: `EQD2_cum = Σ EQD2_i × (1−recovery_i)`。
- 同一 CT 格子・FrameOfReferenceUID を前提。実臨床では deformable registration が必要。
- recovery 目安: 脊髄 6ヶ月で約25%、2年で50% (Nieder 2006 等)。
        """)


main()
