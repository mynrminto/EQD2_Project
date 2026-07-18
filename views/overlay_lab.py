"""重ね合わせ実験室 (ファントム) — 2 プランの累積線量を直列/並列臓器の観点で評価。

同梱の水ファントム線量を「ビーム形状」として、
  Plan A = 原画像 (例 4 Gy × 5 = 20 Gy)
  Plan B = 左右反転 (例 3 Gy × 10 = 30 Gy)
の 2 プランを合成し、中心に置いた円柱 ROI (脊髄など) で累積線量を評価する。

- 直列臓器 (脊髄・消化管): 一部が壊れると全体が機能不全 → **D_max** が律速。
- 並列臓器 (肺・肝): 一定割合/平均線量で機能低下 → **Mean・V_xGy** が律速。

EQD2 は voxel ごとに各プランで換算してから加算する (再照射の累積 EQD2 の標準手順)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dose_io import eqd2_map, compute_dvh  # noqa: E402
from ui_theme import page_header, page_help  # noqa: E402
import viz  # noqa: E402

CORD_COLOR = "#22d3ee"


def _cylinder_mask(shape, px_x, px_y, cx, cy, diam_mm):
    """(nz,ny,nx) の中心軸に沿った直径 diam_mm の円柱 (頭尾方向=z に全長)。"""
    nz, ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r_mm = np.sqrt(((xx - cx) * px_x) ** 2 + ((yy - cy) * px_y) ** 2)
    disk = r_mm <= diam_mm / 2.0
    return np.broadcast_to(disk, (nz, ny, nx))


def main():
    page_header("重ね合わせ実験室 — 2 プランの累積を直列/並列臓器で評価",
                "ファントム線量を左右反転して 2 コースを合成。中心の円柱 ROI で累積線量を検証。")
    page_help(
        "**狙い:** 2 つの治療コースを重ね合わせたとき、間に挟まれた臓器 (例: 脊髄) の"
        "累積線量が許容内に収まるかを、ファントムで手軽に試します。\n\n"
        "**モデル:** 同梱の水ファントム線量を『ビーム形状』として、原画像を Plan A、"
        "左右反転を Plan B とし、それぞれ任意の分割で処方します。EQD2 は voxel ごとに"
        "各プランで換算してから加算します (再照射の累積 EQD2 の標準手順)。\n\n"
        "**直列 vs 並列:** 脊髄・消化管などの**直列臓器**は一部が壊れると全体が"
        "機能不全 → **D_max** が律速。肺・肝などの**並列臓器**は一定割合/平均線量で"
        "機能低下 → **Mean・V_xGy** が律速。臓器タイプで評価指標が切り替わります。")

    ct = viz.get_ct()
    rd_names = viz.get_rtdose_names()
    if not rd_names:
        st.error("RTDOSE がありません。")
        st.stop()
    nz, ny, nx = ct.hu.shape
    st_z = abs(float(ct.z_positions[1]) - float(ct.z_positions[0])) if nz > 1 else 1.0

    # ---------- コントロール ----------
    st.sidebar.markdown("### ビーム形状")
    base_name = st.sidebar.selectbox("線量分布 (RTDOSE)", rd_names,
                                     format_func=viz.rtdose_label,
                                     help="この分布の形をピーク正規化して 2 プランに使います。")
    base = viz.get_dose(base_name).dose_gy.astype(np.float32)
    peak = float(base.max()) or 1e-6

    st.sidebar.markdown("### Plan A (原画像)")
    doseA = st.sidebar.number_input("総線量 A (Gy)", 1.0, 200.0, 20.0, 1.0, key="ol_dA")
    nA = st.sidebar.number_input("分割数 A", 1, 50, 5, 1, key="ol_nA")
    st.sidebar.markdown("### Plan B (左右反転)")
    doseB = st.sidebar.number_input("総線量 B (Gy)", 1.0, 200.0, 30.0, 1.0, key="ol_dB")
    nB = st.sidebar.number_input("分割数 B", 1, 50, 10, 1, key="ol_nB")
    mirror = st.sidebar.checkbox("Plan B を左右反転", True, key="ol_mirror",
                                 help="OFF にすると 2 プランが同じ位置に重なります。")

    st.sidebar.markdown("### 臓器 (円柱 ROI)")
    organ = st.sidebar.radio("臓器タイプ", ["直列 (脊髄)", "並列 (肺・肝)"], key="ol_organ")
    is_serial = organ.startswith("直列")
    diam = st.sidebar.number_input("直径 (mm)", 2.0, 200.0,
                                   10.0 if is_serial else 80.0, 1.0, key="ol_diam")
    ab = st.sidebar.number_input("α/β (Gy)", 0.5, 15.0,
                                 2.0 if is_serial else 3.0, 0.5, key="ol_ab")
    if is_serial:
        limit = st.sidebar.number_input("D_max 制約 EQD2 (Gy)", 1.0, 200.0, 50.0, 1.0, key="ol_lim")
    else:
        v_thr = st.sidebar.number_input("V_x の x (EQD2 Gy)", 1.0, 100.0, 20.0, 1.0, key="ol_vthr")
        limit = st.sidebar.number_input(f"Mean 制約 EQD2 (Gy)", 1.0, 100.0, 20.0, 1.0, key="ol_lim")

    # ---------- 2 プラン合成 & 累積 ----------
    planA = base / peak * doseA
    planB = (np.flip(base, axis=2) if mirror else base.copy()) / peak * doseB
    eA = eqd2_map(planA, int(nA), float(ab))
    eB = eqd2_map(planB, int(nB), float(ab))
    phys_cum = planA + planB
    eqd2_cum = eA + eB

    cx, cy = nx / 2.0, ny / 2.0
    cord = _cylinder_mask(ct.hu.shape, ct.px_x, ct.px_y, cx, cy, diam)

    e_in = eqd2_cum[cord]
    p_in = phys_cum[cord]
    vox_cc = ct.px_x * ct.px_y * st_z / 1000.0
    vol_cc = int(cord.sum()) * vox_cc

    # 最大 voxel での各プラン寄与 (course contribution)
    idx_flat = np.where(cord.reshape(-1), eqd2_cum.reshape(-1), -1).argmax()
    mi = np.unravel_index(int(idx_flat), eqd2_cum.shape)
    aA, aB = float(eA[mi]), float(eB[mi])

    # ---------- 指標 ----------
    st.caption(f"{viz.rtdose_label(base_name)} を形状に使用 ｜ Plan A {doseA:.0f}Gy/"
               f"{int(nA)}fx (peak {planA.max():.1f}Gy, {planA.max()/nA:.1f}Gy/fx) ＋ "
               f"Plan B {doseB:.0f}Gy/{int(nB)}fx {'(左右反転)' if mirror else ''} ｜ "
               f"臓器 {organ} φ{diam:.0f}mm, α/β={ab}")

    if is_serial:
        emax = float(e_in.max()); pmax = float(p_in.max())
        ok = emax < limit
        m = st.columns(4)
        m[0].metric("物理 D_max (累積)", f"{pmax:.1f} Gy")
        m[1].metric("EQD2 D_max (累積)", f"{emax:.1f} Gy")
        m[2].metric(f"制約 {limit:.0f} Gy", "適合 ✅" if ok else "超過 ⚠️",
                    delta=f"{emax-limit:+.1f} Gy", delta_color="inverse")
        m[3].metric("体積", f"{vol_cc:.1f} cc")
        st.markdown(
            f"**直列臓器の評価:** D_max が律速。累積 EQD2 D_max = **{emax:.1f} Gy** "
            f"({'制約内' if ok else '制約超過'})。最大 voxel での寄与は "
            f"**Plan A {aA:.1f} + Plan B {aB:.1f} Gy** "
            f"(A {100*aA/(aA+aB):.0f}% / B {100*aB/(aA+aB):.0f}%)。")
    else:
        mean_e = float(e_in.mean()); mean_p = float(p_in.mean())
        vx = float((e_in >= v_thr).mean() * 100.0)
        ok = mean_e < limit
        m = st.columns(4)
        m[0].metric("物理 Mean (累積)", f"{mean_p:.1f} Gy")
        m[1].metric("EQD2 Mean (累積)", f"{mean_e:.1f} Gy")
        m[2].metric(f"Mean 制約 {limit:.0f} Gy", "適合 ✅" if ok else "超過 ⚠️",
                    delta=f"{mean_e-limit:+.1f} Gy", delta_color="inverse")
        m[3].metric(f"V{v_thr:.0f}Gy (EQD2)", f"{vx:.1f} %")
        st.markdown(
            f"**並列臓器の評価:** Mean・V_xGy が律速 (D_max は重視しない)。"
            f"累積 EQD2 Mean = **{mean_e:.1f} Gy**、V{v_thr:.0f}Gy = **{vx:.1f}%** "
            f"({'制約内' if ok else '制約超過'})。並列臓器では『どれだけの体積が"
            f"一定線量を超えたか』が機能低下と相関します。")

    # ---------- 画像 (累積 EQD2) ----------
    z_mi = int(mi[0])
    vmax = float(eqd2_cum.max()) or 1e-6
    ref = vmax
    st.markdown("---")
    st.markdown(f"**累積 EQD2 分布** — Axial は臓器 D_max スライス (z={z_mi+1}/{nz})、"
                f"Coronal は中心を頭尾に貫く断面。水色輪郭 = 臓器 ROI。")
    imgcols = st.columns([1, 1])
    # Axial at cord-max slice
    ct_ax = viz.window_ct(viz.take_slice(ct.hu, "Axial", z_mi), *viz.WL_PRESETS["軟部 (W400/L40)"])
    ax = viz.overlay(ct_ax, viz.take_slice(eqd2_cum, "Axial", z_mi), vmax, vmax * 0.05, "jet", 0.5)
    ax = viz.apply_isodose_lines(ax, viz.take_slice(eqd2_cum, "Axial", z_mi), ref, viz.ISODOSE_FULL)
    ax = viz.apply_roi_outlines(ax, [(viz.take_slice(cord, "Axial", z_mi), CORD_COLOR)])
    with imgcols[0]:
        viz.show_image(ax, f"Axial z={z_mi+1}")
        viz.colorbar(vmax, "jet")
    # Coronal through center
    y_mi = int(round(cy))
    ct_co = viz.window_ct(viz.take_slice(ct.hu, "Coronal", y_mi), *viz.WL_PRESETS["軟部 (W400/L40)"])
    co = viz.overlay(ct_co, viz.take_slice(eqd2_cum, "Coronal", y_mi), vmax, vmax * 0.05, "jet", 0.5)
    co = viz.apply_isodose_lines(co, viz.take_slice(eqd2_cum, "Coronal", y_mi), ref, viz.ISODOSE_FULL)
    co = viz.apply_roi_outlines(co, [(viz.take_slice(cord, "Coronal", y_mi), CORD_COLOR)])
    with imgcols[1]:
        viz.show_image(co, f"Coronal y={y_mi+1} (2 葉が中心を挟む)")

    # ---------- 臓器 DVH ----------
    st.markdown("---")
    st.subheader("臓器 DVH (累積)")
    fig = go.Figure()
    for arr, name, dash in [(phys_cum, "物理 累積", "solid"),
                            (eqd2_cum, "EQD2 累積", "dash"),
                            (eA, "Plan A のみ (EQD2)", "dot"),
                            (eB, "Plan B のみ (EQD2)", "dot")]:
        d, v = compute_dvh(arr, cord)
        fig.add_trace(go.Scatter(x=d, y=v, name=name, line=dict(dash=dash, width=2)))
    xline = limit if is_serial else v_thr
    fig.add_vline(x=xline, line=dict(color="#ef4444", dash="dash"),
                  annotation_text=f"{'D_max制約' if is_serial else 'V_x'} {xline:.0f}Gy")
    fig.update_layout(xaxis_title="Dose (Gy)", yaxis_title="Volume (%)", yaxis_range=[0, 105],
                      height=420, template="plotly_dark", hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=30, b=20, l=20, r=20),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.3))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("直列臓器は右端 (D_max) を、並列臓器は曲線の中腹 (Mean・V_xGy) を見ます。")


main()
