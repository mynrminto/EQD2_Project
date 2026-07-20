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
from dose_io import (  # noqa: E402
    compute_dvh, dvh_metrics, course_contribution_maps,
    evaluate_goal, default_clinical_goals, GOAL_METRICS,
)
from models import ALPHA_BETA_OPTIONS, ALPHA_BETA_HINT, model_picker, eqd2_volume  # noqa: E402
from ui_theme import apply_theme, page_header, page_help
import viz  # noqa: E402

COURSE_COLORS = ["#3b82f6", "#ef4444", "#22c55e", "#a855f7", "#f59e0b", "#0891b2"]


def _judge(actual: float, op: str, obj: float, var: float, priority) -> str:
    """Eclipse 流の 3 状態判定: Objective 達成 / Variation 内 / 未達。"""
    if str(priority).upper() == "R":
        return "— Report"
    ok = (actual <= obj) if op == "≤" else (actual >= obj)
    if ok:
        return "✅ 達成"
    if var and var > 0:
        ok_var = (actual <= var) if op == "≤" else (actual >= var)
        if ok_var:
            return "⚠️ Variation内"
    return "❌ 未達"


def section_clinical_goals(dose_vol, structures, label: str):
    """Eclipse の Clinical Goals に準拠した構造別の目標判定。"""
    st.markdown("---")
    st.subheader("Clinical Goals")
    if not (structures and structures.rois):
        st.info("RTSTRUCT がありません。")
        return
    st.caption(f"**{label}** に対する構造別の目標判定 (Eclipse の Clinical Goals 準拠)。"
               "P=優先度 (1:Most Important 〜 4:Less Important, R:Report only)、"
               "Objective を満たせば ✅、満たさないが Variation 内なら ⚠️、どちらも外れれば ❌。")
    st.warning("既定値は QUANTEC 等の一般的な文献値に基づく **出発点の目安** です。"
               "分割・併用療法・再照射の既往・施設基準により調整が必須で、"
               "この表のみで臨床判断を行わないでください。", icon="⚠️")

    rois = list(structures.rois.keys())
    base = pd.DataFrame(default_clinical_goals(rois))
    ed = st.data_editor(
        base, hide_index=True, use_container_width=True, key="cg_editor",
        column_config={
            "P": st.column_config.SelectboxColumn("P", options=[1, 2, 3, 4, "R"], width="small"),
            "Structure": st.column_config.TextColumn("Structure", disabled=True),
            "指標": st.column_config.SelectboxColumn("指標", options=list(GOAL_METRICS.keys()),
                                                     help=" / ".join(f"{k}={v}" for k, v in GOAL_METRICS.items())),
            "Param": st.column_config.NumberColumn("Param", min_value=0.0, step=0.01, format="%.2f",
                                                   help="D_cc→体積(cc), D_%→体積(%), V_Gy→線量(Gy)。Dmax/Dmean では未使用。"),
            "演算子": st.column_config.SelectboxColumn("演算子", options=["≤", "≥"], width="small"),
            "Objective": st.column_config.NumberColumn("Objective", min_value=0.0, step=0.1, format="%.2f"),
            "Variation": st.column_config.NumberColumn("Variation", min_value=0.0, step=0.1, format="%.2f",
                                                       help="許容変動。0 なら判定に使いません。"),
        })

    voxel_cc = structures.px_x * structures.px_y * structures.slice_thickness / 1000.0
    rows, n_ok, n_var, n_ng = [], 0, 0, 0
    for _, r in ed.iterrows():
        mask = structures.rois.get(r["Structure"])
        if mask is None or not mask.any():
            continue
        actual, unit = evaluate_goal(dose_vol, mask, str(r["指標"]),
                                     float(r["Param"]), voxel_cc)
        verdict = _judge(actual, str(r["演算子"]), float(r["Objective"]),
                         float(r["Variation"]), r["P"])
        n_ok += verdict.startswith("✅"); n_var += verdict.startswith("⚠️"); n_ng += verdict.startswith("❌")
        m = str(r["指標"])
        obj_txt = (f"{m} {r['演算子']} {float(r['Objective']):.2f} {unit}" if m in ("Dmax", "Dmean")
                   else f"{m}({float(r['Param']):g}) {r['演算子']} {float(r['Objective']):.2f} {unit}")
        rows.append({"P": r["P"], "Structure": r["Structure"], "Objective": obj_txt,
                     "Variation": f"{r['演算子']} {float(r['Variation']):.2f} {unit}"
                                  if float(r["Variation"]) > 0 else "—",
                     "実測": f"{actual:.2f} {unit}", "判定": verdict})
    if not rows:
        st.info("評価可能な ROI がありません。")
        return
    m = st.columns(3)
    m[0].metric("✅ 達成", n_ok)
    m[1].metric("⚠️ Variation内", n_var)
    m[2].metric("❌ 未達", n_ng)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def tab_plan_sum(ct, structures, rd_names, model, params, ab):
    """Eclipse の "Insert New Plan Sum" に準拠した最大 3 プランの合成。"""
    st.caption("Varian Eclipse の **Plan Sum** に準拠。最大 3 プランを "
               "**Operation (+/−)・Plan Weight** つきで合成します。EQD2 は"
               "プランごとに自分の分割数 n で換算してから加算します。")

    n_avail = len(rd_names)
    picks = [rd_names[min(i, n_avail - 1)] for i in range(3)]

    def _fx(name: str) -> int:
        """プラン名から既定の分割数を推定 (過去=通常分割, 今回=寡分割)。"""
        n = name.lower()
        return 25 if "prior" in n else 5

    def _rec(name: str) -> float:
        """過去プランのみ既定で回復を見込む。"""
        return 0.5 if "prior" in name.lower() else 0.0

    base = pd.DataFrame({
        "含める": [True, True, False],
        "Plan ID": picks,
        "Operation": ["+", "+", "+"],
        "Plan Weight": [1.00, 1.00, 1.00],
        "Fractions": [_fx(p) for p in picks],
        "Recovery": [_rec(p) for p in picks],
    })
    ed = st.data_editor(
        base, hide_index=True, use_container_width=True, key="psum_editor",
        column_config={
            "含める": st.column_config.CheckboxColumn("✓", width="small"),
            "Plan ID": st.column_config.SelectboxColumn("Plan ID", options=rd_names, required=True),
            "Operation": st.column_config.SelectboxColumn("Operation", options=["+", "−"], required=True),
            "Plan Weight": st.column_config.NumberColumn("Plan Weight", min_value=0.0,
                                                         max_value=5.0, step=0.05, format="%.2f"),
            "Fractions": st.column_config.NumberColumn("Fractions", min_value=1, max_value=100, step=1),
            "Recovery": st.column_config.NumberColumn("Recovery", min_value=0.0, max_value=1.0,
                                                      step=0.05, format="%.2f",
                                                      help="0=完全蓄積, 1=完全回復 (再照射用)"),
        })

    rows = [r for _, r in ed.iterrows() if bool(r["含める"])]
    if not rows:
        st.warning("少なくとも 1 つのプランを選択してください。")
        return

    plan_sum, per_plan, summary, shapes = None, [], [], set()
    for r in rows:
        phys = viz.get_dose(r["Plan ID"]).dose_gy
        shapes.add(phys.shape)
        n_i, w = int(r["Fractions"]), float(r["Plan Weight"])
        sign = 1.0 if r["Operation"] == "+" else -1.0
        rec = float(r["Recovery"])
        e = eqd2_volume(model, phys, n_i, float(ab), params)
        contrib = sign * w * e * (1.0 - rec)
        plan_sum = contrib if plan_sum is None else plan_sum + contrib
        per_plan.append((str(r["Plan ID"]), e, contrib))
        tot = float(phys.max())
        summary.append({"Plan ID": viz.rtdose_label(str(r["Plan ID"])),
                        "Op": r["Operation"], "Weight": f"{w:.2f}", "Fractions": n_i,
                        "Fraction Dose [Gy]": f"{tot / n_i:.3f}",
                        "Total Dose [Gy]": f"{tot:.3f}", "Recovery": f"{rec:.2f}",
                        "EQD2 peak [Gy]": f"{float(e.max()):.1f}"})
    if len(shapes) > 1:
        st.error("プラン間でグリッド形状が異なります。実臨床では DIR による位置整合が必要です。")
        return

    st.markdown("**Σ Plan Sum の内訳**")
    st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)

    cc = st.columns([1, 1])
    view = cc[0].selectbox("マップ", ["Σ Plan Sum"] + [f"{i+1}. {viz.rtdose_label(p[0])}"
                                                      for i, p in enumerate(per_plan)],
                           key="psum_view")
    axis = cc[1].selectbox("断面", ["Axial", "Coronal", "Sagittal"], key="reirr_axis")
    vol = plan_sum if view.startswith("Σ") else per_plan[int(view.split(".")[0]) - 1][2]

    vmax = max(float(np.max(np.abs(vol))), 1e-6)
    z, y, x = np.unravel_index(int(plan_sum.argmax()), plan_sum.shape)
    default_idx = {"Axial": int(z), "Coronal": int(y), "Sagittal": int(x)}[axis]
    idx = st.slider("スライス", 0, viz.slice_count(ct.hu.shape, axis) - 1,
                    default_idx, key="reirr_idx")

    ct_gray = viz.window_ct(viz.take_slice(ct.hu, axis, idx), 2000, 0)
    img = viz.overlay(ct_gray, viz.take_slice(vol, axis, idx), vmax, vmax * 0.05, "jet", 0.6)

    col_img, col_info = st.columns([3, 1])
    with col_img:
        viz.show_image(img, f"{axis} slice={idx} | {view} (α/β={ab}, {model})")
        viz.colorbar(vmax, "jet", label=f"{view}: 0 … {vmax:.1f} Gy")
    with col_info:
        for i, (name, e, contrib) in enumerate(per_plan):
            st.metric(f"{i+1}. {viz.rtdose_label(name)[:14]}", f"{float(e.max()):.1f} Gy",
                      delta=f"寄与 {float(contrib.max()):+.1f}")
        st.metric("Σ Plan Sum max", f"{float(plan_sum.max()):.1f} Gy")
        st.caption(f"Hotspot: ({ct.origin[0]+x*ct.px_x:+.0f}, "
                   f"{ct.origin[1]+y*ct.px_y:+.0f}, {ct.z_positions[z]:+.0f}) mm")

    if structures and structures.rois:
        roi = "Water" if "Water" in structures.rois else list(structures.rois.keys())[0]
        st.markdown(f"**DVH (ROI: {roi})**")
        fig = go.Figure()
        for i, (name, e, contrib) in enumerate(per_plan):
            d, v = compute_dvh(e, structures.rois[roi])
            fig.add_trace(go.Scatter(x=d, y=v, name=f"{i+1}. {viz.rtdose_label(name)}",
                                     line=dict(color=COURSE_COLORS[i % len(COURSE_COLORS)], width=2)))
        d, v = compute_dvh(plan_sum, structures.rois[roi])
        fig.add_trace(go.Scatter(x=d, y=v, name="Σ Plan Sum",
                                 line=dict(color="#22c55e", width=3)))
        fig.update_layout(xaxis_title="EQD2 (Gy)", yaxis_title="Volume (%)", yaxis_range=[0, 105],
                          height=380, template="plotly_dark", hovermode="x unified",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    section_clinical_goals(plan_sum, structures, f"Σ Plan Sum (累積 EQD2, α/β={ab})")


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
            f = st.selectbox("線量分布", rd_names, index=min(i, len(rd_names) - 1),
                             format_func=viz.rtdose_label, key=f"cc_f_{i}")
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
    page_header("Plan Sum — 累積 EQD2 + コース寄与",
                "Eclipse 準拠の Plan Sum で最大 3 プランを合成 (Operation・Plan Weight・recovery)。")
    page_help(
        "**何ができる:** 複数の線量分布を EQD2 空間で合成し、再照射時の累積線量を評価します。\n\n"
        "**使い方:**\n"
        "1. **Plan Sum** タブ: Varian Eclipse の『Insert New Plan Sum』に準拠した表で、**最大 3 プラン**を選択。"
        "プランごとに **Operation (+/−)・Plan Weight・Fractions・Recovery** を指定すると、"
        "Fraction Dose / Total Dose / EQD2 peak が内訳表に出ます。Σ Plan Sum のマップと DVH を確認。\n"
        "2. **コース寄与マップ**タブ: 各コースがどの voxel にどれだけ寄与したかを、優勢コース map・per-course マップ・hotspot 内訳で確認。\n\n"
        "**recovery の目安:** 0=完全蓄積 (最も厳しい)、1=完全回復。脊髄なら 6ヶ月で約25%、2年で約50% (文献値)。\n\n"
        "**注意:** 線形 recovery の簡易モデルで、同一 CT 格子を前提。実臨床では DIR (変形レジストレーション) で位置整合が必要です。")

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

    t1, t2 = st.tabs(["Plan Sum", "コース寄与マップ"])
    with t1:
        tab_plan_sum(ct, structures, rd_names, model, params, ab)
    with t2:
        tab_contribution(ct, structures, rd_names, model, params, ab)

    with st.expander("⚠️ 注意事項"):
        st.markdown("""
- 加算は線形 recovery モデル: `EQD2_cum = Σ EQD2_i × (1−recovery_i)`。
- 同一 CT 格子・FrameOfReferenceUID を前提。実臨床では deformable registration が必要。
- recovery 目安: 脊髄 6ヶ月で約25%、2年で50% (Nieder 2006 等)。
        """)


main()
