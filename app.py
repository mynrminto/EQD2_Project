"""計算機 — 画像を使わない処方ベースの EQD2/BED 計算。

会議室・ベッドサイドでの素早い計算、複数プラン比較、等EQD2線量表。
モデル選択 (LQ/LQ-L/USC/IR) と α/β はサイドバーで全ページ共通。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models import (
    MODELS, MODEL_COLORS, ALPHA_BETA_OPTIONS, ALPHA_BETA_HINT,
    model_picker, compute_eqd2,
)
from ui_theme import apply_theme, page_header


def bed(D, d, ab):
    return D * (1.0 + d / ab)


def default_plans():
    return [
        {"name": "通常分割", "d": 2.0, "n": 30, "ab": 10.0},
        {"name": "中等度寡分割", "d": 3.0, "n": 20, "ab": 10.0},
        {"name": "SBRT", "d": 12.0, "n": 4, "ab": 10.0},
    ]


def render_plan_sidebar():
    st.sidebar.markdown("### 📋 プラン入力")
    st.sidebar.caption("処方を入力するとリアルタイムで再計算。")
    for i, plan in enumerate(st.session_state.plans):
        with st.sidebar.expander(plan["name"], expanded=(i == 0)):
            plan["name"] = st.text_input("プラン名", plan["name"], key=f"nm_{i}")
            plan["d"] = st.number_input("1回線量 d (Gy)", 0.05, 30.0,
                                        float(plan["d"]), 0.1, key=f"d_{i}")
            plan["n"] = st.number_input("回数 n", 1, 100, int(plan["n"]), 1, key=f"n_{i}")
            plan["ab"] = st.selectbox("α/β (Gy)", ALPHA_BETA_OPTIONS,
                                      index=ALPHA_BETA_OPTIONS.index(float(plan["ab"])),
                                      format_func=lambda x: ALPHA_BETA_HINT[x], key=f"ab_{i}")
            st.caption(f"総線量 D = **{plan['d']*plan['n']:.1f} Gy**")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("＋ 追加", use_container_width=True):
        st.session_state.plans.append({"name": f"プラン{len(st.session_state.plans)+1}",
                                       "d": 2.0, "n": 30, "ab": 10.0})
        st.rerun()
    if c2.button("－ 削除", use_container_width=True,
                 disabled=len(st.session_state.plans) <= 1):
        st.session_state.plans.pop()
        st.rerun()
    if st.sidebar.button("初期化", use_container_width=True):
        st.session_state.plans = default_plans()
        st.rerun()


def results_df(model, params):
    rows = []
    for plan in st.session_state.plans:
        D = plan["d"] * plan["n"]
        rows.append({
            "プラン": plan["name"], "d (Gy)": plan["d"], "n": int(plan["n"]),
            "D (Gy)": round(D, 1), "α/β": plan["ab"],
            "BED (Gy)": round(bed(D, plan["d"], plan["ab"]), 2),
            "EQD2 (Gy)": round(compute_eqd2(model, D, plan["d"], plan["ab"], params), 2),
        })
    return pd.DataFrame(rows)


def tab_calc(model, params):
    df = results_df(model, params)
    st.markdown(f"**選択モデル: {model}**")
    st.dataframe(df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for col, color in [("D (Gy)", "#94a3b8"), ("BED (Gy)", "#60a5fa"), ("EQD2 (Gy)", "#22d3ee")]:
            fig.add_bar(x=df["プラン"], y=df[col], name=col.split()[0],
                        text=df[col], textposition="outside", marker_color=color)
        fig.update_layout(barmode="group", title="総線量 / BED / EQD2",
                          yaxis_title="Gy", height=400, template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=46, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        ab_range = np.linspace(0.5, 15, 80)
        fig = go.Figure()
        pal = ["#22d3ee", "#22c55e", "#ef4444", "#a855f7", "#f59e0b", "#3b82f6"]
        for i, plan in enumerate(st.session_state.plans):
            D = plan["d"] * plan["n"]
            y = [compute_eqd2(model, D, plan["d"], a, params) for a in ab_range]
            fig.add_trace(go.Scatter(x=ab_range, y=y, name=plan["name"],
                                     line=dict(width=2.5, color=pal[i % len(pal)])))
        fig.update_layout(title="EQD2 の α/β 依存性", xaxis_title="α/β (Gy)",
                          yaxis_title="EQD2 (Gy)", height=400, template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=46, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)


def tab_model_compare(params):
    st.caption("同じ処方を4モデルで計算。低線量=IR補正、中線量=一致、高線量=LQ-L/USCがLQの過大評価を補正。")
    rows = []
    for plan in st.session_state.plans:
        D = plan["d"] * plan["n"]
        rows.append({
            "プラン": plan["name"], "処方": f"{plan['d']}Gy×{int(plan['n'])}", "α/β": plan["ab"],
            **{m.split()[0]: round(compute_eqd2(m, D, plan["d"], plan["ab"], params), 2)
               for m in MODELS},
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    fig = go.Figure()
    for m in MODELS:
        key = m.split()[0]
        fig.add_bar(x=df["プラン"], y=df[key], name=key, text=df[key],
                    textposition="outside", marker_color=MODEL_COLORS[m])
    fig.update_layout(barmode="group", title="モデル別 EQD2", yaxis_title="EQD2 (Gy)",
                      height=420, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=46, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)


def _solve_d(model, target, n, ab, params, d_max=40.0):
    lo, hi = 1e-4, d_max
    f_lo = compute_eqd2(model, lo * n, lo, ab, params) - target
    f_hi = compute_eqd2(model, hi * n, hi, ab, params) - target
    if f_lo * f_hi > 0:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = compute_eqd2(model, mid * n, mid, ab, params) - target
        if abs(f_mid) < 1e-6:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def tab_iso(model, params):
    st.caption("基準 EQD2 を保つ処方の早見表。再分割・寡分割化の Optimize 設定時に使用。")
    c1, c2 = st.columns(2)
    target = c1.number_input("基準 EQD2 (Gy)", 1.0, 200.0, 60.0, 1.0)
    ab = c2.selectbox("α/β (Gy)", ALPHA_BETA_OPTIONS, index=ALPHA_BETA_OPTIONS.index(10.0),
                      format_func=lambda x: ALPHA_BETA_HINT[x])
    fx_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 28, 30, 35]
    rows = []
    for n in fx_list:
        d = _solve_d(model, target, n, ab, params)
        rows.append({"分割数 n": n,
                     "1回線量 d (Gy)": round(d, 2) if d else "—",
                     "総線量 D (Gy)": round(d * n, 1) if d else "—"})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    valid = [(r["分割数 n"], r["総線量 D (Gy)"]) for r in rows if r["総線量 D (Gy)"] != "—"]
    if valid:
        xs, ys = zip(*valid)
        fig = go.Figure(go.Scatter(x=list(xs), y=list(ys), mode="lines+markers",
                                   line=dict(color="#22d3ee", width=3)))
        fig.update_layout(title=f"総線量 vs 分割数 (EQD2={target}Gy, {model})",
                          xaxis_title="分割数 n", yaxis_title="総線量 D (Gy)",
                          height=360, template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=46, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
    st.download_button("📄 CSV ダウンロード", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"iso_eqd2_{int(target)}Gy.csv", mime="text/csv")


def main():
    st.set_page_config(page_title="EQD2 Suite — 計算機", page_icon="🧮", layout="wide")
    apply_theme()
    page_header("計算機 — 処方ベース EQD2/BED",
                "画像を使わない素早い計算。複数プラン比較・等EQD2線量表・4モデル対応。",
                badges=["会議室・ベッドサイド", "4モデル", "等EQD2表"])
    if "plans" not in st.session_state:
        st.session_state.plans = default_plans()

    model, params = model_picker()
    render_plan_sidebar()

    t1, t2, t3 = st.tabs(["📊 計算 & グラフ", "🔬 4モデル比較", "📋 等EQD2線量表"])
    with t1:
        tab_calc(model, params)
    with t2:
        tab_model_compare(params)
    with t3:
        tab_iso(model, params)

    st.divider()
    st.caption("画像を使う評価は左サイドバーの各ページへ: "
               "プラン評価 / 画像ビューア / 再照射 / モデル・不確実性。")


if __name__ == "__main__":
    main()
