"""TOP — はじめに / 使い方。説明書を見なくても直感的に使えるようにする案内ページ。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402
from ui_theme import page_header  # noqa: E402


PAGES = [
    ("計算機", "画像なしで素早く計算", "処方(1回線量・回数・α/β)を入れて EQD2/BED を計算。"
     "複数プランの比較、4モデル比較、等EQD2線量表。会議室・ベッドサイド向け。"),
    ("プラン評価", "1プランを画像で総合レビュー", "CT に線量を重ねて表示。"
     "Physical / EQD2 / 差 を横3枚で同時比較。DVH、構造別 α/β も同じページのタブで。"),
    ("画像ビューア", "3断面を同時に見る (TPS 風)", "Axial / Coronal / Sagittal を"
     "クロスヘアで連動表示。アイソドーズ・構造体・カーソル位置の線量を確認。"),
    ("再照射", "過去+今回の累積線量を評価", "2つ以上の線量分布を EQD2 空間で加算。"
     "recovery を考慮した累積マップと、どのコースが効いたかのコース寄与マップ。"),
    ("モデル・不確実性", "モデルとα/βの感度を検討 (QA/教育)", "同じプランを4モデルで並べて比較。"
     "α/β のばらつきによる EQD2 の worst-case を可視化。"),
]

MODELS = [
    ("古典 LQ", "#ef4444", "標準。中線量では妥当だが、8 Gy/fx を超えると過大評価。"),
    ("LQ-L", "#3b82f6", "高線量を補正 (Astrahan)。SBRT/SRS で LQ の過大評価を抑える。"),
    ("USC", "#22c55e", "高線量を補正 (Park/Timmerman)。いわゆるユニバーサル・モデル。"),
    ("IR", "#f59e0b", "低線量の過敏 (HRS) を補正 (Joiner)。散乱線など低線量域で効く。"),
]


def main():
    page_header("EQD2 Biological Dose Suite",
                "異なる分割スケジュール(通常分割・寡分割・SBRT・再照射)を、"
                "生物学的線量 EQD2 で「同じ物差し」で比較する試作ツールです。")

    # --- クイックスタート ---
    st.markdown("#### はじめての方へ (3ステップ)")
    q = st.columns(3)
    for col, (num, title, body) in zip(q, [
        ("1", "計算だけしたい", "左メニューの「計算機」へ。処方を入れると EQD2 が出ます。"),
        ("2", "画像で見たい", "「プラン評価」へ。CT に線量を重ね、Physical / EQD2 / 差を並べて表示。"),
        ("3", "モデルを選ぶ", "各ページ左上の「生物学的モデル」で LQ / LQ-L / USC / IR を切替(全ページ共通)。"),
    ]):
        with col:
            st.markdown(
                f"<div class='eqd2-card' style='height:150px'>"
                f"<div style='color:#3b82f6;font-weight:700;font-size:13px'>STEP {num}</div>"
                f"<div style='font-weight:700;margin:4px 0 6px'>{title}</div>"
                f"<div style='color:#8a9bb5;font-size:13px;line-height:1.6'>{body}</div></div>",
                unsafe_allow_html=True)

    st.markdown("")

    # --- 各ページの説明 ---
    st.markdown("#### 画面の一覧")
    for name, tag, body in PAGES:
        st.markdown(
            f"<div class='eqd2-card' style='display:flex;gap:16px;align-items:flex-start'>"
            f"<div style='min-width:150px'>"
            f"<div style='font-weight:700;font-size:15px'>{name}</div>"
            f"<div style='color:#3b82f6;font-size:12px;margin-top:2px'>{tag}</div></div>"
            f"<div style='color:#c3d0e6;font-size:13.5px;line-height:1.65'>{body}</div></div>",
            unsafe_allow_html=True)

    # --- 用語ミニ解説 ---
    with st.expander("用語ミニ解説 (EQD2 / BED / α/β)"):
        st.markdown(
            "- **EQD2**: 2 Gy/回で照射したと仮定したときの等価線量。異なる分割の「効き目」を"
            "共通の物差しで比べられます。\n"
            "- **BED**: 生物学的効果線量。EQD2 の元になる量。\n"
            "- **α/β**: 組織の放射線感受性の指標。腫瘍は通常 10、脊髄など晩期反応組織は 2〜3。"
            "小さいほど1回線量の大きさに敏感(寡分割の影響大)。")

    # --- 4モデル ---
    st.markdown("#### 生物学的モデル (4種類)")
    st.caption("1回線量が中程度なら4モデルは一致し、高線量・低線量で差が出ます。用途に応じて選択します。")
    m = st.columns(4)
    for col, (name, color, body) in zip(m, MODELS):
        with col:
            st.markdown(
                f"<div class='eqd2-card' style='height:150px;border-top:3px solid {color}'>"
                f"<div style='font-weight:700;font-size:15px'>{name}</div>"
                f"<div style='color:#8a9bb5;font-size:12.5px;line-height:1.6;margin-top:6px'>{body}</div></div>",
                unsafe_allow_html=True)

    st.divider()
    st.caption("研究・教育目的の試作です。臨床判断には使用しないでください。")


main()
