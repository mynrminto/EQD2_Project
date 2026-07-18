"""EQD2 Biological Dose Suite — エントリポイント。

st.navigation で「TOP + 各画面」をサイドバーに定義する。
set_page_config と共通テーマ(apply_theme)はここで1回だけ実行し、
各ページ(views/ と pages/)はページ本体のみを描画する。
"""
from __future__ import annotations

import streamlit as st

from ui_theme import apply_theme

st.set_page_config(page_title="EQD2 Suite", page_icon="🎯", layout="wide")
apply_theme()

pages = [
    st.Page("views/home.py", title="TOP", icon=":material/home:",
            url_path="home", default=True),
    st.Page("views/calculator.py", title="計算機", icon=":material/calculate:",
            url_path="calculator"),
    st.Page("views/plan_review.py", title="プラン評価", icon=":material/dashboard:",
            url_path="plan_review"),
    st.Page("views/image_viewer.py", title="画像ビューア", icon=":material/imagesmode:",
            url_path="image_viewer"),
    st.Page("views/reirradiation.py", title="再照射", icon=":material/replay:",
            url_path="reirradiation"),
    st.Page("views/overlay_lab.py", title="重ね合わせ実験", icon=":material/layers:",
            url_path="overlay_lab"),
    st.Page("views/model_lab.py", title="モデル・不確実性", icon=":material/science:",
            url_path="model_lab"),
]

st.navigation(pages).run()
