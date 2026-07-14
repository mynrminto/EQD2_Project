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
    st.Page("views/home.py", title="TOP", icon=":material/home:", default=True),
    st.Page("views/calculator.py", title="計算機", icon=":material/calculate:"),
    st.Page("pages/1_プラン評価.py", title="プラン評価", icon=":material/dashboard:"),
    st.Page("pages/2_画像ビューア.py", title="画像ビューア", icon=":material/imagesmode:"),
    st.Page("pages/3_再照射.py", title="再照射", icon=":material/replay:"),
    st.Page("pages/4_モデル不確実性.py", title="モデル・不確実性", icon=":material/science:"),
]

st.navigation(pages).run()
