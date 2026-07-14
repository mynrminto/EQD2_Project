"""全ページ共通の UI テーマ / コンポーネント。

Streamlit に統一したデザインシステムを注入する。各ページ冒頭で
`apply_theme()` を呼び、`page_header()` でヘッダを描く。
"""
from __future__ import annotations

import streamlit as st

# ---- カラーパレット (clinical dark) ----
COLORS = {
    "bg": "#0b1220",
    "surface": "#131c2e",
    "surface2": "#1b2740",
    "border": "#2a3a55",
    "text": "#e8eef7",
    "muted": "#8da2c0",
    "accent": "#22d3ee",
    "accent2": "#3b82f6",
    "lq": "#ef4444",
    "lql": "#3b82f6",
    "usc": "#22c55e",
    "ir": "#f59e0b",
}

# モデルごとの色 (全ページ共通)
MODEL_COLORS = {
    "古典LQ": "#ef4444",
    "LQ-L (Astrahan)": "#3b82f6",
    "USC (Park/Timmerman)": "#22c55e",
    "IR (Induced Repair)": "#f59e0b",
}


def apply_theme() -> None:
    """共通 CSS を注入。各ページの先頭で1回呼ぶ。引き算で洗練したフラットデザイン。"""
    st.markdown(
        """
        <style>
        /* ---- ベース: 落ち着いた単色。青1色アクセント ---- */
        .stApp { background: #0a0e16; }
        section.main > div.block-container { padding-top: 2.2rem; max-width: 1180px; }

        /* ---- フラットなページヘッダ ---- */
        .eqd2-hero { padding: 2px 0 16px; margin-bottom: 18px;
            border-bottom: 1px solid #1b2536; }
        .eqd2-hero h1 { color:#eef2f8; font-size: 23px; font-weight: 700;
            letter-spacing: 0; margin:0 0 6px 0; line-height:1.3; }
        .eqd2-hero h1::before { content:""; display:inline-block; width:4px; height:19px;
            background:#3b82f6; border-radius:2px; margin-right:11px; vertical-align:-3px; }
        .eqd2-hero p { color:#8a9bb5; margin:0; font-size:13.5px; line-height:1.5; }

        /* ---- カード ---- */
        .eqd2-card { background:#0f1421; border:1px solid #1b2536; border-radius:12px;
            padding:16px 18px; margin-bottom:14px; }
        .eqd2-card h3 { margin:0 0 10px 0; font-size:14px; color:#eef2f8; font-weight:600;
            border-left:3px solid #3b82f6; padding-left:10px; }

        /* ---- メトリックチップ ---- */
        .chip-row { display:flex; gap:10px; flex-wrap:wrap; }
        .chip { flex:1; min-width:130px; background:#0f1421; border:1px solid #1b2536;
            border-radius:10px; padding:12px 14px; }
        .chip .lbl { font-size:12px; color:#8a9bb5; margin-bottom:4px; }
        .chip .val { font-size:22px; font-weight:700; color:#eef2f8; }
        .chip .sub { font-size:11px; color:#8a9bb5; margin-top:2px; }

        /* ---- サイドバー: 静かに ---- */
        [data-testid="stSidebar"] { background:#0c111b; border-right:1px solid #1b2536; }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color:#c3d0e6; font-size:12.5px; font-weight:600; letter-spacing:.6px;
            text-transform:none; }

        /* ---- タブ: 下線スタイル(フラット) ---- */
        .stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid #1b2536; }
        .stTabs [data-baseweb="tab"] { background:transparent; border-radius:0;
            padding:8px 4px; color:#8a9bb5; font-size:14px; }
        .stTabs [aria-selected="true"] { background:transparent !important; color:#eef2f8 !important;
            border-bottom:2px solid #3b82f6; }

        /* ---- Streamlit ウィジェット微調整 ---- */
        div[data-testid="stMetric"] { background:#0f1421; border:1px solid #1b2536;
            border-radius:10px; padding:10px 14px; }
        div[data-testid="stMetricValue"] { font-size:22px; }
        div[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
        .stButton button { border-radius:8px; border:1px solid #2a3a55; }
        .stSlider [data-baseweb="slider"] { padding-top:4px; }
        h1,h2,h3,h4 { color:#eef2f8; }
        .stCaption, [data-testid="stCaptionContainer"] { color:#7c8ba6; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", badges: list[str] | None = None) -> None:
    """フラットなページヘッダ。badges 引数は後方互換のため受けるが表示しない(シンプル化)。"""
    st.markdown(
        f"""
        <div class="eqd2-hero">
            <h1>{title}</h1>
            {f'<p>{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def chips(items: list[tuple[str, str, str]]) -> None:
    """メトリックチップの行。items = [(label, value, sub), ...]"""
    html = "<div class='chip-row'>"
    for label, value, sub in items:
        html += (
            f"<div class='chip'><div class='lbl'>{label}</div>"
            f"<div class='val'>{value}</div>"
            f"<div class='sub'>{sub}</div></div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def card_open(title: str = "") -> None:
    st.markdown(f"<div class='eqd2-card'>{'<h3>'+title+'</h3>' if title else ''}",
                unsafe_allow_html=True)


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)
