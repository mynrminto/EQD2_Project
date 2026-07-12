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
    """共通 CSS を注入。各ページの先頭で1回呼ぶ。"""
    st.markdown(
        """
        <style>
        /* ---- ベース ---- */
        .stApp { background:
            radial-gradient(1200px 600px at 80% -10%, #16213a 0%, transparent 55%),
            radial-gradient(1000px 500px at -10% 10%, #10243a 0%, transparent 50%),
            #0b1220; }
        section.main > div { padding-top: 1rem; }

        /* ---- ヘッダーバナー ---- */
        .eqd2-hero {
            background: linear-gradient(110deg, #0e7490 0%, #1d4ed8 55%, #4338ca 100%);
            border-radius: 18px; padding: 22px 26px; margin-bottom: 18px;
            box-shadow: 0 8px 30px rgba(29,78,216,0.25);
            border: 1px solid rgba(255,255,255,0.08);
        }
        .eqd2-hero h1 { color:#fff; font-size: 26px; margin:0 0 4px 0; font-weight:800; letter-spacing:.5px; }
        .eqd2-hero p { color: #d6e6ff; margin:0; font-size: 14px; }
        .eqd2-hero .badge {
            display:inline-block; background: rgba(255,255,255,0.16); color:#fff;
            border-radius: 999px; padding: 3px 12px; font-size: 12px; margin-top: 8px;
            margin-right: 6px; backdrop-filter: blur(4px);
        }

        /* ---- カード ---- */
        .eqd2-card {
            background: #131c2e; border: 1px solid #2a3a55; border-radius: 14px;
            padding: 16px 18px; margin-bottom: 14px;
        }
        .eqd2-card h3 { margin:0 0 10px 0; font-size: 15px; color:#e8eef7;
            border-left: 3px solid #22d3ee; padding-left: 10px; }

        /* ---- メトリックチップ ---- */
        .chip-row { display:flex; gap:10px; flex-wrap:wrap; }
        .chip {
            flex:1; min-width: 130px; background:#1b2740; border:1px solid #2a3a55;
            border-radius: 12px; padding: 12px 14px;
        }
        .chip .lbl { font-size: 12px; color:#8da2c0; margin-bottom: 4px; }
        .chip .val { font-size: 22px; font-weight: 800; color:#e8eef7; }
        .chip .sub { font-size: 11px; color:#8da2c0; margin-top: 2px; }

        /* ---- Streamlit ウィジェット微調整 ---- */
        [data-testid="stSidebar"] { background: #0d1626; border-right:1px solid #2a3a55; }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color:#22d3ee; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            background:#131c2e; border-radius: 10px 10px 0 0; padding: 6px 14px;
            color:#8da2c0;
        }
        .stTabs [aria-selected="true"] { background:#1d4ed8 !important; color:#fff !important; }
        div[data-testid="stDataFrame"] { border-radius: 10px; overflow:hidden; }
        .stButton button { border-radius: 10px; }
        h1, h2, h3, h4 { color:#e8eef7; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", badges: list[str] | None = None) -> None:
    """グラデーションのヒーローヘッダ。"""
    badge_html = ""
    if badges:
        badge_html = "<div>" + "".join(f"<span class='badge'>{b}</span>" for b in badges) + "</div>"
    st.markdown(
        f"""
        <div class="eqd2-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            {badge_html}
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
