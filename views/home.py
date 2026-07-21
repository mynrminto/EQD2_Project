"""TOP — はじめに / 使い方。説明書を見なくても直感的に使えるようにする案内ページ。

全体像インフォグラフィックは **インライン SVG/HTML でベクター描画** する
(以前の PNG は拡大時に粗くなるため廃止。ベクターなので常にくっきり・レスポンシブ)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402
from ui_theme import page_header  # noqa: E402


# (名前, サブ, 本文, SVGパス) — 現行6ページ
PAGES = [
    ("計算機", "画像なしで素早く", "処方→EQD2/BED。複数プラン比較、4モデル比較、等EQD2線量表。",
     '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M8 9h8M8 13h5"/>'),
    ("プラン評価", "1プランを総合レビュー", "CT に線量を重ね Physical/EQD2/差 を3枚並列。DVH・構造別α/β・モデル自動切替・DICOM出力。",
     '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="12" cy="12" r="4"/>'),
    ("画像ビューア", "3断面を同時に (TPS風)", "Axial/Coronal/Sagittal をクロスヘア連動。絶対Gyアイソドーズ(線/ウォッシュ排他)。",
     '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 3v18M3 12h18"/>'),
    ("Plan Sum", "累積線量を評価", "Eclipse準拠で最大3プランを合成(Operation・Weight)。コース寄与マップ・Clinical Goals。",
     '<circle cx="9" cy="12" r="6"/><circle cx="15" cy="12" r="6"/>'),
    ("重ね合わせ実験", "直列/並列臓器で検証", "2コースを合成しファントムで累積を検証。脊髄=Dmax / 肺・肝=Mean・Vx。",
     '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 9h16M4 14h16"/>'),
    ("モデル・不確実性", "QA・教育", "4モデルを並べ比較。α/β のばらつきによる worst-case を可視化。",
     '<path d="M3 18 C7 6,11 16,21 4"/><path d="M3 20 C8 12,12 18,21 10"/>'),
]

MODELS = [
    ("古典LQ", "#ef4444"), ("LQ-L", "#3b82f6"), ("USC", "#22c55e"), ("IR", "#f59e0b"),
]

_OVERVIEW_CSS = """
<style>
.ovw{margin:6px 0 4px;font-family:"Hiragino Sans","Yu Gothic",sans-serif;}
.ovw-flow{display:flex;flex-wrap:wrap;gap:12px;align-items:stretch;margin-bottom:14px;}
.ovw-box{background:#0f1421;border:1px solid #1b2536;border-radius:14px;padding:14px 16px;}
.ovw-tag{color:#3b82f6;font-size:12px;font-weight:700;}
.ovw-box h3{font-size:16px;font-weight:800;margin:5px 0 6px;color:#eef2f8;}
.ovw-box p{color:#8a9bb5;font-size:12.5px;line-height:1.5;margin:0;}
.ovw-arw{display:flex;align-items:center;color:#3b82f6;font-size:22px;font-weight:700;}
.ovw-pills{display:flex;flex-wrap:wrap;gap:6px;margin-top:9px;align-items:center;}
.ovw-pill{border-radius:999px;padding:3px 11px;font-size:12px;font-weight:700;color:#fff;}
.ovw-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;}
.ovw-card{background:#0f1421;border:1px solid #1b2536;border-radius:12px;padding:14px 15px;}
.ovw-ico{width:28px;height:28px;margin-bottom:8px;}
.ovw-card h4{font-size:14px;font-weight:800;margin:0;color:#eef2f8;}
.ovw-card .sub{color:#3b82f6;font-size:11px;margin:2px 0 7px;}
.ovw-card p{color:#8a9bb5;font-size:12px;line-height:1.5;margin:0;}
.ovw-foot{display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;
  margin-top:14px;color:#6b7c98;font-size:12px;}
@media (max-width:820px){.ovw-arw{display:none;}}
</style>
"""


def _overview_html() -> str:
    pills = "".join(
        f'<span class="ovw-pill" style="background:{c}">{n}</span>' for n, c in MODELS
    ) + '<span style="color:#8a9bb5;font-size:12px;margin-left:4px">低線量〜高線量まで補正</span>'

    cards = "".join(
        f'<div class="ovw-card">'
        f'<svg class="ovw-ico" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{svg}</svg>'
        f'<h4>{name}</h4><div class="sub">{sub}</div><p>{body}</p></div>'
        for name, sub, body, svg in PAGES
    )

    return _OVERVIEW_CSS + f"""
<div class="ovw">
  <div class="ovw-flow">
    <div class="ovw-box" style="flex:1 1 260px;max-width:340px">
      <div class="ovw-tag">入力</div><h3>処方 または DICOM</h3>
      <p>1回線量・回数・α/β、あるいは<br>CT・線量分布(RTDOSE)・輪郭(RTSTRUCT)</p>
    </div>
    <div class="ovw-arw">→</div>
    <div class="ovw-box" style="flex:2 1 340px">
      <div class="ovw-tag">EQD2 換算 — 生物学的モデル(4種類)</div>
      <h3>voxel 単位 / 処方単位で EQD2・BED を算出</h3>
      <div class="ovw-pills">{pills}</div>
    </div>
    <div class="ovw-arw">→</div>
    <div class="ovw-box" style="flex:1 1 180px;max-width:220px">
      <div class="ovw-tag">6つの画面</div><h3>目的別に<br>可視化・評価</h3>
    </div>
  </div>
  <div class="ovw-cards">{cards}</div>
  <div class="ovw-foot">
    <span>共通設定: 生物学的モデルと α/β は全ページで共有</span>
    <span>研究・教育目的の試作 — 臨床判断には使用しないこと</span>
  </div>
</div>
"""


def main():
    # --- 大タイトル ---
    st.markdown(
        "<div style='padding:6px 0 2px'>"
        "<div style='font-size:38px;font-weight:800;letter-spacing:.3px;line-height:1.2'>"
        "EQD2 <span style='color:#3b82f6'>Biological Dose Suite</span></div>"
        "<div style='color:#8a9bb5;font-size:15px;margin-top:8px'>"
        "放射線治療の異なる分割スケジュールを、生物学的線量 EQD2 で「同じ物差し」で比較する試作ツール</div>"
        "</div>",
        unsafe_allow_html=True)

    # --- 全体像インフォグラフィック (ベクター描画) ---
    st.markdown(_overview_html(), unsafe_allow_html=True)
    st.markdown("")

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

    # --- 用語ミニ解説 ---
    with st.expander("用語ミニ解説 (EQD2 / BED / α/β)"):
        st.markdown(
            "- **EQD2**: 2 Gy/回で照射したと仮定したときの等価線量。異なる分割の「効き目」を"
            "共通の物差しで比べられます。\n"
            "- **BED**: 生物学的効果線量。EQD2 の元になる量。\n"
            "- **α/β**: 組織の放射線感受性の指標。腫瘍は通常 10、脊髄など晩期反応組織は 2〜3。"
            "小さいほど1回線量の大きさに敏感(寡分割の影響大)。")

    st.divider()
    st.caption("研究・教育目的の試作です。臨床判断には使用しないでください。")


main()
