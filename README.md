# EQD2 Biological Dose Suite（試作版・クラウド公開用）

放射線治療の異なる分割スケジュールを生物学的線量 **EQD2** で統一比較する試作 Web アプリ。
古典LQ / LQ-L / USC / IR の 4 モデルに対応し、CT・線量分布(RTDOSE)・輪郭(RTSTRUCT) を読み込んで
EQD2 マップ・DVH・再照射の累積評価を可視化します。

## 公開デモ（Streamlit Community Cloud）
1. このリポジトリを **公開(public)** で GitHub に上げる
2. https://share.streamlit.io にGitHubでログイン → **New app**
3. リポジトリ / ブランチ / メインファイル `app.py` を指定 → **Deploy**
4. 発行された URL を共有すれば、相手はブラウザで開くだけ（インストール不要）

## 画面
- **計算機** … 処方→EQD2/BED、複数プラン比較、等EQD2線量表
- **プラン評価** … 画像オーバーレイ + DVH + 構造別α/β
- **画像ビューア** … Eclipse風 3直交同期ビュー
- **再照射** … 累積EQD2 + コース寄与マップ
- **モデル・不確実性** … 4モデル比較 + α/β worst-case

## データ
同梱の `サンプルデータ/水ファントム` は**合成(架空)データ**で、実患者情報は含みません。
CT は表示軽量化のため 256×256・スライス半減に縮小しています。

## ローカルで動かす場合
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 免責
研究・教育目的の試作。臨床判断には使用しないこと。
