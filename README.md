# 音楽ジャンル分類デモ with Echo State Network

GTZAN Genre Collection を使って、WAV 音声から MFCC を抽出し、Echo State Network + Ridge 回帰でジャンル分類する Streamlit アプリです。

## できること

- GTZAN データセットの読み込み
- MFCC 特徴量の表示
- ESN の学習と評価
- WAV ファイルのアップロード予測
- リザバー内部状態と混同行列の可視化

## 前提

- Python 3.10 以上
- ローカルに GTZAN データを配置
- 学習・評価はローカル実行を想定

データ配置例:

```text
data/gtzan/archive/Data/genres_original/
```

## セットアップ

```bash
uv pip install -r requirements.txt --python .venv/Scripts/python.exe
```

## 起動

```bash
uv run streamlit run app.py
```

## 使い方

1. 左のサイドバーで GTZAN データのパスを確認する
2. 「学習・評価」タブで学習を実行する
3. 「ジャンル予測」タブで WAV をアップロードして予測する
4. 「可視化」タブで MFCC やリザバー状態を確認する

## 配布時の注意

- `.gitignore` で `data/` の中身は無視しています
- `model.pkl` は予測用の事前学習モデルとして同梱できます
- `tmp_upload.wav` は Git 管理しません
- デプロイ先で学習まで行う場合は、データ配置方法を別途用意してください

## 技術スタック

- Streamlit
- librosa
- numpy / scipy
- scikit-learn
- matplotlib
