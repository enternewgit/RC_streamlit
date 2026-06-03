import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle
from scipy.special import softmax
from feature import load_gtzan, extract_mfcc
from train import split_data, train_esn, evaluate

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

st.set_page_config(page_title="音楽ジャンル分類 with ESN", layout="wide")
st.title("🎵 音楽ジャンル分類 with Echo State Network")

# サイドバー
st.sidebar.header("設定")
data_dir = st.sidebar.text_input(
    "GTZANデータのパス",
    value=r"data\gtzan\archive\Data\genres_original"
)

st.sidebar.subheader("ハイパーパラメータ")
n_reservoir = st.sidebar.slider("リザバーノード数", 100, 1000, 500, step=100)
spectral_radius = st.sidebar.slider("スペクトル半径", 0.5, 1.5, 0.95, step=0.05)
sparsity = st.sidebar.slider("結合密度", 0.05, 0.5, 0.1, step=0.05)
ridge_alpha = st.sidebar.select_slider(
    "Ridge正則化強度",
    options=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    value=1.0
)

# タブ
tab1, tab2, tab3 = st.tabs(["📊 学習・評価", "🎵 ジャンル予測", "🔍 可視化"])

# -------- Tab1: 学習・評価 --------
with tab1:
    st.header("学習・評価")

    if st.button("▶ 学習スタート"):
        with st.spinner("GTZANデータを読み込み中..."):
            X, y, genres = load_gtzan(data_dir)
            if len(X) == 0:
                st.error("指定されたデータパスにWAVが見つかりません。ローカルのGTZANデータを配置してください。")
                st.stop()
            st.success(f"データ読み込み完了！{len(X)}曲 / {len(genres)}ジャンル")

        with st.spinner("Train/Val/Testに分割中..."):
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
            st.info(f"Train: {len(X_train)}曲 / Val: {len(X_val)}曲 / Test: {len(X_test)}曲")

        with st.spinner("ESN学習中...（数分かかります）"):
            esn, clf, val_acc = train_esn(
                X_train, y_train, X_val, y_val,
                n_reservoir=n_reservoir,
                spectral_radius=spectral_radius,
                sparsity=sparsity,
                ridge_alpha=ridge_alpha
            )
            st.success(f"Validation Accuracy: {val_acc:.4f}")

        with st.spinner("テストデータで評価中..."):
            test_acc, cm = evaluate(esn, clf, X_test, y_test)

        st.metric("✅ Test Accuracy", f"{test_acc:.4f}")

        # 混同行列
        st.subheader("混同行列")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(len(genres)))
        ax.set_yticks(range(len(genres)))
        ax.set_xticklabels(genres, rotation=45, ha='right')
        ax.set_yticklabels(genres)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("True label")
        plt.colorbar(im, ax=ax)
        for i in range(len(genres)):
            for j in range(len(genres)):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        # モデル保存
        with open("model.pkl", "wb") as f:
            pickle.dump((esn, clf, genres), f)
        st.info("モデルを model.pkl に保存しました")

# -------- Tab2: ジャンル予測 --------
with tab2:
    st.header("ジャンル予測")

    uploaded = st.file_uploader("WAVファイルをアップロード", type=["wav"])

    if uploaded is not None:
        if not os.path.exists("model.pkl"):
            st.warning("先に「学習・評価」タブでモデルを学習してください")
        else:
            with open("model.pkl", "rb") as f:
                obj = pickle.load(f)
                if isinstance(obj, tuple) and len(obj) == 4:
                    esn, clf, scaler, genres = obj
                elif isinstance(obj, tuple) and len(obj) == 3:
                    esn, clf, genres = obj
                    scaler = None
                else:
                    raise ValueError("Unexpected model.pkl format")

            tmp_path = "tmp_upload.wav"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())

            with st.spinner("特徴量抽出中..."):
                mfcc = extract_mfcc(tmp_path)
                feature = esn.transform(mfcc[np.newaxis, ...])[0]
                scores = clf.decision_function([feature])
                pred = int(np.argmax(scores, axis=1)[0])
                confidence = float(softmax(scores[0])[pred])

            st.success(f"予測ジャンル: **{genres[pred]}**")
            st.metric("確信度", f"{confidence:.2%}")

            # MFCC表示
            st.subheader("MFCCの波形")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(mfcc.T, aspect='auto', origin='lower', cmap='magma')
            ax.set_xlabel("frame")
            ax.set_ylabel("MFCC")
            plt.tight_layout()
            st.pyplot(fig)

# -------- Tab3: 可視化 --------
with tab3:
    st.header("リザバー内部状態の可視化")

    if not os.path.exists("model.pkl"):
        st.warning("先に「学習・評価」タブでモデルを学習してください")
    else:
        with open("model.pkl", "rb") as f:
            obj = pickle.load(f)
            if isinstance(obj, tuple) and len(obj) == 4:
                esn, clf, scaler, genres = obj
            elif isinstance(obj, tuple) and len(obj) == 3:
                esn, clf, genres = obj
                scaler = None
            else:
                raise ValueError("Unexpected model.pkl format")

        sample_path = st.text_input(
            "可視化したいWAVファイルのパス",
            value=r"data/gtzan/archive/Data/genres_original/blues/blues.00000.wav"
        )

        if st.button("▶ 可視化"):
            with st.spinner("リザバー状態を計算中..."):
                mfcc = extract_mfcc(sample_path)
                states = esn._run_reservoir(mfcc)

            st.subheader("リザバーノードの時系列（先頭10ノード）")
            fig, ax = plt.subplots(figsize=(10, 4))
            for i in range(10):
                ax.plot(states[:, i], alpha=0.7, label=f"node {i}")
            ax.set_xlabel("frame")
            ax.set_ylabel("node state")
            ax.legend(loc='upper right', fontsize=7, ncol=2)
            plt.tight_layout()
            st.pyplot(fig)