#MFCC特徴量抽出
import os
import numpy as np
import librosa

def extract_mfcc(file_path, n_mfcc=13, max_len=130):
    """
    WAVファイルからMFCCを抽出する
    
    Parameters:
        file_path: WAVファイルのパス
        n_mfcc: MFCCの次元数
        max_len: 時系列の最大長（フレーム数）
    
    Returns:
        mfcc: shape (max_len, n_mfcc) のnumpy配列
    """
    y, sr = librosa.load(file_path, sr=22050, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # (time, n_mfcc)

    # 長さをmax_lenに統一
    if mfcc.shape[0] < max_len:
        pad = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)))
    else:
        mfcc = mfcc[:max_len]

    return mfcc


def load_gtzan(data_dir, n_mfcc=13, max_len=130):
    """
    GTZANデータセットを全件読み込む
    
    Returns:
        X: shape (n_samples, max_len, n_mfcc)
        y: shape (n_samples,) ラベルの整数配列
        genres: ジャンル名のリスト
    """
    genres = sorted(os.listdir(data_dir))
    X, y = [], []

    for label, genre in enumerate(genres):
        genre_dir = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_dir):
            continue
        for fname in sorted(os.listdir(genre_dir)):
            if not fname.endswith('.wav'):
                continue
            fpath = os.path.join(genre_dir, fname)
            try:
                mfcc = extract_mfcc(fpath, n_mfcc=n_mfcc, max_len=max_len)
                X.append(mfcc)
                y.append(label)
            except Exception as e:
                print(f"スキップ: {fpath} ({e})")

    return np.array(X), np.array(y), genres