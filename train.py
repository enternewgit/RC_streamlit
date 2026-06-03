#学習・評価
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from esn import EchoStateNetwork


def split_data(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """
    層化サンプリングでTrain/Val/Testに分割

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_size + test_size,
        random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_esn(X_train, y_train, X_val, y_val,
              n_reservoir=500, spectral_radius=0.95,
              sparsity=0.1, ridge_alpha=1.0):
    """
    ESN + RidgeClassifierで学習

    Returns:
        esn: 学習済みESN
        clf: 学習済み分類器
        val_acc: Validation accuracy
    """
    esn = EchoStateNetwork(
        n_inputs=X_train.shape[2],
        n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        sparsity=sparsity
    )

    print("リザバー変換中（Train）...")
    X_train_esn = esn.transform(X_train)

    print("リザバー変換中（Val）...")
    X_val_esn = esn.transform(X_val)

    clf = RidgeClassifier(alpha=ridge_alpha)
    clf.fit(X_train_esn, y_train)

    val_acc = accuracy_score(y_val, clf.predict(X_val_esn))
    print(f"Validation Accuracy: {val_acc:.4f}")

    return esn, clf, val_acc


def evaluate(esn, clf, X_test, y_test):
    """
    Testデータで最終評価

    Returns:
        test_acc: Test accuracy
        cm: 混同行列
    """
    print("リザバー変換中（Test）...")
    X_test_esn = esn.transform(X_test)

    preds = clf.predict(X_test_esn)
    test_acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc, cm