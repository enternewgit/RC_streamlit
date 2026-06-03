#ESN実装
import numpy as np
from scipy import sparse

class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir=500, spectral_radius=0.95,
                 sparsity=0.1, noise=0.001, random_state=42):
        """
        Parameters:
            n_inputs: 入力次元数（MFCCの次元数）
            n_reservoir: リザバーのノード数
            spectral_radius: スペクトル半径
            sparsity: リザバーの結合密度
            noise: ノイズの強さ
            random_state: 乱数シード
        """
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.rng = np.random.RandomState(random_state)

        self._init_weights()

    def _init_weights(self):
        # 入力重み（入力層 → リザバー）
        self.W_in = self.rng.uniform(-1, 1, (self.n_reservoir, self.n_inputs))

        # リザバー重み（スパース行列）
        W = sparse.random(
            self.n_reservoir, self.n_reservoir,
            density=self.sparsity,
            random_state=self.rng
        ).toarray()
        W[W != 0] = self.rng.uniform(-1, 1, W[W != 0].shape)

        # スペクトル半径で正規化
        eigenvalues = np.linalg.eigvals(W)
        sr = np.max(np.abs(eigenvalues))
        if sr > 0:
            W = W * (self.spectral_radius / sr)
        self.W = W

    def _run_reservoir(self, X):
        """
        時系列Xをリザバーに流して内部状態を取得

        Parameters:
            X: shape (time_steps, n_inputs)
        Returns:
            states: shape (time_steps, n_reservoir)
        """
        states = np.zeros((X.shape[0], self.n_reservoir))
        x = np.zeros(self.n_reservoir)

        for t, u in enumerate(X):
            x = np.tanh(
                self.W_in @ u + self.W @ x
                + self.noise * self.rng.randn(self.n_reservoir)
            )
            states[t] = x

        return states

    def transform(self, X_seq):
        """
        複数の時系列をリザバーに流して特徴量を取得（平均pooling）

        Parameters:
            X_seq: shape (n_samples, time_steps, n_inputs)
        Returns:
            features: shape (n_samples, n_reservoir)
        """
        features = []
        for x in X_seq:
            states = self._run_reservoir(x)
            # 時間方向に平均を取って固定長ベクトルに
            features.append(states.mean(axis=0))
        return np.array(features)