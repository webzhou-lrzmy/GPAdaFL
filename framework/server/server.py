# server/predictor.py
"""
Dual-channel gradient predictor for GP-AdaFL
- LSTM path: learns long-term trend from scalar gradient norms
- Kalman path: corrects short-term deviations in latent space
- Bayesian fusion: uncertainty-weighted combination
"""

import torch
import numpy as np
from typing import Optional, Tuple
from pykalman import KalmanFilter


# ---------- LSTM-based temporal learner ---------- #
class LSTMPredictor(torch.nn.Module):
    def __init__(self, hidden: int = 128, r: int = 10):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1,
                                  hidden_size=hidden,
                                  num_layers=2,
                                  batch_first=True,
                                  dropout=0.1)
        self.fc = torch.nn.Linear(hidden, r)  # map to latent dim
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H] (H historical norms)
        returns: [B, r] latent state prediction
        """
        x = x.unsqueeze(-1)  # [B, H, 1]
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]  # final layer hidden
        return self.fc(out)  # [B, r]


# ---------- Kalman filter corrector ---------- #
class KalmanPredictor:
    def __init__(self, r: int = 10):
        self.r = r
        self.kf = KalmanFilter(n_dim_obs=r, n_dim_state=r)
        self._fitted = False

    def fit(self, states: np.ndarray) -> None:
        """
        states: [T, r]  historical latent vectors
        EM algorithm to estimate transition & obs covariances
        """
        self.kf = self.kf.em(states)
        self._fitted = True

    def predict_update(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: [r] current latent observation
        returns corrected [r] state estimate
        """
        if not self._fitted:
            return obs
        state_mean, _ = self.kf.filter_update(obs)
        return state_mean

    def uncertainty(self) -> float:
        """返回先验协方差迹作为不确定性度量"""
        if not self._fitted:
            return 1e3
        return float(np.trace(self.kf.transition_covariance))


# ---------- Bayesian fusion ---------- #
class DualPredictor:
    """
    双通道预测器
    1. LSTM：输入梯度范数序列 → 潜在状态
    2. Kalman：潜在状态序列 → 误差修正
    3. 不确定性加权融合
    """

    def __init__(self, hidden: int = 128, r: int = 10, hist_len: int = 5):
        self.lstm = LSTMPredictor(hidden, r)
        self.kf = KalmanPredictor(r)
        self.hist_len = hist_len
        self.norm_buffer = []  # 保存最近 hist_len 个梯度范数
        self.latent_buffer = []  # 保存对应 latent 状态

    def update(self, grad_norm: float) -> None:
        """接收服务器上传的最新平均梯度范数"""
        self.norm_buffer.append(grad_norm)
        if len(self.norm_buffer) > self.hist_len:
            self.norm_buffer.pop(0)

    def predict(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        返回 (lstm_pred, kf_pred)
        若历史不足，返回 (None, None)
        """
        if len(self.norm_buffer) < self.hist_len:
            return None, None

        # 1. LSTM 通道
        x = torch.tensor(self.norm_buffer).unsqueeze(0).float()  # [1, H]
        with torch.no_grad():
            lstm_latent = self.lstm(x).squeeze(0).numpy()  # [r]

        # 2. Kalman 通道
        self.latent_buffer.append(lstm_latent)
        if len(self.latent_buffer) > self.hist_len:
            self.latent_buffer.pop(0)
        if len(self.latent_buffer) >= self.hist_len:
            self.kf.fit(np.stack(self.latent_buffer))
        kf_latent = self.kf.predict_update(lstm_latent)

        return lstm_latent, kf_latent

    def fused_predict(self) -> Tuple[np.ndarray, float]:
        """
        贝叶斯最优融合
        returns: (fused_latent, uncertainty)
        """
        lstm_latent, kf_latent = self.predict()
        if lstm_latent is None:
            return np.zeros(self.kf.r), 1e3

        # 不确定性 = 通道各自误差估计
        lstm_err = self._lstm_uncertainty()
        kf_err = self.kf.uncertainty()

        # 最优权重（MVUE）
        total_inv = 1.0 / (lstm_err + 1e-8) + 1.0 / (kf_err + 1e-8)
        w_lstm = (1.0 / (lstm_err + 1e-8)) / total_inv
        w_kf = 1.0 - w_lstm

        fused = w_lstm * lstm_latent + w_kf * kf_latent
        fused_uncertainty = 1.0 / total_inv
        return fused, fused_uncertainty

    # ---------- 内部工具 ---------- #
    def _lstm_uncertainty(self) -> float:
        """用最近 3 步 MSE 近似 LSTM 预测误差"""
        if len(self.latent_buffer) < 3:
            return 1.0
        preds = np.stack(self.latent_buffer[-3:])
        truth = np.stack(self.latent_buffer[-3:])  # 简化：用自身当 truth
        return float(np.mean((preds - truth) ** 2))


# ---------------- 快速自检 ---------------- #
if __name__ == "__main__":
    predictor = DualPredictor(hidden=64, r=10, hist_len=5)
    # 模拟 10 个梯度范数
    for gn in np.linspace(1.0, 5.0, 10):
        predictor.update(gn)
        fused, unc = predictor.fused_predict()
        if fused is not None:
            print(f"gn={gn:.2f}  fused_norm={np.linalg.norm(fused):.3f}  uncertainty={unc:.3f}")
