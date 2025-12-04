# client/client.py
"""
Client-side logic for GP-AdaFL
- local_train: 执行本地训练并注入动态 DP 噪声
- grad_norm: 上传前计算梯度 L2 范数（供预测器使用）
- quality_score: 计算本地数据质量 Q_k^(t)
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from utils.dp_utils import scale_gaussian_noise
from utils.metrics import data_quality


class LocalClient:
    """
    轻量级 Client 实现
    所有计算在 device 上完成，返回 CPU 副本
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        cfg: Dict,
        device: torch.device,
    ):
        self.cid = client_id
        self.model = model.to(device)
        self.loader = train_loader
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.SGD(
            self.model.parameters(),
            lr=cfg["lr"],
            momentum=0.9,
            weight_decay=1e-4,
        )
        self._global_dict: Optional[Dict[str, torch.Tensor]] = None

    # ------------- 对外接口 ------------- #

    def local_train(
        self, global_dict: Dict[str, torch.Tensor], noise_scale: float
    ) -> Dict[str, torch.Tensor]:
        """执行 E 轮本地训练 → 返回加噪后的 state_dict"""
        self._sync_global(global_dict)
        C = self.cfg["clip_norm"]  # 裁剪阈值
        sigma = scale_gaussian_noise(
            self.cfg["sigma_max"], noise_scale, C
        )  # 动态 σ

        self.model.train()
        for _ in range(self.cfg["E"]):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad()
                loss = nn.functional.cross_entropy(self.model(x), y)
                loss.backward()

                # 1. 梯度裁剪
                nn.utils.clip_grad_norm_(self.model.parameters(), C)

                # 2. 动态噪声注入
                with torch.no_grad():
                    for p in self.model.parameters():
                        p.grad += torch.randn_like(p) * (sigma * C)

                self.opt.step()

        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def grad_norm(self, global_dict: Dict[str, torch.Tensor]) -> float:
        """计算当前梯度 L2 范数（供服务器预测器）"""
        self._sync_global(global_dict)
        self.model.zero_grad()
        # 前向-反向一次即可
        for x, y in self.loader:
            x, y = x.to(self.device), y.to(self.device)
            loss = nn.functional.cross_entropy(self.model(x), y)
            loss.backward()
            break  # 仅取一个 batch 近似
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total**0.5

    def quality_score(self) -> float:
        """数据质量 Q_k^(t)：1 - 平均预测熵"""
        return data_quality(self.model, self.loader, self.device)

    # ------------- 内部工具 ------------- #

    def _sync_global(self, global_dict: Dict[str, torch.Tensor]) -> None:
        """拉取全局参数"""
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in global_dict.items()}
        )
