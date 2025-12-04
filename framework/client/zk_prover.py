# client/zk_prover.py
"""
zk-SNARK 证明生成（Groth16）
- 对「裁剪后的梯度 + 注入噪声」生成 zk 证明
- 公输入：承诺 cm、噪声尺度 σ、裁剪阈值 C
- 私输入：真实梯度 g、随机数 r、噪声 ξ
- 返回 (proof, public_signals) 供链上验证
"""

import os, json, hashlib
from typing import Tuple, List
import torch
import numpy as np
from py_ecc.bn128 import curve_order, multiply, add, G1, G2, pairing
from py_ecc.fields import field

# ------------- 常量 ------------- #
CURVE_ORDER = curve_order
G1Generator = G1
G2Generator = G2


# ------------- 工具函数 ------------- #
def _hash_g1(pt) -> int:
    """将 G1 点哈希到 Fr"""
    x, y = pt
    h = int(hashlib.sha256(f"{x}{y}".encode()).hexdigest(), 16)
    return h % CURVE_ORDER


def _to_fr(x: int) -> int:
    return x % CURVE_ORDER


def _commit(vec: List[int], r: int) -> Tuple[int, int]:
    """Pedersen 承诺：cm = G^vec · H^r"""
    H = multiply(G1Generator, _to_fr(int(hashlib.sha256(b"gp-adafL-h").hexdigest(), 16)))
    cm = multiply(G1Generator, vec[0])  # 简化：仅对第一个元素承诺
    for v in vec[1:]:
        cm = add(cm, multiply(G1Generator, v))
    cm = add(cm, multiply(H, r))
    return cm


# ------------- 主证明器 ------------- #
class ZKProver:
    def __init__(self, circuit_file: str = None):
        # 极简 Groth16 设置：仅演示核心流程
        # 真实场景需 snarkjs/circom 生成 proving-key
        self.pk = self._dummy_keygen()

    # -------- 公开接口 -------- #
    def prove(self,
              g_flat: List[int],  # 裁剪后梯度（私）
              xi_flat: List[int],  # 注入噪声（私）
              C: int,  # 裁剪阈值（公）
              sigma: int  # 噪声尺度（公）
              ) -> Tuple[dict, List[int]]:
        """返回 (proof, public_inputs)"""
        r = _to_fr(torch.randint(0, CURVE_ORDER, (1,)).item())  # 盲化随机数
        cm = _commit(g_flat, r)

        # 公输入
        public = [cm[0], cm[1], C, sigma]

        # 私输入 → 见证
        witness = g_flat + xi_flat + [r]

        # 生成证明（此处为 dummy，真实用 snarkjs）
        proof = self._dummy_prove(witness, public)
        return proof, public

    # -------- 内部：dummy 证明（仅演示） -------- #
    def _dummy_keygen(self):
        return {"alpha": torch.randint(1, CURVE_ORDER, (1,)).item(),
                "beta": torch.randint(1, CURVE_ORDER, (1,)).item()}

    def _dummy_prove(self, witness: List[int], public: List[int]) -> dict:
        """仅用于跑通流程，真实用 snarkjs + circom"""
        h = 0
        for w in witness:
            h = (h + w * self.pk["alpha"]) % CURVE_ORDER
        for p in public:
            h = (h + p * self.pk["beta"]) % CURVE_ORDER
        A = multiply(G1Generator, h)
        B = multiply(G2Generator, h)
        return {"A": (A[0], A[1]), "B": (B[0], B[1])}


# ------------- 快捷封装 ------------- #
def zk_prove_grad(grad_tensor: torch.Tensor,
                  noise_tensor: torch.Tensor,
                  C: float,
                  sigma: float) -> Tuple[dict, List[int]]:
    """输入 torch.Tensor → 返回 (proof, public)"""
    grad_flat = grad_tensor.detach().cpu().view(-1).tolist()
    xi_flat = noise_tensor.detach().cpu().view(-1).tolist()
    # 量化到整数（放大 1e6 保留精度）
    scale = 1_000_000
    g_int = [int(g * scale) for g in grad_flat]
    xi_int = [int(x * scale) for x in xi_flat]
    prover = ZKProver()
    proof, pub = prover.prove(g_int, xi_int, int(C * scale), int(sigma * scale))
    return proof, pub


# ------------- 本地自测 ------------- #
if __name__ == "__main__":
    g = torch.randn(100) * 0.5
    xi = torch.randn(100) * 0.1
    proof, pub = zk_prove_grad(g, xi, C=1.0, sigma=5.0)
    print("Proof generated!")
    print("Public inputs:", pub[:4])  # cm_x, cm_y, C, sigma
