# blockchain/scpb.py
"""
Smart-Contract Proxy for Privacy Budget (SCPB)
- Interfaces with an on-chain Groth16 verifying contract
- allocateWindow: locks ε for the next W rounds
- consume: submits Δε + zk-proof, updates on-chain accumulator
- getRemaining: queries remaining budget in active window
All monetary values are in Wei (smallest Ether unit).
"""

import json, math, os
from typing import Dict, Any, Tuple
from web3 import Web3
from eth_account import Account
from eth_utils import to_checksum_address


class SCPB:
    """
    Lightweight proxy to the Solidity SCPB contract
    ABI must contain:
        - allocateWindow(windowId, epsilonWei)
        - consume(deltaEpsilonWei, proofA, proofB, inputs)
        - getRemaining(windowId) → uint256
    """

    def __init__(self,
                 rpc_url: str,
                 contract_addr: str,
                 abi_path: str,
                 private_key: str):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise RuntimeError("Cannot connect to RPC")
        self.account = Account.from_key(private_key)
        self.address = to_checksum_address(contract_addr)
        with open(abi_path, "r") as f:
            self.abi = json.load(f)
        self.contract = self.w3.eth.contract(address=self.address, abi=self.abi)
        self.chain_id = self.w3.eth.chain_id

    # ------------- high-level APIs ------------- #

    def allocate_window(self,
                        window_id: int,
                        epsilon_wei: int,
                        gas_limit: int = 200_000) -> str:
        """
        Lock ε_wei for window window_id.
        Returns tx-hash.
        """
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        tx = self.contract.functions.allocateWindow(window_id, epsilon_wei).build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "gas": gas_limit,
            "gasPrice": self.w3.eth.gas_price,
            "chainId": self.chain_id
        })
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        return tx_hash.hex()

    def consume(self,
                delta_epsilon_wei: int,
                proof: Dict[str, List[int]],
                public_inputs: List[int],
                gas_limit: int = 300_000) -> str:
        """
        Submit Δε and zk-proof to the contract.
        proof = {"A": [x, y], "B": [x, y]}  (G1 & G2 points)
        public_inputs = [cm_x, cm_y, C, sigma, ...]
        Returns tx-hash.
        """
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        tx = self.contract.functions.consume(
            delta_epsilon_wei,
            proof["A"],  # G1 point
            proof["B"],  # G2 point
            public_inputs
        ).build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "gas": gas_limit,
            "gasPrice": self.w3.eth.gas_price,
            "chainId": self.chain_id
        })
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        return tx_hash.hex()

    def get_remaining(self, window_id: int) -> int:
        """Query remaining budget (in Wei) for active window."""
        return self.contract.functions.getRemaining(window_id).call()

    # ------------- helper conversions ------------- #

    @staticmethod
    def epsilon_to_wei(epsilon: float, scale: int = 10 ** 18) -> int:
        """Convert floating ε to fixed-point Wei."""
        return int(epsilon * scale)

    @staticmethod
    def wei_to_epsilon(wei: int, scale: int = 10 ** 18) -> float:
        """Convert Wei back to floating ε."""
        return wei / scale


# ---------------- quick test ---------------- #
if __name__ == "__main__":
    import os
    rpc = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    addr = os.getenv("SCPB_ADDR", "0x...")
    abi_file = os.path.join(os.path.dirname(__file__), "scpb_abi.json")
    pk = os.getenv("PRIVATE_KEY")  # 0x...
    scpb = SCPB(rpc, addr, abi_file, pk)

    # 1. allocate
    tx1 = scpb.allocate_window(window_id=1, epsilon_wei=SCPB.epsilon_to_wei(10.0))
    print("allocate tx:", tx1)

    # 2. consume
    dummy_proof = {"A": [1, 2], "B": [1, 2]}  # replace with real Groth16 proof
    public = [3, 4, 100, 5]  # cm, C, sigma
    tx2 = scpb.consume(delta_epsilon_wei=SCPB.epsilon_to_wei(0.1),
                       proof=dummy_proof,
                       public_inputs=public)
    print("consume tx:", tx2)

    # 3. query
    rem = scpb.get_remaining(1)
    print("remaining ε:", SCPB.wei_to_epsilon(rem))
