# blockchain/deploy.py
"""
Hardhat deployment script for SCPB (Privacy-Budget Sliding-Window Contract)
- compiles Solidity contracts
- deploys SCPB to target network
- saves ABI & address to disk for backend/python layer
Usage:
    python deploy.py --network localhost --gas-price 20000000000
"""

import json, os, sys
from pathlib import Path
import subprocess
import argparse


def run(cmd: list, check=True) -> str:
    """Run a shell command and return stdout."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=check)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        sys.exit(1)
    return result.stdout.strip()


def compile_contracts(contract_dir: Path):
    """Compile Solidity contracts with Hardhat."""
    os.chdir(contract_dir)
    run(["npx", "hardhat", "compile"])


def deploy_scpb(network: str, gas_price: str) -> dict:
    """Deploy SCPB and return deployment JSON."""
    script = f"""
        const SCPB = await ethers.getContractFactory("SCPB");
        const scpb = await SCPB.deploy({{ gasPrice: {gas_price} }});
        await scpb.deployed();
        console.log(JSON.stringify({
            address: scpb.address,
            abi: SCPB.interface.format("json")
        }));
    """
    # 使用 Hardhat console 执行内联脚本
    cmd = ["npx", "hardhat", "console", "--network", network]
    output = run(cmd, input=script, text=True)
    # 提取 JSON 行（最后一行）
    lines = output.splitlines()
    json_line = next((l for l in reversed(lines) if l.startswith("{")), "{}")
    return json.loads(json_line)


def save_deployment(deploy: dict, out_dir: Path):
    """Save ABI and address for Python layer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    abi_path = out_dir / "scpb_abi.json"
    addr_path = out_dir / "scpb_address.json"
    with abi_path.open("w") as f:
        json.dump(json.loads(deploy["abi"]), f, indent=2)
    with addr_path.open("w") as f:
        json.dump({"address": deploy["address"]}, f, indent=2)
    print(f"✓ ABI saved -> {abi_path}")
    print(f"✓ Address saved -> {addr_path}")


def main():
    parser = argparse.ArgumentParser(description="Deploy SCPB contract")
    parser.add_argument("--network", default="localhost", help="Hardhat network alias")
    parser.add_argument("--gas-price", type=str, default="20000000000", help="Gas price in wei")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "artifacts",
                        help="Output directory for ABI & address")
    args = parser.parse_args()

    contract_dir = Path(__file__).parent
    compile_contracts(contract_dir)
    deploy = deploy_scpb(args.network, args.gas_price)
    save_deployment(deploy, args.out)
    print("🎉 SCPB deployment finished!")


if __name__ == "__main__":
    main()
