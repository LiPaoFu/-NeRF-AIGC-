import os
import sys
import platform
import subprocess
from datetime import datetime

def run(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"[FAILED] {e}"

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def try_import(name):
    try:
        __import__(name)
        mod = sys.modules[name]
        ver = getattr(mod, "__version__", "unknown")
        print(f"[OK] import {name} (version={ver})")
        return True
    except Exception as e:
        print(f"[FAIL] import {name}: {e}")
        return False

def main():
    section("BASIC INFO")
    print("Time:", datetime.now().isoformat(timespec="seconds"))
    print("OS:", platform.platform())
    print("Python:", sys.version.replace("\n", " "))
    print("Python exe:", sys.executable)
    print("CWD:", os.getcwd())

    section("PIP ENV")
    print("pip:", run("python -m pip --version"))
    # 只列出关键包，避免输出太长
    print("torch:", run("python -m pip show torch | findstr /R \"^Name ^Version\"") if os.name == "nt"
          else run("python -m pip show torch | egrep '^(Name|Version):'"))
    print("torchvision:", run("python -m pip show torchvision | findstr /R \"^Name ^Version\"") if os.name == "nt"
          else run("python -m pip show torchvision | egrep '^(Name|Version):'"))

    section("IMPORT CHECK (required packages)")
    required = ["torch", "torchvision", "numpy", "sklearn", "matplotlib", "pandas", "tqdm"]
    for pkg in required:
        try_import(pkg)

    section("CUDA / GPU CHECK (PyTorch)")
    try:
        import torch
        print("torch.__version__:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            idx = 0
            print("Current device:", torch.cuda.current_device())
            print("Device name:", torch.cuda.get_device_name(idx))
            props = torch.cuda.get_device_properties(idx)
            print("Total VRAM (GB):", round(props.total_memory / (1024**3), 2))

            # 真正分配一次显存 + 跑一次矩阵乘，确认不是“假可用”
            x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
            y = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            _ = (x @ y).sum().item()
            torch.cuda.synchronize()

            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"CUDA memory allocated (MB): {allocated:.1f}")
            print(f"CUDA memory reserved  (MB): {reserved:.1f}")
            print("[OK] CUDA compute test passed.")
        else:
            print("[WARN] torch.cuda.is_available() is False. You are likely on CPU-only torch.")
    except Exception as e:
        print("[FAIL] CUDA check error:", e)

    section("SYSTEM NVIDIA CHECK (optional)")
    # 这一步不是必须，但能看出系统是否装了驱动 & CUDA runtime
    print("nvidia-smi:\n", run("nvidia-smi"))

    section("DONE")
    print("If anything shows [FAIL] or CUDA available=False, paste this output to me.")

if __name__ == "__main__":
    main()