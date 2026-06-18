import subprocess
import sys
import platform

def detect_gpu():
    # 1. First, check for NVIDIA via nvidia-smi (fastest and standard for CUDA)
    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, timeout=2)
        if res.returncode == 0:
            return "NVIDIA"
    except Exception:
        pass

    system = platform.system().lower()

    # 2. Windows specific controller search
    if system == "windows":
        try:
            cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if res.returncode == 0:
                gpu_names = res.stdout.lower()
                if "nvidia" in gpu_names:
                    return "NVIDIA"
                elif "intel" in gpu_names:
                    return "INTEL"
                elif "amd" in gpu_names or "radeon" in gpu_names:
                    return "AMD"
        except Exception:
            pass

    # 3. Linux specific controller search
    elif system == "linux":
        try:
            res = subprocess.run(["lspci"], capture_output=True, text=True, timeout=3)
            if res.returncode == 0:
                gpu_names = res.stdout.lower()
                if "nvidia" in gpu_names:
                    return "NVIDIA"
                elif "intel" in gpu_names:
                    return "INTEL"
                elif "amd" in gpu_names or "radeon" in gpu_names:
                    return "AMD"
        except Exception:
            pass

    # 4. macOS specific check
    elif system == "darwin":
        if platform.machine() == "arm64":
            return "MPS"

    return "CPU"

if __name__ == "__main__":
    print(detect_gpu())
