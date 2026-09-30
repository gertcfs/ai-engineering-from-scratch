import time

def check_gpu():
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Run: pip install torch")
        return

    print("=== GPU Check ===\n")
    print(f"PyTorch version: {torch.__version__}")
    
    # 1. Detectar el mejor hardware disponible
    if torch.cuda.is_available():
        device_type = "cuda"
        print("Hardware detected: NVIDIA GPU (CUDA)")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        total_mem_gb = props.total_memory / 1e9
        print(f"Memory: {total_mem_gb:.1f} GB")
        print(f"Compute capability: {props.major}.{props.minor}")
    elif torch.backends.mps.is_available():
        device_type = "mps"
        print("Hardware detected: Apple Silicon GPU (MPS)")
        # En Mac no hay 'device_properties' idéntico a CUDA, pero sabemos que tienes un M2 Max
        print("GPU: Apple M2 Max Integrated Graphics")
        # Al usar memoria unificada, estimamos un uso seguro para gráficos (p.e. 75% de tus 64GB)
        total_mem_gb = 64.0  
        print(f"Unified Memory (Total System): {total_mem_gb:.1f} GB")
    else:
        device_type = "cpu"
        print("\nNo GPU detected (Using CPU). That's fine for most lessons.")
        print("For GPU-heavy lessons, use Google Colab (free).")
        return

    device = torch.device(device_type)

    print("\n=== CPU vs GPU Benchmark ===\n")
    size = 4000

    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # Benchmark en CPU
    start = time.time()
    _ = a @ b
    cpu_time = time.time() - start
    print(f"CPU matrix multiply ({size}x{size}): {cpu_time:.3f}s")

    # Mover a la GPU disponible (sea CUDA o MPS)
    a_gpu = a.to(device)
    b_gpu = b.to(device)
    
    # Sincronizar hardware gráfico para medir bien el tiempo
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()

    start = time.time()
    _ = a_gpu @ b_gpu
    
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()
        
    gpu_time = time.time() - start
    print(f"GPU ({device_type.upper()}) matrix multiply ({size}x{size}): {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")

    # Estimación de parámetros
    params_fp16 = total_mem_gb * 1e9 / 2
    params_billions = params_fp16 / 1e9
    print(f"\nEstimated max model size (fp16): ~{params_billions:.0f}B parameters")

if __name__ == "__main__":
    check_gpu()
