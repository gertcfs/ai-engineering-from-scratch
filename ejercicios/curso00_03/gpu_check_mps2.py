import time
import torch

def real_benchmark():
    device = torch.device("mps")
    size = 4000
    print("Calentando motores y compilando kernels de Apple Silicon...")
    
    # Creamos las matrices directamente en el dispositivo para evitar tiempos de copia
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    a_mps = torch.randn(size, size, device=device)
    b_mps = torch.randn(size, size, device=device)

    # --- PASO DE CALENTAMIENTO (Warm-up) ---
    # Esto compila los shaders de Metal en tu Mac para que no falseen la medición
    _ = a_mps @ b_mps
    torch.mps.synchronize()
    
    print("\n=== Ejecutando Test Real (Media de 10 iteraciones) ===")
    
    # Benchmark CPU
    start = time.time()
    for _ in range(10):
        _ = a_cpu @ b_cpu
    cpu_time = (time.time() - start) / 10
    print(f"CPU promedio por multiplicación: {cpu_time:.4f}s")

    # Benchmark GPU (MPS)
    start = time.time()
    for _ in range(10):
        _ = a_mps @ b_mps
    torch.mps.synchronize()
    gpu_time = (time.time() - start) / 10
    print(f"GPU (MPS) promedio por multiplicación: {gpu_time:.4f}s")
    
    print(f"Speedup Real de la GPU: {cpu_time / gpu_time:.1f}x")

if __name__ == "__main__":
    real_benchmark()
