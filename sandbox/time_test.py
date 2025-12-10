import time
from collections.abc import Callable
from itertools import repeat

import numpy as np
from scanning_optimized import scanning_optimized


def benchmark(func:Callable, name:str, *args, **kwargs):
    print("\n" + "=" * 70)
    print(f"BENCHMARK: {name}")
    print("=" * 70)

    start_time = time.perf_counter()
    _ = func(*args, **kwargs)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print("Done")
    print(f"Execution time: {elapsed_time:.6f} seconds ({elapsed_time * 1000:.2f} ms)")
    print("=" * 70 + "\n")

    return elapsed_time

def benchmark_process_contours():
    """Замер производительности process_contours_optimized"""
    parallelepiped = np.array([[x, y, z]
                               for x in range(0, 100)
                               for y in range(0, 100)
                               for z in range(0, 100)], dtype=np.float32)

    contour = (np.array([[100,
                          np.sin(n*2*np.pi/360),
                          np.cos(n*2*np.pi/360)]
                         for n in range(0, 360)], dtype=np.float32),
               np.array([[0, 0, 1],
                         [0, 1, 0],
                         [-1, 0, 0]], dtype=np.float32))

    contours = list(repeat(contour, 20))

    elapsed_time = benchmark(scanning_optimized.process_contours_optimized, 'process_contours_optimized', parallelepiped, contours)

    return elapsed_time

def benchmark_creating_mesh():
    """Замер производительности creating_mesh"""
    parallelepiped = np.array([[x, y, z]
                               for x in range(0, 100)
                               for y in range(0, 100)
                               for z in range(0, 100)], dtype=np.float32)
    elapsed_time = benchmark(scanning_optimized.build_voxel_mesh, 'build_voxel_mesh',parallelepiped, 0.01)
    return elapsed_time

def benchmark_multiple_runs(func, runs=5):
    """Запускает benchmark несколько раз для получения средних значений"""
    print("\n" + "=" * 70)
    print(f"RUNNING {runs} BENCHMARK ITERATIONS")
    print("=" * 70 + "\n")

    times = []
    for i in range(runs):
        print(f"Run {i + 1}/{runs}")
        elapsed = func()
        times.append(elapsed)

    print("=" * 70)
    print(f"Average time: {np.mean(times):.6f} seconds ({np.mean(times) * 1000:.2f} ms)")
    print(f"Std deviation: {np.std(times):.6f} seconds ({np.std(times) * 1000:.2f} ms)")
    print(f"Min time: {np.min(times):.6f} seconds ({np.min(times) * 1000:.2f} ms)")
    print(f"Max time: {np.max(times):.6f} seconds ({np.max(times) * 1000:.2f} ms)")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    benchmark_multiple_runs(benchmark_process_contours, runs=5)
    benchmark_multiple_runs(benchmark_creating_mesh, runs=5)
