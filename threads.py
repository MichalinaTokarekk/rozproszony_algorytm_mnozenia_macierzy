import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Process, Array
from numba import cuda

class MatrixMultiplication:
    def __init__(self, size, num_processes):
        self.size = size
        self.num_processes = num_processes
        self.A, self.B = self.generate_matrices(size)
        self.C = Array('d', size * size)  # Shared memory array

    def generate_matrices(self, size):
        A = np.random.uniform(0, 10, (size, size))
        B = np.random.uniform(0, 10, (size, size))
        return A, B

    def sequential_multiply(self):
        C = np.zeros((self.size, self.size))
        start_compute = time.time()
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    C[i, j] += self.A[i, k] * self.B[k, j]
        compute_time = time.time() - start_compute
        return compute_time, 0  # No communication in sequential multiply

    def worker(self, start, end, A, B, C, size):
        A_np = np.ctypeslib.as_array(A).reshape((size, size))
        B_np = np.ctypeslib.as_array(B).reshape((size, size))
        C_np = np.ctypeslib.as_array(C).reshape((size, size))
        for i in range(start, end):
            for j in range(size):
                for k in range(size):
                    C_np[i, j] += A_np[i, k] * B_np[k, j]

    def multiprocess_multiply(self):
        processes = []
        chunk_size = self.size // self.num_processes

        # Shared memory arrays
        A_shared = Array('d', self.A.flatten(), lock=False)
        B_shared = Array('d', self.B.flatten(), lock=False)

        # Start communication timer
        start_communication = time.time()
        for i in range(self.num_processes):
            start = i * chunk_size
            end = self.size if i == self.num_processes - 1 else (i + 1) * chunk_size
            p = Process(target=self.worker, args=(start, end, A_shared, B_shared, self.C, self.size))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        communication_time = time.time() - start_communication

        # Simulate computation time (not directly measurable in this model)
        compute_time = communication_time * 0.5  # Approximation for computation time
        return compute_time, communication_time

    @staticmethod
    @cuda.jit
    def gpu_kernel(A, B, C):
        i, j = cuda.grid(2)
        if i < C.shape[0] and j < C.shape[1]:
            temp = 0
            for k in range(A.shape[1]):
                temp += A[i, k] * B[k, j]
            C[i, j] = temp

    def hybrid_multiply(self):
        gpu_rows = self.size // 2
        cpu_rows = self.size - gpu_rows

        # GPU computation
        start_gpu = time.time()
        A_gpu = cuda.to_device(self.A[:gpu_rows])
        B_gpu = cuda.to_device(self.B)
        C_gpu = cuda.device_array((gpu_rows, self.size))

        threads_per_block = (16, 16)
        blocks_per_grid_x = int(np.ceil(gpu_rows / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil(self.size / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        MatrixMultiplication.gpu_kernel[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_gpu)
        gpu_compute_time = time.time() - start_gpu

        # CPU computation
        start_cpu = time.time()
        C_cpu = np.zeros((cpu_rows, self.size))
        for i in range(gpu_rows, self.size):
            for j in range(self.size):
                for k in range(self.size):
                    C_cpu[i - gpu_rows, j] += self.A[i, k] * self.B[k, j]
        cpu_compute_time = time.time() - start_cpu

        # Simulate communication overhead
        communication_time = 0.05  # Example fixed overhead for data transfers

        total_compute_time = gpu_compute_time + cpu_compute_time
        return total_compute_time, communication_time


if __name__ == "__main__":
    sizes = [100, 200, 400]  # Matrix sizes to test
    num_processes_list = [2, 4, 8]  # Different numbers of processes
    times = {"sequential": [], "hybrid": [], **{num_processes: [] for num_processes in num_processes_list}}

    for size in sizes:
        print(f"Testing matrix size: {size}x{size}")

        # Multiprocessing multiply
        for num_processes in num_processes_list:
            print(f"Testing with {num_processes} processes...")
            mm = MatrixMultiplication(size, num_processes)
            compute_time, communication_time = mm.multiprocess_multiply()
            total_time = compute_time + communication_time
            times[num_processes].append((total_time, compute_time, communication_time))
            print(f"{num_processes} processes: total = {total_time:.4f}s, compute = {compute_time:.4f}s, communication = {communication_time:.4f}s")

        # Sequential multiply
        print("Running sequential multiplication...")
        mm = MatrixMultiplication(size, 1)
        compute_time, communication_time = mm.sequential_multiply()
        total_time = compute_time + communication_time
        times["sequential"].append((total_time, compute_time, communication_time))
        print(f"Sequential: total = {total_time:.4f}s, compute = {compute_time:.4f}s, communication = {communication_time:.4f}s")

        # Hybrid multiply
        print("Running hybrid CPU+GPU multiplication...")
        mm = MatrixMultiplication(size, 1)
        try:
            compute_time, communication_time = mm.hybrid_multiply()
            total_time = compute_time + communication_time
            times["hybrid"].append((total_time, compute_time, communication_time))
            print(f"Hybrid: total = {total_time:.4f}s, compute = {compute_time:.4f}s, communication = {communication_time:.4f}s")
        except Exception as e:
            print(f"Hybrid multiplication failed: {e}")
            times["hybrid"].append((None, None, None))

    # Memory complexity
    memory_sequential = [size**2 * 8 for size in sizes]  # Sequential: O(n^2)
    memory_hybrid = [size**2 * 8 * 1.3 for size in sizes]  # Extra 30% for hybrid GPU+CPU
    memory_multiprocessing = {num_processes: [size**2 * 8 * num_processes for size in sizes] for num_processes in num_processes_list}

    # Compute memory speedup and efficiency
    memory_speedup = {num_processes: [] for num_processes in num_processes_list}
    memory_efficiency = {num_processes: [] for num_processes in num_processes_list}

    for num_processes in num_processes_list:
        for i, size in enumerate(sizes):
            sm = memory_sequential[i] / memory_multiprocessing[num_processes][i]
            em = sm / num_processes
            memory_speedup[num_processes].append(sm)
            memory_efficiency[num_processes].append(em)

    # Hybrid memory speedup and efficiency
    hybrid_memory_speedup = [memory_sequential[i] / memory_hybrid[i] for i in range(len(sizes))]
    hybrid_memory_efficiency = [s / 2 for s in hybrid_memory_speedup]  # Assuming 2 resources (CPU + GPU)

    # Plot time complexity
    plt.figure()
    plt.plot(sizes, times["sequential"], label="Sequential")
    if not all(t is None for t in times["hybrid"]):
        plt.plot(sizes, times["hybrid"], label="Hybrid")
    for num_processes in num_processes_list:
        plt.plot(sizes, times[num_processes], label=f"{num_processes} processes")
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.title("Time Complexity with Sequential, Hybrid, and Multiprocessing")
    plt.show()

    # Plot speedup
    plt.figure()
    if not all(t is None for t in times["hybrid"]):
        hybrid_speedups = [times["sequential"][i] / t if t else None for i, t in enumerate(times["hybrid"])]
        plt.plot(sizes, hybrid_speedups, label="Hybrid")
    for num_processes in num_processes_list:
        speedups = [times["sequential"][i] / times[num_processes][i] for i in range(len(sizes))]
        plt.plot(sizes, speedups, label=f"{num_processes} processes")
    plt.xlabel("Matrix Size")
    plt.ylabel("Speedup")
    plt.legend()
    plt.title("Speedup with Multiprocessing and Hybrid")
    plt.show()

    # Plot efficiency
    plt.figure()
    if not all(t is None for t in times["hybrid"]):
        hybrid_efficiency = [s / 2 if s else None for s in hybrid_speedups]
        plt.plot(sizes, hybrid_efficiency, label="Hybrid")
    for num_processes in num_processes_list:
        speedups = [times["sequential"][i] / times[num_processes][i] for i in range(len(sizes))]
        efficiencies = [s / num_processes for s in speedups]
        plt.plot(sizes, efficiencies, label=f"{num_processes} processes")
    plt.xlabel("Matrix Size")
    plt.ylabel("Efficiency")
    plt.legend()
    plt.title("Efficiency with Multiprocessing and Hybrid")
    plt.show()

    # Plot memory complexity
    plt.figure()
    memory_sequential = [size**2 * 8 for size in sizes]  # Sequential: O(n^2)
    plt.plot(sizes, memory_sequential, label="Memory Complexity (Sequential)")

    if not all(t is None for t in times["hybrid"]):
        memory_hybrid = [size**2 * 8 * 1.3 for size in sizes]  # Extra 30% for hybrid GPU+CPU
        plt.plot(sizes, memory_hybrid, label="Memory Complexity (Hybrid)")

    for num_processes in num_processes_list:
        memory_multiprocessing = [size**2 * 8 * num_processes for size in sizes]
        plt.plot(sizes, memory_multiprocessing, label=f"Memory Complexity ({num_processes} processes)")

    plt.xlabel("Matrix Size")
    plt.ylabel("Memory Usage (bytes)")
    plt.legend()
    plt.title("Memory Complexity for Sequential, Hybrid, and Multiprocessing")
    plt.show()

    # Plot communication times
    plt.figure()
    plt.plot(sizes, communication_times["sequential"], label="Communication Time (Sequential)")
    plt.plot(sizes, communication_times["hybrid"], label="Communication Time (Hybrid)")
    for num_processes in num_processes_list:
        plt.plot(sizes, communication_times[num_processes], label=f"Communication Time ({num_processes} processes)")
    plt.xlabel("Matrix Size")
    plt.ylabel("Communication Time (s)")
    plt.legend()
    plt.title("Communication Time for Sequential, Hybrid, and Multiprocessing")
    plt.show()

    # Plot communication speedup
    plt.figure()
    for num_processes in num_processes_list:
        plt.plot(sizes, communication_speedup[num_processes], label=f"Communication Speedup ({num_processes} processes)")
    plt.plot(sizes, hybrid_communication_speedup, label="Communication Speedup (Hybrid)", linestyle="--")
    plt.xlabel("Matrix Size")
    plt.ylabel("Communication Speedup")
    plt.legend()
    plt.title("Communication Speedup for Multiprocessing and Hybrid")
    plt.show()

    # Plot communication efficiency
    plt.figure()
    for num_processes in num_processes_list:
        plt.plot(sizes, communication_efficiency[num_processes], label=f"Communication Efficiency ({num_processes} processes)")
    plt.plot(sizes, hybrid_communication_efficiency, label="Communication Efficiency (Hybrid)", linestyle="--")
    plt.xlabel("Matrix Size")
    plt.ylabel("Communication Efficiency")
    plt.legend()
    plt.title("Communication Efficiency for Multiprocessing and Hybrid")
    plt.show()
