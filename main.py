import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Process, Array

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
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    C[i, j] += self.A[i, k] * self.B[k, j]
        return C

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

        for i in range(self.num_processes):
            start = i * chunk_size
            end = self.size if i == self.num_processes - 1 else (i + 1) * chunk_size
            p = Process(target=self.worker, args=(start, end, A_shared, B_shared, self.C, self.size))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

if __name__ == "__main__":
    sizes = [10, 30, 60, 100, 200, 400, 500]  # Matrix sizes to test
    num_processes_list = [2, 4, 8]  # Different numbers of processes
    times = {"sequential": [], **{num_processes: [] for num_processes in num_processes_list}}

    for size in sizes:
        print(f"Testing matrix size: {size}x{size}")

        # Sequential multiply
        mm = MatrixMultiplication(size, 1)
        start_time = time.time()
        mm.sequential_multiply()
        elapsed_time = time.time() - start_time
        times["sequential"].append(elapsed_time)
        print(f"Sequential completed in {elapsed_time:.4f} seconds")

        # Multiprocessing multiply
        for num_processes in num_processes_list:
            mm = MatrixMultiplication(size, num_processes)
            start_time = time.time()
            mm.multiprocess_multiply()
            elapsed_time = time.time() - start_time
            times[num_processes].append(elapsed_time)
            print(f"{num_processes} processes completed in {elapsed_time:.4f} seconds")

    # Plot time complexity
    plt.figure()
    plt.plot(sizes, times["sequential"], label="Sequential")
    for num_processes in num_processes_list:
        plt.plot(sizes, times[num_processes], label=f"{num_processes} processes")
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.title("Time Complexity with Sequential and Multiprocessing")
    plt.show()
