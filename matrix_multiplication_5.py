import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
from mpi4py import MPI
import time
import psutil

# Kernel CUDA do mnożenia macierzy
kernel_code = """
__global__ void MatrixMulKernel(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float Cvalue = 0;
        for (int k = 0; k < n; ++k) {
            Cvalue += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = Cvalue;
    }
}
"""

# Funkcja do mnożenia macierzy na GPU (sekwencyjnie)
def gpu_matrix_multiply_sequential(A, B):
    n = A.shape[0]

    # Przygotowanie pamięci na GPU
    A_gpu = cuda.mem_alloc(A.nbytes)
    B_gpu = cuda.mem_alloc(B.nbytes)
    C_gpu = cuda.mem_alloc(A.nbytes)
    
    cuda.memcpy_htod(A_gpu, A)
    cuda.memcpy_htod(B_gpu, B)

    # Kompilacja kodu CUDA
    mod = SourceModule(kernel_code)
    matrixmul = mod.get_function("MatrixMulKernel")

    # Ustawienie wymiarów bloków i siatki
    block_size = 16
    grid_size = (n + block_size - 1) // block_size
    matrixmul(A_gpu, B_gpu, C_gpu, np.int32(n), block=(block_size, block_size, 1), grid=(grid_size, grid_size))

    # Pobranie wyników z GPU
    C = np.empty_like(A)
    cuda.memcpy_dtoh(C, C_gpu)

    return C

# Funkcja do mnożenia macierzy na CPU (sekwencyjnie)
def cpu_matrix_multiply_sequential(A, B):
    return np.dot(A, B)

# Funkcja do mnożenia macierzy w środowisku rozproszonym (CPU)
def distributed_matrix_multiply_cpu(A, B, comm, rank, size):
    n = A.shape[0]
    sqrt_p = int(np.sqrt(size))

    if sqrt_p ** 2 != size:
        raise ValueError("Liczba procesów musi być kwadratem liczby całkowitej")

    block_size = n // sqrt_p
    sub_A = np.zeros((block_size, block_size), dtype=np.float64)
    sub_B = np.zeros((block_size, block_size), dtype=np.float64)
    sub_C = np.zeros((block_size, block_size), dtype=np.float64)

    row_comm = comm.Split(rank // sqrt_p)
    col_comm = comm.Split(rank % sqrt_p)

    for i in range(sqrt_p):
        row_comm.Bcast(sub_A, root=i)
        sub_C += cpu_matrix_multiply_sequential(sub_A, sub_B)

    return sub_C

# Funkcja do mnożenia macierzy w środowisku rozproszonym (GPU)
def distributed_matrix_multiply_gpu(A, B, comm, rank, size):
    n = A.shape[0]
    sqrt_p = int(np.sqrt(size))

    if sqrt_p ** 2 != size:
        raise ValueError("Liczba procesów musi być kwadratem liczby całkowitej")

    block_size = n // sqrt_p
    sub_A = np.zeros((block_size, block_size), dtype=np.float32)
    sub_B = np.zeros((block_size, block_size), dtype=np.float32)
    sub_C = np.zeros((block_size, block_size), dtype=np.float32)

    row_comm = comm.Split(rank // sqrt_p)
    col_comm = comm.Split(rank % sqrt_p)

    for i in range(sqrt_p):
        row_comm.Bcast(sub_A, root=i)
        sub_C += gpu_matrix_multiply_sequential(sub_A, sub_B)

    return sub_C

# Funkcja mierząca zużycie pamięci
def memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1e6  # Zwracamy zużycie pamięci w MB

# Funkcja do testowania różnych metod
def run_tests(matrix_sizes, num_nodes):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for matrix_size in matrix_sizes:
        if rank == 0:
            # Testowanie rozproszonych metod
            print(f"\nTesting for matrix size: {matrix_size}x{matrix_size}")

            # Distributed CPU
            start_time = time.time()
            A = np.random.rand(matrix_size, matrix_size).astype(np.float32)
            B = np.random.rand(matrix_size, matrix_size).astype(np.float32)
            sub_C = distributed_matrix_multiply_cpu(A, B, comm, rank, size)
            print(f"Distributed CPU: {time.time() - start_time} seconds")

            # Distributed GPU
            start_time = time.time()
            sub_C = distributed_matrix_multiply_gpu(A, B, comm, rank, size)
            print(f"Distributed GPU: {time.time() - start_time} seconds")

            # Sekwencyjne CPU
            start_time = time.time()
            C_cpu = cpu_matrix_multiply_sequential(A, B)
            print(f"Sequential CPU: {time.time() - start_time} seconds")

            # Sekwencyjne GPU
            start_time = time.time()
            C_gpu = gpu_matrix_multiply_sequential(A, B)
            print(f"Sequential GPU: {time.time() - start_time} seconds")

if __name__ == '__main__':
    # Rozmiary macierzy do przetestowania
    matrix_sizes = [256, 512]
    run_tests(matrix_sizes, MPI.COMM_WORLD.Get_size())
