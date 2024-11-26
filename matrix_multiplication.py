from mpi4py import MPI
import numpy as np
import cupy as cp
import time
import os

# Sekwencyjne mnożenie macierzy na CPU
def sequential_matrix_multiplication(A, B):
    n, m = len(A), len(B[0])
    result = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

# Sekwencyjne mnożenie macierzy na GPU
def sequential_gpu_matrix_multiplication(A, B):
    d_A = cp.array(A)
    d_B = cp.array(B)
    d_C = cp.dot(d_A, d_B)  # Mnożenie na GPU
    return cp.asnumpy(d_C)

# Rozproszone mnożenie macierzy na CPU
def distributed_matrix_multiplication(local_A, B):
    local_result = []
    for row in local_A:
        row_result = []
        for j in range(len(B[0])):
            sum = 0
            for k in range(len(B)):
                sum += row[k] * B[k][j]
            row_result.append(sum)
        local_result.append(row_result)
    return local_result

# Mnożenie macierzy na GPU
def gpu_matrix_multiplication(A, B):
    d_A = cp.array(A)
    d_B = cp.array(B)
    d_C = cp.dot(d_A, d_B)
    return cp.asnumpy(d_C)

# Rozproszone mnożenie macierzy z GPU
def distributed_gpu_matrix_multiplication(local_A, B):
    d_A = cp.array(local_A)
    d_B = cp.array(B)
    d_C = cp.dot(d_A, d_B)
    return cp.asnumpy(d_C)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n = 500  # Rozmiar macierzy
    if rank == 0:
        A = np.random.randint(0, 10, (n, n))
        B = np.random.randint(0, 10, (n, n))

        # Sekwencyjne mnożenie na CPU
        print("Rozpoczynam sekwencyjne mnożenie macierzy na CPU...")
        start_time = time.time()
        sequential_result = sequential_matrix_multiplication(A, B)
        print(f"Sekwencyjne wykonanie na CPU zajęło: {time.time() - start_time:.2f} s")

        # Sekwencyjne mnożenie na GPU
        print("Rozpoczynam sekwencyjne mnożenie macierzy na GPU...")
        start_time = time.time()
        sequential_gpu_result = sequential_gpu_matrix_multiplication(A, B)
        print(f"Sekwencyjne wykonanie na GPU zajęło: {time.time() - start_time:.2f} s")

        # GPU mnożenie
        print("Rozpoczynam mnożenie macierzy na GPU...")
        start_time = time.time()
        gpu_result = gpu_matrix_multiplication(A, B)
        print(f"Mnożenie na GPU zajęło: {time.time() - start_time:.2f} s")
        
    else:
        A = None
        B = None

    # Rozsyłanie macierzy B do wszystkich procesów
    B = comm.bcast(B, root=0)

    # Dzielenie macierzy A pomiędzy procesy
    local_A = np.array_split(A, size, axis=0) if rank == 0 else None
    local_A = comm.scatter(local_A, root=0)

    # Rozproszone mnożenie na CPU
    start_time = time.time()
    local_result = distributed_matrix_multiplication(local_A, B)
    gathered_results = comm.gather(local_result, root=0)

    if rank == 0:
        distributed_result = np.vstack(gathered_results)
        print(f"Rozproszone wykonanie na CPU zajęło: {time.time() - start_time:.2f} s")

    # Rozproszone mnożenie na GPU
    start_time = time.time()
    local_result_gpu = distributed_gpu_matrix_multiplication(local_A, B)
    gathered_results_gpu = comm.gather(local_result_gpu, root=0)

    if rank == 0:
        distributed_result_gpu = np.vstack(gathered_results_gpu)
        print(f"Rozproszone wykonanie na GPU zajęło: {time.time() - start_time:.2f} s")

    # Analiza wyników
    if rank == 0:
        # Testowanie poprawności wyników
        assert np.allclose(sequential_result, distributed_result), "Wyniki różnią się!"
        assert np.allclose(sequential_result, distributed_result_gpu), "Wyniki różnią się!"
        assert np.allclose(sequential_gpu_result, distributed_result_gpu), "Wyniki różnią się!"
        print("Wyniki są zgodne!")

        # Badania
        print("\nPorównanie wydajności:")
        print(f"Rozmiar macierzy: {n}x{n}")
        print(f"Liczba węzłów w rozproszonym systemie: {size}")
        
        # Sekwencyjne wykonanie na CPU
        print(f"Sekwencyjne wykonanie na CPU zajęło: {time.time() - start_time:.2f} s")
        
        # Sekwencyjne wykonanie na GPU
        print(f"Sekwencyjne wykonanie na GPU zajęło: {time.time() - start_time:.2f} s")
        
        # Mnożenie rozproszone na CPU
        print(f"Rozproszone wykonanie na CPU zajęło: {time.time() - start_time:.2f} s")
        
        # Mnożenie rozproszone na GPU
        print(f"Rozproszone wykonanie na GPU zajęło: {time.time() - start_time:.2f} s")
