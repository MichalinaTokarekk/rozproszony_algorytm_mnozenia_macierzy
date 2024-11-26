from mpi4py import MPI
import numpy as np
import cupy as cp
import time
import psutil
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
def sequential_gpu_matrix_multiplication_custom(A, B):
    # Konwersja macierzy na GPU
    d_A = cp.array(A)
    d_B = cp.array(B)
    n, m, p = d_A.shape[0], d_B.shape[1], d_A.shape[1]

    # Tworzenie macierzy wynikowej na GPU
    d_C = cp.zeros((n, m), dtype=cp.float64)

    # Implementacja własnego mnożenia na GPU
    for i in range(n):
        for j in range(m):
            for k in range(p):
                d_C[i, j] += d_A[i, k] * d_B[k, j]

    # Konwersja wyników z GPU na CPU
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


# Rozproszone mnożenie macierzy na GPU
def distributed_gpu_matrix_multiplication(local_A, B):
    # Konwersja danych na GPU
    d_A = cp.array(local_A)
    d_B = cp.array(B)
    
    # Wymiary macierzy
    n, m, p = d_A.shape[0], d_B.shape[1], d_A.shape[1]

    # Tworzenie pustej macierzy wynikowej na GPU
    d_C = cp.zeros((n, m), dtype=cp.float64)

    # Implementacja własnego algorytmu mnożenia
    for i in range(n):
        for j in range(m):
            for k in range(p):
                d_C[i, j] += d_A[i, k] * d_B[k, j]

    # Konwersja wyników z GPU na CPU
    return cp.asnumpy(d_C)


# Funkcja do mierzenia zużycia pamięci
def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # W MB


# Złożoność pamięciowa (szacunkowa)
def memory_complexity(n):
    # Macierze A, B i C: O(n^2) każda
    return 3 * n * n * 8 / (1024 * 1024)  # W MB (zakładając float64, czyli 8 bajtów na element)

# Złożoność komunikacyjna
def communication_complexity(size, n):
    # Rozsyłanie danych (bcast i scatter): O(n^2 / size)
    # Zbieranie wyników (gather): O(n^2)
    return (n * n / size) + n * n  # O(n^2) w rozproszonym systemie, zależne od liczby węzłów

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
        print(f"Zużycie pamięci na CPU: {memory_usage():.2f} MB")
        print(f"Złożoność pamięciowa na CPU: {memory_complexity(n):.2f} MB")

        # Sekwencyjne mnożenie na GPU
        print("Rozpoczynam sekwencyjne mnożenie macierzy na GPU...")
        start_time = time.time()
        sequential_gpu_result = sequential_gpu_matrix_multiplication_custom(A, B)
        print(f"Sekwencyjne wykonanie na GPU zajęło: {time.time() - start_time:.2f} s")
        
    else:
        A = None
        B = None

    # Rozsyłanie macierzy B do wszystkich procesów
    start_time = time.time()
    B = comm.bcast(B, root=0)
    if rank == 0:
        print(f"Czas komunikacji (bcast B): {time.time() - start_time:.2f} s")

    # Dzielenie macierzy A pomiędzy procesy
    local_A = np.array_split(A, size, axis=0) if rank == 0 else None
    start_time = time.time()
    local_A = comm.scatter(local_A, root=0)
    if rank == 0:
        print(f"Czas komunikacji (scatter A): {time.time() - start_time:.2f} s")

    # Rozproszone mnożenie na CPU
    start_time = time.time()
    local_result = distributed_matrix_multiplication(local_A, B)
    gathered_results = comm.gather(local_result, root=0)

    if rank == 0:
        distributed_result = np.vstack(gathered_results)
        print(f"Rozproszone wykonanie na CPU zajęło: {time.time() - start_time:.2f} s")
        print(f"Zużycie pamięci na CPU: {memory_usage():.2f} MB")
        print(f"Złożoność pamięciowa na CPU: {memory_complexity(n):.2f} MB")
        print(f"Złożoność komunikacyjna: {communication_complexity(size, n):.2f} MB")

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
