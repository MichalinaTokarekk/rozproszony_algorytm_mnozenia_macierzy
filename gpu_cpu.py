from multiprocessing import Process, Pipe
import cupy as cp
import numpy as np

def cpu_worker(A, B, row_start, row_end, conn):
    rows_A = len(A)
    cols_B = len(B[0])
    cols_A = len(A[0])

    local_result = [[0] * cols_B for _ in range(row_end - row_start)]
    for i in range(row_start, row_end):
        for j in range(cols_B):
            for k in range(cols_A):
                local_result[i - row_start][j] += A[i][k] * B[k][j]
    conn.send(local_result)
    conn.close()

def gpu_worker(A_gpu, B_gpu, row_start, row_end, conn):
    A_gpu_segment = A_gpu[row_start:row_end]
    local_result_gpu = cp.dot(A_gpu_segment, B_gpu)
    conn.send(local_result_gpu.get().tolist())
    conn.close()

class MatrixMultiplierDistributed:
    """
    A class to perform matrix multiplication using both CPU (multiprocessing) and GPU (CUDA).
    """
    @staticmethod
    def multiply_matrices_distributed(A, B):
        """
        Static method to multiply two matrices A and B using both CPU and GPU.

        Parameters:
        A (list of list of int/float): Matrix A.
        B (list of list of int/float): Matrix B.

        Returns:
        list of list of int/float: Resultant matrix after multiplication.
        """
        rows_A = len(A)
        cols_A = len(A[0])
        rows_B = len(B)
        cols_B = len(B[0])

        if cols_A != rows_B:
            raise ValueError("Number of columns in A must be equal to number of rows in B.")

        result = [[0] * cols_B for _ in range(rows_A)]

        mid = rows_A // 2

        # Convert matrices to GPU arrays
        A_gpu = cp.array(A)
        B_gpu = cp.array(B)

        # Set up pipes for communication
        parent_conn1, child_conn1 = Pipe()
        parent_conn2, child_conn2 = Pipe()

        # Start CPU process
        cpu_process = Process(target=cpu_worker, args=(A, B, 0, mid, child_conn1))

        # Start GPU process
        gpu_process = Process(target=gpu_worker, args=(A_gpu, B_gpu, mid, rows_A, child_conn2))

        cpu_process.start()
        gpu_process.start()

        # Receive results from processes
        part1 = parent_conn1.recv()
        part2 = parent_conn2.recv()

        cpu_process.join()
        gpu_process.join()

        # Combine results
        result[:mid] = part1
        result[mid:] = part2

        return result

# Example usage
if __name__ == "__main__":
    from generate_matrix import MatrixGenerator

    # Generate matrices using MatrixGenerator from another file
    A = MatrixGenerator.generate_random_matrix(4, 4)
    B = MatrixGenerator.generate_random_matrix(4, 4)

    print("Matrix A:")
    for row in A:
        print(row)

    print("\nMatrix B:")
    for row in B:
        print(row)

    # Perform matrix multiplication using distributed computation (CPU + GPU)
    result = MatrixMultiplierDistributed.multiply_matrices_distributed(A, B)

    print("\nResultant Matrix (Distributed - CPU + GPU):")
    for row in result:
        print(row)
