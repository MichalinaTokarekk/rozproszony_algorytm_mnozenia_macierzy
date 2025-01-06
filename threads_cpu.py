from multiprocessing import Process, Pipe

def worker(A, B, row_start, row_end, cols_B, cols_A, conn):
    local_result = [[0] * cols_B for _ in range(row_end - row_start)]
    for i in range(row_start, row_end):
        for j in range(cols_B):
            for k in range(cols_A):
                local_result[i - row_start][j] += A[i][k] * B[k][j]
    conn.send(local_result)
    conn.close()

class MatrixMultiplierMultiprocessing:
    @staticmethod
    def multiply_matrices_multiprocessing(A, B, num_threads=2):
        rows_A = len(A)
        cols_A = len(A[0])
        rows_B = len(B)
        cols_B = len(B[0])

        if cols_A != rows_B:
            raise ValueError("Number of columns in A must be equal to number of rows in B.")

        result = [[0] * cols_B for _ in range(rows_A)]
        step = rows_A // num_threads
        processes = []
        pipes = []

        for i in range(num_threads):
            row_start = i * step
            row_end = (i + 1) * step if i != num_threads - 1 else rows_A
            
            # Upewnij się, że nie wychodzimy poza zakres
            if row_start >= rows_A:
                break

            parent_conn, child_conn = Pipe()
            pipes.append(parent_conn)
            process = Process(target=worker, args=(A, B, row_start, row_end, cols_B, cols_A, child_conn))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        parts = [pipe.recv() for pipe in pipes]
        current_row = 0
        for part in parts:
            for row in part:
                result[current_row] = row
                current_row += 1

        return result
