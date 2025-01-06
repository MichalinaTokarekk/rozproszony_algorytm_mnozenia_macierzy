class MatrixMultiplier:
    @staticmethod
    def multiply_matrices_sequential(A, B):
        rows_A = len(A)
        cols_A = len(A[0])
        rows_B = len(B)
        cols_B = len(B[0])

        if cols_A != rows_B:
            raise ValueError("Number of columns in A must be equal to number of rows in B.")

        result = [[0] * cols_B for _ in range(rows_A)]

        for i in range(rows_A):  # Iteracja po wierszach macierzy A
            for j in range(cols_B):  # Iteracja po kolumnach macierzy B
                for k in range(cols_A):  # Iteracja po wspólnym wymiarze A i B
                    result[i][j] += A[i][k] * B[k][j]  # Obliczanie wartości w wynikowej macierzy

        return result
