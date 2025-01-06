class MatrixGenerator:
    @staticmethod
    def generate_random_matrix(rows, cols, min_val=0, max_val=10):
        from random import randint

        return [[randint(min_val, max_val) for _ in range(cols)] for _ in range(rows)]

# Example usage
if __name__ == "__main__":
    generator = MatrixGenerator()
    matrix = generator.generate_random_matrix(3, 3)
    print("Generated Matrix:")
    for row in matrix:
        print(row)
