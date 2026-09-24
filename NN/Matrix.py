class Matrix:
    def __init__(self, data):
        """
        WIP
         Matrix class without
        using list of lists
        fixed missing functions
        """
        if not all(isinstance(row, list) for row in data):
            raise ValueError("Matrix must be a list of lists.")
        if not data:
            self.rows = 0
            self.cols = 0
            self.data = []
            return
            # Note for self: Len means length
        self.rows = len(data)
        self.cols = len(data[0])
        if not all(len(row) == self.cols for row in data):
            raise ValueError("All rows in the matrix must have the same number of columns.")

        self.data = [list(row) for row in data]  # mutation issue fixer

    def __str__(self):
        """
        Returns a string representation of the matrix, so I don't have to make a "Print" function
        """
        return "\n".join([" ".join(map(str, row)) for row in self.data])

    def addition(self, other):
        """
        Performs matrix addition.
        """
        if not isinstance(other, Matrix) or self.rows != other.rows or self.cols != other.cols:
            raise ValueError("must have the same dimensions")

        result_data = []
        for r in range(self.rows):
            new_row = [self.data[r][c] + other.data[r][c] for c in range(self.cols)]
            result_data.append(new_row)
        return Matrix(result_data)

    def to_scalar(self):
        if self.rows == 1 and self.cols == 1:
            return self.data[0][0]
        raise ValueError("Matrix is not 1x1")

    def multiplication(self, other):
        """
         matrix multiplication
        """
        if isinstance(other, (int, float)):  # Scalar multiplication
            result_data = []
            for r in range(self.rows):
                new_row = [self.data[r][c] * other for c in range(self.cols)]
                result_data.append(new_row)
            return Matrix(result_data)

        elif isinstance(other, Matrix):  # Matrix multiplication
            if self.cols != other.rows:
                raise ValueError(
                    "Number of columns in the first matrix must be the same as the number of rows in the second")

            result_data = [[0 for _ in range(other.cols)] for _ in range(self.rows)]
            for i in range(self.rows):
                for j in range(other.cols):
                    for k in range(self.cols):
                        result_data[i][j] += self.data[i][k] * other.data[k][j]
            return Matrix(result_data)
        else:
            raise TypeError("Type Error")

    #  Operator overloads
    def __add__(self, other):
        if isinstance(other, Matrix):
            return self.addition(other)
        elif isinstance(other, (int, float)):  # scalar addition
            result_data = [[self.data[r][c] + other for c in range(self.cols)] for r in range(self.rows)]
            return Matrix(result_data)
        else:
            raise TypeError("Unsupported type for addition")

    def __sub__(self, other):
        if isinstance(other, Matrix):
            result_data = [[self.data[r][c] - other.data[r][c] for c in range(self.cols)] for r in range(self.rows)]
            return Matrix(result_data)
        elif isinstance(other, (int, float)):  # scalar subtraction
            result_data = [[self.data[r][c] - other for c in range(self.cols)] for r in range(self.rows)]
            return Matrix(result_data)
        else:
            raise TypeError("Unsupported type for subtraction")

    def __mul__(self, other):
        return self.multiplication(other)  # delegate to multiplication()

    def __rmul__(self, other):
        return self.multiplication(other)  # scalar * Matrix support


#matrix_a = Matrix([[1, 2], [3, 4]])

#print(matrix_a)