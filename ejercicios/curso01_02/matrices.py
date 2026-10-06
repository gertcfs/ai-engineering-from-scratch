import random


class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    # Imprimir con print
    def __repr__(self):
        return f"Vector({self.data})"

    # Sumar con a + b
    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    # Restar con a - b
    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    # Multiplicar con a * scalar
    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    # Producto de matrices a · b con a.dot(b)
    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    # Magnitud de un vector (longitud)
    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5

    # Normalizar un vector (convertirlo en un vector unitario)
    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.data])


class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data] # Datos de la matriz
        self.rows = len(self.data)              # Número de filas
        self.cols = len(self.data[0])           # Número de columnas
        self.shape = (self.rows, self.cols)     # Dimensión de la matriz

    # Imprimir con print
    def __repr__(self):
        col_widths = []
        for j in range(self.cols):
            width = max(len(f"{self.data[i][j]:.4f}") for i in range(self.rows))
            col_widths.append(width)
        lines = []
        for i in range(self.rows):
            row_str = "  ".join(
                f"{self.data[i][j]:{col_widths[j]}.4f}" for j in range(self.cols)
            )
            bracket_l = "|" if 0 < i < self.rows - 1 else ("/" if i == 0 else "\\")
            bracket_r = "|" if 0 < i < self.rows - 1 else ("\\" if i == 0 else "/")
            lines.append(f"  {bracket_l} {row_str} {bracket_r}")
        header = f"Matrix {self.rows}x{self.cols}:"
        return header + "\n" + "\n".join(lines)

    # Suma de matrices con A + B
    def __add__(self, other):
        if isinstance(other, Matrix):
            if other.shape == self.shape:
                return Matrix([
                    [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
                    for i in range(self.rows)
                ])
            if other.rows == 1 and other.cols == self.cols:
                return Matrix([
                    [self.data[i][j] + other.data[0][j] for j in range(self.cols)]
                    for i in range(self.rows)
                ])
            if other.cols == 1 and other.rows == self.rows:
                return Matrix([
                    [self.data[i][j] + other.data[i][0] for j in range(self.cols)]
                    for i in range(self.rows)
                ])
        raise ValueError(f"Cannot add shapes {self.shape} and {other.shape}")

    # Resta de matrices con A - B
    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    # Multiplicación de matriz por unescalar
    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    # Multiplicación de matrices elemento a elemento
    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        if self.cols != other.rows:
            raise ValueError(
                f"Cannot multiply shapes {self.shape} and {other.shape}: "
                f"inner dimensions {self.cols} != {other.rows}"
            )
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    # Multiplicación de matrices con A @ B
    def __matmul__(self, other):
        return self.matmul(other)

    # Transposición de matrices
    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    @property   # Atributo de solo lectura para la transposición, se invoca con A.T
    def T(self):
        return self.transpose()

    # Determinante de una matriz cuadrada
    def determinant(self):
        if self.rows != self.cols:
            raise ValueError("Determinante solo definido para matrices cuadradas")
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        if self.shape != (2, 2):
            raise ValueError("Este método solo funciona para matrices 2x2")
        det = self.determinant()
        if abs(det) < 1e-10:
            raise ValueError("La matriz es singular, no existe inversa")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    # Método estático, crea una matriz identidad, se invoca con Matrix.identity(n)
    # n es el tamaño de la matriz identidad (n x n)
    @staticmethod   
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])

    # Método estático, crea una matriz de ceros, se invoca con Matrix.zeros(rows, cols)
    @staticmethod
    def zeros(rows, cols):
        return Matrix([[0] * cols for _ in range(rows)])

    # Método estático, crea una matriz de unos, se invoca con Matrix.ones(rows, cols)
    @staticmethod
    def ones(rows, cols):
        return Matrix([[1] * cols for _ in range(rows)])

    # Método estático, crea una matriz con valores aleatorios, 
    # se invoca con Matrix.random(rows, cols, low=-1.0, high=1.0)  
    @staticmethod
    def random(rows, cols, low=-1.0, high=1.0):
        return Matrix([
            [random.uniform(low, high) for _ in range(cols)]
            for _ in range(rows)
        ])

# Función de activación ReLU aplicada a una matriz
def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])


def demo_basic_operations():
    print("=" * 60)
    print("OPERACIONES BÁSICAS CON MATRICES")
    print("=" * 60)

    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])

    print("\nA =")
    print(A)
    print("\nB =")
    print(B)

    print("\nA + B =")
    print(A + B)

    print("\nA - B =")
    print(A - B)

    print("\nA * 3 (scalar) =")
    print(A.scalar_multiply(3))

    print("\nA * B (elemento a elemento) =")
    print(A.element_wise_multiply(B))

    print("\nA @ B (multiplicación de matrices) =")
    print(A @ B)

    print("\nA^T =")
    print(A.T)


def demo_determinant_inverse():
    print("\n" + "=" * 60)
    print("DETERMINANTE Y MATRIZ INVERSA")
    print("=" * 60)

    A = Matrix([[4, 7], [2, 6]])
    print("\nA =")
    print(A)
    print(f"\ndet(A) = {A.determinant()}")

    A_inv = A.inverse_2x2()
    print("\nA^-1 =")
    print(A_inv)

    print("\nA @ A^-1 (debe ser la matriz identidad) =")
    print(A @ A_inv)

    I = Matrix.identity(3)
    print("\nMatriz Identidad 3x3 =")
    print(I)


def demo_broadcasting():
    print("\n" + "=" * 60)
    print("BROADCASTING")
    print("=" * 60)

    output = Matrix([[1, 2, 3], [4, 5, 6]])
    bias = Matrix([[10, 20, 30]])

    print("\nOutput =")
    print(output)
    print("\nBias =")
    print(bias)
    print("\nOutput + Bias (broadcast) =")
    print(output + bias)

# - Capa de entrada: no se cuenta como capa de la red neuronal porque en ese punto
#   no se produce computación
# - Capa oculta (hidden layer): cualquier capa situada entre la capa de entrada y 
#   la de salida 
# - Capa de salida: donde queda el resultado de la red neuronal
# Esta red neuronal tiene una entrada y 2 capas, la oculta y la de salida
# En todas las capas (oculta y salida) se aplica el mismo proceso:
# - se multiplican por los pesos, suma el bias y aplica una función de activación
def demo_neural_network_layer():
    print("\n" + "=" * 60)
    print("PASO ADELANTE DE UNA CAPA DE RED NEURONAL")
    print("=" * 60)

    random.seed(42)     # Fijar la semilla para reproducibilidad

    input_size = 3      # Tamaño de la entrada  (neuronas = 3)
    hidden_size = 4     # Tamaño de la capa oculta  (neuronas = 4)
    output_size = 2     # Tamaño de la capa de salida  (neuronas = 2)

    x = Matrix([[0.5], [0.8], [0.2]])       # Entrada de la red neuronal (3x1)
    W1 = Matrix.random(hidden_size, input_size)  # Pesos de la primera capa (4x3)
    b1 = Matrix([[0.0]] * hidden_size)      # Sesgos de la primera capa (4x1)
    W2 = Matrix.random(output_size, hidden_size) # Pesos de la segunda capa (2x4)
    b2 = Matrix([[0.0]] * output_size)      # Sesgos de la segunda capa (2x1)

    print(f"\nInput x: {x.shape}")
    print(f"W1: {W1.shape}")
    print(f"W2: {W2.shape}")

    z1 = (W1 @ x) + b1      # Matriz entrada por pesos + sesgo
    h1 = relu_matrix(z1)    # a lo anterior le pasamos la función de activación ReLU
    print(f"\nPre-activacion de la capa oculta z1: {z1.shape}")
    print(z1)
    print(f"\nCapa oculta post-ReLU h1: {h1.shape}")
    print(h1)

    z2 = (W2 @ h1) + b2     # Capa de salida: matriz capa oculta por pesos + sesgo
    print(f"\nOutput z2: {z2.shape}")
    print(z2)
    print("\nEste es un pase hacia adelante completo de una red neuronal de 2 capas.")
    #               W1.   input x   b1.      z1.              h1
    print("Capa 1: (4x3) @ (3x1) + (4x1) -> (4x1) -> ReLU -> (4x1)")
    #               W2.      h1.    b2.     output z2
    print("Capa 2: (2x4) @ (4x1) + (2x1) -> (2x1)")


def demo_vectors():
    print("\n" + "=" * 60)
    print("OPERACIONES CON VECTORES")
    print("=" * 60)

    v = Vector([3, 4])
    w = Vector([1, 2])

    print(f"\nv = {v}")
    print(f"w = {w}")
    print(f"v + w = {v + w}")
    print(f"v - w = {v - w}")
    print(f"v * 2 = {v * 2}")
    print(f"v . w = {v.dot(w)}")
    print(f"|v| = {v.magnitude()}")
    print(f"v normalizado = {v.normalize()}")
    print(f"|v normalizado| = {v.normalize().magnitude()}")


def demo_weight_matrix_intuition():
    print("\n" + "=" * 60)
    print("COMPRENDER LA MATRIZ DE PESOS W")
    print("=" * 60)

    print("\nUna matriz de pesos transforma las features de entrada en features de salida.")
    print("Cada fila extrae un patrón de la entrada.\n")

    W = Matrix([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ])
    x = Matrix([[0.8], [0.6], [0.1]])

    print("Pesos de la matriz W (3 detectors, 3 inputs):")
    print(W)
    print("\nInput x:")
    print(x)
    print("\nW @ x =")
    result = W @ x
    print(result)
    print("\nFila 0 de W = [1, 0, 0]: copia de la entrada la feature 0")
    print("Fila 1 de W = [0, 1, 0]: copia de la entrada la feature 1")
    print("Fila 2 de W = [0.5, 0.5, 0]: media de features 0 y 1")


if __name__ == "__main__":
    demo_vectors()
    demo_basic_operations()
    demo_determinant_inverse()
    demo_broadcasting()
    demo_weight_matrix_intuition()
    demo_neural_network_layer()
