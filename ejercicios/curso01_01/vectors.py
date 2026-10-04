class Vector:
    def __init__(self, components):
        self.components = list(components) # Esto es la definicion de un campo de la clase Vector
        self.dim = len(self.components) # Otro campo que es la longitud de la lista de componentes

    # Suma de clases, es estandar en Python definir el metodo __add__ para poder usar el operador 
    # + entre objetos de la clase Vector 
    # self es el objeto actual, other es el otro objeto con el que se esta sumando
    # Devuelve un nuevo objeto Vector con la suma de los componentes
    # Explicacion de [a + b for a, b in zip(self.components, other.components)]:
    # - zip(self.components, other.components) devuelve un iterador de tuplas, donde cada tupla 
    # contiene un elemento de self.components y un elemento de other.components
    # - [a + b for a, b in zip(self.components, other.components)] es una lista por comprension 
    # que recorre cada tupla (a, b) y devuelve a + b, es decir, la suma de los componentes 
    # correspondientes
    # Por ejemplo, si self.components = [1, 2, 3] y other.components = [4, 5, 6], entonces 
    # zip(self.components, other.components) devuelve [(1, 4), (2, 5), (3, 6)], y la lista 
    # por comprension devuelve [1 + 4, 2 + 5, 3 + 6] = [5, 7, 9]
    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    # Restar dos vectores es similar a sumar, pero cambiando el signo de la segunda componente
    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    # Multiplicar un vector por un escalar es simplemente multiplicar cada componente por ese escalar
    # Aqui self.components devuelve cada elemento del vector y le llamamos x
    # Luego multiplicamos x por el escalar y vamos componiendo el nuevo vector
    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.components])

    # dot es el producto escalar entre dos vectores
    # zip(self.components, other.components) genera tuplas de 2 componentes a los que llamamos a y b
    # luego hacemos a*b y sumamos todos los resultados con sum() que suma los elementos de un iterable
    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    # Magnitud de un vector es la raiz cuadrada de la suma de los cuadrados de sus componentes
    # Asignamos x a cada componente del vector y hacemos x**2 para elevarlo al cuadrado
    # Luego sumamos todos los cuadrados y le aplicamos la raiz cuadrada con ** 0.5
    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    # Normalizar un vector es dividir cada componente por la magnitud del vector
    # Primero calculamos la magnitud del vector con self.magnitude()
    # Luego hacemos una lista por comprension donde dividimos cada componente x por la magnitud
    # Finalmente devolvemos un nuevo objeto Vector con los componentes normalizados
    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    # Entran dos vectores, se calcula el producto escalar y se divide entre el producto de las magnitudes
    # Esto nos da el coseno del angulo entre los dos vectores (similitud coseno). Significado:
    # Valor 1: Los vectores apuntan hacia el mismo lado; el significado o contenido es idéntico
    # Valor 0: Los vectores son perpendiculares, no tienen relación entre sí; no comparten información
    # Valor -1: Los vectores apuntan exactamente en direcciones opuestas; son contrarios
    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    # Calculamos el ángulo entre dos vectores en grados
    # Calculamos cos_theta = coseno entre los dos vectores con la funcion cosine_similarity()
    # Luego corregimos cos_theta para que esté entre -1 y 1, ya que acos() solo acepta valores en ese rango
    # Finalmente con acos() obtenemos el ángulo en radianes y lo convertimos a grados con math.degrees()
    def angle_between(self, other):
        import math
        cos_theta = self.cosine_similarity(other)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.degrees(math.acos(cos_theta))

    # Proyectar un vector sobre otro es calcular la componente del 1er vector en la dirección del 2º
    # Primero calculamos el escalar = self.dot(other) / other.dot(other)
    # Luego devolvemos un nuevo vector con cada componente de other multiplicada por el escalar
    def project_onto(self, other):
        scalar = self.dot(other) / other.dot(other)
        return Vector([scalar * x for x in other.components])

    # Representación en cadena del vector, para que al imprimirlo se vea bonito
    def __repr__(self):
        return f"Vector({self.components})"

# Entrada: una lista de vectores
# Salida: True si los vectores son linealmente independientes, False si no lo son
def is_independent(vectors):
    n = len(vectors)
    if n == 0:
        return True
    dim = vectors[0].dim  # coger la dimensión del primer vector
    # Extrae los componentes de cada vector y los pone en una lista de listas
    rows = [v.components[:] for v in vectors] # [:] slicing (copia de la lista)
    rank = 0
    for col in range(dim): 
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n    # Si el rango es igual al número de vectores devuelve true

# Entrada: una lista de vectores
# Salida: una lista de vectores ortonormales (ortogonales y de magnitud 1) 
# que forman una base del espacio generado por los vectores de entrada
def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = w.project_onto(u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal

# Entrada: una lista de filas (listas de números) que representan una matriz
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows] # por cada fila de la matriz, hacemos una copia de la lista para no modificar la original
        self.shape = (len(self.rows), len(self.rows[0])) # shape = dimension (filas, columnas)

    # Definimos el operador @ para multiplicar matriz por vector o matriz por matriz 
    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    # Transponer una matriz es intercambiar filas por columnas
    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    # Calcular el rango de una matriz es contar el número de filas linealmente independientes
    # Usamos el mismo algoritmo que en is_independent, pero devolvemos el rango en
    def rank(self):
        rows = [row[:] for row in self.rows]
        m, n = self.shape
        r = 0
        for col in range(n):
            pivot = None
            for row in range(r, m):
                if abs(rows[row][col]) > 1e-10:
                    pivot = row
                    break
            if pivot is None:
                continue
            rows[r], rows[pivot] = rows[pivot], rows[r]
            scale = rows[r][col]
            rows[r] = [x / scale for x in rows[r]]
            for row in range(m):
                if row != r and abs(rows[row][col]) > 1e-10:
                    factor = rows[row][col]
                    rows[row] = [rows[row][j] - factor * rows[r][j] for j in range(n)]
            r += 1
        return r

    def __repr__(self):
        return f"Matrix({self.rows})"


if __name__ == "__main__":
    print("=== Vectores ===")
    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}  (suma de vectores)")
    print(f"a - b = {a - b}  (resta de vectores)")
    print(f"a * 3 = {a * 3}  (producto escalar)")
    print(f"a · b = {a.dot(b)}  (producto de vectores = · = dot)")
    print(f"|a| = {a.magnitude():.4f}  (magnitud)")
    print(f"â (normalized) = {a.normalize()}  (vector normalizado)")
    print(f"cosine_similarity(a, b) = {a.cosine_similarity(b):.4f}")

    print("\n=== Matrices ===")
    rotation_90 = Matrix([[0, -1], [1, 0]])
    point = Vector([3, 1])
    rotated = rotation_90 @ point
    print(f"Matriz de rotación de 90° {rotation_90}")
    print(f"Punto original: {point}")
    print("Hacemos operación @ => Matriz x Vector =>")
    print(f"Rotación de {point} en 90° → {rotated}")

    print("\n=== Ángulo Entre Vectores ===")
    v1 = Vector([1, 0])
    v2 = Vector([0, 1])
    v3 = Vector([1, 1])
    print(f"Ángulo entre {v1} y {v2}: {v1.angle_between(v2):.1f} grados")
    print(f"Ángulo entre {v1} y {v3}: {v1.angle_between(v3):.1f} grados")
    print(f"Ángulo entre {v1} y {v1}: {v1.angle_between(v1):.1f} grados")

    print("\n=== Proyección ===")
    a = Vector([3, 4])
    b = Vector([1, 0])
    proj = a.project_onto(b)
    residual = a - proj
    print(f"a = {a}")
    print(f"b = {b}")
    print("Proyección de a sobre b. La proyeccion elimina la componente y")
    print(f"proj_b(a) = {proj}")
    print("El residuo es perpendicular a b, por lo que su producto escalar será 0")
    print(f"residual = {residual}")
    print(f"residual dot b = {residual.dot(b):.6f}")

    print("\n=== Independencia Lineal ===")
    e1 = Vector([1, 0, 0])
    e2 = Vector([0, 1, 0])
    e3 = Vector([0, 0, 1])
    print("Vectores estándar e1, e2, e3:")
    print(f"e1 = {e1}")
    print(f"e2 = {e2}")
    print(f"e3 = {e3}")
    print("Vector dependiente 2*e1 + e2:")
    dep = Vector([2, 1, 0])
    print(f"dep = {dep}")
    print(f"{{e1, e2, e3}} independencia: {is_independent([e1, e2, e3])}")
    print(f"{{e1, e2, 2*e1+e2}} independencia: {is_independent([e1, e2, dep])}")

    print("\n=== Ortogonalización de Gram-Schmidt ===")
    u1 = Vector([1, 1, 0])
    u2 = Vector([1, 0, 1])
    u3 = Vector([0, 1, 1])
    print("Vectores originales:")
    print(f"u1 = {u1}")
    print(f"u2 = {u2}")
    print(f"u3 = {u3}")
    basis = gram_schmidt([u1, u2, u3])
    print("Base ortonormal conseguida con Gram-Schmidt a partir de los vectores originales:")
    for i, vec in enumerate(basis):
        print(f"u{i+1} = {vec}")
    print(" Producto escalar = · = dot , debe ser 0 para vectores ortogonales (perpendiculares)")
    print(f"u1 dot u2 = {basis[0].dot(basis[1]):.6f}")
    print(f"u1 dot u3 = {basis[0].dot(basis[2]):.6f}")
    print(f"u2 dot u3 = {basis[1].dot(basis[2]):.6f}")
    print(" Magnitud = |u| = magnitude , debe ser 1 para vectores normalizados")
    for i, vec in enumerate(basis):
        print(f"|u{i+1}| = {vec.magnitude():.6f}")

    print("\n=== Rango de Matrices ===")
    full_rank = Matrix([[1, 0], [0, 1]])
    rank_deficient = Matrix([[1, 2], [2, 4]])
    rectangular = Matrix([[1, 0, 0], [0, 1, 0]])
    print(f"Matriz identidad de 2x2, rango: {full_rank.rank()}")
    print(f"Matriz [[1,2],[2,4]] rango: {rank_deficient.rank()}")
    print(f"Matriz [[1,0,0],[0,1,0]] rango: {rectangular.rank()}")

    print("\n=== Capa de Red Neuronal (Matriz x Vector) ===")
    import random
    random.seed(42)
    weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
    input_vec = Vector([1.0, 0.5, -0.3])
    output = weights @ input_vec
    print(f"Entrada vector (3D):  {input_vec}")
    print(f"Pesos matriz (2x3): {weights}")
    print(f"Salida (2D) (entrada @ pesos): {output}  (@=matriz x matriz/vector)")
    print("^ Esto es literalmente lo que hace una capa de red neuronal.")
