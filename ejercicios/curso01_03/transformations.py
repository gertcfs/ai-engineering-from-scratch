import math

# Devuelve una matriz 2D que aplica una rotación de angulo theta
# Una rotación bidimensional mediante un ángulo theta mantiene intactas las 
# distancias y los ángulos. Mueve cada punto a lo largo de un arco circular.
def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

# Devuelve una matriz 3D que aplica una rotación de angulo theta alrededor del eje z
def rotation_3d_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]

# Devuelve una matriz 3D que aplica una rotación de angulo theta alrededor del eje x
def rotation_3d_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]

# Devuelve una matriz 3D que aplica una rotación de angulo theta alrededor del eje y
def rotation_3d_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]

# Devuelve una matriz 2D que aplica un escalado sx, sy
# El escalado estira o comprime a lo largo de cada eje de forma independiente.
def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]

# Devuelve una matriz 2D que aplica un cizallamiento kx, ky
# El corte inclina un eje mientras mantiene el otro fijo. 
# Convierte los rectángulos en paralelogramos.
def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]

# Devuelve una matriz 2D que aplica una reflexión a través del eje x
def reflection_x():
    return [[1, 0], [0, -1]]

# Devuelve una matriz 2D que aplica una reflexión a través del eje y
def reflection_y():
    return [[-1, 0], [0, 1]]

# Multiplicar una matriz por un vector
def mat_vec_mul(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

# Multiplicar dos matrices
def mat_mul(a, b):
    rows_a, cols_b = len(a), len(b[0])
    cols_a = len(a[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]

# Calcular el determinante de una matriz 2x2
def det_2x2(m):
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]

# Calcular el determinante de una matriz 3x3
def det_3x3(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )

# Calcular Valores propios / Autovalores de una matriz 2x2
def eigenvalues_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        real = trace / 2
        imag = (-discriminant) ** 0.5 / 2
        return (complex(real, imag), complex(real, -imag))
    sqrt_disc = discriminant ** 0.5
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)

# Calcular Vector propio / Autovector de una matriz 2x2 y su autovalor
def eigenvector_2x2(matrix, eigenvalue):
    a, b = matrix[0]
    c, d = matrix[1]
    if abs(b) > 1e-10:
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        if abs(a - eigenvalue) < 1e-10:
            v = [1, 0]
        else:
            v = [0, 1]
    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return [v[0] / mag, v[1] / mag]

# Redondea a 4 decimales o a otros si se le pasa el parametro
def fmt(v, decimals=4):
    if isinstance(v, list):
        return [round(x, decimals) for x in v]
    return round(v, decimals)


def demo_basic_transformations():
    print("=" * 60)
    print("TRANSFORMACIONES BASICAS")
    print("=" * 60)

    point = [1.0, 0.0]          # un punto representado como un vector de 2 numeros
    theta = math.pi / 4         # 2*pi=360 => pi=180 => pi/2=90 => pi/4 = 45 grados

    # La ROTACION mueve cada punto a lo largo de un arco circular.
    # Primero obtenemos la matriz de rotacion 2d correspondiente al angulo theta
    # Luego multiplicamos la matriz de rotación por el punto (1,0)
    rotated = mat_vec_mul(rotation_2d(theta), point)
    print(f"\nRotación del punto (1,0) en 45 grados: {fmt(rotated)}")

    # El ESCALADO estira o comprime a lo largo de cada eje de forma independiente.
    # Primero obtenemos la matriz de escalado 2d correspondiente a sx=2 sy=3
    # Luego multiplicamos la matriz de escalado por el punto(1,1)
    scaled = mat_vec_mul(scaling_2d(2, 3), [1.0, 1.0])
    print(f"Escalando el punto (1,1) en (2,3): {fmt(scaled)}")

    # El CIZALLAMIENTO inclina un eje mientras mantiene el otro fijo. 
    # Convierte los rectángulos en paralelogramos.
    # Matrices de corte:
    #   Shx = [[1, k], [0, 1]]desplaza x en k * y
    #   Shy = [[1, 0], [k, 1]]desplaza y en k * x
    # Primero obtenemos la matriz de cizallamiento 2d con kx=1 y ky=0
    # Luego multiplicamos la matriz de cizallamiento por el punto (1,1)
    sheared = mat_vec_mul(shearing_2d(1, 0), [1.0, 1.0])
    print(f"Cizallamiento del punto (1,1) con kx=1: {fmt(sheared)}")

    # La REFLEXION refleja puntos a través de un eje o línea.
    # Matrices de reflexión:
    #   Reflejar a través del eje y:[[-1, 0], [0, 1]]
    #   Reflejar a través del eje x:[[1, 0], [0, -1]]
    # Primero obtenemos la matriz de reflexion sobre el eje y
    # Luego multiplicamos la matriz de reflexion por el punto (2,1)
    reflected = mat_vec_mul(reflection_y(), [2.0, 1.0])
    print(f"Reflexión del punto (2,1) sobre el eje y: {fmt(reflected)}")
    # Reflexión sobre el eje y
    reflected_x = mat_vec_mul(reflection_x(), [2.0, 1.0])
    print(f"Reflexión del punto (2,1) sobre el eje x: {fmt(reflected_x)}")

def demo_unit_square():
    print("\n" + "=" * 60)
    print("TRANSFORMACION SOBRE UN CUADRADO UNITARIO")
    print("=" * 60)

    square = [[0, 0], [1, 0], [1, 1], [0, 1]]   # Vector con las esquinas del cuadrado
    labels = ["origin", "right", "top-right", "top"]

    print("\nCuadrado original:")
    for label, pt in zip(labels, square):
        print(f"  {label}: {pt}")

    transforms = [
        ("Rotar 45 grados", rotation_2d(math.pi / 4)),
        ("Escalar (2, 0.5)", scaling_2d(2, 0.5)),
        ("Cizallar kx=0.5", shearing_2d(0.5, 0)),
        ("Reflejar en eje y", reflection_y()),
    ]

    rotado = []
    for name, matrix in transforms:
        print(f"\n{name}:")
        for label, pt in zip(labels, square):
            result = mat_vec_mul(matrix, pt)
            print(f"  {label}: {pt} -> {fmt(result)}")
            if name == "Rotar 45 grados":
                rotado.append(fmt(result))
        print(f"  det = {fmt(det_2x2(matrix))}")

    for i, j in zip([0,1,2,3], [1,2,3,0]):
        print(f"Distancias {square[i]} a {square[j]} = "
              f"{math.dist(square[i], square[j])}; "
              f"{rotado[i]} a {rotado[j]} = "
              f"{round(math.dist(rotado[i], rotado[j]),4)}")


def demo_composition():
    print("\n" + "=" * 60)
    print("COMPOSICION DE TRANSFORMACIONES")
    print("=" * 60)

    R = rotation_2d(math.pi / 2)        # Rotacion
    S = scaling_2d(2, 0.5)              # Escalado

    # Multiplicar matrices de transformación para encadenar operaciones. 
    # El orden importa: B @ A significa aplicar primero A y luego B.
    rotate_then_scale = mat_mul(S, R)   # Rotar y después escalar S @ R (al reves)
    scale_then_rotate = mat_mul(R, S)   # Escalar y después rotar R @ S (al reves)

    point = [1.0, 0.0]      # Lista (vector) de 2 numeros que representa un punto

    # Multiplica la matriz de transformación por el vector
    result1 = mat_vec_mul(rotate_then_scale, point)
    result2 = mat_vec_mul(scale_then_rotate, point)

    print(f"\nPunto: {point}")
    print(f"Rota 90 despues escala (2, 0.5): {fmt(result1)}")
    print(f"Escala (2, 0.5) despues rota 90: {fmt(result2)}")
    print("El orden importa.")

    print(f"\ndet(R) = {fmt(det_2x2(R))}")
    print(f"det(S) = {fmt(det_2x2(S))}")
    print(f"det(S @ R) = {fmt(det_2x2(rotate_then_scale))}")
    print(f"det(S) * det(R) = {fmt(det_2x2(S) * det_2x2(R))}")
    print("Determinante de la composicion = producto de determinantes.")


def demo_3d_rotations():
    print("\n" + "=" * 60)
    print("ROTACIONES 3D")
    print("=" * 60)

    point = [1.0, 0.0, 0.0]     # Punto 3D: Lista (vector) con 3 numeros
    theta = math.pi / 2         # Angulo = 180/2 = 90 grados

    # En cada eje x,y,z aplicamos una rotación 3D de 90 grados al punto (1,0,0)
    rz = mat_vec_mul(rotation_3d_z(theta), point)
    rx = mat_vec_mul(rotation_3d_x(theta), point)
    ry = mat_vec_mul(rotation_3d_y(theta), point)

    print(f"\nPunto: {point}")
    print(f"Rotacion 90 sobre z: {fmt(rz)}")
    print(f"Rotacion 90 sobre x: {fmt(rx)}")
    print(f"Rotacion 90 sobre y: {fmt(ry)}")

    print(f"\ndet(Rz) = {fmt(det_3x3(rotation_3d_z(theta)))}")
    print(f"det(Rx) = {fmt(det_3x3(rotation_3d_x(theta)))}")
    print(f"det(Ry) = {fmt(det_3x3(rotation_3d_y(theta)))}")
    print("Todos los determinantes de la rotacion = 1 (el volumen no cambia).")

def demo_eigenvalues_from_scratch():
    print("\n" + "=" * 60)
    print("VALORES Y VECTORES PROPIOS (DESDE CERO, 2x2)")
    print("=" * 60)

    matrices = [
        ("Simétrica", [[2, 1], [1, 2]]),
        ("Triangular superior", [[3, 1], [0, 2]]),
        ("Escalado", [[3, 0], [0, 5]]),
        ("Rotación 90", [[0, -1], [1, 0]]),
    ]
    # Si multiplicamos la matriz @ autovector vemos que se produce un escalado
    # de dicho vector, nunca se produce un giro
    # - El autovector no cambia de dirección 
    # - El factor de escalado es el autovalor
    # Para demostrar esto cogemos una matriz (ejemplo: A = [[2, 1], [1, 2]])
    # Extraemos los autovalores (eigenvalues_2x2 nos da : 3 y 1)
    # Por cada autovalor:
    # - recupera su autovector (eigenvector_2x2 nos da: 
    #    autovectores: para 3 => [1,1] ; para 1 => [1, -1])
    # - multiplicamos la matriz @ autovector y nos da:
    #    A @ [1,1] = [3, 3]  y   A @ [1,-1] = [1, -1] => misma dirección, escalado x3
    # - comprobamos que el escalado nos da los mismos resultados:
    #    autovalor*autovector => 3 * [1, 1] y 1 * [1, -1] => el autovalor es el factor de escalado
    # Se demuestra que los autovectores no cambian de dirección y su factor de
    # escalado es su autovalor
    # La 'Rotación 90' se ve que no tiene autovalores precisamente por ser una rotación
    for name, A in matrices:
        vals = eigenvalues_2x2(A)   # Extrae autovalores de cada matriz de 'matrices'
        print(f"\n{name}: {A}")     # Imprime la matriz
        print(f"  Autovalores: {vals[0]}, {vals[1]}")   # Imprime los autovalores
        # isinstance verifica que cada autovalor v de vals sea int o float
        # all() devuelve true si todos los isinstace dan true, es decir que 
        # el if solo pasa si todos los autovalores son int o float
        if all(isinstance(v, (int, float)) for v in vals):
            for val in vals:        # recorre todos los autovalores y por cada uno:
                vec = eigenvector_2x2(A, val)   # recuperamos el autovector
                result = mat_vec_mul(A, vec)    # multiplica la matriz @ autovector
                scaled = [val * vec[0], val * vec[1]]   # [autovalor*autovector[0], autovalor*autovector[1]]
                print(f"  lambda={fmt(val)}, v={fmt(vec)}") # autovalor, autovector
                print(f"    A @ v = {fmt(result)}") # Matriz @ autovector
                print(f"    l * v = {fmt(scaled)}") # vector escalado
        else:
            print("  Autovalores complejos: rotación pura, sin autovectores reales.")

# La diagonalización de matrices transforma matrices complejas en su forma más simple,
# que es una matriz diagonal. Estas matrices tienen multiples ventajas computacionales
# Para encontrar la matriz diagonal hay que conseguir sus autovalores
# Con los autovalores podemos calcular los autovectores y con ellos la matriz V
# El tema es que la matriz original A = V @ D @ V_inv, por lo que podemos trabajar con 
# matrices Diagonalizadas y luego reconstruir la matriz original
def demo_eigendecomposition():
    print("\n" + "=" * 60)
    print("DESCOMPOSICION EN AUTOVALORES (2x2, DESDE 0)")
    print("=" * 60)

    # Tenemos una matriz 2D
    # Obtenemos sus autovalores y, por cada autovalor, sus autovectores
    A = [[3, 1], [0, 2]]            # matrix 2D
    vals = eigenvalues_2x2(A)       # obtener los autovalorees de la matriz A
    v0 = eigenvector_2x2(A, vals[0])    # obtiene autovector v0 de la matriz A
    v1 = eigenvector_2x2(A, vals[1])    # obtiene autovector v1 de la matriz A

    # Matriz V compuesta con los valores de los autovectores v0 y v1
    V = [[v0[0], v1[0]], [v0[1], v1[1]]]

    # Matriz D (es la matriz diagonalizada)
    # Se compone con los autovalores (vals) en la diagonal y el resto con 0's
    D = [[vals[0], 0], [0, vals[1]]]

    # Matriz Inversa de V
    det_v = det_2x2(V)                      # determinante de V
    V_inv = [                               # matriz inversa de V
        [V[1][1] / det_v, -V[0][1] / det_v],
        [-V[1][0] / det_v, V[0][0] / det_v],
    ]

    # Reconstruccion: (V @ D) @ V^-1 = Matriz original
    reconstructed = mat_mul(mat_mul(V, D), V_inv)

    print(f"\nA = {A}")
    print(f"Autovalores: {fmt(vals[0])}, {fmt(vals[1])}")
    print("V (autovectores como columnas):")
    for row in V:
        print(f"  {fmt(row)}")
    print("D (autovalores sobre diagonal):")
    for row in D:
        print(f"  {fmt(row)}")
    print("Reconstrucción de A = V @ D @ V^-1:")
    for row in reconstructed:
        print(f"  {fmt(row)}")

# El determinante de la matriz nos dice como aumenta el espacio/volumen
def demo_determinant_meaning():
    print("\n" + "=" * 60)
    print("DETERMINANTE COMO FACTOR DE ESCALADO DE VOLUMEN")
    print("=" * 60)

    cases = [
        ("Rotacion 45 grados", rotation_2d(math.pi / 4)),
        ("Escalado (2, 3)", scaling_2d(2, 3)),
        ("Cizallamiento kx=1", shearing_2d(1, 0)),
        ("Reflexion eje y", reflection_y()),
        ("Singular [[1,2],[2,4]]", [[1, 2], [2, 4]]),
    ]

    print()
    for name, m in cases:
        d = det_2x2(m)
        if d == 0:
            meaning = "espacio colapsa, irreversible"
        elif d < 0:
            meaning = "orientacion invertida"
        elif abs(d - 1.0) < 1e-10:
            meaning = "area preserved"
        else:
            meaning = f"area escalado por {abs(d):.1f}x"
        print(f"det({name}) = {fmt(d):>8}  ({meaning})")


def demo_numpy_comparison():
    print("\n" + "=" * 60)
    print("COMPARACION CON NUMPY")
    print("=" * 60)

    try:
        import numpy as np
    except ImportError:
        print("\nNumPy not installed. Skipping.")
        return

    theta = math.pi / 4
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])

    point = np.array([1.0, 0.0])
    print(f"\nRotate (1,0) by 45 deg: {R @ point}") # Rotación de 45 s/punto

    A = np.array([[2, 1], [1, 2]], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(A)    # autovalores y autovectores
    print(f"\nA = {A.tolist()}")
    print(f"Eigenvalues (numpy): {eigenvalues}")
    print(f"Eigenvectors (numpy, columns):\n{eigenvectors}")

    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        print(f"  A @ v{i} = {A @ v}, lambda * v{i} = {lam * v}")

    B = np.array([[3, 1], [0, 2]], dtype=float)
    vals, vecs = np.linalg.eig(B)       # autovalores y autovectores
    D = np.diag(vals)       # D = matriz diagonal a partir de los autovalores
    V = vecs                # V = matriz de autovectores
    reconstructed = V @ D @ np.linalg.inv(V)    # reconstrucción con la inversa de V
    print(f"\nEigendecomposition of {B.tolist()}:")
    print(f"  Reconstructed: {reconstructed.tolist()}")

    Rz = np.array(rotation_3d_z(math.pi / 2))  # matriz rotación 3D angulo 90 grados
    point_3d = np.array([1.0, 0.0, 0.0])       # punto 3d (1,0,0)
    print(f"\n3D rotate (1,0,0) 90 deg around z: {np.round(Rz @ point_3d, 4)}")

    cov = np.array([[2.0, 1.0], [1.0, 3.0]])
    vals, vecs = np.linalg.eig(cov)     # autovalores y autovectores
    print(f"\nCovariance matrix: {cov.tolist()}")
    print(f"Principal components (eigenvectors): columns of\n{vecs}")
    print(f"Variance along each (eigenvalues): {vals}")
    print("PCA picks the eigenvectors with the largest eigenvalues.")


if __name__ == "__main__":
    demo_basic_transformations()
    demo_unit_square()
    demo_composition()
    demo_3d_rotations()
    demo_eigenvalues_from_scratch()
    demo_eigendecomposition()
    demo_determinant_meaning()
    demo_numpy_comparison()
