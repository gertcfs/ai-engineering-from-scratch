import matrices as m

# Implementar la inversa de 3x3. 
# Extender la clase Matrix para calcular inversas para matrices de 3x3 
# usando el método de la matriz adjunta. 
# - La adjunta (o matriz adjugada) de una matriz 3x3 se calcula obteniendo 
#   la transpuesta de su matriz de cofactores.
# Probarlo contra NumPy np.linalg.inv

class Matrix_extension(m.Matrix):
    def __init__(self, data):
        super().__init__(data)
    def obtener_submatriz_2x2(self, fila_eliminar, col_eliminar):
        """Devuelve la submatriz resultante de eliminar una fila y una columna"""
        return [
            [self.data[f][c] for c in range(3) if c != col_eliminar]
            for f in range(3) if f != fila_eliminar
        ]
    def inversa_3x3(self):
        if self.shape != (3, 3):
            raise ValueError("Este método solo funciona para matrices 3x3")
        det = self.determinant()
        if det == 0:
            raise ValueError("Matriz sin inversa, determinante 0")
        if abs(det) < 1e-10:
            raise ValueError(f"La matriz es singular, no existe inversa, det={det}")
        # 1. Calcular la matriz de cofactores
        cofactores = []
        for f in range(3):
            fila_cofactores = []
            for c in range(3):
                submatriz = self.obtener_submatriz_2x2(f, c)
                m2 = m.Matrix(submatriz)
                # El signo se alterna según la posición (-1)^(f+c)
                signo = 1 if (f + c) % 2 == 0 else -1
                valor_cofactor = signo * m2.determinant()   # determinante_2x2(submatriz)
                fila_cofactores.append(valor_cofactor)
            cofactores.append(fila_cofactores)
        # 2. Transponer la matriz de cofactores para obtener la Adjunta
        m3 = m.Matrix(cofactores)
        adjunta = m3.T
        # 3. Dividir la adjunta por el determinante para obtener la Inversa
        inversa = m.Matrix([[adjunta.data[f][c] / det for c in range(3)] for f in range(3)])
        return inversa  # det, cofactores, adjunta
    def inversa_3x3_Numpy(self):
        import numpy as np
        return m.Matrix(np.linalg.inv(self.data))

A = Matrix_extension([[4,3,2],[1,5,6],[8,1,9]])
print(f"Matriz de entrada: {A}")
B = A.inversa_3x3()
print(f"Matriz inversa sin Numpy: {B}")
C = A.inversa_3x3_Numpy()
print(f"Matriz inversa con Numpy: {C}")
print("Matriz de entrada @ Matriz inversa (debe ser la unitaria)")
print(A @ B)
print(A @ C)
