import vectors


# funcion para escalar un vector por 2 en x y 3 en y
def escalado_2x_3y(vector):
    matriz_escalado = vectors.Matrix([[2, 0], [0, 3]])
    return matriz_escalado @ vector

cx = int(input("Introduce la coordenada x: "))
cy = int(input("Introduce la coordenada y: "))

x = vectors.Vector([cx, cy])
print(f"Escalado de vector {x} por 2 en x y 3 en y: {escalado_2x_3y(x)}")