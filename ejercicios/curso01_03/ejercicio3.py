import math
import transformations as t

# - Crea una composición de tres transformaciones 
#   (rotación de 30 grados, escalado por [1.5, 0.8], cizallamiento con kx=0.3) 
#   y aplícala a 8 puntos dispuestos en un círculo. 
# - Imprime las coordenadas antes y después de la transformación. 
# - Calcula el determinante de la matriz compuesta y verifica que sea igual al producto 
#   de los determinantes individuales.

# Composicion de las 3 transformaciones
R = t.rotation_2d(30)
E = t.scaling_2d(1.5, 0.8)
C = t.shearing_2d(0.3, 0)

R_E = t.mat_mul(E, R)
R_E_C = t.mat_mul(C, R_E)

# Calcular los puntos de la circunferencia y aplicar la transformación a cada uno
num_puntos = 8
radio = 1
centro = [0,0]

for i in range(num_puntos):
    # Calcular el ángulo en radianes para cada uno de los puntos
    angulo = 2 * math.pi * i / num_puntos
    # Calcular coordenadas X e Y
    x = centro[0] + radio * math.cos(angulo)
    y = centro[1] + radio * math.sin(angulo)
    # Imprimir resultados antes y después de la composicion de transformaciones
    tr = t.mat_vec_mul(R_E_C, [x,y])
    print(f"Punto {i + 1}: (x: {round(x, 4)}, y: {round(y, 4)}) "
        f"transformado: (x: {round(tr[0], 4)}, y: {round(tr[1], 4)})")

# Calcular el determinante de la matriz compuesta
det1 = t.det_2x2(R_E_C)
det2 = t.det_2x2(R) * t.det_2x2(E) * t.det_2x2(C)
print(f"Determinante composicion: {round(det1, 4)} = {round(det2, 4)}")
