import numpy as np
from transformations import eigenvalues_2x2

# Calcula manualmente los valores propios de la matriz [[4, 2], [1, 3]] utilizando 
# la ecuación característica. Luego, verifica el resultado con tu función 
# personalizada y con NumPy.
# - det(A - lamda*I) = 0 => det (|a-lambda    b   | = (a-lambda)(d-lambda) -bc = 
#					             |   c    d-lambda|)
#   = lambda^2 - (a+d)lambda + (ad-bc)
# - sustituyendo e igualando a 0 => 
#   lambda^2 - (4+3)lambda + 4*3-2*1 = lambda^2 - 7lambda + 10 = 0  => lambda = 5 y 2
# - valores propios o autovalores = 5 y 2

matriz = [[4,2],[1,3]]

print(f"Autovalores usando transformations : {eigenvalues_2x2(matriz)}")
eigenvalues, eigenvectors = np.linalg.eig(matriz)
print(f"Autovalores usando Numpy : {eigenvalues}")
