from matrices import Matrix, relu_matrix

# Construye una red de dos capas. 
# Utilizando únicamente la clase Matrix (sin NumPy), crea una red neuronal de dos capas: 
# entrada (3) -> capa oculta (4) -> salida (2). 
# Inicializa pesos aleatorios, realiza una pasada hacia adelante y verifica 
# que todas las dimensiones sean correctas.

neuronas_input = 3
neuronas_capa_oculta = 4
neuronas_output = 2

input = Matrix([[2],[4],[6]])   # 3x1

# Capa 1:
# Se multiplica la matriz de pesos por la capa input: W1 @ input
# La matriz de pesos W1 es de 4 x N y la matriz input es 3 x 1
# Entonces N tiene que ser = 3 para que la multiplicación de matrices se pueda hacer
# Luego los pesos W1 tienen que ser 4x3 y así quedará 4x3 @ 3x1 = 4 x 1
W1 = Matrix.random(neuronas_capa_oculta, neuronas_input)    # 4x3
# La salida 4 x 1 se suma al bias 4 x 1 y da otra 4 x 1
b1 = Matrix([[0.0]]*neuronas_capa_oculta)
# A esto le pasamos la funcion Relu y nos da otra 4 x 1
capa1 = relu_matrix((W1 @ input) + b1)

# Capa 2 (output):
# Se multiplica la matriz de pesos W2 por la capa1
# La matriz de pesos W2 es 2 x N y hay que multiplicarla por un 4 x 1
# Por tanto W2 tiene que ser 2 x 4 para poder hacer W2 @ capa1 = 2 x 4 @ 4 x 1 = 2 x 1
W2 = Matrix.random(neuronas_output, neuronas_capa_oculta)   # 2x4
# La salida 2 x 1 se suma al bias 2 x 1
b2 = Matrix([[0.0]]*neuronas_output)
# A esto le pasamos la funcion Relu y nos da otra 2 x 1
capa2 = relu_matrix((W2 @ capa1) + b2)

print(f"Entrada {input.shape}: {input}")
print(f"Pesos W1 {W1.shape}: {W1}")
print(f"Resultado capa1 {capa1.shape}: {capa1}")
print(f"Pesos W2 {W2.shape}: {W2}")
print(f"Resultado final {capa2.shape}: {capa2}")
