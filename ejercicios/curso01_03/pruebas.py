# COMPRENSION DE LISTAS
# Esto se utiliza en python para poder poner los bucles en una sola linea
# En esta funcion, el bucle de abajo del todo, for i ..., se podría poner a continuación
# del final de la linea sum ... así estaría todo en una sola linea
# ¿como se lee esto?
# siempre empezamos por la derecha del todo y vamos yendo hacia la izquierda
# en este caso el primer bucle es for i .. luego viene for j .. y al final 
# la instruccion que ya usa i y j 
# entre el primer y el segundo bucle hay que hacer una suma (sum)
def mat_vec_mul(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

# Esta es la misma funciona escrita con bucles tradicionales
def mat_vec_mul2(matrix, vector):
    resultado = []  # Aquí guardaremos el vector final
    
    # BUCLE EXTERNO (i): Recorre cada fila de la matriz
    for i in range(len(matrix)):
        suma_posicion = 0
        
        # BUCLE INTERNO (j): Recorre cada columna/elemento de esa fila y del vector
        for j in range(len(vector)):
            # Multiplica y acumula el resultado
            suma_posicion += matrix[i][j] * vector[j]
            
        # Guarda el total de la fila en la lista final
        resultado.append(suma_posicion)
        
    return resultado

matrix = [ [1,3,7], [4,5,6], [2,2,2] ]
vector = [1,2,3]

print(f"Total1 = {mat_vec_mul2(matrix,vector)}")
print(f"Total2 = {mat_vec_mul(matrix,vector)}")

for i, j in zip(range(3), range(1, 4)):
    print(f"i={i} j={j}")