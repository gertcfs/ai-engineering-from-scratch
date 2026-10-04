import vectors


def vector_aleatorio_50():
    """Genera un vector aleatorio de dimensión 50 con componentes entre 1 y 100000."""
    import random
    return vectors.Vector([random.uniform(1, 100000) for _ in range(50)])

# lista de 5 vectores aleatorios de dimensión 50
vectores = [vector_aleatorio_50() for _ in range(5)]

# Encontrar los dos vectores mas similares utilizando la similitud coseno
# Comparamos cada uno con los otros 4 con el resultado de vector.cosine_similarity
max_similitud = -1
vector1 = None
vector2 = None
for i in range(len(vectores)):
    for j in range(i + 1, len(vectores)):
        similitud = vectores[i].cosine_similarity(vectores[j])
        if similitud > max_similitud:
            max_similitud = similitud
            vector1 = vectores[i]
            vector2 = vectores[j]

# Imprimir los resultados
print("Vectores generados:")
for i, vec in enumerate(vectores):
    print(f"Vector {i + 1}: {vec}")
print(f"\nLos dos vectores más similares son:\n{vector1}\ny\n{vector2}\ncon una similitud coseno de {max_similitud:.4f}")