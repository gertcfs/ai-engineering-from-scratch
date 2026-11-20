import bayes as b

'''
Pruebas múltiples. Un paciente da positivo dos veces en pruebas independientes 
(ambas con un 99 % de precisión, prevalencia de la enfermedad de 1 en 10.000).
¿Cual es la P(enfermo) después de ambas pruebas? 
Utilice la distribución posterior de la primera prueba como distribución previa para la segunda.
'''

# Haciendolo con doble llamada a la función bayes
result = b.bayes(prior=b.bayes(0.0001, likelihood=0.99, false_positive_rate=0.01),
                    likelihood=0.99,
                    false_positive_rate=0.01)
print(f"P(enfermo|positivo) = {result:.4f}")

# Haciendolo con la función sequential_bayes
result = b.sequential_bayes(0.0001, 0.99, 0.01, 2)
print(f"P(enfermo|positivo) = {result:.4f}")
