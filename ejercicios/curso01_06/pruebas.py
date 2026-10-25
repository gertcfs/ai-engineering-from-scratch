import probability as pro

# Problema Poisson
# En una oficina de atención al cliente se reciben en promedio (lambda = 5) llamadas 
# por hora. ¿Cuál es la probabilidad de recibir exactamente (3) llamadas en una hora?
result = pro.poisson_pmf(3,5)
print(f"Resultado problema: {result:.4f} = {result*100:.2f}%")

# Problema de función de densidad de probabilidad (FDP) de una DISTRIBUCION NORMAL:
# Una máquina automatizada empaqueta bolsas de café cuyo peso sigue una distribución normal 
# con una media (mu) de 500 gramos y una desviación estándar (sigma) de 10 gramos.
# Calcula el valor exacto de la función de densidad (f(x) para una bolsa que pesa 
# exactamente 510 gramos.
# No calcula la probabilidad, calcula el punto de la curva de la función, la probabilidad
# es el area bajo la curva y para eso se necesita hacer la integral de la función
result = pro.normal_pdf(510, 500, 10)
print(f"Resultado problema: {result:.4f} = {result*100:.2f}%")

print("*** CON NUMPY Y SCIPY ***")
import numpy as np
from scipy import stats
normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Media: {np.mean(samples):.4f}, Estandar: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")
logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")

