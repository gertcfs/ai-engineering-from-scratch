
def bayes2(
    probabilidad_a_priori: float,
    probabilidad_b_dado_a: float,
    probabilidad_b_dado_no_a: float
) -> float:
    """
    Calcula la probabilidad de A dado B usando el teorema de Bayes.

    Args:
        probabilidad_a_priori (float): La probabilidad a priori de A.
        probabilidad_b_dado_a (float): La probabilidad de B dado A.
        probabilidad_b_dado_no_a (float): La probabilidad de B dado no A.

    Returns:
        float: La probabilidad de A dado B.
    """
    # Calculamos la probabilidad total de B
    probabilidad_b = (
        probabilidad_b_dado_a * probabilidad_a_priori +
        probabilidad_b_dado_no_a * (1 - probabilidad_a_priori)
    )
    
    # Aplicamos el teorema de Bayes
    probabilidad_a_dado_b = (
        (probabilidad_b_dado_a * probabilidad_a_priori) / probabilidad_b
    )
    
    return probabilidad_a_dado_b
'''
EJEMPLO 1. ENFERMEDAD MEDICA    
1-Una enfermedad afecta a 1 de cada 1000 personas
2-Las pruebas detectan la enfermedad correctamente el 99% de las veces
  y dan falsos positivos el 1% de las veces
3-Cual es la probabilidad de que una persona tenga la enfermedad si 
  su prueba dio positiva?
Se pide la probabilidad de 'enfermo=true' sabiendo que 'positivo=true': 
  P(enfermo|positivo)
De cada punto sacamos lo necesario para aplicar el teorema de Bayes:
1-La probabilidad de que una persona este enferma es de 1/10000 = 0,0001
  P(enfermo) = 0,0001
2-Si me hago la prueba:
  - Si estoy enfermo, la probabilidad de dar positivo es de 0,99
    - Probabilidad positivo sabiendo que estoy enfermo:
      P(positivo|enfermo) = 0,99
 - Si no estoy enfermo, la probabilidad de dar positivo es de 0,01
    - Probabilidad positivo sabiendo que no estoy enfermo:
      P(positivo|no_enfermo) = 0,01
Entonces tenemos:
- A PRIORI:   P(enfermo) = 0,0001             P(A)
- LIKELIHOOD: P(positivo|enfermo) = 0,99      P(B|A)
- EVIDENCIA:  P(positivo|no_enfermo) = 0,01   P(B|¬A)
- A POSTERIORI: P(enfermo|positivo) = ?       P(A|B)
Para aplicar el teorema de Bayes necesitamos calcular tambien la 
probabilidad total de dar positivo P(B) = P(B|A)·P(A) + P(B|not A)·P(not A)
=> P(B) = 0,99·0,0001 + 0,01·(1-0,0001) = 0,010098
Ahora ya podemos aplicar el teorema de Bayes:
P(A|B) = P(B|A)·P(A)/P(B) = 0,99·0,0001/0,010098 = 0,0098
'''

def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior

# EJEMPLO 1
result = bayes(prior=0.0001, 
               likelihood=0.99, 
               false_positive_rate=0.01)
print(f"P(enfermo|positivo) = {result:.4f}")

result = bayes2(probabilidad_a_priori=0.0001, 
                probabilidad_b_dado_a=0.99, 
                probabilidad_b_dado_no_a=0.01)
print(f"P(enfermo|positivo) = {result:.4f}")

'''
EJEMPLO 2. Detección de spam
- Un correo electrónico es spam en el 20% de los casos
- Si un correo es spam, la probabilidad de que contenga la palabra "gratis"
  es del 70%
- Si un correo no es spam, la probabilidad de que contenga la palabra "gratis" es del 10%
- Se desea calcular la probabilidad de que un correo sea spam dado que contiene la palabra "gratis"
- P(spam|gratis) = ?
- A PRIORI:   P(spam) = 0,2             P(A)
- LIKELIHOOD: P(gratis|spam) = 0,7      P(B|A)
- EVIDENCIA:  P(gratis|no_spam) = 0,1   P(B|¬A)
- A POSTERIORI: P(spam|gratis) = ?       P(A|B)
Para aplicar el teorema de Bayes necesitamos calcular tambien la probabilidad total de dar positivo P(B) = P(B|A)·P(A) + P(B|not A)·P(not A)
=> P(B) = 0,7·0,2 + 0,1·(1-0,2) = 0,14 + 0,08 = 0,22
Ahora ya podemos aplicar el teorema de Bayes:
P(A|B) = P(B|A)·P(A)/P(B) = 0,7·0,2/0,22 = 0,14/0,22 = 0,6364
'''

# EJEMPLO 2
result = bayes(prior=0.2, 
               likelihood=0.7, 
               false_positive_rate=0.1)
print(f"P(spam|gratis) = {result:.4f}")

result = bayes2(probabilidad_a_priori=0.2, 
                probabilidad_b_dado_a=0.7, 
                probabilidad_b_dado_no_a=0.1)
print(f"P(spam|gratis) = {result:.4f}")

'''
EJEMPLO 3
Quiero hayar la probabilidad de que la puerta del jardin se abra si se que va a hacer viento
P(A) => Probabilidad de que la puerta del jardin se abra
P(B) => Probabilidad de que haga viento
P(B|A) => Probabilidad de que haga viento sabiendo que la puerta del jardin se ha abierto
P(B|¬A) => Probabilidad de que haga viento sabiendo que la puerta del jardin no se ha abierto
P(A|B) => Probabilidad de que la puerta del jardin se abra sabiendo que va a hacer viento
----------
P(A): la puerta del jardin se abre el 1% de las veces
P(B|A): si la puerta del jardin se abre, la probabilidad de que haga viento es del 90%
P(B|¬A): si la puerta del jardin no se abre, la probabilidad de que haga viento es del 10%
----------
P(B) = P(B|A)·P(A) + P(B|¬A)·P(¬A) = 0,9·0,01 + 0,1·(1-0,01) = 0,009 + 0,099 = 0,108
P(A|B) = P(B|A)·P(A)/P(B) = 0,9·0,01/0,108 = 0,009/0,108 = 0,0833
'''
# EJEMPLO 3
result = bayes(prior=0.01, 
               likelihood=0.9, 
               false_positive_rate=0.1)
print(f"P(puerta|viento) = {result:.4f}")

result = bayes2(probabilidad_a_priori=0.01, 
                probabilidad_b_dado_a=0.9, 
                probabilidad_b_dado_no_a=0.1)
print(f"P(puerta|viento) = {result:.4f}")