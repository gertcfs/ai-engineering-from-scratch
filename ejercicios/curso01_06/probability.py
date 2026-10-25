import math
import random

random.seed(42)

# Calcular un factorial
def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# Numeros combinatorios (n sobre k) (elementos de n tomados de k en k)
def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

# P(A | B) significa la probabilidad de A condicionada a que suceda B
# P(A|B) = P(A and B) / P(B)
def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

# Distribución de Bernouilli
# Si exito   => k=1 => devuelve p
# Si fracaso => k=0 => devuelve (1-p)
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

# Distribución Categórica
# Entra una lista probs y un indice k
# Devuelve el elemento probs(k)
def categorical_pmf(k, probs):
    return probs[k]

# Distribución Poisson
#  k = numero de ocurrencias del evento
#  lam: Tasa media o valor esperado de ocurrencias por intervalo
# Devuelve la probabilidad
def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

# Distribución uniforme
# Todos los resultados tienen la misma probabilidad
# Se utiliza para la inicialización aleatoria
# Entrada: rango desde 'a' hasta 'b' , x tiene que estar dentro del rango
# Salida: 1 / (b - a)
def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

# Función de densidad de probabilidad (FDP/PDF) de una distribución normal 
# Esta es la función que nos da la curva de densidad, devuelve un punto de la curva
# La probabilidad es el area debajo de la curva hasta el eje x y para hallarla tenemos
# que hacer la integral de ese area, pero como es muy dificil calcularla, usamos las
# tablas de Z y todo lo de la tipificación, etc.
# Entrada: mu = media ; sigma = varianza ; x = variable aleatoria
def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)

# Valor esperado: es el resultado promedio ponderado
# Entradas:
# - values: lista de valores
# - probabilities: lista de probabilidades correspondientes a cada valor que suman 1
# Salida: el sumatorio de todos los valores por su probabilidad => Suma(valor*prob)
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

# Varianza: mide la dispersión en torno a la media
# Entradas:
# - values: lista de valores
# - probabilities: lista de probabilidades correspondientes a cada valor que suman 1
# Salida: aplica la fórmula 
#  p * (v - mu)^2
# donde:
#  v = un valor de la lista de valores de entrada
#  p = la probabilidad del valor anterior recogida de la lista de probabilidades de entrada
#  mu = valor esperado calculado con la funcion anterior expected_value
def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

# Muestreo distribución bernouilli
# Entrada:
# - p: un numero entre 0 y 1
# - n: número de muestras a generar
# Salida: devuelve una lista de n valores 0's o 1's que se calculan así:
# - se calcula un nº aleatorio, si p es mayor que ese nº devuelve 0, si no devuelve 1
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

# Muestreo distribución categórica
# Entrada:
# - p: una lista de números entre 0 y 1
# - n: número de muestras a generar
# Salida: una lista de n números entre 0 y la len de la lista de entrada propbs
def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p                  # acumula un nº de la lista probs en el campo total
        cumulative.append(total)    # añade el campo total a la lista cumulative
    samples = []
    for _ in range(n):              # bucle de n veces (n viene en la entrada)
        r = random.random()
        for i, c in enumerate(cumulative):  # i es el indice y c es el valor de la lista
            if r <= c:              # si el nº random es <= que el valor de la lista
                samples.append(i)   # añade el indice a la lista samples
                break
    return samples


def sample_uniform(a, b, n=1):
    return [a + (b - a) * random.random() for _ in range(n)]

# El método de Box-Muller ​ es un método de generación de números aleatorios independientes
# con distribución normal "estándar", a partir de números aleatorios uniformemente distribuidos.
# Entrada:
# - mu: es la media
# - sigma: es la varianza
# - n: es el número de ocurrencias de la muestra que queremos generar
# Salida: Lista de n numeros aleatorios que siguen la curva de Gauss generados con 
#         el método Box-Mulleer
def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()    # numero aleatorio en el intervalo [0,1]
        u2 = random.random()    # igual
        # Formula de la transformada de Box-Muller
        # z será un nº aleatorio que sigue la curva de Gauss (distribución normal)
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)  # añada a samples el nº (mu+sigma+z)
    return samples

# La función Softmax es una operación matemática que convierte un vector de números reales 
# sin procesar (llamados logits) en un vector de probabilidades cuya suma total es 
# exactamente 1 (logits: vector de puntuaciones sin procesar)
# Fórmula: softmax(z_i) = exp(z_i) / sum(exp(z_j) para todo j)
# Entrada: Lista de numeros logits
# Calcula el logit mas alto y se lo resta a todos los logits creando la lista shifted
# (truco de estabilidad numérica)(para que no haya desbordamiento en caso de logits grandes)
# Crea exps como lista que contiene el exponencial (e elevado a cada elemento) de shifted
# Crea total como el sumatorio de los elementos de exps
# Divide cada elemento de exps por el total y devuelve una lista con los resultados
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

# La función log_softmax combina softmax y logaritmo para lograr estabilidad numérica. 
# PyTorch lo utiliza internamente para la pérdida de entropía cruzada
# Entrada: Lista de numeros logits (vector de puntuaciones sin procesar)
# Calcula el logit mas alto y se lo resta a todos los logits creando la lista shifted
# (truco de estabilidad numérica)(para que no haya desbordamiento en caso de logits grandes)
# Por cada elemento de shifted calcula su exponencial (e elevado al elemento) y va
# sumando los resultados, lo que nos da un número del cual hacemos su logaritmo
# luego le sumamos el logit mas alto calculado anteriormente y nos da log_sum_exp
# Devuelve:
# - por cada elemento de la lista logits de entrada se le resta el logit mas alto y
#   se devuelve la lista así generada
def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

# La pérdida de entropía cruzada es el logaritmo negativo de la probabilidad 
# de la clase correcta
# Entrada: lista de logits y un numero que es un indice
# - calcula log_probs que es la lista que devuelve la función log_softmax con los logits
# Devuelve el elemento target_index de la lista log_probs en negativo
def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]

# Entrada: joint es una matriz de 2 dimensiones que representa una distribución conjunta
# es decir, una distribución con 2 variables aleatorias (pe: clima, paraguas)
# Distribución marginal es ver como se comporta cada variable aleatoria por si misma
# La función suman los numeros de las filas por un lado y las columnas por otro
# Devuelve una lista con las sumas de cada fila y otra lista con las sumas de cada columna
def joint_to_marginals(joint):
    rows = len(joint)
    cols = len(joint[0])
    marginal_x = [sum(joint[i][j] for j in range(cols)) for i in range(rows)]
    marginal_y = [sum(joint[i][j] for i in range(rows)) for j in range(cols)]
    return marginal_x, marginal_y

# Ver si 2 sucesos son independientes
# Entrada: 
# - matriz de probabilidades de la distribución conjunta, distribuciones marginales
# - el campo tol, supongo que se refiere a tolerancia, pero no lo usa aquí
# Devuelve: true o false
# los dados son independientes solo si cada celda es igual al producto de sus dos marginales
def check_independence(joint, marginal_x, marginal_y, tol=1e-9):
    for i in range(len(marginal_x)):
        for j in range(len(marginal_y)):
            if abs(joint[i][j] - marginal_x[i] * marginal_y[j]) > tol:
                return False
    return True

# Función para la demostración del Teorema del Limite Central (clt)
# Hay un bucle que se ejecuta 'n_averages' veces y por cada vez hace:
# - se llena la lista samples con lo que devuelve la función dist_fn
# - se generan n_per_sample elementos en la lista
# - se saca la media y se va añadiendo a la lista averages
# Devuelve la lista averages
def demonstrate_clt(dist_fn, n_per_sample, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_per_sample)]
        averages.append(sum(samples) / len(samples))
    return averages


if __name__ == "__main__":
    print("=" * 60)
    print("PROBABILIDAD Y DISTRIBUCIONES")
    print("=" * 60)

# Probabilidad de que, si sale una figura (B), esta sea un rey (A)
# En la baraja de 52 cartas hay 4 reyes, la probabilidad de que salga un rey es 4/52
# Tambien hay 12 figuras (sota, caballo y rey x 4), la probabilidad es 12/52
# A and B => A union B => 'que sea rey y que sea figura' => 4/52
# Probabilidad condicionada P(A|B)= P(A and B) / PB = 4/52 : 12/52 = 1/3 = 0,33
    print("\n--- Probabilidad Condicional ---")
    p_king_given_face = conditional_probability(4 / 52, 12 / 52)
    print(f"P(Rey | Figura) = {p_king_given_face:.4f}")

# Distribución Bernouilli
# Se le pasan 2 valores en k, 0 y 1
# Cuando se le pasa 0 (fracaso) => la función devuelve 1 -0,7 = 0,3
# Cuando se le pasa 1 (exito) => la función devuelve 0,7
    print("\n--- PMF: Bernoulli (p=0.7) ---")
    for k in [0, 1]:
        print(f"  P(X={k}) = {bernoulli_pmf(k, 0.7):.4f}")

# Distribución Categórica
# Entra cat_probs => una lista de probabilidades que suman 1
# Llamamos a la funcion categorical_pmf 1 vez por cada elemento de la lista
# En cada llamada se le pasa la lista cat_probs y el indice (k) de un elemento
# La función devuelve la probabilidad del elemento del indice (k)
    print("\n--- PMF: Categorical ---")
    cat_probs = [0.1, 0.3, 0.4, 0.2]
    for k, p in enumerate(cat_probs):
        print(f"  P(X={k}) = {categorical_pmf(k, cat_probs):.4f}")

# Distribución de Poisson
# Se llama a la función 10 veces pasandole:
# - lambda = 3 => tasa media
# - k = un numero del 0 al 9 
# En cada llamada devuelve la probabilidad de que se produzca un evento que ocurre k veces
# sabiendo que la tasa media de ocurrencias es lambda
    print("\n--- PMF: Poisson (lambda=3) ---")
    for k in range(10):
        print(f"  P(X={k}) = {poisson_pmf(k, 3):.4f}")

# Función de densidad de probabilidad (PDF) de una Distribución Normal estandar (0,1)
# Esto no calcula la probabilidad, sino el punto de la curva de densidad representada por f(x)
# Entrada: mu = media ; sigma = varianza ; x = variable aleatoria
# Entra:
# - x: variable aleatoria, en cada llamada se pasa un número -3, -2, -1, 0, 1, 2, 3
# - mu: es la media, se le pasa siempre 0
# - sigma: es la varianza, se le pasa siempre 1
    print("\n--- PDF: Normal (mu=0, sigma=1) ---")
    for x in [-3, -2, -1, 0, 1, 2, 3]:
        print(f"  f({x:+d}) = {normal_pdf(x, 0, 1):.4f}")

# Entradas:
# - die_values => lista con los 6 valores del dado
# - die_probs  => lista con 6 numeros que representan la probabilidad de cada numero = 1/6
# Valor Esperado: es la media ponderada => Función expected_value 
# Devuelve la suma de cada valor por su probabilidad (1*1/6) + (2*1/6) +...+ (6*1/6) = 3,5
# Varianza: mide la dispersión en torno a la media => Función variance
# Devuelve la varianza según la fórmula => probabilidad * (valor - valor_esperado)^2
    print("\n--- Valores esperados & Varianza ---")
    die_values = [1, 2, 3, 4, 5, 6]
    die_probs = [1 / 6] * 6
    mu = expected_value(die_values, die_probs)
    var = variance(die_values, die_probs)
    print(f"  Dado no trucado: E[X] (valor esperado) = {mu:.4f}, Var(X) (varianza) = {var:.4f}")
    print(f"                   SD (desviación estandar o típica = raiz(varianza)) = {var ** 0.5:.4f}")

# Muestreo Bernouilli
# Se genera una lista de 20 números 0's o 1's aleatorios con la función sample_bernouilli
# Esta lista son las muestras y con ella calculamos la Media empírica que es:
# - la suma de los elementos de la lista dividido por el nº de elementos
    print("\n--- Muestreo: Bernoulli (p=0.3, n=20) ---")
    bern_samples = sample_bernoulli(0.3, 20)
    print(f"  Muestras: {bern_samples}")
    print(f"  Media empírica: {sum(bern_samples) / len(bern_samples):.4f} (expected 0.3)")

# La función sample_categorical devuelve una lista de 1000 elementos (2º parámetro)
# cada elemento es un número comprendido entre 0 y 4 (la len del 1er parámentro)
# counts es el resultado de contar las veces que aparecen los números del 0 a 3 en la lista
# Recuentos a partir de 1000 muestras saca counts (las repeticiones de 0's, 1's, 2's y 3's)
# Empírico saca cada número de counts dividido por 1000
# Esperado saca lo esperado, que está hardcodeado
    print("\n--- Muestreo: Categorical ---")
    cat_samples = sample_categorical([0.1, 0.3, 0.4, 0.2], 1000)
    counts = [cat_samples.count(i) for i in range(4)]
    print(f"  Recuentos a partir de 1000 muestras: {counts}")
    print(f"  Empirico: {[c / 1000 for c in counts]}")
    print("  Esperado:  [0.1, 0.3, 0.4, 0.2]")

# Se genera una lista de 10000 números aleatorios obtenidos por el método Box-Muller (norm_sample)
# Se suman los números de la lista anterior y se dividen por el nº de elementos de la lista
# generandose un número que es la media en el campo sample_mean
# Se coge cada elemento de la lista de 10000 numeros norm_samples, se le resta la media anterior
# el resultado se eleva al cuadrado y se divide por el número de elementos de la lista norm_samples
# cada uno de los resultados anteriores se suma y nos da el resultado en sample_var
    print("\n--- Muestreo: Normal (Box-Muller) ---")
    norm_samples = sample_normal_box_muller(0, 1, 10000)
    sample_mean = sum(norm_samples) / len(norm_samples)
    sample_var = sum((x - sample_mean) ** 2 for x in norm_samples) / len(norm_samples)
    print("  10000 muestras de N(0,1):")
    print(f"  Media muestral: {sample_mean:.4f} (expected 0)")
    print(f"  Variable de muestra:  {sample_var:.4f} (expected 1)")

# Demostración de la función Softmax que convierte la lista de entrada logits
# en una lista de salida donde cada elemento representa su probabilidad y el total suma 1 
    print("\n--- Softmax ---")
    logits = [2.0, 1.0, 0.1]
    probs = softmax(logits)
    print(f"  Logits:  {logits}")
    print(f"  Softmax: [{', '.join(f'{p:.4f}' for p in probs)}]")
    print(f"  Suma:    {sum(probs):.4f}")

# Ahora se prueba la función Sotmax con numeros grandes para demostrar que el algoritmo
# controla que no haya desbordamiento restando el máximo antes de hacer el cálculo
    print("\n--- Softmax con logits grandes (prueba de estabilidad) ---")
    large_logits = [100, 101, 102]
    probs_large = softmax(large_logits)
    print(f"  Logits:  {large_logits}")
    print(f"  Softmax: [{', '.join(f'{p:.4f}' for p in probs_large)}]")
    print("  (No hay desbordamiento porque restamos el máximo antes de la expresión.)")

# La función LogSoftmax calcula el logaritmo natural de la función Softmax: aplica 
# Softmax a un vector de puntuaciones sin procesar (llamadas logits) y luego toma el 
# logaritmo de esas probabilidades
# Verify exp coge cada numero de log_probs y saca su exponencial (e elevado a lp) 
    print("\n--- Probabilidades logarítmicas ---")
    log_probs = log_softmax(logits)
    print(f"  Logits:      {logits}")
    print(f"  Log-softmax: [{', '.join(f'{lp:.4f}' for lp in log_probs)}]")
    print(f"  Verify exp:  [{', '.join(f'{math.exp(lp):.4f}' for lp in log_probs)}]")

# La pérdida de entropía cruzada es el logaritmo negativo de la probabilidad 
# de la clase correcta
    print("\n--- Pérdida de entropía cruzada ---")
    ce = cross_entropy_loss([2.0, 1.0, 0.1], target_index=0)
    print("  Logits: [2.0, 1.0, 0.1], target: 0")
    print(f"  Pérdida de entropía cruzada: {ce:.4f}")

# Las probabilidades brutas causan problemas numéricos. Al multiplicar muchas 
# probabilidades pequeñas rápidamente resulta en un desbordamiento negativo.
# - la probabilidad bruta empieza a crecer hacia 0,00000000000....00001
# Las probabilidades logarítmicas lo solucionan, las multiplicaciones se convierten en sumas
# - se consiguen números mas manejables
    print("\n--- Por qué son importantes las probabilidades logarítmicas ---")
    probabilidad_palabra = 0.01
    numero_palabras = 50
    probabilidad_bruta = probabilidad_palabra ** numero_palabras
    suma_logaritmos = numero_palabras * math.log(probabilidad_palabra)
    print(f"  P(palabra)^{numero_palabras} = {probabilidad_palabra}^{numero_palabras}")
    print(f"  Probabilidad bruta: {probabilidad_bruta:.2e} (desbordamientos por defecto con más términos)")
    print(f"  Suma de logaritmos:     {suma_logaritmos:.4f} (estable)")
    print(f"  Recuperado:   {math.exp(suma_logaritmos):.2e}")

# Una distribución conjunta P(X ,Y) describe dos variables aleatorias conjuntamente.
# Ejemplo de PMF conjunta (X = clima, Y = paraguas):
# 		         Y=0 (no paraguas)	    Y=1 (paraguas)		    Marginal P(X)
# X=0 (sol)	             0,40			    0,10			    P(X=0) = 0,50
# X=1 (lluvia)	         0,05			    0,45			    P(X=1) = 0,50
# Marginal P(Y)	         P(Y=0) = 0,45		P(Y=1) = 0,55		1,00
# Distribución Marginal: Se suman las probabilidades de la distribución conjunta sobre 
# todos los posibles valores de la variable que se desea descartar
# Los totales de filas y columnas en la tabla anterior son los márginales
# marginal_x es la suma de las filas x=0 y x=1, lo que nos da (0.50 , 0.50)
# marginal_y es la suma de las columnas y=0 e y=1 lo que da (0.45 , 0,55)
# Se pasa una matriz de 2x2 a la función joint_to_marginals que devuelve las sumas
# de las filas en marginal_x y la sumas de las columnas en marginal_y
# El último print verifica si son sucesos independientes llamando a la función
    print("\n--- Distribuciones conjuntas y marginales ---")
    joint = [
        [0.40, 0.10],
        [0.05, 0.45],
    ]
    marginal_x, marginal_y = joint_to_marginals(joint)
    print("  Distribución conjunta (tiempo x paraguas):")
    print(f"    Sol, sin paraguas: {joint[0][0]}")
    print(f"    Sol, con paraguas:    {joint[0][1]}")
    print(f"    Lluvia, sin paraguas: {joint[1][0]}")
    print(f"    Lluvia, con paraguas:    {joint[1][1]}")
    print(f"  Marginal X (meteorología):  {marginal_x}")
    print(f"  Marginal Y (paraguas): {marginal_y}")
    print(f"  Sucesos Independientes? {check_independence(joint, marginal_x, marginal_y)}")

# Demostrar el teorema del límite central
# - Teorema del límite central: la suma (o promedio) de muchas variables aleatorias 
#   independientes converge a una distribución normal, independientemente de la 
#   distribución original.
# Por cada numero 1, 2, 5, 30
# - Se genera una lista con 10000 numeros con la función demostrate_clt
# - Se saca la media de cada elemento de la lista y se suman todas las medias en avg_mean
# - Luego calcula la desviación estándar del conjunto de medias (promedios) en avg_std
# - Imprime la media y la desviación estandar para que veamos como a medida que n aumenta
#   disminuye la desviación estandar y la distribución se va convirtiendo en normal
# Fórmula utilizada:
# En estadística y probabilidad, cuando aplicas esa fórmula al desglose de varias medias, 
# estás calculando lo que matemáticamente se conoce como el error estándar de la media (EEM)
# o simplemente la desviación estándar de la distribución muestral.
# => (x - avg_mean) ** 2: Calcula la diferencia al cuadrado entre cada media individual (x) 
# y el promedio de todas las medias (avg_mean). Esto mide la dispersión.
# => sum(...) / len(avgs): Suma todas esas diferencias al cuadrado y las divide entre el 
# número total de medias. Esto nos da la varianza.
# => ** 0.5: Hace la raíz cuadrada, lo que transforma la varianza de vuelta a las 
# unidades originales, convirtiéndola en la desviación estándar.
# ¿Para qué se usa? Si tienes muchos grupos de datos, calculas el promedio de cada grupo, 
# y luego metes esos promedios en tu lista avgs, esta fórmula te dice qué tan variables 
# son los promedios entre sí. Si el resultado es muy cercano a cero, significa que todas 
# las muestras que tomaste tienen medias muy parecidas.
    print("\n--- Teorema del límite central ---")
    print("  Promediado de muestras uniformes en [0,1):")
    for n in [1, 2, 5, 30]:
        avgs = demonstrate_clt(random.random, n, 10000)
        avg_mean = sum(avgs) / len(avgs)
        avg_std = (sum((x - avg_mean) ** 2 for x in avgs) / len(avgs)) ** 0.5
        print(f"    n={n:2d}: media={avg_mean:.4f}, std={avg_std:.4f}")
    print("  A medida que n aumenta, la desviación estándar disminuye y la distribución se aproxima a una distribución normal..")

    print("\n--- Visualización ---")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        ax = axes[0][0]
        ax.set_title("Bernoulli PMF (p=0.7)")
        ax.bar([0, 1], [bernoulli_pmf(0, 0.7), bernoulli_pmf(1, 0.7)],
               color=["#4a90d9", "#d94a4a"], width=0.4)
        ax.set_xlabel("k")
        ax.set_ylabel("P(X=k)")
        ax.set_xticks([0, 1])

        ax = axes[0][1]
        ax.set_title("Poisson PMF (lambda=3)")
        ks = list(range(12))
        ax.bar(ks, [poisson_pmf(k, 3) for k in ks], color="#4a90d9", width=0.6)
        ax.set_xlabel("k")
        ax.set_ylabel("P(X=k)")

        ax = axes[0][2]
        ax.set_title("Normal PDF")
        xs = [i * 0.01 - 5 for i in range(1001)]
        for mu_val, sigma_val, label in [(0, 1, "N(0,1)"), (0, 2, "N(0,2)"), (2, 0.5, "N(2,0.5)")]:
            ys = [normal_pdf(x, mu_val, sigma_val) for x in xs]
            ax.plot(xs, ys, label=label, linewidth=2)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()

        ax = axes[1][0]
        ax.set_title("Uniform PDF [a=1, b=4]")
        xs_u = [i * 0.01 - 1 for i in range(701)]
        ys_u = [uniform_pdf(x, 1, 4) for x in xs_u]
        ax.plot(xs_u, ys_u, color="#4a90d9", linewidth=2)
        ax.fill_between(xs_u, ys_u, alpha=0.3, color="#4a90d9")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_ylim(0, 0.5)

        ax = axes[1][1]
        ax.set_title("Teorema del Límite Central")
        for n_val, color in [(1, "#aaaaaa"), (2, "#88aacc"), (5, "#4a90d9"), (30, "#d94a4a")]:
            avgs = demonstrate_clt(random.random, n_val, 10000)
            ax.hist(avgs, bins=50, alpha=0.5, color=color, label=f"n={n_val}", density=True)
        ax.set_xlabel("Media muestral")
        ax.set_ylabel("Densidad")
        ax.legend()

        ax = axes[1][2]
        ax.set_title("Salida Softmax")
        logit_sets = [
            ([1, 1, 1], "equal [1,1,1]"),
            ([2, 1, 0], "spread [2,1,0]"),
            ([10, 1, 0], "sharp [10,1,0]"),
        ]
        x_positions = range(3)
        width = 0.25
        for idx, (lg, label) in enumerate(logit_sets):
            sm = softmax(lg)
            offset = (idx - 1) * width
            ax.bar([x + offset for x in x_positions], sm, width=width, label=label)
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(["Class 0", "Class 1", "Class 2"])
        ax.legend()

        plt.tight_layout()
        plt.savefig("probability_distributions.png", dpi=150)
        print("  Saved: probability_distributions.png")
        plt.close()

    except ImportError:
        print("  matplotlib no está disponible; se omite la visualización.")

    print("\n" + "=" * 60)
    print("Todos los cálculos de probabilidad han concluido..")
    print("=" * 60)
