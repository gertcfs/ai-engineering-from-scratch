import math
from collections import defaultdict


# Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)
# P(B) = evidence
# P(A) = prior
# P(B|A) = likelihood
# P(A|B) = posterior
def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior

# Actualización bayesiana secuencial (o inferencia bayesiana recursiva)
# En machine learning, se refiere a modelos que se entrenan de forma incremental con 
# flujos de datos en tiempo real en lugar de usar un conjunto de datos estático.
# - Las actualizaciones bayesianas son un aprendizaje continuo. 
#   - La distribución posterior de hoy se convierte en la distribución previa de mañana. 
#   - Cuando el modelo detecta nuevos datos, actualiza sus creencias de forma 
#     incremental en lugar de volver a entrenarlo desde cero.
# • Paso 1 (Inicial). Partes de una creencia inicial (prior) y observas un primer dato. 
#   Aplicas el teorema de Bayes para obtener una creencia actualizada (posterior).
# • Paso 2 (Iteración). Cuando llega un segundo dato, el posterior del paso anterior 
#   se convierte automáticamente en tu nuevo prior.
# • Paso 3 (Repetición). Repites este ciclo indefinidamente con cada nuevo dato o 
#   evidencia que recibe
def sequential_bayes(prior, likelihood, false_positive_rate, num_tests):
    current = prior
    for i in range(num_tests):
        current = bayes(current, likelihood, false_positive_rate)
        print(f"  Despues del test {i + 1}: P(enfermo|positivo) = {current:.6f}")
    return current


class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing #suavizado de Laplace para evitar probabilidades cero
        self.class_counts = defaultdict(int)         #contador clase/num.documentos
        self.word_counts = defaultdict(lambda: defaultdict(int)) #contador clase/palabra/num.veces
        self.class_word_totals = defaultdict(int)    #contador totales clase/num.palabras
        self.vocab = set() #vocabulario total de palabras sin duplicados

    # Entrenamiento del clasificador Naive Bayes
    # Aquí se procesan los documentos de entrenamiento y sus clases correspondientes.
    # Para cada documento y clase: 
    # - se incrementa el contador de documentos para esa clase, 
    # - se tokeniza el documento en palabras,
    # - por cada palabra:
    #   - se incrementa el contador de la palabra concreta dentro de la clase
    #   - se incrementa el total de palabras dentro de la clase
    #   - se añade la palabra al vocabulario si no existia antes
    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = doc.lower().split()
            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    # Probabilidad de una clase 'cls' => P(clase) => (A PRIORI)
    # - se obtiene el total de documentos de todas las clases total_docs
    # - se halla la probabilidad de la clase cls: contador cls / total documentos
    # - se devuelve el logaritmo de la probabilidad de la clase cls
    def _log_prior(self, cls):
        total_docs = sum(self.class_counts.values()) #total docs todas clases
        return math.log(self.class_counts[cls] / total_docs) 

    # Probabilidad P(palabra|clase) => (LIKELIHOOD)
    # - se obtiene el contador de la palabra 'word' en una clase 'cls'
    # - se obtiene el total de palabras en la clase 'cls'
    # - se obtiene el tamaño del vocabulario (número de palabras únicas)
    # - se aplica suavizado de Laplace para evitar probabilidades cero
    # - se devuelve el logaritmo de la probabilidad de la palabra dado la clase
    def _log_likelihood(self, word, cls):
        count = self.word_counts[cls].get(word, 0)
        total = self.class_word_totals[cls]
        vocab_size = len(self.vocab)
        return math.log(
            (count + self.smoothing) / (total + self.smoothing * vocab_size)
        )

    # Función para predecir la clase de un documento dado
    # - se tokeniza el documento de entrada en palabras
    # - recorremos todas las clases existentes del entrenamiento
    # - para cada clase calculamos una puntuacion (score):
    #   - score = probabilidad a priori de la clase P(clase) (log_prior)
    #   - por cada palabra del documento de entrada
    #     - score = score + P(palabra|clase) (log_likelihood)
    # - mira si la puntuación obtenida es mayor que best_score y si es así la sustituye
    # - se devuelve la clase con la puntuación más alta (best_class)
    def predict(self, document):
        words = document.lower().split()
        best_class = None
        best_score = float("-inf")

        for cls in self.class_counts:   #por cada una de las clases existentes
            score = self._log_prior(cls)    #obtiene la probabilidad a priori de la clase
            for word in words:          #por cada palabra del documento de entrada
                score += self._log_likelihood(word, cls) #acumula la probabilidad (palabra|clase)
            if score > best_score:  #actualiza la clase con la puntuación más alta
                best_score = score  #mejor puntuación
                best_class = cls    #mejor clase

        return best_class

    # Dado un documento, devuelve una tabla clase/probabilidad de pertenecer a esa clase
    # - se tokeniza el documento en palabras
    # - se calcula la puntuación para cada clase: log_prior + sum(log_likelihood)
    # - se aplica la función softmax para convertir las puntuaciones en probabilidades
    # - se devuelve un diccionario con las probabilidades de cada clase
    def predict_proba(self, document):
        words = document.lower().split()
        scores = {}

        for cls in self.class_counts: # por cada una de las clases existentes
            score = self._log_prior(cls)    #score=probabilidad a priori de la clase P(clase) (log_prior)
            for word in words:        #por cada palabra del documento de entrada
                score += self._log_likelihood(word, cls)  #score=score+P(palabra|clase) (log_likelihood)
            scores[cls] = score     #almacena la puntuación para cada clase

        # Calcula la probabilidad de pertenecer a cada clase usando softmax (ver curso 01_06)
        max_score = max(scores.values())    # obtiene la puntuación (logit) mas alta
        # - Resta el logit máximo antes de exponenciar para evitar el desbordamiento
        exp_scores = {cls: math.exp(s - max_score) for cls, s in scores.items()}
        total = sum(exp_scores.values())    # sumatorio de los elementos de exp_scores
        # - Divide cada elemento de exp_scores por el total para obtener la probabilidad de cada clase
        return {cls: exp_scores[cls] / total for cls in exp_scores} 
        # Se devuelve un diccionario con las probabilidades de cada clase

    # Función para obtener las palabras más indicativas de una clase
    # Devuelve las 10 palabras con mayor probabilidad de la clase
    # - se calcula la probabilidad de cada palabra dado la clase
    # - se ordenan las palabras por probabilidad descendente
    # - se devuelve una lista de las n palabras más indicativas de la clase
    def top_words(self, cls, n=10):
        vocab_size = len(self.vocab) # tamaño del vocabulario
        total = self.class_word_totals[cls] # palabras totales de la clase 'cls'
        probs = {}
        for word in self.vocab: # por cada palabra (word) del vocabulario (vocab)
            count = self.word_counts[cls].get(word, 0) # nº veces de 'word' en 'cls'
            probs[word] = (count + self.smoothing) / ( # calculo de la probabilidad
                total + self.smoothing * vocab_size    # con suavizado
            )
        return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:n]

# Demo del Teorema de Bayes
# El ejemplo de la enfermedad rara
# Calcula el posterior con la función bayes
# Al final llama a sequential_bayes para una demo de actualización bayesiana
def demo_bayes_theorem():
    print("=" * 60)
    print("BAYES' THEOREM: MEDICAL TEST")
    print("=" * 60)

    prior = 0.0001
    likelihood = 0.99
    fpr = 0.01

    posterior = bayes(prior, likelihood, fpr)
    print(f"\n  Prevalencia de la enfermedad (prior):   {prior}")
    print(f"  Sensibilidad del test (likelihood): {likelihood}")
    print(f"  Tasa de falsos positivos:           {fpr}")
    print(f"  P(enfermo | positivo):            {posterior:.4f} ({posterior*100:.2f}%)")
    print(f"\n  A pesar de la precisión del 99%, solo el {posterior*100:.2f}% de los positivos son realmente enfermos.")

    print("\n  Pruebas secuenciales (2 positive tests):")
    sequential_bayes(prior, likelihood, fpr, 2)

# Otro ejemplo del Teorema de Bayes
# El ejemplo del filtro de spam
# No usa la función bayes()
def demo_spam_filter():
    print("\n" + "=" * 60)
    print("BAYES' THEOREM: SPAM FILTER")
    print("=" * 60)

    p_spam = 0.3
    p_lottery_given_spam = 0.05
    p_lottery_given_ham = 0.001

    p_lottery = p_lottery_given_spam * p_spam + p_lottery_given_ham * (1 - p_spam)
    p_spam_given_lottery = p_lottery_given_spam * p_spam / p_lottery

    print(f"\n  P(spam):                 {p_spam}")
    print(f"  P('lotería' | spam):     {p_lottery_given_spam}")
    print(f"  P('lotería' | not spam): {p_lottery_given_ham}")
    print(f"  P(spam | 'lotería'):     {p_spam_given_lottery:.4f} ({p_spam_given_lottery*100:.1f}%)")

# Demo de Naive Bayes - Clasificador de spam
# Se le pasan una serie de frases o documentos de entrenamiento al objeto
# Se le pasa también la clase que hemos asignado a cada documento
# Con esto se realiza el entrenamiento del objeto
# Ahora simulamos que nos llegan una serie de documentos que tenemos que analizar
# - sacamos las predicciones que nos dicen las probabilidades de cada documento
#   de pertenecer a algunas de nuestras clases
# - al final sacamos las 5 palabras con mas probabilidad de cada clase (mas indicadoras)
def demo_naive_bayes():
    print("\n" + "=" * 60)
    print("NAIVE BAYES SPAM CLASSIFIER")
    print("=" * 60)

    train_docs = [
        "win free money now",
        "free lottery ticket winner",
        "claim your prize today free",
        "urgent offer free cash",
        "congratulations you won free",
        "meeting tomorrow at noon",
        "project update attached",
        "can we schedule a call",
        "quarterly report review",
        "lunch on thursday sounds good",
        "team standup notes attached",
        "please review the pull request",
    ]

    train_labels = [
        "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham", "ham", "ham",
    ]

    classifier = NaiveBayes(smoothing=1.0)
    classifier.train(train_docs, train_labels)

    print(f"\n  Entrenamiento: {len(train_docs)} documentos ({sum(1 for l in train_labels if l == 'spam')} spam, {sum(1 for l in train_labels if l == 'ham')} ham)")
    print(f"  Tamaño del vocabulario: {len(classifier.vocab)}")

    test_messages = [
        "free money waiting for you",
        "meeting rescheduled to friday",
        "you won a free prize",
        "please review the attached report",
        "urgent free offer claim now",
        "can we discuss the project update",
    ]

    print("\n  Predicciones:")
    for msg in test_messages:
        prediction = classifier.predict(msg)
        proba = classifier.predict_proba(msg)
        confidence = proba[prediction]
        print(f"    '{msg}'")
        print(f"      -> {prediction} (confianza: {confidence:.3f})")

    print("\n  Top 5 palabras indicadoras de spam:")
    for word, prob in classifier.top_words("spam", 5):
        print(f"    {word}: {prob:.4f}")

    print("\n  Top 5 palabras indicadoras de ham:")
    for word, prob in classifier.top_words("ham", 5):
        print(f"    {word}: {prob:.4f}")

# ESTIMACION DE MAXIMA VEROSIMILITUD  (Maximum likelihood estimation (MLE)
# MAXIMUM A POSTERIORI (MAP)
'''
EJEMPLO MLE vs MAP

Tenemos una moneda con probabilidad p desconocida
p = probabilidad de que salga cara
- se lanza la moneda 10 veces (7 caras y 3 cruces)

Usando MLE
- se busca un p que maximiza la probabilidad de haber observado esos datos
p = 7 /10 = 0,7
- Interpretación: Con solo lo que he visto, la mejor estimación es 0.7. 
  - No tiene en cuenta ninguna creencia previa. Es puramente empírico.

Usando MAP
- Aquí incorporas un conocimiento previo (un prior) sobre la moneda. 
  - Usamos un prior Beta(a, b), la conjugada natural del binomial
- la fórmula para MAP es = caras + a - 1 / total + a + b - 2
- a y b son los numeros de la distribución Beta(a,b) que usamos como conocimiento previo

- Ejemplo con la distribución Beta(2,2)
  - Significa información previa pobre: 2 + 2 = 4
  - Por eso el resultado es muy parecido a MLE
  - p = 7 + 2 - 1 / 10 + 2 + 2 - 2  =  8 / 12 = 0,666

- Ejemplo con la distribución Beta(10,10)
  - a+b=20 => mucha información previa (es como si ya hubieras lanzado 20 veces con 10 caras y 10 cruces)
  - Es un prior informativo y fuerte: confías mucho en que la moneda es justa
  - p = 7 + 10 - 1 / 10 + 10 + 10 - 2  =  0,571
'''
def demo_mle_vs_map():
    print("\n" + "=" * 60)
    print("ESTIMACION MLE vs MAP ESTIMATION")
    print("=" * 60)

    heads = 7   # head=cara ; tail=cruz; flips=lanzar la moneda al aire
    total = 10

    mle = heads / total
    print(f"\n  Observado: {heads} caras en {total} tiradas de moneda")
    print(f"  MLE estimado: {mle:.4f}")

    # En la Estimación Máxima a Posteriori (MAP), cuando se utiliza una distribución 
    # a priori tipo Beta (muy común para problemas binarios como Binomial o Bernoulli),
    # alfa y beta son los parámetros de dicha distribución a priori
    # Representan el conocimiento previo antes de observar los datos
    # alfa representa el conteo previo de un resultado (ej. caras al tirar moneda)
    # beta representa el conteo del otro resultado (ej. cruces al tirar la moneda)
    alpha = 2
    beta = 2
    map_estimate = (heads + alpha - 1) / (total + alpha + beta - 2)
    print(f"\n  Beta({alpha},{beta}) prior (ligera inclinación hacia 0.5)")
    print(f"  MAP estimado: {map_estimate:.4f}")

    alpha = 10
    beta = 10
    map_strong = (heads + alpha - 1) / (total + alpha + beta - 2)
    print(f"\n  Beta({alpha},{beta}) prior (fuerte inclinación hacia 0.5)")
    print(f"  MAP estimado: {map_strong:.4f}")

    print("\n  Una distribución a priori más fuerte desplaza la estimación hacia 0,5 (la media a priori).")
    print("  Este es el mismo efecto que la regularización L2 que lleva los pesos hacia cero.")

# Actualización a posteriori de priores conjugados
def beta_update(alpha, beta_param, successes, failures):
    return alpha + successes, beta_param + failures

# Esta función demuestra la actualización bayesiana secuencial de una Beta-Binomial/Bernoulli
# Inicializa una prior Beta(1, 1), que es uniforme: no favorece ni éxito ni fracaso.
# Define cuatro lotes de datos batches de datos con diferentes resultados de lanzamiento de dados.
# Actualiza la prior Beta(1, 1) con cada lote de datos para obtener una estimación posterior.
# Al final imprime la posterior final, la media final y compara lo que sería si se 
# actualizaran todos los datos a la vez en un solo lote
# La idea clave es que en un modelo bayesiano con Beta prior y datos binarios (éxito/fracaso), 
# la actualización secuencial y la actualización por lotes dan exactamente el mismo 
# resultado final, porque la actualización es asociativa: el orden no importa cuando 
# todos los datos están incluidos.
def sequential_update_demo():
    print("\n" + "=" * 60)
    print("ACTUALIZACION BAYESIANA SECUENCIAL")
    print("=" * 60)

    alpha, beta_param = 1, 1    # prior inicial uniforme 
    print(f"\n  Prior inicial: Beta({alpha}, {beta_param})")
    print(f"  Media a priori: {alpha / (alpha + beta_param):.4f}")

    batches = [     # 4 lotes con alfa, beta, descripción
        (7, 3, "Día 1: 7 caras, 3 cruces"),
        (5, 5, "Día 2: 5 caras, 5 cruces"),
        (3, 7, "Día 3: 3 caras, 7 cruces"),
        (6, 4, "Día 4: 6 caras, 4 cruces"),
    ]

    # Por cada lote llamamos a beta_update pasandole exitos/fallos y alfa/beta que
    # nos devuelve el alfa/beta actualizado aque usaremos en la siguiente iteración
    # Es decir, cada nuevo lote refuerza la evidencia y va cambiando la distribución posterior
    # Al final se calcula la media, la varianza y la desviación estándar de cada iteración
    alpha, beta_param = 1, 1
    for successes, failures, description in batches:
        alpha, beta_param = beta_update(alpha, beta_param, successes, failures)
        mean = alpha / (alpha + beta_param)
        print(f"\n  {description}")
        print(f"  Posterior: Beta({alpha}, {beta_param})")
        print(f"  Posterior media: {mean:.4f}")
        variance = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))
        std = variance ** 0.5   # desviación estandar o típica
        print(f"  Posterior desviación std:  {std:.4f}")

    print(f"\n  Creencia final después de todos los datos: Beta({alpha}, {beta_param})")
    print(f"  Media = {alpha / (alpha + beta_param):.4f}")

    alpha_batch, beta_batch = 1, 1
    total_s = sum(s for s, _, _ in batches)
    total_f = sum(f for _, f, _ in batches)
    alpha_batch += total_s
    beta_batch += total_f
    print(f"\n  Actualización por lotes (todos los datos a la vez): Beta({alpha_batch}, {beta_batch})")
    print(f"  Media = {alpha_batch / (alpha_batch + beta_batch):.4f}")
    print(f"  Secuencial y por lotes dan el mismo resultado: {alpha == alpha_batch and beta_param == beta_batch}")

# Estás probando dos colores de botón. Variante A (azul) y variante B (verde) en tu
# página web. Quieres saber cuál recibe más clics
def ab_test_demo():
    print("\n" + "=" * 60)
    print("PRUEBA A/B BAYESIANA")
    print("=" * 60)

    import random as rng
    rng.seed(42)

    a_clicks, a_views = 50, 1000
    b_clicks, b_views = 65, 1000

    # Empieza con Beta(1,1) y calcula el posterior de cada variante
    a_alpha, a_beta = 1 + a_clicks, 1 + (a_views - a_clicks)
    b_alpha, b_beta = 1 + b_clicks, 1 + (b_views - b_clicks)

    # Calcula la media e imprime todo
    print(f"\n  Variante A: {a_clicks}/{a_views} clicks")
    print(f"  Variant B: {b_clicks}/{b_views} clicks")
    print(f"\n  Posterior A: Beta({a_alpha}, {a_beta}), media = {a_alpha / (a_alpha + a_beta):.4f}")
    print(f"  Posterior B: Beta({b_alpha}, {b_beta}), media = {b_alpha / (b_alpha + b_beta):.4f}")

    # Método de Monte Carlo con 100.000 muestras
    # Extrae 100.000 muestras de Beta variante A y otras 100.000 de Beta variante B
    # Si la muestra de B es mayor que la muestra de A, aumenta el contador b_wins
    n_samples = 100000
    b_wins = 0
    for _ in range(n_samples):
        sample_a = _beta_sample(a_alpha, a_beta, rng)
        sample_b = _beta_sample(b_alpha, b_beta, rng)
        if sample_b > sample_a:
            b_wins += 1

    # Calcula la probabilidad de que B sea mejor que A
    p_b_better = b_wins / n_samples
    print(f"\n  Ejemplos de Monte Carlo: {n_samples}")
    print(f"  P(B > A) = {p_b_better:.4f}")

    if p_b_better > 0.95:
        print("  Decision: Enviar variante B")
    elif p_b_better < 0.05:
        print("  Decision: Enviar variante A")
    else:
        print("  Decision: Sigue recopilando datos")

    # Estimación del incremento de la variante B sobre la variante A
    # Igual que el anterior, pero ahora calculamos el incremento relativo (lift) de B sobre A
    print("\n  Estimación del incremento:")
    lifts = []
    rng.seed(42)
    for _ in range(n_samples):
        sa = _beta_sample(a_alpha, a_beta, rng)
        sb = _beta_sample(b_alpha, b_beta, rng)
        if sa > 0:
            lifts.append((sb - sa) / sa)
    lifts.sort()
    median_lift = lifts[len(lifts) // 2]
    low = lifts[int(len(lifts) * 0.05)]
    high = lifts[int(len(lifts) * 0.95)]
    print(f"  Incremento relativo medio: {median_lift:.1%}")
    print(f"  90% intervalo de credibilidad: [{low:.1%}, {high:.1%}]")


def _beta_sample(alpha, beta_param, rng_module):
    x = _gamma_sample(alpha, rng_module)
    y = _gamma_sample(beta_param, rng_module)
    if x + y == 0:
        return 0.5
    return x / (x + y)


def _gamma_sample(shape, rng_module):
    if shape <= 0:
        raise ValueError("Gamma shape parameter must be positive")
    if shape < 1:
        return _gamma_sample(shape + 1, rng_module) * rng_module.random() ** (1.0 / shape)

    d = shape - 1.0 / 3.0
    c = 1.0 / (9.0 * d) ** 0.5

    while True:
        x = rng_module.gauss(0, 1)
        v = (1 + c * x) ** 3
        if v <= 0:
            continue
        u = rng_module.random()
        if u < 1 - 0.0331 * x ** 4:
            return d * v
        if math.log(u) < 0.5 * x ** 2 + d * (1 - v + math.log(v)):
            return d * v


if __name__ == "__main__":
    demo_bayes_theorem()
    demo_spam_filter()
    demo_naive_bayes()
    demo_mle_vs_map()
    sequential_update_demo()
    ab_test_demo()
