import math
import bayes as b
from collections import defaultdict


'''
Agregar características. 

Extender la clase NaiveBayes para que también utilice la longitud del mensaje 
(corto/largo) como una característica junto con el recuento de palabras. 

Estimar P(corto/spam) y P(largo/ham) a partir de los datos de entrenamiento e 
incorporarlos a la puntuación de predicción.

Hecho con IA
'''

class NaiveBayesLength(b.NaiveBayes):
    def __init__(self, smoothing=1.0, limite_long=4):
        super().__init__(smoothing=smoothing)
        self.limite_long = limite_long
        self.contador_cortos = defaultdict(int)   # P(corto | clase)
        self.contador_largos  = defaultdict(int)   # P(largo | clase)

    # Función sobrescrita para entrenar el clasificador con la longitud del mensaje
    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = doc.lower().split()
            # --- palabras (igual que antes) ---
            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)
            # --- NUEVO: longitud del mensaje ---
            if len(words) <= self.limite_long:
                self.contador_cortos[label] += 1
            else:
                self.contador_largos[label] += 1

    # Función nueva para calcular la probabilidad logarítmica de la longitud del mensaje dado la clase
    # Los métodos que empiezan con _ son internos y no deberían llamarse desde fuera de la clase
    def _log_likelihood_length(self, length_label, cls):
        if length_label == 'corto':
            count = self.contador_cortos[cls]
        else:
            count = self.contador_largos[cls]
        total = self.class_counts[cls]
        # 2 categorías → añadimos smoothing * 2 al denominador
        return math.log((count + self.smoothing) / (total + 2 * self.smoothing))

    # Método estático para determinar si un documento es corto o largo según el umbralx
    @staticmethod
    def _length_of(document, threshold):
        """Devuelve 'corto' o 'largo' según el nº de palabras."""
        n_words = len(document.lower().split())
        return 'corto' if n_words <= threshold else 'largo'

    # Función sobrescrita para predecir la clase de un documento, incorporando la longitud
    def predict(self, document):
        words = document.lower().split()
        length_label = self._length_of(document, self.limite_long)
        best_class = None
        best_score = float('-inf')
        for cls in self.class_counts:
            # prior
            score = self._log_prior(cls)
            # likelihood de palabras
            for word in words:
                score += self._log_likelihood(word, cls)
            # NUEVO: likelihood de longitud
            score += self._log_likelihood_length(length_label, cls)
            if score > best_score:
                best_score = score
                best_class = cls
        return best_class

    # Función sobrescrita para predecir la probabilidad de cada clase dado un documento, incorporando la longitud
    def predict_proba(self, document):
        words = document.lower().split()
        length_label = self._length_of(document, self.limite_long)

        scores = {}
        for cls in self.class_counts:
            score = self._log_prior(cls)
            for word in words:
                score += self._log_likelihood(word, cls)
            score += self._log_likelihood_length(length_label, cls)
            scores[cls] = score

        max_score = max(scores.values())
        exp_scores = {cls: math.exp(s - max_score) for cls, s in scores.items()}
        total = sum(exp_scores.values())
        return {cls: exp_scores[cls] / total for cls in exp_scores}

    # Función nueva para inspeccionar la probabilidad de que un mensaje sea corto dado la clase
    # Devuelve P(corto | cls) sin log, para inspección
    def prob_short(self, cls):
        total = self.class_counts[cls]
        return (self.contador_cortos[cls] + self.smoothing) / \
            (total + 2 * self.smoothing)


docs = [
    "hola",
    "ganar dinero gratis",
    "hola como estas",
    "oferta especial compra ya",
    "te invito a un café mañana",
    "descarga ahora gratis",
    "buenos dias profesor",
    "click aqui para ganar",
    "te veo en la reunion",
    "promocion unica no te la pierdas",
]
labels = ["ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam"]

model = NaiveBayesLength(smoothing=1.0, limite_long=4)
model.train(docs, labels)

print(f"P(corto | spam): {model.prob_short('spam'):.4f}")
print(f"P(corto | ham): {model.prob_short('ham'):.4f}")
print(f"Predicción para 'hola': {model.predict('hola')}")
print(f"Predicción para 'ganar dinero gratis ahora': {model.predict('ganar dinero gratis ahora')}")
