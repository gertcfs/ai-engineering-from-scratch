from collections import defaultdict
'''
defaultdict es un diccionario que devuelve un valor predeterminado para claves 
que no existen (El diccionario estandar devuelve error en esos casos). 
Se utiliza para contar palabras y clases sin tener que verificar si la clave 
existe previamente. 
'''

class_counts = defaultdict(int)
word_counts = defaultdict(lambda: defaultdict(int))
class_word_totals = defaultdict(int)
vocab = set()

def entrenar(doc, label):
    class_counts[label] += 1
    words = doc.split()
    for word in words:
        print(f"Procesando palabra: {word} , para la clase: {label}")
        word_counts[label][word] += 1
        class_word_totals[label] += 1
        vocab.add(word)

entrenar("este es un correo de prueba", "spam")
entrenar("este es un correo de prueba", "spam")
entrenar("este es otro igual o parecido", "ham")

print(f"Contador clase/num.frases: {dict(class_counts)}")
print(f"Contador clase/palabra/num.veces: {dict(word_counts)}")
print(f"Totales clase/num.palabras: {dict(class_word_totals)}")
print(f"Vocabulario: {vocab}")

'''
RESULTADOS DE LA EJECUCION:
class_counts: Contador clase/num.frases
    {'spam': 2, 'ham': 1}
word_counts: Contador clase/palabra/num.veces
    {'spam': defaultdict(<class 'int'>, 
        {'este': 2, 'es': 2, 'un': 2, 'correo': 2, 'de': 2, 'prueba': 2}), 
    'ham': defaultdict(<class 'int'>, 
        {'este': 1, 'es': 1, 'otro': 1, 'igual': 1, 'o': 1, 'parecido': 1})}
class_word_totals: Totales clase/num.palabras
    {'spam': 12, 'ham': 6}
vocab: Vocabulario
    {'parecido', 'un', 'o', 'otro', 'de', 'igual', 'este', 'es', 'prueba', 'correo'}
'''