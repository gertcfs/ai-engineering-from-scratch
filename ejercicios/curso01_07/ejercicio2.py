import bayes as b

def demo_naive_bayes(suavizado=1.0):
    print("\n" + "=" * 60)
    print(f"NAIVE BAYES SPAM CLASIFICADOR CON SUAVIZADO = {suavizado}")
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

    classifier = b.NaiveBayes(smoothing=suavizado)
    classifier.train(train_docs, train_labels)

    '''
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
    '''

    print("\n  Top 5 palabras indicadoras de spam:")
    for word, prob in classifier.top_words("spam", 5):
        print(f"    {word}: {prob:.4f}")

    print("\n  Top 5 palabras indicadoras de ham:")
    for word, prob in classifier.top_words("ham", 5):
        print(f"    {word}: {prob:.4f}")

'''
Impacto del suavizado. Ejecute el clasificador de spam con valores de suavizado 
de 0,01, 0,1, 1,0 y 10,0. 
¿Cómo cambian las probabilidades de las palabras principales? 
¿Qué sucede con suavizado=0 y una palabra que aparece solo en correos no deseados?

Este ejercicio se trata de ejecutar la función bayes.naive_bayes_demo() con 
diferentes valores de suavizado y observar los resultados.

Voy a copiar la función naive_bayes_demo() aquí para que puedas modificarla y 
ejecutarla con diferentes valores de suavizado.
'''
demo_naive_bayes(suavizado=0.01)
demo_naive_bayes(suavizado=0.1)
demo_naive_bayes(suavizado=1.0)
demo_naive_bayes(suavizado=10.0)

'''
RESULTADOS
- cuanto mayor es el suavizado, más baja la probabilidad de las palabras principales 
  de spam y ham, ya que el suavizado reduce la influencia de las palabras raras.
- con suavizado=0 pueden salir probabilidades 0
Sin suavizado, una palabra que nunca aparece en los datos de entrenamiento de spam 
obtiene P(palabra|spam)=0, lo que hace que el producto total sea cero, independientemente 
de otros indicadores fuertes de spam. Sumar 1 a cada recuento evita esto.
============================================================
NAIVE BAYES SPAM CLASIFICADOR CON SUAVIZADO = 0.01
============================================================

  Top 5 palabras indicadoras de spam:
    free: 0.2337
    today: 0.0471
    offer: 0.0471
    prize: 0.0471
    winner: 0.0471

  Top 5 palabras indicadoras de ham:
    review: 0.0683
    attached: 0.0683
    we: 0.0343
    the: 0.0343
    quarterly: 0.0343

============================================================
NAIVE BAYES SPAM CLASIFICADOR CON SUAVIZADO = 0.1
============================================================

  Top 5 palabras indicadoras de spam:
    free: 0.2008
    today: 0.0433
    offer: 0.0433
    prize: 0.0433
    winner: 0.0433

  Top 5 palabras indicadoras de ham:
    review: 0.0629
    attached: 0.0629
    we: 0.0329
    the: 0.0329
    quarterly: 0.0329

============================================================
NAIVE BAYES SPAM CLASIFICADOR CON SUAVIZADO = 1.0
============================================================

  Top 5 palabras indicadoras de spam:
    free: 0.0923
    today: 0.0308
    offer: 0.0308
    prize: 0.0308
    winner: 0.0308

  Top 5 palabras indicadoras de ham:
    review: 0.0411
    attached: 0.0411
    we: 0.0274
    the: 0.0274
    quarterly: 0.0274

============================================================
NAIVE BAYES SPAM CLASIFICADOR CON SUAVIZADO = 10.0
============================================================

  Top 5 palabras indicadoras de spam:
    free: 0.0325
    today: 0.0239
    offer: 0.0239
    prize: 0.0239
    winner: 0.0239

  Top 5 palabras indicadoras de ham:
    review: 0.0256
    attached: 0.0256
    we: 0.0235
    the: 0.0235
    quarterly: 0.0235
'''