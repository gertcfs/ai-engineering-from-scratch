import math
import random


# Calculo de la derivada por el Método numérico o de "diferencia finita centrada"
# Las derivadas numéricas son mas lentas pero funcionan para cualquier función
# - f es una función lambda que contiene la función polinómica 1D a derivar
# - x es el punto del eje X donde queremos obtener la derivada
# - h es el incremento o paso (usamos un número muy pequeño)
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

# Calculo del gradiente de una función de pérdida 2D en un punto
# El grandiente es un vector que contiene todas las derivadas parciales
# - f es una función que tiene 2 variables (x e y) por tanto es 2D
# - point es el punto donde queremos obtener el gradiente [x,y]
# - h es el incremento o paso que será siempre un número muy pequeño cercano a 0
# Salida: el vector que contiene las derivadas parciales (gradiente)
# El bucle hace tantas pasadas como cantidad de números tiene el punto (dos)
# - Primera pasada con el primer numero (x) y segunda pasada con el 2º (y)
#   - Calcula la derivada parcial (campo partial)
#       - se copia el punto de entrada (point) a point_plus y point_minus
#       - se hace point_plus[0 o 1]=x + h    point_minus[0 o 1]=x - h
#       - se restan ambos resultados, se pasan por la función f y se divide por 2h
#       - el resultado es la derivada parcial (partial)
#       - la derivada parcial se añade al vector gradient
def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)    # copia la lista en point_plus => [x,y]
        point_minus = list(point)   # lo mismo pero en point_minus
        point_plus[i] += h          # incrementa la x de point_plus en h
        point_minus[i] -= h         # incrementa la x de point_minus en h
        # Pasa el punto mayor y el menor por la función
        # Calcula la diferencia de lo que da la función en ambos casos y divide por 2h
        # Esto es, en definitiva, calcular la derivada en el punto
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        # añade la derivada recien calculada al vector gradienrte
        gradient.append(partial)
    return gradient

# Esta funcion no se utiliza en este programa
def gradient_descent_1d(f, df, x0, lr=0.1, steps=20):
    x = x0
    history = []
    for step in range(steps):
        grad = df(x)
        x = x - lr * grad
        history.append((step, x, f(x)))
    return x, history

# Esta función no se utiliza en este programa
def gradient_descent_nd(f, x0, lr=0.1, steps=100):
    point = list(x0)
    history = []
    for step in range(steps):
        grad = numerical_gradient(f, point)
        point = [p - lr * g for p, g in zip(point, grad)]
        history.append((step, list(point), f(point)))
    return point, history

# Comparación entre derivada numérica y analítica
# Numérica es la que se calcula con la fórmula (por definición)
# Analítica la que se calcula con las reglas manualmente
def demo_numerical_vs_analytical():
    print("=" * 55)
    print("DERIVADA NUMERICA vs ANALITICA")
    print("=" * 55)

    # Tabla de funciones de 1D (1 variable) compuesta por tres columnas:
    #  nombre de la función "x al cuadrado" para la primera fila
    #  función: función lambda que calcula x al cuadrado
    #  derivada de la función: función lambda que calcula 2*x
    # Qué es una función lambda en python:
    # - es una función anónima que se define de forma rápida sin def
    # - ejemplo => lambda x: x**2
    #   - función que recibe un parámetro x y devuelve x**2
    test_cases = [
        ("x^2",    lambda x: x**2,        lambda x: 2*x),
        ("x^3",    lambda x: x**3,        lambda x: 3*x**2),
        ("sin(x)", lambda x: math.sin(x), lambda x: math.cos(x)),
        ("e^x",    lambda x: math.exp(x), lambda x: math.exp(x)),
        ("1/x",    lambda x: 1/x,         lambda x: -1/x**2),
    ]

    x = 2.0     # coordenada x del punto en el que se calculan las derivadas
    # Cabeceras:
    # < significa alineación a la izquierda; > significa alineación a la derecha
    # 12 son los espacios de ancho que se han de reservar para el texto
    # .6f indica que se mostrarán 6 decimales exactos
    # .2e convierte el número a notación científica (exponencial) mostrando 2 decimales
    print(f"\nEn x = {x}:")
    print(f"{'Función':<12} {'Numérica':>12} {'Analitica':>12} {'Error':>12}")
    print("-" * 50)
    for name, f, df in test_cases:       # nombre, función, funcion derivada
        num = numerical_derivative(f, x) # calc derivada numérica (funcion, punto x=2)
        ana = df(x)                      # calc derivada analítica (pasa la x por la funcion derivada)
        err = abs(num - ana)             # difª entre derivada numerica y analitica
        print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")

# Esta función compara las gradientes numérica y analítica
# Gradiente de función de pérdida 2D, es decir, tiene 2 variables 'x' e 'y' (2 pesos)
# Derivada parcial es la que se obtiene tomando 1 de las variables como constante
# Entonces hay 2 derivadas, la derivada respecto a 'x' y respecto a 'y'
# El gradiente es un vector que contiene las 2 derivadas
# Este vector apunta en la dirección que incrementa la función más rápidamente, como lo
# que buscamos es minimizar la función de pérdida hay que aplicar la dirección contraria
def demo_gradient():
    print("\n" + "=" * 55)
    print("GRADIENTE (VECTOR DE DERIVADAS PARCIALES)")
    print("=" * 55)

    # Esta función recibe un punto [x,y] y le pasa la funcion polinómica x^2+3xy+y^2
    def f(point):
        x, y = point
        return x**2 + 3*x*y + y**2

    point = [1.0, 2.0]      # punto en el que se calcula el gradiente
    # calcular el gradiente numérica con la función f y el punto point
    grad = numerical_gradient(f, point)
    # Ahora calcula el gradiente directamente con las derivadas
    # - derivada con respecto a x = 2x+3y, sustituyendo=2+6 = 8
    # - derivada con respecto a y = 3x+2y, sustituyendo=3+4 = 7
    analytical = [2*point[0] + 3*point[1], 3*point[0] + 2*point[1]]

    print("\nf(x,y) = x^2 + 3xy + y^2")
    print(f"En el punto ({point[0]}, {point[1]}):")
    print(f"  Gradiente numérico:  [{grad[0]:.4f}, {grad[1]:.4f}]")
    print(f"  Gradiente analítico: [{analytical[0]:.1f}, {analytical[1]:.1f}]")

# Implementar el descenso de gradiente desde cero para minimizar una 
# función de pérdida en 1D (x^2)
# El Descenso de Gradiente es una técnica de optimización de la función de pérdida
# - se calcula el gradiente de la función de pérdida
# - se aplica a cada peso la formula: nuevo peso = peso actual - (lr x gradiente)
# En esta demo se empieza por x=5 que se supone que es un peso asignado aleatoriamente
# Se dan 20 pasos y en cada paso:
# - se obtiene el gradiente: la derivada de x^2 es 2x => 2*5=10 en el paso 0
# - se calcula el nuevo peso: x = x -lr * grad => 5-(0,1*10)=4 en el paso 0
# - se itera un nuevo paso (en 20 pasos casi se llega al mínimo)
# En este caso el entrenamiento termina en 20 pasos pero se supone que el objetivo
# es minimizar la función de pérdida y aquí lo que minimiza es el valor de x que
# entiendo es el valor del peso. Lo de que el mínimo es x=0 no lo entiendo
# Tal vez x no es un peso?
def demo_gradient_descent_1d():
    print("\n" + "=" * 55)
    print("DESCENSO DE GRADIENTE: f(x) = x^2")
    print("=" * 55)

    # Partiendo de x=5, cada paso nos acerca más a x=0 (el mínimo).
    x = 5.0
    lr = 0.1    # lr=learning rate=tasa de aprendizaje)
    print(f"\nComienzo: x={x}, lr(tasa de aprendizaje)={lr}")
    for step in range(20):
        grad = 2 * x        # 2x es la derivada de x^2 => (0) 2*5=10 (1) 2*4=8 
        x = x - lr * grad   # (0) x=5-0,1*10=5-1=4 (1)=4-0,1*8=3,2 
        if step % 4 == 0 or step == 19:
            # 2d:2 espacios decimales; 8.4f:8 espacios y 4 decimales; 10 espacios y 6 decimales
            print(f"  step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
    print(f"Minimo encontrado en x={x:.6f} (verdadero minimo: x=0)")

# Implementar el descenso de gradiente desde cero para encontrar el mínimo de la
# función de pérdida 2D f(x,y)=(x^2 + y^2)
# Esta función la tengo comentada y con prints en ejercicio2.py
def demo_gradient_descent_2d():
    print("\n" + "=" * 55)
    print("DESCENSO DE GRADIENTE: f(x,y) = x^2 + y^2")
    print("=" * 55)

    def f(point):
        x, y = point
        return x**2 + y**2

    point = [4.0, 3.0]  # x=4 y=3
    lr = 0.1            # lr=learning rate=tasa de aprendizaje)
    print(f"\nComienzo: ({point[0]}, {point[1]}), lr={lr}")
    for step in range(30):
        # Gradiente numérico de funcion 2D, devuelve un vector
        grad = numerical_gradient(f, point)
        point = [p - lr * g for p, g in zip(point, grad)]
        loss = f(point)
        if step % 5 == 0 or step == 29:
            print(f"  step {step:2d}  ({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
    print(f"Minimo encontrado en ({point[0]:.4f}, {point[1]:.4f}) (true: (0, 0))")

# Devuelve la matriz hessiana de una función de pérdida 2D 
# La matriz hessiana mide la curvatura de la funcion de pérdida
# es la matriz de derivadas parciales de 2º orden, es cuadrada n x n
# Ejemplo: Funcion x^2 - y^2    (funcion silla de montar)
#   Primera derivada: df/dx = 2x ; df/dy = 2y
#   Segunda derivada: d2f/d2x = 2 ; d2f/d2y = 2   ; d2f/dxdy = 0
#	Matriz hessiana:
#			H = | 2   0 |
#			    | 0  -2 |
#	Autovalores: 2 y -2 (uno positivo, uno negativo)
#	--> Punto de silla en (0, 0)#
# Cómo lo calcula la función:
#  Diagonal principal:
#  - fxx : segunda derivada de la función respecto a x
#  - fyy : segunda derivada de la función respecto a y
#  Fuera de la diagonal
#  - fxy : segunda derivada mixta (1ª respecto a x y 2ª respecto a y (o al reves))
def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]

# Aproximación de Taylor
# Devuelve un número
# f esta recibiendo math.exp, math.sin y math.cos
# f_prime lo mismo
# f_double_prime esta recibiendo math.exp o una función lambda
# x0 es un número, un 1 o un 0
# h es un valor 0.1, 0.5, 1.0, 2.0
# order puede ser 1, 2, 3
def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

# Devuelve los autovalores (valores propios, eigenvalues) de la matriz hessiana
# Entra una matriz, la diagonaliza y obtiene sus autovalores
def hessian_eigenvalues(H):
    a, b = H[0][0], H[0][1]
    c, d = H[1][0], H[1][1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        return None, None
    sqrt_disc = discriminant ** 0.5
    return (trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2

# Tenemos 3 funciones: 
#   saddle=silla        x^2 - y^2                   punto (0,0)
#   bowl=cuenco         x^2 + y^2                   punto (0,0)
#   rosenblock=banana   (1-x)^2 + 100*(y-x^2)^2     punto (1,1)
# De cada una sacamos: matriz hessiana y autovalores
# Comparando los signos de los autovalores nos da:
#   Distintos: Punto de silla
#   Positivos: Mínimo local
def demo_hessian():
    print("\n" + "=" * 55)
    print("MATRIZ HESSIANA: PUNTO DE SILLA vs. MÍNIMO")
    print("=" * 55)

    def saddle(x, y):
        return x ** 2 - y ** 2

    def bowl(x, y):
        return x ** 2 + y ** 2

    print("\nf(x,y) = x^2 - y^2 (funcion de silla)")
    H = hessian_2d(saddle, 0.0, 0.0)
    e1, e2 = hessian_eigenvalues(H)
    print("  Hessiana en (0,0):")
    print(f"    [{H[0][0]:6.2f}  {H[0][1]:6.2f}]")
    print(f"    [{H[1][0]:6.2f}  {H[1][1]:6.2f}]")
    print(f"  Autovalores: {e1:.2f}, {e2:.2f}")
    print("  Signos distintos --> PUNTO DE SILLA")

    print("\nf(x,y) = x^2 + y^2 (funcion cuenco)")
    H = hessian_2d(bowl, 0.0, 0.0)
    e1, e2 = hessian_eigenvalues(H)
    print("  Hessiana en (0,0):")
    print(f"    [{H[0][0]:6.2f}  {H[0][1]:6.2f}]")
    print(f"    [{H[1][0]:6.2f}  {H[1][1]:6.2f}]")
    print(f"  Autovalores: {e1:.2f}, {e2:.2f}")
    print("  Ambos positivos --> MINIMO LOCAL")

    def rosenbrock(x, y):
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    print("\nFunción banana Rosenbrock f(x,y) = (1-x)^2 + 100*(y-x^2)^2")
    H = hessian_2d(rosenbrock, 1.0, 1.0)
    e1, e2 = hessian_eigenvalues(H)
    print("  Hessiana en minimo (1,1):")
    print(f"    [{H[0][0]:8.2f}  {H[0][1]:8.2f}]")
    print(f"    [{H[1][0]:8.2f}  {H[1][1]:8.2f}]")
    print(f"  Autovalores: {e1:.2f}, {e2:.2f}")
    print("  Ambos positivos --> MINIMO LOCAL (confirmado)")

# Aproximación por serie de Taylor
# Convierte una función compleja y difícil de calcular en una suma de polinomios 
# mucho más sencilla de manejar
def demo_taylor():
    print("\n" + "=" * 55)
    print("APROXIMACION POR SERIE DE TAYLOR")
    print("=" * 55)

    x0 = 1.0
    print(f"\nAproximando f(x) = e^x cerca de x0 = {x0}")
    print(f"{'h':>8}  {'True f(x0+h)':>14}  {'Orden 0':>10}  {'Orden 1':>10}  {'Orden 2':>10}")
    print("-" * 60)

    for h in [0.1, 0.5, 1.0, 2.0]:
        true_val = math.exp(x0 + h)
        t0 = taylor_approx(math.exp, math.exp, math.exp, x0, h, order=0)
        t1 = taylor_approx(math.exp, math.exp, math.exp, x0, h, order=1)
        t2 = taylor_approx(math.exp, math.exp, math.exp, x0, h, order=2)
        print(f"{h:8.1f}  {true_val:14.6f}  {t0:10.6f}  {t1:10.6f}  {t2:10.6f}")

    print("\nAproximando f(x) = sin(x) cerca de x0 = 0")
    print(f"{'h':>8}  {'True sin(h)':>14}  {'Orden 0':>10}  {'Orden 1':>10}  {'Orden 2':>10}")
    print("-" * 60)

    for h in [0.1, 0.5, 1.0, 2.0]:
        true_val = math.sin(h)
        t0 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), 0.0, h, order=0)
        t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), 0.0, h, order=1)
        t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), 0.0, h, order=2)
        print(f"{h:8.1f}  {true_val:14.6f}  {t0:10.6f}  {t1:10.6f}  {t2:10.6f}")

    print("\nIdea clave: más términos = mejor aproximación cerca de x0,")
    print("pero todas las aproximaciones de Taylor divergen lejos de x0.")

# Regresión lineal
# 
def demo_linear_regression():
    print("\n" + "=" * 55)
    print("DESCENSO DE GRADIENTE: REGRESIÓN LINEAL y = 2x + 1")
    print("=" * 55)

    random.seed(42)         # fijar semilla aleatoria
    w = random.gauss(0, 1)  # genera un nº aleatorio que sigue una distribuicion normal
    b = random.gauss(0, 1)  #  estándar (campana de gauss) de media 0 y desviación 1
    lr = 0.01               # lr=learning rate= tasa de aprendizaje

    xs = [1.0, 2.0, 3.0, 4.0, 5.0]  # Entradas (inputs) en la red neuronal
    ys = [3.0, 5.0, 7.0, 9.0, 11.0] # Salidas previsas

    for epoch in range(200):    # 200 épocas
        total_loss = 0
        dw = 0
        db = 0
        for x, y in zip(xs, ys):    # x => entradas, y => salidas previstas
            pred = w * x + b        # predicción: peso(w) * entrada + sesgo(b)
            error = pred - y        # error = predicción - salida prevista
            total_loss += error ** 2  # total pérdida += error al cuadrado
            dw += 2 * error * x     # dw = 2x*error (derivada de y=2x+1 = >2x)
            db += 2 * error         # db = 2*error
        dw /= len(xs)           # dividir dw por cantidad de entradas (5)
        db /= len(xs)           # dividir db por cantidad de entradas
        total_loss /= len(xs)   # dividir total pérdida por cantidad de entradas
        w -= lr * dw            # peso(w) = peso actual - lr * dw
        b -= lr * db            # sesgo(b) = sesgo anteriro - lr * db
        if epoch % 40 == 0 or epoch == 199:
            print(f"  época {epoch:3d}  w={w:.4f}  b={b:.4f}  pérdida={total_loss:.6f}")

    print(f"\nAprendido: y = {w:.2f}x + {b:.2f}")
    print("Actual   :  y = 2.00x + 1.00")


if __name__ == "__main__":
    demo_numerical_vs_analytical()
    demo_gradient()
    demo_gradient_descent_1d()
    demo_gradient_descent_2d()
    demo_hessian()
    demo_taylor()
    demo_linear_regression()
