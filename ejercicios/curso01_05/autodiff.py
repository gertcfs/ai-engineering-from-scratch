import random

# Un motor de autograd necesita tres cosas:
#  - wrapping (envoltura) de valores: Envuelve cada número en un objeto que almacena 
#    su valor y gradiente.
#  - Registrar grafo. Cada operación registra sus entradas y la función de gradiente 
#    local.
#  - Paso hacia atrás. Ordena topológicamente el grafo y luego lo recorre en sentido 
#    inverso, aplicando la regla de la cadena en cada nodo.

# Clase que implementa el wrapping de valores
# Envuelve cada número en un objeto que almacena su valor y gradiente
class Value:
    # Entrada:
    #  data:        el valor
    #  grad:        el gradiente
    #  _backward:   funcion a llamar en backward  (el prefijo _ significa que es privado)
    #   todos los que han sido creados con una operacion tienen una función backward
    #  _prev:       referencia a los objetos padres del objeto
    #  _op:         la operación que generó el objeto
    def __init__(self, data, children=(), op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)  # set convierte children en un set de datos (elimina duplicados)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # Operación + (suma dos objetos Value c = a + b)
    # Devuelve un nuevo objeto Value con:
    # - data = suma de los data de los padres c=a+b
    # - _prev = guarda referencias a los objetos padres de la operación a y b
    # - _op = guarda el signo '+'
    # Carga en el campo _backward la dirección a una función que se utilizará en otra
    # función llamada backward. 
    # La función hace lo siguiente
    # - suma al campo grad de los objetos 'a' y 'b' lo que hay en el campo grad de 'c'
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+') # crear out
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    # define el comportamiento del operador de suma + cuando tu objeto se encuentra en el 
    # lado derecho de la operación 
    def __radd__(self, other):
        return self.__add__(other)

    # Operación * (multiplicar objetos Value c = a * b)
    # Genera un nuevo objeto que tiene
    # - c.data = a.data * b.data
    # - c_prev = referencias a los objetos a y b
    # - c._op = '*'
    # - c._backward = dirección a una función que se usa en 'backward'
    # Esta función hace lo siguiente:
    # - grad de cada multiplicando (a o b) de la operacion =
    #       data del otro multiplicando (a o b) * grad del resultado (c)
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # multiplicación cuando el objeto está en la derecha
    def __rmul__(self, other):
        return self.__mul__(other)

    # define el comportamiento cuando se usa el operador -objeto
    # Multiplica el data del objeto por -1 (se usa en la resta)
    def __neg__(self):
        return self * -1

    # operador resta
    # la resta trabaja como si fuera una suma, con su backward y demás, pero en negativo
    def __sub__(self, other):
        return self + (-other)

    # resta cuando el objeto está a la derecha
    def __rsub__(self, other):
        return other + (-self)

    # Operación potencia **
    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    # Operación división real /
    # La división trabaja como si fuera una multiplicación, con su backward y demás pero en negativo
    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    # Operación relu sobre un objeto Value
    # El data del objeto de salida es 0 o mayor que 0, se eliminan negativos
    # _prev = (self,)
    # _op = 'relu'
    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # La función math.tanh calcula la tangente hiperbólica de un número.
    # Se usa como función de activación en las capas ocultas porque centra las salidas 
    # alrededor del cero (a diferencia de la función sigmoide, que va de 0 a 1), ayudando a 
    # que los modelos converjan más rápido.
    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # Operación exponencial e**x
    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    # Operación logaritmo
    def log(self):
        import math
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # Paso atras
    def backward(self):
        topo = []                       # se crea la lista topo vacia
        visited = set()                 # se crea el set visited vacio
        def build_topo(v):
            if v not in visited:        # si el objeto no está en el set visited
                visited.add(v)          # añadimos el objeto al set visited
                for child in v._prev:   # recorremos todos los hijos (_prev) del objeto
                    build_topo(child)   # por cada objeto hijo ejecutamos build_topo con el
                topo.append(v)          # añadimos el objeto a la lista topo
        build_topo(self)                # Llama al bucle build_top con un objeto Value
        # se crea una lista topo con todos los objetos Value participantes en operaciones
        # relacionadas con el objeto principal hacia atrás
        self.grad = 1.0
        for v in reversed(topo):    # recorrer los objetos en el orden real (el inverso)
            v._backward()           # ejecuta la función que tiene el objeto en _backward

# Tenemos la funcion y= x1*x2 + 1 con los valores x1=2 y x2=3
# La derivada de y con respecto a x1 = dy/dx1 = x2 = 3
# La derivada de y con respecto a x2 = dy/dx2 = x1 = 2
# La secuencia es:
# - primero hacemos a= x1*x2
# - luego b = a + 1
# - por ultimo y = b.relu()
# Esto podría ser que la entrada x1=2 la multiplicamos por el peso x2=3, 
# sumamos el bias = 1 y a todo le aplicamos la función de activación relu
# El resultado de estas operaciones es y=7
# El y.backward() calcula los gradientes hacia atrás (las derivadas)
def demo_basic():
    print("=== Basic: y = relu(x1 * x2 + 1) ===")
    x1 = Value(2.0)
    x2 = Value(3.0)
    a = x1 * x2
    b = a + Value(1.0)
    y = b.relu()
    y.backward()

    print("  x1 = 2.0, x2 = 3.0")
    print(f"  y  = {y.data}")
    print(f"  dy/dx1 = {x1.grad}  (expected 3.0 = x2)")
    print(f"  dy/dx2 = {x2.grad}  (expected 2.0 = x1)")
    assert abs(x1.grad - 3.0) < 1e-6
    assert abs(x2.grad - 2.0) < 1e-6
    print("  PASSED\n")

# Aquí tenemos una potencia y=x**3 con x=2
# La derivada dy/dx = 3x**2 =3*2**2 = 12
# Esta demo solo demuestra que backward() calcula bien el gradiente (derivada)
def demo_power():
    print("=== Potencia: y = x^3, dy/dx at x=2 ===")
    x = Value(2.0)
    y = x ** 3
    y.backward()

    print("  x = 2.0")
    print(f"  y = {y.data}  (expected 8.0)")
    print(f"  dy/dx = {x.grad}  (expected 12.0 = 3*x^2)")
    assert abs(x.grad - 12.0) < 1e-6
    print("  PASSED\n")

# Esta demo solo muestra el calculo de unas derivadas sencillas con backward
def demo_complex():
    print("=== Complex: f = relu(a*b + c) ===")
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    f = (a * b + c).relu()
    f.backward()

    print("  a=2, b=-3, c=10")
    print(f"  f = {f.data}  (expected 4.0)")
    print(f"  df/da = {a.grad}  (expected -3.0 = b)")
    print(f"  df/db = {b.grad}  (expected 2.0 = a)")
    print(f"  df/dc = {c.grad}  (expected 1.0)")
    assert abs(a.grad - (-3.0)) < 1e-6
    assert abs(b.grad - 2.0) < 1e-6
    assert abs(c.grad - 1.0) < 1e-6
    print("  PASSED\n")

# Neurona sencilla
# Entradas x1=3 y x2=2; pesos w1=0,5 y w2=-1,5; bias=0,1
# y = relu(3*0,5 + 2*-1,5 + 0,1) = relu(-1,4)
def demo_neuron():
    print("=== Single neuron: y = relu(w1*x1 + w2*x2 + b) ===")
    w1 = Value(0.5)
    w2 = Value(-1.5)
    x1 = Value(3.0)
    x2 = Value(2.0)
    b = Value(0.1)

    y = (w1 * x1 + w2 * x2 + b).relu()
    y.backward()

    print("  w1=0.5, w2=-1.5, x1=3.0, x2=2.0, b=0.1")
    print(f"  pre_act = {w1.data*x1.data + w2.data*x2.data + b.data}")
    print(f"  y = {y.data}")
    print(f"  dy/dw1 = {w1.grad}")
    print(f"  dy/dw2 = {w2.grad}")
    print(f"  dy/dx1 = {x1.grad}")
    print(f"  dy/dx2 = {x2.grad}")
    print(f"  dy/db  = {b.grad}")

    pre = w1.data * x1.data + w2.data * x2.data + b.data
    if pre > 0:
        assert abs(w1.grad - x1.data) < 1e-6
        assert abs(w2.grad - x2.data) < 1e-6
        assert abs(x1.grad - w1.data) < 1e-6
        assert abs(x2.grad - w2.data) < 1e-6
        assert abs(b.grad - 1.0) < 1e-6
        print("  PASSED (relu active)\n")
    else:
        assert abs(w1.grad) < 1e-6
        assert abs(w2.grad) < 1e-6
        print("  PASSED (relu inactive, all grads zero)\n")

# Clase Neurona
# w (pesos) lista de tantos Values con numeros aleatorios como diga n_inputs 
# Hay un peso por cada entrada, por lo tanto pesos = entradas
# b (bias) es un Value(0)
class Neuron:
    def __init__(self, n_inputs):
        #print(f"Neurona de {n_inputs} Values (entradas que recibe cada neurona)")
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        #print(f"self.w={self.w}")
        self.b = Value(0.0)

    # Al definir __call__ dentro de una clase, el objeto resultante puede 
    # recibir parámetros directamente usando paréntesis
    # Por ejemplo: se puede definir una variable
    #  a = Neuron(2)  
    #  y luego podemos hacer: a(5) que activa este método
    # Esta funcion hace lo siguiente:
    # - multiplica cada uno de los pesos por cada uno de los valores de la lista x
    # - suma estas multiplicaciones y al final le suma el bias
    # Los pesos y el bias son objetos Value, por eso el resultado act también es un Value
    # Al final se devuelve act.tanh() que es un método de Value que es lo que se usa como
    # funcion de activación para capas ocultas
    # Resumiendo: esto calcula tanh(pesos*input + bias), es decir, un pase adelante
    def __call__(self, x):
        #[print(f"Pase adelante: wi={wi} xi={xi}") for wi, xi in zip(self.w, x)]
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

# Clase Capa
# Es una lista que contiene n_outputs Neuronas y cada una con n_input valores
# n_outputs: numero de neuronas de la capa
class Layer:
    def __init__(self, n_inputs, n_outputs):
        #print("capa")
        #[print(f"Neuron n_inputs={n_inputs} n_outputs{x}") for x in range(n_outputs)]
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    # Se recorre todas las neuronas de la capa y por cada una ejecuta el proceso
    # __call__ de Neuron, que lo que hace es un pase adelante de cada neurona
    def __call__(self, x):
        out = [n(x) for n in self.neurons]  # aquí llama al __call__ de Neuron con n(x)
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

# Clase MLP: Perceptrón Multicapa (Multi-Layer Perceptron)
# Crea una red neuronal que tiene varias capas
class MLP:
    def __init__(self, sizes):
        #[print(x) for x in range(len(sizes) - 1)]
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    # Recorre las capas de la red neuronal y por cada una ejecuta el proceso
    # __call__ de Layer, que hace un pase adelante en la capa
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
"""
Ejemplo MLP(2,4,1)
- 2 entradas, 4 neuronas ocultas, 1 salida
- significa: 2 entradas, una capa oculta de 4 neuronas y una salida de 1 neurona)
- las parejas de tamaños entre capas (llamadas a Layer) son (2, 4) y (4, 1).
- La primera capa recibe 2 entradas y tiene 4 neuronas (Layer(2, 4)). 
  (como tiene 2 entradas, cada neurona tiene 2 values(pesos)
- La segunda capa recibe 4 entradas y tiene 1 neurona (Layer(4, 1)).
1 capa con 4 neuronas de 2 values (2 values = 2 pesos = 2 entradas, por tanto hay 2 entradas y 4 neuronas ocultas)
- la capa tiene 4 neuronas trabajando en paralelo que reciben las mismas entradas
1 capa con 1 neurona de 4 values (4 pesos que reciben las entradas)
- la ultima capa devuelve la salida

Entrada Bruta: [ x1, x2 ]  (2 valores)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Capa 1: Layer(in=2, out=4)                           │
│ Contiene 4 neuronas en paralelo. Cada una recibe       │
│ las 2 entradas y calcula su propia activación.         │
│                                                        │
│ ├─ Neurona 1 ──► (w1·x1 + w2·x2 + b) ──► salida o1     │
│ ├─ Neurona 2 ──► (w3·x1 + w4·x2 + b) ──► salida o2     │
│ └─ Neurona 3 ──► (w5·x1 + w6·x2 + b) ──► salida o3     │
│ └─ Neurona 4 ──► (w7·x1 + w8·x2 + b) ──► salida o4     │
└────────────────────────────────────────────────────────┘
       │
       ▼  (La salida de la Capa 1 pasa a ser la entrada de la Capa 2)
   [ o1, o2, o3, o4 ]  (4 valores intermedios)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Capa 2 (Salida): Layer(in=4, out=1)                    │
│ Contiene 1 sola neurona. Recibe los 4 valores          │
│ anteriores como sus nuevas entradas.                   │
│                                                        │
│ └─ Neurona 1 ──►                                       |
|       (w9·o1 + w10·o2 + w11·o3 + w12·o4 + b) ──► y     │
└────────────────────────────────────────────────────────┘
       │
       ▼
Salida Final de la Red: y  (1 único valor, ej. la predicción)
(en realidad la salida son los 4 valores de la neurona según demo_mlp_training() )

Se calcula la pérdida con la fórmula del error cuadratico medio MSE (predicción - real)^2

Descenso de gradiente:
- Después, con backward() se calculan los gradientes hacia atrás
- Y una vez calculados, se reajustan los pesos de las neuronas
- cada peso se modifica con la fórmula nuevo peso = peso actual - lr*gradiente
"""

# Comprobar un gradiente, lo calcula por el método numérico y por el de la clase Value
# Recibe una funcion y un valor
# devuelve los dos gradientes y su diferencia
def gradient_check(build_expr, x_val, h=1e-7):
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff

# Demo de entrenamiento de MLP: Perceptrón Multicapa (Multi-Layer Perceptron)
# La red neuronal que se crea está explicada un poco mas arriba
# - 2 entradas, 1 capa con 4 neuronas cuyas salidas pasan a otra capa con 4 neuronas y 1 salida
# Hay un bucle que se ejecuta 100 veces
# - xs son las entradas, se hacen pasar por la red neuronal en un pase adelante
# - la salida es preds (predicciones), los 4 valores de la capa de salida (1 neurona)
# - se calcula la pérdida loss (diferencia entre predicción y real al cuadrado)
# - se calculan los gradientes hacia atrás con loss.backward()
# - se reajustan los pesos con la fórmula peso nuevo = peso actual - lr * gradiente
def demo_mlp_training():
    print("=== Mini MLP Training on XOR ===")
    random.seed(42)
    model = MLP([2, 4, 1])  # 2 inputs, 4 hidden neurons, 1 output

    # xs es una lista de 4 tuplas de 2 Values: 2 entradas para cada una de las 4 neuronas
    xs = [[Value(0), Value(0)], [Value(0), Value(1)],
          [Value(1), Value(0)], [Value(1), Value(1)]]
    # Esta es la salida prevista: 4 valores de las 4 neuronas de la capa final
    ys = [-1.0, 1.0, 1.0, -1.0] # XOR pattern (using -1/1 for tanh)

    for step in range(100):
        # Esto ejecuta un pase adelante
        preds = [model(x) for x in xs]  # esto llama al __call__ de MLP y le pasa cada xs
        # Calcula la pérdida (diferencia entre lo calculado y lo previsto al cuadrado)
        #loss = sum((p + Value(-y)) ** 2 for p, y in zip(preds, ys)) # linea original
        loss: Value = sum(          
            ((p + Value(-y)) ** 2 for p, y in zip(preds, ys)),
            start=Value(0.0),
        )
        for p in model.parameters():    # Esto parece que pone a 0 los gradientes
            p.grad = 0.0
        loss.backward()                 # Calcula todos los gradientes

        lr = 0.05                       # learning rate = tasa de aprendizaje

        for p in model.parameters():    # actualizar todos los pesos con la fórmula
            p.data -= lr * p.grad       # nuevo peso = peso actual - lr * gradiente

        if step % 20 == 0 or step == 99:
            print(f"  paso {step:3d}  pérdida = {loss.data:.4f}")

    print("\n  Predicciones después del entrenamiento:")
    for x, y in zip(xs, ys):
        pred = model(x)
        sign = "+" if pred.data > 0 else "-"
        print(f"    entrada=[{x[0].data:.0f},{x[1].data:.0f}]  objetivo={y:+.0f}  pred={pred.data:+.3f} ({sign})")
    print("  DONE\n")

# Demo de gradiente check, la función que verifica las derivadas (gradientes)
# La función calcula la derivada por el método numérico y por el método Value
# Saca si hay diferencias o no 
def demo_gradient_check():
    print("=== Gradient Checking ===")

    expressions = [
        ("x^3 + 2x + 1",       lambda x: x ** 3 + x * 2 + 1),
        ("tanh(x^2)",           lambda x: (x ** 2).tanh()),
        ("(x+1) / (x^2+1)",    lambda x: (x + 1) * ((x ** 2 + 1) ** -1)),
        ("exp(x) * x",         lambda x: x.exp() * x),
        ("log(x^2 + 1)",       lambda x: (x ** 2 + 1).log()),
    ]

    print(f"  {'Expression':<22} {'Autodiff':>12} {'Numerical':>12} {'Diff':>12}")
    print("  " + "-" * 60)

    all_passed = True
    for name, expr in expressions:
        ad, num, diff = gradient_check(expr, 0.5)
        status = "OK" if diff < 1e-5 else "FAIL"
        if diff >= 1e-5:
            all_passed = False
        print(f"  {name:<22} {ad:12.8f} {num:12.8f} {diff:12.2e}  {status}")

    if all_passed:
        print("  ALL CHECKS PASSED\n")
    else:
        print("  SOME CHECKS FAILED\n")

# Demo de derivada exponencial y logaritmica
# Calcula ambas derivadas por el método numérico y por el método Value y compara
def demo_exp_log():
    print("=== Exp and Log operations ===")
    x = Value(2.0)
    y = x.exp()
    y.backward()
    import math
    print(f"  exp(2.0) = {y.data:.4f}  (expected {math.exp(2.0):.4f})")
    print(f"  d/dx exp(x) at x=2 = {x.grad:.4f}  (expected {math.exp(2.0):.4f})")
    assert abs(x.grad - math.exp(2.0)) < 1e-4
    print("  PASSED\n")

    x = Value(3.0)
    y = x.log()
    y.backward()
    print(f"  log(3.0) = {y.data:.4f}  (expected {math.log(3.0):.4f})")
    print(f"  d/dx log(x) at x=3 = {x.grad:.4f}  (expected {1/3:.4f})")
    assert abs(x.grad - 1.0 / 3.0) < 1e-4
    print("  PASSED\n")

# Verificación de calculos de derivada con Value contra Pytorch
def demo_verify_pytorch():
    print("=== Verify against PyTorch ===")
    try:
        import torch
    except ImportError:
        print("  PyTorch not installed, skipping verification.\n")
        return

    x1_v = Value(2.0)
    x2_v = Value(3.0)
    y_v = (x1_v * x2_v + 1.0).relu()
    y_v.backward()

    x1_t = torch.tensor(2.0, requires_grad=True)
    x2_t = torch.tensor(3.0, requires_grad=True)
    y_t = torch.relu(x1_t * x2_t + 1.0)
    y_t.backward()

    print(f"  Our engine: dy/dx1={x1_v.grad}, dy/dx2={x2_v.grad}")
    print(f"  PyTorch:    dy/dx1={x1_t.grad.item()}, dy/dx2={x2_t.grad.item()}")
    assert abs(x1_v.grad - x1_t.grad.item()) < 1e-6
    assert abs(x2_v.grad - x2_t.grad.item()) < 1e-6
    print("  MATCH\n")


if __name__ == "__main__":
    demo_basic()
    demo_power()
    demo_complex()
    demo_neuron()
    demo_exp_log()
    demo_gradient_check()
    demo_mlp_training()
    demo_verify_pytorch()
    print("All demos passed.")
