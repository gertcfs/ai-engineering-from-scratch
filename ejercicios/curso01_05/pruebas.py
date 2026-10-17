import autodiff as a


class Value:
    def __init__(self, data, children=(), op='', nom=''):
        self.data = float(data)
        self.grad = 0.0
        self.nom = nom
        self._backward = lambda: None
        self._prev = set(children)  # set convierte children en un set de datos (elimina duplicados)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f}, nom={self.nom})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            print(f"out.grad={out.grad} self.grad={self.grad} other.grad={other.grad}")
            self.grad += out.grad
            other.grad += out.grad
            print("pasa add")
            print(f"out.grad={out.grad} self.grad={self.grad} other.grad={other.grad}")
            print(f"out.data={out.data} self.data={self.data} other.data={other.data}")
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            print(f"out.grad={out.grad} self.grad={self.grad} other.grad={other.grad}")
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            print("pasa mul")
            print(f"out.grad={out.grad} self.grad={self.grad} other.grad={other.grad}")
            print(f"out.data={out.data} self.data={self.data} other.data={other.data}")
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    # Operación potencia **
    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            print(f"out.grad={out.grad} self.grad={self.grad} ")
            self.grad += n * (self.data ** (n - 1)) * out.grad
            print("pasa pow")
            print(f"out.grad={out.grad} self.grad={self.grad} ")
            print(f"out.data={out.data} self.data={self.data} ")
        out._backward = _backward
        return out
    
    def nombre(self, nom):
        self.nom = nom

    # Ejemplo: 
    # - tenemos a, b, c, d con valores 3, 4, 5, 6 el resto de campos sin info
    # - e = a + b
    #   - se crea e con valor 7, _prev= ref-a, ref-b, op=+, _backguard=_backward-add
    # - f = e * c
    #   - se crea f con valor 7*5=35, _prev=ref-e, ref-c, op=*. backguard=_backward-mul
    # Hacemos backward(f)
    # build_topo(f) (añade f a topo)
    # -visited.add(f)
    # - recorrer f._prev
    #   - build_topo(e) (añade e a topo)
    #     - recorrer e._prev
    #       - build_topo(a) (añade 'a' a topo)
    #         - recorrer a.prev
    #         - vacio; seguimos por e._prev
    #       - build_topo(b) (añade b a topo)
    #         - recorrer b._prev
    #         - vacio; seguimos por e.prev
    #     - vacio; seguimos por f._prev
    #   - build_topo(c) (añade c a topo)
    #     - recorrer c._prev
    #     - vacio, seguimos por f.prev
    # - vacio
    # - f.grad = 1
    # - recorrer topo en inversa (f,e,b,a,c)
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
            print(f"Hago backward de: {v.nom}")
            v._backward()           # ejecuta la función que tiene el objeto en _backward

def prueba1():
    va = Value(3, nom='a')
    vb = Value(4, nom='b')
    vc = Value(5, nom='c')
    ve = va + vb            # = 7
    ve.nombre('e')
    vf = ve * vc            # = 35
    vf.nombre('f')
    vg = vf**2
    vg.nombre('g')
    vg.backward()

"""
Displays:
Hago backward de: g                 (g=f**2 ; g.data=1225 f.data=35)
out.grad=1.0 self.grad=0.0          (se asigna 1 a la derivada de la ultima operacion, grad=1)
pasa pow                            (g se originó con una operacion potencia)
out.grad=1.0 self.grad=70.0         (g.grad=1, calcula derivada de y=x**2=2x = f.grad=35*2=70)
out.data=1225.0 self.data=35.0      (g.data=1225 f.data=35)
-------------------
Hago backward de: f
out.grad=70.0 self.grad=0.0 other.grad=0.0
pasa mul                           
out.grad=70.0 self.grad=350.0 other.grad=490.0  
out.data=35.0 self.data=7.0 other.data=5.0      
    (f.data=35, e.data=7, c.data=5)
    (operacion f=e*c)(f.data=35, e.data=7, c.data=5)
    (derivada de y=x*z, con respecto a x = z y con respecto a z = x)
    (f.grad=70 ; calculo derivadas => e.grad = 70*5 = 350 ; c.grad = 70*7 = 490)
-------------------
Hago backward de: c
Hago backward de: e
out.grad=350.0 self.grad=0.0 other.grad=0.0
pasa add
out.grad=350.0 self.grad=350.0 other.grad=350.0
out.data=7.0 self.data=3.0 other.data=4.0
    (e.data = 7, a.data = 3, b.data = 4)
    (operacion e = a + b => 7 = 3 + 4)
    (derivada de y=x+z, con respecto a x = 1, con respecto a z = 1)
    (e.grad=350 ; calculo derivadas => a.grad = 350 , b.grad = 350)
Hago backward de: a
Hago backward de: b
"""

# prueba1()

def prueba2():
    w = [1,2]
    x = [3,4]
    b = 3
    # Esto hace 1*3 + 2*4 + 3 = 14
    # zip(w,x) devuelve los numero en pares primero 1 y 3 y luego 2 y 4
    # por cada par que devuelve, lo cogen wi y xi y se multiplican
    # y luego el primer par se suma al segundo par 
    # y quedaría sum(11, 3) que nos da = 14 
    act = sum((wi * xi for wi, xi in zip(w, x)), b) # = 14
    print(act)
    n_inputs=3
    n_outputs=4
    # crear una lista que contiene n_output Neuronas y cada una con n_input valores
    neurons = [a.Neuron(n_inputs) for _ in range(n_outputs)]
    for x in neurons:
        print(x.w)

def prueba3():
    # mlp = a.MLP([1,2])  # crea 1 capa con 2 neuronas de 1 value
    # mlp = a.MLP([1,3])  # crea 1 capa con 3 neuronas de 1 value
    # mlp = a.MLP([1,2,3])    # crea 1 capa con 2 neuronas de 1v + 1capa 3 neuronas 2v
    # mlp = a.MLP([1,2,3,4])  # 1 capa 2 neuronas de 1v + 1 capa 3n de 2v + 1 capa 4n de 3v
    # mlp = a.MLP([3,2,1])    # crea 1 capa con 2 neuronas de 3v + 1 capa con 1 neurona de 2v
    mlp = a.MLP([2,4,1])    # 1 capa con 4 neuronas de 2v + 1 capa con 1 neur de 4v
    # 2 entradas, 4 neuronas ocultas, 1 salida
    # self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    #  capas =[ Layer(3, 4) for i en el rango 2]
    # hace esto 2 veces: Layer(0,1) y Layer(1,2)
    """
    for x in mlp.layers:
        print(f"Capa {x.neurons}")
        for y in x.neurons:
            print(y.w)
    """

# a.demo_mlp_training()
# a.demo_gradient_check()

# Ejercicio 1
# y = x^3 => derivada 3x^2 => sustituyendo x=2 => 3*2^2=12
x = a.Value(2)
y=x**3
y.backward()
print(x.grad)

# Ejercicio 3
w1 = a.Value(1.0)
w2 = a.Value(1.1)
k1 = (w1*2 + w2*3 + 1.0)
k2 = k1.relu()
k2.backward()
print(f"gradiente w1 = {w1.grad}")
print(f"gradiente w2 = {w2.grad}")
print(f"gradiente k1 = {k1.grad}")
print(f"gradiente k2 = {k2.grad}")

import torch

w1t = torch.tensor(1.0, requires_grad=True)
w2t = torch.tensor(1.1, requires_grad=True)
k1t = (w1t*2 + w2t*3 + 1)
k2t = torch.relu(k1t)
k2t.backward()
print(f"gradiente w1t = {w1t.grad.item()}")
print(f"gradiente w2t = {w2t.grad.item()}")
# print(f"gradiente k1t = {k1t.grad.item()}")
# print(f"gradiente k2t = {k2t.grad.item()}")
