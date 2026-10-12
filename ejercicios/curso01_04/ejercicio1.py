import derivatives as d

# Implementar 'numerical_second_derivative(f, x)' llamando dos veces a 
# 'numerical_derivative'. Verificar que la segunda derivada de x^3 en x=2 sea 12.

# Esta función me la implementó la IA porque a mi no me salia
# Tiene una funcion lambda dentro de otra funcion
# h=1e-5 en vez de 1e-7 porque si no la derivada da 11,92 en vez de 12
def numerical_second_derivative(f, x, h=1e-5):
    # Definimos la primera derivada como una función que depende de x
    f_prime = lambda val: d.numerical_derivative(f, val, h)
    
    # Calculamos la derivada de esa nueva función f_prime
    return d.numerical_derivative(f_prime, x, h)

def f(x):
    return x**3

x=2
segunda_derivada = numerical_second_derivative(f,x)
print(f"Derivada de x al cubo en el punto x={x} = {round(segunda_derivada,4)}")
