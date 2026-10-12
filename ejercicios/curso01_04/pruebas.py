import derivatives as d


# Esta función recibe un punto [x,y] y le pasa la funcion polinómica x^2+3xy+y^2
# Esta función polinómica tiene dos variables, x e y, luego es 2D
# - suponiendo el punto [1,2], esto devuelve 1+6+4=11
# Si calculamos las derivadas parciales
# - derivada de esta funcion con respecto a x = 2x+3y, sustituyendo=2+6 = 8
# - derivada con respecto a y = 3x+2y, sustituyendo=3+4 = 7
def f(point):
    x, y = point
    return x**2 + 3*x*y + y**2

point = [1.0, 2.0]
print(d.numerical_gradient(f , point))

# Calculo que hace la funcion numerical_gradient
# La funcion polinómica f tiene 2 variables x, y (es 2D); el punto es el [1,2]
# La derivada respecto a x es 8 y respecto a y es 7 como se ve mas arriba
# Aquí se calculan las derivadas parciales, con respecto a x=8, con respecto a y=7
# El bucle hace tantas pasadas como cantidad de números tiene el punto (dos)
# - Primera pasada con el primer numero (x) y segunda pasada con el 2º (y)
#   - Calcula la derivada parcial (campo partial)
#       - se copia el punto de entrada (point) a point_plus y point_minus
#       - se hace point_plus[0 o 1]=x + h    point_minus[0 o 1]=x - h
#       - se restan ambos resultados, se pasan por la función f y se divide por 2h
#       - el resultado es la derivada parcial (partial)
#       - la derivada parcial se añade al vector gradient
h=1e-7
gradient = []
for i in range(len(point)):         # recibe el punto [1,2] len=2
    point_plus = list(point)        # point_plus = [1,2] las 2 veces
    point_minus = list(point)       # point_minus = [1,2] las 2 veces
    point_plus[i] += h              # point_plus = [1.0000001, 2] y [1.0, 2.0000001]
    print(f"{i} point_plus: {point_plus}")
    point_minus[i] -= h             # point_minus = [0.9999999, 2] y [1.0, 1.9999999]
    print(f"{i} point_minus: {point_minus}")
    partial = (f(point_plus) - f(point_minus)) / (2 * h)    # = 7.9999 y 6.9999
    gradient.append(partial)
    print(f"{i} gradient: {gradient}")