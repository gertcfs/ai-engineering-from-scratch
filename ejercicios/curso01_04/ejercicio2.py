import derivatives as d

# Utilizar el método de descenso de gradiente para encontrar el mínimo de 
# f(x, y) = (x - 3)^2 + (y + 1)^2. 
# Comenzar desde (0, 0). La solución debe converger a (3, -1).

# Se trata de implementar una función similar a demo_gradient_descent
# He copiado la función de derivatives.py y solo voy a cambiar la función f
# y el point inicial que será el [0,0]
# Todo lo demás es igual excepto que he metido muchos comentarios y prints
def demo_gradient_descent_2d():
    print("\n" + "=" * 55)
    print("DESCENSO DE GRADIENTE: f(x, y) = (x - 3)^2 + (y + 1)^2")
    print("=" * 55)
    def f(point):
        x, y = point
        return (x-3)**2 + (y+1)**2              # x**2 + y**2
    point = [0, 0]                              # [4.0, 3.0]  # x=4 y=3
    lr = 0.1            # lr=learning rate=tasa de aprendizaje)
    print(f"\nComienzo: ({point[0]}, {point[1]}), lr={lr}")
    for step in range(30):
        print(f"Entra {step}, point actual = ({point[0]:5.2f}, {point[1]:5.2f})", end=" ")
        # Gradiente numérico de la funcion, devuelve un vector con 2 derivadas
        # parciales, la derivada respecto a x y la derivada respecto a y
        grad = d.numerical_gradient(f, point)
        print(f"grad=({grad[0]:5.2f}, {grad[1]:5.2f})")
        # Calcula el nuevo point = point anterior - (lr * gradiente)1)
        # En la 1a iteración esta instrucción, ejecuta dos operaciones, 
        # - dado que point y grad son vectores 2D que valen [4,3] y [8,6]
        # - en la primera p=4 y g=8 => point[0] = 4 - (0.1 * 8) = 4 * 0,8 = 3,2
        # - en la segunda p=3 y g=6 => point[1] = 3 - (0,1 * 6) = 3 * 0,6 = 2,4
        # - el resultado es que el nuevo 'point' vale [3.2, 2.4]
        point = [p - lr * g for p, g in zip(point, grad)]
        loss = f(point)
        #print(f"Nuevo point={point} pérdida={loss}")
        #if step % 5 == 0 or step == 29:
        print(f"  step {step:2d}  ({point[0]:7.4f}, {point[1]:7.4f})  pérdida f={loss:.6f}")
    print(f"Minimo encontrado en ({point[0]:.4f}, {point[1]:.4f}) (true: (0, 0))")

demo_gradient_descent_2d()

