import matrices as m

# Verifica la inversa. Multiplica A @ A.inverse_2x2()y confirma que obtienes la matriz 
# identidad. Inténtalo con tres matrices 2x2 diferentes. 
# ¿Qué sucede cuando el determinante es cero?

# Matriz |1 2|         
#.       |3 4|
m1 = m.Matrix( [ [1,2], [3,4] ] )

# Matriz |6 7|         
#.       |8 9|
m2 = m.Matrix( [ [6,7], [8,9] ] )

# Matriz |0 1|         
#.       |3 4|
m3 = m.Matrix( [ [0,1], [3,4] ] )

print(f"Matriz m1: {m1}")
m1_det = m1.determinant()
print(f"Determinante de m1: {m1_det}")
if m1_det:
    m1_inv = m1.inverse_2x2()
    print(f"Inversa de m1: {m1_inv}")
    print(f"Prueba m1 @ m1^-1 = {m1 @ m1_inv}")
else:
    print("Determinante 0, no existe la matriz inversa")

print(f"Matriz m2: {m2}")
m2_det = m2.determinant()
print(f"Determinante de m2: {m2_det}")
if m2_det:
    m2_inv = m2.inverse_2x2()   
    print(f"Inversa de m2: {m2_inv}")
    print(f"Prueba m2 @ m2^-1 = {m2 @ m2_inv}")
else:
    print("Determinante 0, no existe la matriz inversa")

print(f"Matriz m3: {m3}")
m3_det = m3.determinant()
print(f"Determinante de m3: {m3_det}")
if m3_det:
    m3_inv = m3.inverse_2x2()
    print(f"Inversa de m3: {m3_inv}")
    print(f"Prueba m3 @ m3^-1 = {m3 @ m3_inv}")
else:
    print("Determinante 0, no existe la matriz inversa")
