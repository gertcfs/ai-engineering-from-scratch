import vectors as v

a = v.Vector([1, 2, 3])
b = v.Vector([4, 5, 6])
c = v.Vector([7, 8, 9])
print(f"a = {a}")
print(f"components = {a.components}")
print(f"dim = {a.dim}")

result = v.is_independent([a, b, c])
print(f"Is independent: {result}")

print("Gram-Schmidt process:")
orthonormal_vectors = v.gram_schmidt([a, b, c])
for i, vec in enumerate(orthonormal_vectors):
    print(f"Orthonormal vector {i}: {vec}") 

m = v.Matrix([[4,5,7], [4,5,6], [0,0,1]])
print(f"Matrix m:\n{m} rango = {m.rank()}")

d = v.Vector([1, 2, 3])
e = v.Vector([1, 1, 1])
proj = d.project_onto(e)
print(f"Proyección de {d} sobre {e} = {proj}")
