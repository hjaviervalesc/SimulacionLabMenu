import taichi as ti

#Constantes físicas
g = ti.Vector([0.0, -10.0, 0.0])
v_viento = ti.Vector([-12.5, 0.0, 0.0])
m = 1.0
d = 0.4

#Parámetros
N = 100
dt = 0.01

#Campos
v_prev = ti.Vector.field(3, dtype=ti.f32, shape=N)
x_prev = ti.Vector.field(3, dtype=ti.f32, shape=N)

#Inicialización
@ti.kernel
def init():
    for i in range(N):
        x_prev[i] = ti.Vector([0.0, 100.0, 0.0])

        v_prev[i] = ti.Vector([
            ti.random() * 100 - 50,
            ti.random() * 100,
            ti.random() * 100 - 50
        ])

#Física
@ti.func
def a_func(v):
    return g - (d/m) * (v - v_viento)

#Step Euler
@ti.kernel
def step():
    for i in range(N):
        a = a_func(v_prev[i])

        v_new = v_prev[i] + a * dt
        x_new = x_prev[i] + v_new * dt

        # suelo
        if x_new[1] < 0:
            x_new[1] = 0
            v_new[1] *= -0.6

        v_prev[i] = v_new
        x_prev[i] = x_new

#Reset
def reset():
    init()
