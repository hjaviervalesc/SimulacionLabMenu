import taichi as ti
ti.init(arch=ti.cpu)

#constantes 
g = ti.Vector([0.0, -10.0, 0.0])
v_viento = ti.Vector([-12.5, 0.0, 0.0])
m=1
d=0.4

#tiempos
total_time = 5.0
dt = 0.03
t = 0.0

#campos
v_prev = ti.Vector.field(3, dtype=ti.f32, shape=())
x_prev = ti.Vector.field(3, dtype=ti.f32, shape=())

#condiciones iniciales
v_prev[None] = ti.Vector([10.0, 0.0, 30.0])
x_prev[None] = ti.Vector([0.0, 100.0, 0.0])


@ti.func
def a_func(v, v_viento):
    return g - d/m * (v - v_viento)

@ti.func
def v_func(v, a):
    return v + a * dt

@ti.func
def x_func(x, v):
    return x + v * dt

@ti.kernel
def step():
    #calcular aceleración
    a = a_func(v_prev[None], v_viento)

    #actualizar velocidad y posición
    v_prev[None] += a * dt
    x_prev[None] += v_prev[None] * dt


def reset():
    v_prev[None] = ti.Vector([10.0, 0.0, 30.0])
    x_prev[None] = ti.Vector([0.0, 100.0, 0.0])

if __name__ == "__main__":
    print(f"t: {t:.2f} s,\t v: {v_prev[None]} m/s\t x: {x_prev[None]} m")

    while t < total_time:
        t += dt
        step()
        print(f"t: {t:.2f} s,\t v: {v_prev[None]} m/s\t x: {x_prev[None]} m")

    

