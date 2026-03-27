import taichi as ti
import numpy as np
ti.init(arch=ti.cpu)


g = ti.Vector([0.0, -10.0, 0.0])
a_helper = g


v_prev = ti.Vector.field(3, dtype=ti.f32, shape=())
x_prev = ti.Vector.field(3, dtype=ti.f32, shape=())

v_prev[None] = ti.Vector([10.0, 0.0, 30.0])
x_prev[None] = ti.Vector([0.0, 100.0, 0.0])


m = 1 #Masa
d = 0.4 #Coeficiente de resistencia al aire
total_time = 5
dt = 1
t = 0.0



@ti.func
def v_func(v, a):
    return v + a * dt

@ti.func
def x_func(x, v):
    return x + v * dt

@ti.func
def g_func(v):
    a = g- ((d/m) * v)
    return a

@ti.kernel
def step():
    a_helper = g_func(v_prev[None])
    v_helper = v_func(v_prev[None], a_helper)
    x_helper = x_func(x_prev[None], v_prev[None])
    
    v_prev[None] = v_helper
    x_prev[None] = x_helper

def reset():
    v_prev[None] = ti.Vector([10.0, 0.0, 30.0])
    x_prev[None] = ti.Vector([0.0, 100.0, 0.0])
    t = 0.0

if __name__ == "__main__":
    print(f"t: {t:.2f} s,\t v: {np.round(v_prev[None],1)} m/s\t x: {np.round(x_prev[None],1)} m")

    while t < total_time:
        t += dt
        step()  
        print(f"t: {t:.2f} s,\t v: {np.round(v_prev[None],1)} m/s\t x: {np.round(x_prev[None],1)} m")

