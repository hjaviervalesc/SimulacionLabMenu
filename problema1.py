import taichi as ti
ti.init(arch=ti.cpu)


g = ti.Vector([0.0, -10.0, 0.0])


v_prev = ti.Vector.field(3, dtype=ti.f32, shape=())
x_prev = ti.Vector.field(3, dtype=ti.f32, shape=())

v_prev[None] = ti.Vector([10.0, 0.0, 30.0])
x_prev[None] = ti.Vector([0.0, 100.0, 0.0])

total_time = 5
dt = 0.03
t = 0.0



@ti.func
def v_func(v):
    return v + g * dt

@ti.func
def x_func(x, v):
    return x + v * dt

@ti.kernel
def step():
    v_helper = v_func(v_prev[None])
    x_helper = x_func(x_prev[None], v_prev[None])

    v_prev[None] = v_helper
    x_prev[None] = x_helper

def reset():
    v_prev[None] = ti.Vector([10.0, 0.0, 30.0])
    x_prev[None] = ti.Vector([0.0, 100.0, 0.0])
    t = 0.0

if __name__ == "__main__":
    print(f"t: {t:.2f} s,\t v: {v_prev[None]} m/s\t x: {x_prev[None]} m")

    while t < total_time:
        t += dt
        step()  
        print(f"t: {t:.2f} s,\t v: {v_prev[None]} m/s\t x: {x_prev[None]} m")
