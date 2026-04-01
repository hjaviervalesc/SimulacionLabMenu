import taichi as ti
ti.init(arch=ti.cpu)

#Constantes físicas
g = ti.Vector([0.0, -10.0, 0.0])
v_viento = ti.Vector([-12.5, 0.0, 0.0])
m = 1.0
d = 0.4

#Parámetros de la simulación
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
            ti.random() * 100 - 50,   # x [-50, 50]
            ti.random() * 100,        # y [0, 100]
            ti.random() * 100 - 50    # z [-50, 50]
        ])

#Física
@ti.func
def a_func(v):
    return g - (d/m) * (v - v_viento)

@ti.func
def v_func(v, a):
    return v + a * dt

@ti.func
def x_func(x, v):
    return x + v * dt

# Step Euler
@ti.kernel
def step():
    for i in range(N):
        a = a_func(v_prev[i])

        # Symplectic Euler
        v_new = v_func(v_prev[i], a)
        x_new = x_func(x_prev[i], v_new)

        # suelo
        if x_new[1] < 0:
            x_new[1] = 0
            v_new[1] *= -0.6

        v_prev[i] = v_new
        x_prev[i] = x_new

#VISUALIZACIÓN
if __name__ == "__main__":
    init()

    window = ti.ui.Window("Sistema de Partículas", (1024, 1024))
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = window.get_scene()

    camera = ti.ui.Camera()
    #cámara
    camera.position(0.0, 100.0, 400.0)
    camera.lookat(0.0, 100.0, 0.0)
    scene.set_camera(camera)

    while window.running:
        step()

        scene.point_light(pos=(0, 100, 200), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        scene.particles(x_prev, radius=0.9, color=(0.5, 0.42, 0.8))

        canvas.scene(scene)
        window.show()
