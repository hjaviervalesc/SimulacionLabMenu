import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

N = 1000
dt = 0.01

g = np.array([0.0, -10.0, 0.0])
d = 0.4
m = 1.0
v_wind = np.array([-12.5, 0.0, 0.0])

# Campos Taichi
ball_radius = 0.1
positions = ti.Vector.field(3, dtype=float, shape=(N,))
velocities = ti.Vector.field(3, dtype=float, shape=(N,))

# Inicializa
for i in range(N):
    positions[i] = [0, 100, 0]

    vx = np.random.uniform(-50, 50)
    vy = np.random.uniform(0, 100)
    vz = np.random.uniform(-50, 50)

    velocities[i] = [vx, vy, vz]

# la ventanna
window = ti.ui.Window("Sistema de Partículas", (1024, 1024))
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = window.get_scene()

camera = ti.ui.Camera()
camera.position(0.0, 50.0, 200)
camera.lookat(0.0, 50.0, 0)
scene.set_camera(camera)

# el loop
while window.running:
    for i in range(N):
        v = velocities[i].to_numpy()
        x = positions[i].to_numpy()

        # aceleración
        a = g - (d/m) * (v - v_wind)

        # euler
        v = v + a * dt
        x = x + v * dt

        # para suelo
        if x[1] < 0:
            x[1] = 0
            v[1] *= -0.6

        velocities[i] = v
        positions[i] = x

    scene.point_light(pos=(0, 100, 200), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.particles(positions, radius=ball_radius, color=(0.5, 0.42, 0.8))

    canvas.scene(scene)
    window.show()
