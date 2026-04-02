import taichi as ti
ti.init(arch=ti.cpu)
import problema4

N = problema4.N

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(N,))

window = ti.ui.Window("Explosion de Partículas", (1024, 1024))
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = window.get_scene()

camera = ti.ui.Camera()
camera.position(0.0, 100.0, 300.0)
camera.lookat(0.0, 100.0, 0.0)
scene.set_camera(camera)

problema4.init()

while window.running:

    if window.get_event(ti.ui.PRESS):
        if window.event.key == "r":
            problema4.reset()

    problema4.step()

    for i in range(N):
        ball_center[i] = problema4.x_prev[i]

    scene.point_light(pos=(0, 100, 200), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

    scene.particles(ball_center, radius=ball_radius, color=(0.5, 0.42, 0.8))

    canvas.scene(scene)
    window.show()
