import taichi as ti
ti.init(arch=ti.cpu)
import problema1

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 100, 0]

window = ti.ui.Window("Bola 3D", (1024, 1024))
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = window.get_scene()

camera = ti.ui.Camera()
camera.position(35, 60, 120)
camera.lookat(0.0, 60, 0)
scene.set_camera(camera)

initialized= False

while window.running:
    if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == "r":
                problema1.reset()

    problema1.step()

    ball_center[0] = problema1.x_prev[None]
    
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.particles(ball_center, radius=ball_radius, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()