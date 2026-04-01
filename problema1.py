import taichi as ti
ti.init(arch=ti.cpu)

#Por la segunda ley de Newton, la aceleración de la bola es a = F/m = −g, que es independiente de la masa de la bola.
g = ti.Vector([0.0, -10.0, 0.0]) #gravedad

v_prev = ti.Vector.field(3, dtype=ti.f32, shape=()) #vector de velocidad
x_prev = ti.Vector.field(3, dtype=ti.f32, shape=()) #vector de posición

#La bola se deja caer desde x0 = [0,100,0] con velocidad inicial v0 = [10,0,30]
v_prev[None] = ti.Vector([10.0, 0.0, 30.0]) #vector para guardar la velocidad anterior
x_prev[None] = ti.Vector([0.0, 100.0, 0.0]) #vector para guardar la posición anterior

total_time = 5 
dt = 0.03
t = 0.0


@ti.func
def v_func(v):
    '''
    Función para calcular la velocidad actual utilizando Forward Euler
    '''
    return v + g * dt

@ti.func
def x_func(x, v):
    '''
    Función para calcular la posición actual utilizando Forward Euler
    '''
    return x + v * dt

@ti.kernel
def step():
    '''
    Función step para calcular la velocidad y posición actuales.
    En cada paso, avanzamos la velocidad y la posición usando aproximaciones de las integrales.
    '''
    v_helper = v_func(v_prev[None])
    x_helper = x_func(x_prev[None], v_prev[None])

    #Guardar los valores actuales para la siguiente iteración
    v_prev[None] = v_helper
    x_prev[None] = x_helper

def reset():
    '''
    Función para reiniciar la simulación
    '''
    v_prev[None] = ti.Vector([10.0, 0.0, 30.0])
    x_prev[None] = ti.Vector([0.0, 100.0, 0.0])
    t = 0.0

if __name__ == "__main__":
    print(f"t: {t:.2f} s,\t v: {v_prev[None]} m/s\t x: {x_prev[None]} m")

    while t < total_time:
        t += dt
        step()  
        print(f"t: {t:.2f} s,\t v: {v_prev[None]} m/s\t x: {x_prev[None]} m")
