import taichi as ti
ti.init(arch=ti.cpu)

# IDEA DEL EJERCICIO
# En este problema se extiende el modelo anterior (Problema 3)
# a un sistema de muchas partículas (1000 bolas).
# Todas las partículas parten de la misma posición inicial,
# pero con velocidades aleatorias, generando una "explosión".
# Se tienen en cuenta las siguientes fuerzas:
# - Gravedad
# - Resistencia del aire
# - Viento
# Se utiliza el método de integración Symplectic Euler.

#Constantes físicas
g = ti.Vector([0.0, -10.0, 0.0])  #la gravedad en 10
v_viento = ti.Vector([-12.5, 0.0, 0.0]) #velocidad del viento
m = 1.0 # la masa
d = 0.3 #coeficiente de resistencia del aire, el rozamiento

#Parámetros
N = 1000 #número de partículas
dt = 0.01 #paso del tiempo

#Campos
# Cada partícula tiene su propia posición y velocidad
v_prev = ti.Vector.field(3, dtype=ti.f32, shape=N) #velocidades
x_prev = ti.Vector.field(3, dtype=ti.f32, shape=N) #posiciones

#Inicialización
@ti.kernel
def init():

    #inicia todas las partículas en
    #- la misma posición
    #- velocidades aleatorias
    for i in range(N):
        x_prev[i] = ti.Vector([0.0, 100.0, 0.0]) #posicion inicial común
        # velocidades aleatorias:
        # x,z en [-50, 50] y en [0, 100]
        v_prev[i] = ti.Vector([
            ti.random() * 100 - 50,
            ti.random() * 100,
            ti.random() * 100 - 50
        ])

#Funcion de la aceleración
@ti.func
def a_func(v):
    #Calcula la aceleración total de una partícula teniendo en cuenta
    #-gravedad
    #-resistencia al aire
    #-viento
    return g - (d/m) * (v - v_viento)

#Step Euler
@ti.kernel
def step():
    #Función principal de actualización. Se aplica Euler para:
    #-actualizar velocidad usando la aceleración
    #-actualizar posición se usa la velocidad
    for i in range(N):
        # calcular aceleración
        a = a_func(v_prev[i])

        # Symplectic Euler
        v_new = v_prev[i] + a * dt
        x_new = x_prev[i] + v_new * dt

        # colisión con el suelo
        if x_new[1] < 0:
            x_new[1] = 0
            v_new[1] *= -0.6
        # actualizo los valores
        v_prev[i] = v_new
        x_prev[i] = x_new

#Reset 
def reset():
    #Reinicia la simulación, volviendo a generar posiciones y velocidades aleatorias
    init()

#Se simula un sistema de 1000 partículas con condiciones iniciales comunes y velocidades aleatorias
#integradas mediante el método Symplectic Euler bajo la acción de gravedad, resistencia aerodinámica y viento