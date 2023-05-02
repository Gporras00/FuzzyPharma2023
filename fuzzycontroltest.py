import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Definir el universo de discurso (range) para la entrada "temperature" y la salida "power"
temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
power = ctrl.Consequent(np.arange(0, 101, 1), 'power')

# Definir las funciones de membresía para la entrada "temperature"
temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 50])
temperature['warm'] = fuzz.trimf(temperature.universe, [0, 50, 100])
temperature['hot'] = fuzz.trimf(temperature.universe, [50, 100, 100])

# Definir las funciones de membresía para la salida "power"
power['low'] = fuzz.trimf(power.universe, [0, 0, 50])
power['medium'] = fuzz.trimf(power.universe, [0, 50, 100])
power['high'] = fuzz.trimf(power.universe, [50, 100, 100])

# Visualizar las funciones de membresía
temperature.view()
power.view()

# Definir las reglas difusas
rule1 = ctrl.Rule(temperature['cold'], power['low'])
rule2 = ctrl.Rule(temperature['warm'], power['medium'])
rule3 = ctrl.Rule(temperature['hot'], power['high'])

# Definir el sistema de inferencia difusa
power_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

# Crear la simulación del sistema de inferencia difusa
power_sim = ctrl.ControlSystemSimulation(power_ctrl)

# Establecer la entrada del sistema de inferencia difusa
power_sim.input['temperature'] = 70

# Realizar la inferencia
power_sim.compute()

# Obtener la salida del sistema de inferencia difusa
print(power_sim.output['power'])

# Visualizar la salida difusa
power.view(sim=power_sim)

# Mostrar las gráficas
plt.show()
