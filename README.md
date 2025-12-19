# Advanced Equation Solver 🧮
Una potente aplicación web para resolver sistemas de ecuaciones algebraicas, trigonométricas y transcendentes. Combina la precisión simbólica de SymPy con la velocidad numérica de SciPy, todo envuelto en una interfaz moderna con NiceGUI.

# 🚀 Características Clave
Soporte Matemático Total: Álgebra, trigonometría, cálculo (derivadas/integrales) y funciones especiales.

Modos de Ángulo Inteligentes: Alterna entre RAD (matemático) y DEG (ingeniería) con conversión automática (ej: sin(30) en DEG = 0.5).

Resolución Híbrida: Estrategia en cascada que prioriza soluciones simbólicas exactas y recurre a métodos numéricos robustos (fsolve, Newton, Levenberg-Marquardt) si es necesario.

Paralelismo: Ejecución concurrente para evitar bloqueos en la UI.

Interfaz Reactiva: Visualización de progreso, estadísticas de error y scroll infinito para soluciones múltiples.




x^2 + y^2 = 25       # Círculo
sin(x) + cos(y) = 1  # Trigonométrica
