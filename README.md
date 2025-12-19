<img width="360" height="360" alt="Gemini_Generated_Image_rzw4p2rzw4p2rzw4" src="https://github.com/user-attachments/assets/d6959838-de02-4675-883e-0c9793257274" />


#  Advanced Equation Solver

Una aplicación web avanzada para resolver sistemas de ecuaciones algebraicas, trigonométricas y trascendentes con soporte completo para modos grados/radianes.

---

##  Características Principales

###  Modos de Ángulo Inteligentes
* **RAD (Radianes):** Modo por defecto para cálculos matemáticos.
* **DEG (Grados):** Conversión automática de constantes trigonométricas (ej: `sin(30)` = 0.5).
* **Cambio dinámico:** Alterna entre modos con un solo clic sin perder datos.

###  Soporte Matemático Completo
* **Álgebra:** Ecuaciones polinómicas, racionales e irracionales.
* **Trigonometría:** Funciones directas, inversas e hiperbólicas.
* **Cálculo:** Derivadas, integrales, sumatorias y productorias.
* **Funciones Especiales:** Gamma, factorial, error y Bessel.

###  Métodos de Resolución
1. **Simbólico (SymPy):** Para obtener soluciones exactas y analíticas.
2. **Numérico (SciPy):** Implementación de `fsolve`, `root`, `Newton` y mínimos cuadrados.
3. **Estrategia en Cascada:** Ejecuta métodos rápidos primero y complejos después.
4. **Paralelismo:** Uso de `ThreadPoolExecutor` para ejecución concurrente.

---

##  Interfaz de Usuario
* **Diseño Moderno:** UI reactiva con CSS personalizado y animaciones suaves.
* **Feedback en Tiempo Real:** Visualización del progreso del solver.
* **Panel de Estadísticas:** Muestra el número de soluciones, métodos usados y error de aproximación.

---

##  Instalación y Inicio Rápido

```bash
# 1. Clonar el repositorio
git clone https://github.com/dagc1948/AES.git
cd AES

# 2. Instalar dependencias
pip install nicegui sympy numpy scipy

# 3. Ejecutar aplicación
python main.py
```

La aplicación estará disponible en: http://localhost:8080

---

##  Ejemplos de Uso

Escribe tus ecuaciones (una por línea), el sistema detectará las variables automáticamente:

### Sistema Trigonométrico (Modo DEG):
```
sin(x) = 0.5
cos(y) = 1
```

### Cálculo y Álgebra:
```
diff(x^3, x) = 12
x^2 + y^2 = 25
```

---

##  Detalles Técnicos

### Manejo de Casos Especiales
* **Conversión DEG:** `sin(30)` se procesa internamente como `sin(30 * pi/180)`.
* **Derivadas:** Soporta sintaxis corta como `x' -> diff(x)` y `x'' -> diff(x, 2)`.
* **Validaciones:** Control de sintaxis inválida, variables no detectadas y timeouts de convergencia.

### Arquitectura
* **Parser:** Normalización de sintaxis y detección de variables.
* **Solver Core:** Filtrado de soluciones únicas y generación de puntos iniciales adaptativos.
* **Frontend:** NiceGUI para una experiencia de usuario fluida.

---

##  Licencia

Este proyecto está bajo la Licencia MIT.

**Desarrollado por dagc1948**
