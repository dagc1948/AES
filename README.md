Advanced Equation Solver
Una aplicación web avanzada para resolver sistemas de ecuaciones algebraicas, trigonométricas y transcendentes con soporte completo para modos grados/radianes.

🌟 Características Principales
🔄 Modos de Ángulo Inteligentes
RAD (Radianes): Modo por defecto para cálculos matemáticos

DEG (Grados): Conversión automática de constantes trigonométricas

Cambio dinámico con un clic: sin(30) = 0.5 en modo DEG

📐 Soporte Matemático Completo
Álgebra: Ecuaciones polinómicas, racionales, irracionales

Trigonometría: Funciones directas, inversas e hiperbólicas

Cálculo: Derivadas, integrales, sumatorias, productorias

Funciones especiales: Gamma, factorial, error, Bessel

Operadores: +, -, *, /, ^, **, min(), max()

🎯 Múltiples Métodos de Resolución
Simbólico (SymPy) para soluciones exactas

Numérico: fsolve, root, Newton, mínimos cuadrados

Estrategia en cascada: Métodos rápidos primero, complejos después

Paralelismo: ThreadPoolExecutor para ejecución concurrente

🖥️ Interfaz Moderna y Reactiva
Diseño elegante con CSS personalizado

Visualización en tiempo real del progreso

Animaciones suaves y feedback visual

Panel de estadísticas completo

Scroll infinito para múltiples soluciones

🚀 Instalación Rápida
bash
# 1. Clonar repositorio
git clone https://github.com/dagc1948/AES/main.py
cd advanced-equation-solver

# 2. Instalar dependencias
pip install nicegui sympy numpy scipy

# 3. Ejecutar aplicación
python main.py
La aplicación se abrirá automáticamente en http://localhost:8080

📖 Uso Básico
1. Escribir Ecuaciones
python
# Una ecuación por línea
x^2 + y^2 = 25
sin(x) + cos(y) = 1
exp(x) - log(y) = 2
2. Seleccionar Modo
RAD: Para cálculos matemáticos estándar

DEG: Para problemas con ángulos en grados

3. Resolver
Click en el botón ▶ para iniciar

Observar progreso en tiempo real

Ver soluciones en panel derecho

📚 Ejemplos Incluidos
Álgebra Básica
text
x^2 - 4 = 0
x^3 + 2x^2 - 5x - 6 = 0
Trigonometría
text
sin(x) = 0.5        # x = 30° (DEG) o π/6 (RAD)
cos(2x) + sin(x) = 0
tan(x) = 1
Sistemas de Ecuaciones
text
x + y = 5
x^2 + y^2 = 13
Cálculo Diferencial
text
diff(sin(x), x) = cos(x)
diff(x^3, x, 2) = 6x
🔧 Características Técnicas
Algoritmos Implementados
Solución Simbólica (SymPy) - para sistemas pequeños

fsolve (SciPy) - método híbrido rápido

root - múltiples variantes (hybr, lm, broyden)

Newton Personalizado - con aproximación numérica de Jacobiano

Mínimos Cuadrados - para sistemas sobredeterminados

Manejo de Casos Especiales
python
# Conversión automática en modo DEG
sin(30) → sin(30*pi/180)  # = 0.5
cos(45) → cos(45*pi/180)  # = √2/2

# Manejo de funciones inversas
asin(0.5) → 30° en DEG, π/6 en RAD

# Soporte para derivadas
x' → diff(x)
x'' → diff(x, 2)
📊 Resultados y Visualización
Panel de Estadísticas
Número total de soluciones

Métodos utilizados

Variables encontradas

Error de aproximación

Formato de Salida
json
{
  "solution": {"x": 2.0, "y": 3.0},
  "method": "fsolve",
  "error": 1.2e-10,
  "angle_mode": "DEG"
}
🛠️ Arquitectura
Componentes Principales
text
AdvancedEquationSolver/
├── Parser de Ecuaciones
│   ├── Detección automática de variables
│   ├── Conversión DEG→RAD para constantes
│   └── Normalización de sintaxis
├── Sistema de Resolución
│   ├── Cascada de métodos numéricos
│   ├── Generación de puntos iniciales
│   └── Filtrado de soluciones únicas
└── Interfaz de Usuario
    ├── Panel de ecuaciones
    ├── Panel de resultados
    └── Controles de modo
Flujo de Datos
Entrada: Ecuaciones en texto plano

Parsing: Conversión a expresiones SymPy

Resolución: Aplicación secuencial de métodos

Post-proceso: Filtrado y formateo

Salida: Soluciones formateadas en UI

🚨 Manejo de Errores
Validaciones Automáticas
✅ Ecuaciones vacías

✅ Variables no detectadas

✅ Sintaxis inválida

✅ Tiempos de espera

✅ Convergencia numérica

Mensajes Informativos
text
✅ "Solución encontrada con error: 1e-12"
⚠️  "No hay variables para resolver"
❌  "Error de sintaxis en ecuación 3"
📈 Rendimiento
Optimizaciones
ThreadPoolExecutor: Paralelización de métodos

Caché de funciones: Reutilización de lambdas

Guesses inteligentes: Puntos iniciales adaptativos

Timeout por método: Evita bloqueos infinitos

Límites Recomendados
Ecuaciones: ≤ 20

Variables: ≤ 10

Complejidad: Sistemas no lineales mixtos

🎨 Personalización
Modificación de CSS
css
/* Temas personalizados */
.theme-dark { background: #1a1a1a; }
.theme-light { background: #ffffff; }

/* Animaciones personalizadas */
@keyframes custom-spin { ... }
Configuración del Solver
python
# Ajustes en tiempo de ejecución
solver.timeout = 15  # segundos por método
solver.executor.max_workers = 5
🤝 Contribuir
Reportar Problemas
Revisar modo actual (DEG/RAD)


📄 Licencia
MIT License - Ver LICENSE para detalles completos.

🙏 Agradecimientos
SymPy: Biblioteca de matemática simbólica

SciPy: Herramientas numéricas avanzadas

NiceGUI: Framework web minimalista

NumPy: Operaciones numéricas eficientes
