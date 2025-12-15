from nicegui import ui
import sympy as sp
import numpy as np
from scipy.optimize import fsolve, least_squares, root
import re
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, List, Any, Optional

class AdvancedEquationSolver:
    def __init__(self):
        self.equations = []
        self.variables = set()
        self.timeout = 10  # segundos por método
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    def parse_equation(self, eq_text: str):
        """Convierte una ecuación en formato string a expresión sympy"""
        eq_text = eq_text.strip()
        eq_text = eq_text.replace('^', '**')
        
        # Manejar multiplicación implícita
        eq_text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', eq_text)
        eq_text = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', eq_text)
        
        # Manejar igualdad
        if '=' in eq_text:
            lhs, rhs = eq_text.split('=', 1)
            expr = f"({lhs.strip()}) - ({rhs.strip()})"
        else:
            expr = eq_text
            
        return sp.sympify(expr)
    
    def generate_initial_guesses(self, n_vars: int, n_guesses: int = 10):
        """Genera puntos iniciales para métodos numéricos"""
        guesses = []
        
        # Puntos fijos comunes
        fixed_guesses = [
            [0.0] * n_vars,
            [1.0] * n_vars,
            [-1.0] * n_vars,
            [0.5] * n_vars,
            [-0.5] * n_vars,
        ]
        
        guesses.extend(fixed_guesses)
        
        # Puntos aleatorios
        for _ in range(n_guesses - len(fixed_guesses)):
            guess = np.random.uniform(-5, 5, n_vars).tolist()
            guesses.append(guess)
            
        # Puntos basados en distribuciones
        for scale in [0.1, 2.0, 10.0]:
            guess = np.random.randn(n_vars) * scale
            guesses.append(guess.tolist())
            
        return guesses
    
    def solve_system(self, equations_text: list):
        """Resuelve el sistema de ecuaciones usando múltiples métodos en orden de consumo computacional"""
        try:
            self.equations = []
            self.variables = set()
            
            # Parsear ecuaciones
            for eq_text in equations_text:
                if eq_text.strip():
                    expr = self.parse_equation(eq_text)
                    self.equations.append(expr)
                    self.variables.update({str(s) for s in expr.free_symbols})
            
            if not self.equations:
                return {"error": "No hay ecuaciones"}
            
            if not self.variables:
                return {"error": "No hay variables"}
            
            var_list = sorted(list(self.variables))
            
            # Preparar funciones para métodos numéricos
            var_symbols = [sp.Symbol(v) for v in var_list]
            numpy_funcs = [sp.lambdify(var_symbols, eq, ('numpy', 'math')) for eq in self.equations]
            
            def equations_func(vars):
                return [float(f(*vars)) for f in numpy_funcs]
            
            # Resultados de todos los métodos
            all_solutions = []
            methods_used = []
            
            # ORDEN DE EJECUCIÓN POR CONSUMO COMPUTACIONAL (de menor a mayor):
            
            # 1. Método más simple: Intento de solución simbólica (menos consumo)
            symbolic_result = self.try_symbolic_solution()
            if symbolic_result:
                all_solutions.extend(symbolic_result)
                methods_used.append("simbólica")
                # Si encontramos solución simbólica, podemos retornar temprano
                if len(self.equations) == len(var_list):  # Sistema determinado
                    print("Solución simbólica encontrada, omitiendo métodos numéricos")
                    unique_solutions = self.filter_unique_solutions(all_solutions)
                    return {
                        "success": True, 
                        "solutions": unique_solutions,
                        "methods_used": list(set(methods_used)),
                        "total_solutions": len(unique_solutions)
                    }
            
            # 2. Método rápido: fsolve con pocas iteraciones
            fsolve_result = self.try_fsolve(var_list, equations_func)
            if fsolve_result:
                all_solutions.extend(fsolve_result)
                methods_used.append("fsolve")
            
            # 3. Método intermedio: root con métodos simples
            root_result = self.try_root_methods_simple(var_list, equations_func)
            if root_result:
                all_solutions.extend(root_result)
                methods_used.append("root-simple")
            
            # 4. Método más costoso: Newton personalizado
            newton_result = self.try_newton_method(var_list, equations_func)
            if newton_result:
                all_solutions.extend(newton_result)
                methods_used.append("newton")
            
            # 5. Método costoso: least_squares
            least_sq_result = self.try_least_squares(var_list, equations_func)
            if least_sq_result:
                all_solutions.extend(least_sq_result)
                methods_used.append("mínimos cuadrados")
            
            # 6. Método más costoso: root con métodos complejos (solo si no se encontraron soluciones)
            if not all_solutions:
                root_complex_result = self.try_root_methods_complex(var_list, equations_func)
                if root_complex_result:
                    all_solutions.extend(root_complex_result)
                    methods_used.append("root-complex")
            
            # Filtrar soluciones duplicadas
            unique_solutions = self.filter_unique_solutions(all_solutions)
            
            if unique_solutions:
                return {
                    "success": True, 
                    "solutions": unique_solutions,
                    "methods_used": list(set(methods_used)),
                    "total_solutions": len(unique_solutions)
                }
            else:
                return {"error": "No se encontraron soluciones con ningún método"}
                    
        except Exception as e:
            return {"error": f"Error: {str(e)}"}
    
    def try_symbolic_solution(self):
        """Intenta solución simbólica (método más liviano)"""
        try:
            var_list = sorted(list(self.variables))
            # Limitar el tiempo para sistemas grandes
            if len(var_list) > 5 or len(self.equations) > 5:
                print("Sistema demasiado grande para solución simbólica rápida")
                return None
                
            solutions = sp.solve(self.equations, var_list, dict=True, manual=True, simplify=False)
            
            if solutions:
                result = []
                for sol in solutions:
                    float_sol = {}
                    for var, value in sol.items():
                        try:
                            if value.is_number:
                                num_val = float(sp.N(value))
                                if abs(num_val) < 1e-12:
                                    num_val = 0.0
                                float_sol[str(var)] = num_val
                            else:
                                float_sol[str(var)] = str(value)
                        except:
                            float_sol[str(var)] = str(value)
                    
                    result.append({
                        "solution": float_sol,
                        "method": "simbólica",
                        "error": 0.0
                    })
                
                return result
        except Exception as e:
            print(f"Solución simbólica falló: {e}")
        return None
    
    def try_fsolve(self, var_list, equations_func):
        """Intenta solución con fsolve (método rápido)"""
        solutions = []
        # Usar menos guesses para este método rápido
        guesses = self.generate_initial_guesses(len(var_list), n_guesses=5)
        
        for guess in guesses:
            try:
                def timeout_wrapper():
                    return fsolve(equations_func, guess, full_output=True, maxfev=100)
                
                future = self.executor.submit(timeout_wrapper)
                x_sol, info, ier, msg = future.result(timeout=self.timeout)
                
                if ier == 1:
                    residuals = equations_func(x_sol)
                    error = np.linalg.norm(residuals)
                    
                    if error < 1e-6:
                        sol_dict = {var: float(val) for var, val in zip(var_list, x_sol)}
                        solutions.append({
                            "solution": sol_dict,
                            "method": "fsolve",
                            "error": error
                        })
            except TimeoutError:
                continue
            except Exception as e:
                continue
        
        return solutions
    
    def try_root_methods_simple(self, var_list, equations_func):
        """Intenta métodos root simples (hybr es el más eficiente)"""
        solutions = []
        guesses = self.generate_initial_guesses(len(var_list), n_guesses=3)
        
        # Solo métodos simples y rápidos
        methods = ['hybr', 'lm']
        
        for method in methods:
            for guess in guesses:
                try:
                    result = root(equations_func, guess, method=method, 
                                 options={'maxfev': 200})
                    
                    if result.success:
                        error = np.linalg.norm(result.fun)
                        if error < 1e-6:
                            sol_dict = {var: float(val) for var, val in zip(var_list, result.x)}
                            solutions.append({
                                "solution": sol_dict,
                                "method": f"root-{method}",
                                "error": error
                            })
                except:
                    continue
        
        return solutions
    
    def try_root_methods_complex(self, var_list, equations_func):
        """Métodos root más complejos y costosos (solo como último recurso)"""
        solutions = []
        guesses = self.generate_initial_guesses(len(var_list), n_guesses=2)
        
        # Métodos más costosos
        methods = ['broyden1', 'broyden2', 'anderson']
        
        for method in methods:
            for guess in guesses:
                try:
                    result = root(equations_func, guess, method=method, 
                                 options={'maxiter': 50})
                    
                    if result.success:
                        error = np.linalg.norm(result.fun)
                        if error < 1e-6:
                            sol_dict = {var: float(val) for var, val in zip(var_list, result.x)}
                            solutions.append({
                                "solution": sol_dict,
                                "method": f"root-{method}",
                                "error": error
                            })
                except:
                    continue
        
        return solutions
    
    def try_newton_method(self, var_list, equations_func):
        """Método de Newton personalizado (consumo intermedio)"""
        solutions = []
        guesses = self.generate_initial_guesses(len(var_list), n_guesses=3)
        
        for guess in guesses:
            try:
                x = np.array(guess, dtype=float)
                n = len(x)
                max_iter = 50  # Reducir iteraciones
                tol = 1e-10
                
                for _ in range(max_iter):
                    f_val = np.array(equations_func(x))
                    
                    if np.linalg.norm(f_val) < tol:
                        break
                    
                    J = np.zeros((len(self.equations), n))
                    h = 1e-8
                    
                    for j in range(n):
                        x_plus = x.copy()
                        x_plus[j] += h
                        f_plus = np.array(equations_func(x_plus))
                        J[:, j] = (f_plus - f_val) / h
                    
                    try:
                        dx = np.linalg.solve(J, -f_val)
                    except np.linalg.LinAlgError:
                        dx = -np.linalg.pinv(J) @ f_val
                    
                    alpha = 1.0
                    for _ in range(5):  # Reducir búsqueda de línea
                        x_new = x + alpha * dx
                        f_new = np.array(equations_func(x_new))
                        
                        if np.linalg.norm(f_new) < np.linalg.norm(f_val):
                            x = x_new
                            break
                        alpha *= 0.5
                    else:
                        break
                
                error = np.linalg.norm(equations_func(x))
                if error < 1e-6:
                    sol_dict = {var: float(val) for var, val in zip(var_list, x)}
                    solutions.append({
                        "solution": sol_dict,
                        "method": "newton",
                        "error": error
                    })
            except:
                continue
        
        return solutions
    
    def try_least_squares(self, var_list, equations_func):
        """Intenta solución con least squares (método costoso)"""
        solutions = []
        guesses = self.generate_initial_guesses(len(var_list), n_guesses=3)
        
        for guess in guesses:
            try:
                def residuals_func(vars):
                    return np.array(equations_func(vars))
                
                result = least_squares(residuals_func, guess, 
                                      bounds=(-10, 10), 
                                      method='trf',
                                      max_nfev=200)  # Reducir evaluaciones
                
                if result.success:
                    error = np.linalg.norm(result.fun)
                    if error < 1e-4:
                        sol_dict = {var: float(val) for var, val in zip(var_list, result.x)}
                        solutions.append({
                            "solution": sol_dict,
                            "method": "mínimos cuadrados",
                            "error": error
                        })
            except:
                continue
        
        return solutions
    
    def filter_unique_solutions(self, solutions):
        """Filtra soluciones duplicadas"""
        unique = []
        tolerance = 1e-4
        
        for sol in solutions:
            is_unique = True
            sol_vector = np.array([sol["solution"][var] for var in sorted(sol["solution"].keys())])
            
            for uniq in unique:
                uniq_vector = np.array([uniq["solution"][var] for var in sorted(uniq["solution"].keys())])
                if np.linalg.norm(sol_vector - uniq_vector) < tolerance:
                    is_unique = False
                    break
            
            if is_unique:
                unique.append(sol)
        
        return unique


# Variables globales para comunicación entre hilos
result_queue = queue.Queue()
status_queue = queue.Queue()  # Nueva cola para estado del método actual
is_solving = False
current_method = ""
# Usamos un lock para proteger el acceso a is_solving
solving_lock = threading.Lock()
# Timer para verificar resultados
check_timer = None
# Timer para verificar estado
status_timer = None

solver = AdvancedEquationSolver()

# CSS mejorado con animación para métodos activos
ui.add_head_html('''
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body, html {
        height: 100vh;
        width: 100vw;
        overflow: hidden;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
        background: #fafafa;
    }
    
    .app-container {
        height: 100vh;
        width: 100vw;
        display: flex;
        flex-direction: column;
        padding: 20px;
        gap: 20px;
    }
    
    .top-bar {
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 4px;
    }
    
    .app-title {
        font-size: 22px;
        font-weight: 600;
        color: #1a1a1a;
        letter-spacing: -0.5px;
    }
    
    .solve-btn {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: #000;
        color: white;
        border: none;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        position: relative;
    }
    
    .solve-btn:hover:not(.loading) {
        background: #333;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .solve-btn.loading {
        background: #666;
        cursor: not-allowed;
    }
    
    .solve-btn .btn-spinner {
        position: absolute;
        width: 32px;
        height: 32px;
        border: 3px solid transparent;
        border-top-color: white;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        opacity: 0;
    }
    
    .solve-btn.loading .btn-spinner {
        opacity: 1;
    }
    
    .solve-btn.loading .play-icon {
        opacity: 0;
    }
    
    .play-icon {
        font-size: 18px;
        margin-left: 2px;
        transition: opacity 0.2s;
    }
    
    .main-grid {
        flex: 1;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        min-height: 0;
    }
    
    .panel {
        display: flex;
        flex-direction: column;
        background: white;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #f0f0f0;
    }
    
    .panel-header {
        padding: 18px 20px;
        border-bottom: 1px solid #f0f0f0;
        background: white;
    }
    
    .panel-title {
        font-size: 13px;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .panel-body {
        flex: 1;
        overflow: hidden;
        padding: 20px;
        background: #fdfdfd;
    }
    
    /* Área de ecuaciones */
    .equation-box {
        width: 100%;
        height: 100%;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e0e0e0;
        background: #fcfcfc;
        position: relative;
    }
    
    .equation-textarea {
        width: 100%;
        height: 100%;
        border: none;
        outline: none;
        resize: none;
        background: transparent;
        font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
        font-size: 15px;
        line-height: 1.8;
        color: #1a1a1a;
        padding: 16px;
    }
    
    .equation-textarea::placeholder {
        color: #aaa;
    }
    
    .equation-info {
        position: absolute;
        bottom: 8px;
        right: 8px;
        font-size: 11px;
        color: #999;
        background: rgba(255, 255, 255, 0.9);
        padding: 2px 6px;
        border-radius: 3px;
    }
    
    /* Área de resultados */
    .results-box {
        width: 100%;
        height: 100%;
        overflow-y: auto;
        padding-right: 4px;
    }
    
    .results-box::-webkit-scrollbar {
        width: 6px;
    }
    
    .results-box::-webkit-scrollbar-track {
        background: transparent;
    }
    
    .results-box::-webkit-scrollbar-thumb {
        background: #e0e0e0;
        border-radius: 3px;
    }
    
    .results-box::-webkit-scrollbar-thumb:hover {
        background: #d0d0d0;
    }
    
    /* Solución */
    .solution-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        border: 1px solid #f0f0f0;
    }
    
    .solution-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .solution-title {
        font-size: 16px;
        font-weight: 600;
        color: #1a1a1a;
    }
    
    .method-badge {
        background: #f0f0f0;
        color: #666;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Variables apiladas */
    .variables-stacked {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    
    .variable-line {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
        font-size: 15px;
        padding: 8px 0;
        border-bottom: 1px solid #f8f8f8;
    }
    
    .variable-line:last-child {
        border-bottom: none;
    }
    
    .var-name {
        font-weight: 600;
        color: #2563eb;
        min-width: 40px;
    }
    
    .var-equals {
        color: #666;
    }
    
    .var-value {
        color: #059669;
        font-weight: 500;
        margin-left: 4px;
    }
    
    .solution-info {
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px dashed #f0f0f0;
        font-size: 12px;
        color: #888;
    }
    
    /* Estado de error */
    .error-state {
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 20px;
    }
    
    .error-icon {
        font-size: 48px;
        color: #ff4444;
        margin-bottom: 16px;
        opacity: 0.8;
    }
    
    .error-title {
        font-size: 16px;
        font-weight: 600;
        color: #ff4444;
        margin-bottom: 8px;
    }
    
    .error-message {
        font-size: 14px;
        color: #666;
        max-width: 300px;
        line-height: 1.5;
    }
    
    /* Loading state mejorado */
    .loading-state {
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 20px;
    }
    
    .solving-spinner {
        width: 40px;
        height: 40px;
        position: relative;
        margin-bottom: 20px;
    }
    
    .solving-spinner .ring {
        position: absolute;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        border: 3px solid transparent;
    }
    
    .solving-spinner .ring-1 {
        border-top-color: #000;
        animation: spin 1s linear infinite;
    }
    
    .solving-spinner .ring-2 {
        border-right-color: #666;
        animation: spin 1.2s linear infinite reverse;
        width: 70%;
        height: 70%;
        top: 15%;
        left: 15%;
    }
    
    .solving-spinner .ring-3 {
        border-bottom-color: #999;
        animation: spin 0.8s linear infinite;
        width: 40%;
        height: 40%;
        top: 30%;
        left: 30%;
    }
    
    .solving-text {
        font-size: 14px;
        color: #666;
        margin-bottom: 8px;
    }
    
    .solving-subtext {
        font-size: 12px;
        color: #999;
        max-width: 200px;
        margin-bottom: 16px;
    }
    
    .solving-methods {
        margin-top: 16px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
        max-width: 300px;
    }
    
    .method-tag {
        font-size: 10px;
        padding: 2px 8px;
        background: #f0f0f0;
        border-radius: 4px;
        color: #666;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .method-tag.active {
        background: #2563eb;
        color: white;
        transform: scale(1.05);
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }
    
    .method-tag.active::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        animation: shimmer 1.5s infinite;
    }
    
    .method-tag.completed {
        background: #10b981;
        color: white;
    }
    
    .method-counter {
        font-size: 11px;
        color: #666;
        margin-bottom: 8px;
        text-align: center;
    }
    
    /* Resultados estadísticos */
    .stats-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        border: 1px solid #e9ecef;
    }
    
    .stats-title {
        font-size: 13px;
        font-weight: 600;
        color: #666;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 10px;
    }
    
    .stat-item {
        display: flex;
        flex-direction: column;
    }
    
    .stat-label {
        font-size: 11px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    .stat-value {
        font-size: 14px;
        font-weight: 600;
        color: #1a1a1a;
        font-family: 'SF Mono', monospace;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes shimmer {
        100% { left: 100%; }
    }
    
    .solution-card {
        animation: fadeIn 0.3s ease-out;
    }
    
    .method-progress {
        width: 100%;
        height: 4px;
        background: #f0f0f0;
        border-radius: 2px;
        margin: 12px 0;
        overflow: hidden;
    }
    
    .method-progress-bar {
        height: 100%;
        background: #2563eb;
        border-radius: 2px;
        transition: width 0.5s ease;
    }
</style>
''')

# Contenedor principal
with ui.element('div').classes('app-container'):
    # Barra superior
    with ui.element('div').classes('top-bar'):
        ui.label('Advanced Equation Solver').classes('app-title')
        with ui.button('', on_click=lambda: solve_equations()).classes('solve-btn') as play_btn:
            ui.element('div').classes('btn-spinner')
            ui.label('▶').classes('play-icon')
    
    # Grid de dos columnas
    with ui.element('div').classes('main-grid'):
        # Panel izquierdo - Ecuaciones
        with ui.element('div').classes('panel'):
            with ui.element('div').classes('panel-header'):
                ui.label('Ecuaciones').classes('panel-title')
            
            with ui.element('div').classes('panel-body'):
                with ui.element('div').classes('equation-box'):
                    equation_area = ui.textarea(
                        value='x + y = 5\nx - y = 1',
                        placeholder='Una ecuación por línea\n\nEjemplos avanzados:\nx + y + z = 10\nx^2 + y^2 = 25\nsin(x) + cos(y) = 1\nexp(x) - y = 0\nx*y*z = 1',
                        on_change=lambda e: update_equation_info()
                    ).classes('equation-textarea').props('borderless')
                    
                    equation_info = ui.label('').classes('equation-info')
        
        # Panel derecho - Resultados
        with ui.element('div').classes('panel'):
            with ui.element('div').classes('panel-header'):
                ui.label('Resultados').classes('panel-title')
            
            with ui.element('div').classes('panel-body'):
                results_container = ui.element('div').classes('results-box')

def update_equation_info():
    """Actualiza la información sobre las ecuaciones"""
    eq_texts = [line.strip() for line in equation_area.value.split('\n') if line.strip()]
    var_set = set()
    
    for eq in eq_texts:
        matches = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', eq)
        var_set.update(matches)
    
    math_keywords = {'sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt', 'pi', 'e', 'abs'}
    variables = [v for v in var_set if v.lower() not in math_keywords and not v.isdigit()]
    
    equation_info.text = f'{len(eq_texts)} ecuaciones, {len(variables)} variables'

# Variables para controlar el estado de resolución
current_method_index = 0
method_tags = []

def show_solving_state():
    """Muestra el estado de resolución con conteo de métodos"""
    global current_method_index, method_tags
    
    results_container.clear()
    with results_container:
        with ui.element('div').classes('loading-state'):
            with ui.element('div').classes('solving-spinner'):
                ui.element('div').classes('ring ring-1')
                ui.element('div').classes('ring ring-2')
                ui.element('div').classes('ring ring-3')
            
            ui.label('Buscando soluciones...').classes('solving-text')
            ui.label('Probando múltiples métodos en orden de eficiencia').classes('solving-subtext')
            
            # Contador de métodos
            method_counter = ui.label('Método 1 de 6').classes('method-counter')
            
            # Barra de progreso
            with ui.element('div').classes('method-progress'):
                progress_bar = ui.element('div').classes('method-progress-bar').style('width: 16.66%')
            
            # Mostrar métodos en el orden en que se ejecutan
            methods = [
                "1. Solución simbólica",
                "2. Método fsolve", 
                "3. Métodos root simples",
                "4. Newton personalizado",
                "5. Mínimos cuadrados",
                "6. Métodos complejos"
            ]
            
            with ui.element('div').classes('solving-methods') as methods_container:
                method_tags = []
                for i, method in enumerate(methods):
                    tag = ui.label(method).classes('method-tag')
                    if i == 0:
                        tag.classes('active')
                    method_tags.append(tag)

def update_solving_state(method_index):
    """Actualiza el estado de resolución mostrando el método actual"""
    global current_method_index, method_tags
    
    if method_index < len(method_tags):
        # Quitar clase 'active' de todos los métodos
        for tag in method_tags:
            tag.classes(remove='active')
            tag.classes(remove='completed')
        
        # Marcar métodos anteriores como completados
        for i in range(method_index):
            if i < len(method_tags):
                method_tags[i].classes('completed')
        
        # Marcar método actual como activo
        method_tags[method_index].classes('active')
        
        # Actualizar contador
        counter_text = f'Método {method_index + 1} de {len(method_tags)}'
        if hasattr(update_solving_state, 'counter_label'):
            update_solving_state.counter_label.text = counter_text
        
        # Actualizar barra de progreso
        progress = ((method_index + 1) / len(method_tags)) * 100
        if hasattr(update_solving_state, 'progress_bar'):
            update_solving_state.progress_bar.style(f'width: {progress}%')

def update_results(result):
    """Actualiza los resultados en la UI"""
    results_container.clear()
    
    if "error" in result:
        with results_container:
            with ui.element('div').classes('error-state'):
                ui.label('⚠️').classes('error-icon')
                ui.label('Error').classes('error-title')
                ui.label(result["error"]).classes('error-message')
    else:
        with results_container:
            # Mostrar estadísticas
            with ui.element('div').classes('stats-box'):
                ui.label('ESTADÍSTICAS').classes('stats-title')
                with ui.element('div').classes('stats-grid'):
                    with ui.element('div').classes('stat-item'):
                        ui.label('Soluciones').classes('stat-label')
                        ui.label(str(result.get('total_solutions', 0))).classes('stat-value')
                    
                    with ui.element('div').classes('stat-item'):
                        ui.label('Métodos usados').classes('stat-label')
                        methods = result.get('methods_used', [])
                        ui.label(str(len(methods))).classes('stat-value')
                    
                    with ui.element('div').classes('stat-item'):
                        ui.label('Variables').classes('stat-label')
                        if result['solutions']:
                            vars_count = len(result['solutions'][0]['solution'])
                        else:
                            vars_count = 0
                        ui.label(str(vars_count)).classes('stat-value')
            
            # Mostrar soluciones
            for i, solution_data in enumerate(result["solutions"]):
                solution = solution_data.get("solution", {})
                method = solution_data.get("method", "desconocido")
                error = solution_data.get("error", 0.0)
                
                with ui.element('div').classes('solution-card'):
                    with ui.element('div').classes('solution-header'):
                        if len(result["solutions"]) > 1:
                            ui.label(f'Solución {i+1}').classes('solution-title')
                        else:
                            ui.label('Solución').classes('solution-title')
                        ui.label(method).classes('method-badge')
                    
                    # Variables apiladas verticalmente
                    with ui.element('div').classes('variables-stacked'):
                        for var, value in sorted(solution.items()):
                            with ui.element('div').classes('variable-line'):
                                ui.label(var).classes('var-name')
                                ui.label('=').classes('var-equals')
                                
                                if isinstance(value, (int, float)):
                                    # Siempre mostrar con 5 decimales
                                    formatted = f"{value:.5f}"
                                    # Corregir -0.00000 a 0.00000
                                    if formatted == "-0.00000":
                                        formatted = "0.00000"
                                else:
                                    formatted = str(value)
                                
                                ui.label(formatted).classes('var-value')
                    
                    # Información adicional
                    with ui.element('div').classes('solution-info'):
                        ui.label(f'Método: {method} | Error: {error:.2e}')

def solve_in_background(eq_texts):
    """Función que se ejecuta en el hilo de fondo"""
    try:
        print(f"Iniciando resolución de {len(eq_texts)} ecuaciones...")
        
        # Simular progreso de métodos (en una implementación real, esto vendría del solver)
        methods_order = [
            "simbólica",
            "fsolve", 
            "root-simple",
            "newton",
            "mínimos cuadrados",
            "root-complex"
        ]
        
        # Enviar actualizaciones de progreso
        for i, method in enumerate(methods_order):
            time.sleep(0.5)  # Simular tiempo entre métodos
            if status_queue:
                status_queue.put(i)  # Enviar índice del método actual
        
        # Resolver realmente
        result = solver.solve_system(eq_texts)
        print(f"Resultado obtenido: {result}")
        result_queue.put(result)
        
    except Exception as e:
        print(f"Error en el hilo de fondo: {e}")
        result_queue.put({"error": f"Error en el hilo de fondo: {str(e)}"})

def check_result():
    """Función que verifica si hay resultados en la cola"""
    global is_solving, check_timer, status_timer
    
    if not result_queue.empty():
        result = result_queue.get()
        
        # Actualizar el estado de resolución
        with solving_lock:
            is_solving = False
        
        # Actualizar la UI
        play_btn.classes(remove='loading')
        update_results(result)
        
        # Detener los timers
        if check_timer:
            check_timer.deactivate()
            check_timer = None
        
        if status_timer:
            status_timer.deactivate()
            status_timer = None
        
        return False  # Detener el timer
    
    return True  # Continuar verificando

def check_status():
    """Función que verifica si hay actualizaciones de estado"""
    if not status_queue.empty():
        method_index = status_queue.get()
        update_solving_state(method_index)
    
    return True

def solve_equations():
    """Función principal para resolver ecuaciones"""
    global is_solving, check_timer, status_timer, current_method_index
    
    # Verificar si ya se está resolviendo
    with solving_lock:
        if is_solving:
            print("Ya se está resolviendo, espera...")
            return
        is_solving = True
    
    eq_texts = [line.strip() for line in equation_area.value.split('\n') if line.strip()]
    
    if not eq_texts:
        ui.notify('No hay ecuaciones para resolver', type='warning')
        with solving_lock:
            is_solving = False
        return
    
    print(f"Iniciando resolución de {len(eq_texts)} ecuaciones...")
    play_btn.classes('loading')
    
    # Reiniciar índice del método
    current_method_index = 0
    
    # Mostrar estado de resolución
    show_solving_state()
    
    # Guardar referencias a los elementos de la UI para actualizarlos
    if results_container:
        with results_container:
            # Buscar y guardar referencias a los elementos de progreso
            for element in results_container.default_slot.children:
                if hasattr(element, 'text') and 'Método 1 de 6' in element.text:
                    update_solving_state.counter_label = element
                elif hasattr(element, 'style') and 'method-progress-bar' in element.classes:
                    update_solving_state.progress_bar = element
    
    # Detener timers anteriores si existen
    if check_timer:
        check_timer.deactivate()
        check_timer = None
    
    if status_timer:
        status_timer.deactivate()
        status_timer = None
    
    # Iniciar hilo de fondo para resolver
    thread = threading.Thread(target=solve_in_background, args=(eq_texts,), daemon=True)
    thread.start()
    
    # Iniciar timer para verificar resultados
    check_timer = ui.timer(0.5, check_result)
    
    # Iniciar timer para verificar estado (más rápido para animación suave)
    status_timer = ui.timer(0.1, check_status)
    
    # También iniciar una simulación de progreso visual
    def simulate_progress():
        global current_method_index
        if current_method_index < 6:
            update_solving_state(current_method_index)
            current_method_index += 1
    
    # Iniciar simulación de progreso (cada 1 segundo cambia de método)
    ui.timer(1.0, lambda: simulate_progress() if is_solving else False)

# Inicializar información de ecuaciones
update_equation_info()

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Advanced Equation Solver",
        port=8080,
        reload=False,
        show=True,
        dark=False,
        favicon="🔢"
    )
