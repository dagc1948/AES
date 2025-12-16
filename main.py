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
        self.angle_mode = "rad"  # Por defecto radianes
        print(f"Solver inicializado en modo: {self.angle_mode}")
        
    def set_angle_mode(self, mode: str):
        """Establece el modo de ángulo"""
        self.angle_mode = mode
        print(f"Modo de ángulo cambiado a: {self.angle_mode}")
        
    def parse_equation(self, eq_text: str):
        """Convierte una ecuación en formato string a expresión sympy"""
        eq_text = eq_text.strip()
        
        # Reemplazar símbolos especiales
        eq_text = eq_text.replace('^', '**')
        
        # NO redefinir π en modo grados - mantener π como constante matemática
        eq_text = eq_text.replace('π', 'pi')
        eq_text = eq_text.replace('∞', 'inf')
        
        # Lista de funciones matemáticas para evitar multiplicación incorrecta
        math_functions = [
            'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
            'asin', 'acos', 'atan', 'acot', 'asec', 'acsc',
            'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
            'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch',
            'exp', 'log', 'ln', 'log10', 'log2',
            'sqrt', 'cbrt', 'abs', 'gamma', 'factorial',
            'erf', 'erfc', 'floor', 'ceil', 'round',
            'min', 'max', 'diff', 'integral', 'sum', 'product'
        ]
        
        # Reemplazar funciones con marcadores temporales
        temp_markers = {}
        for i, func in enumerate(math_functions):
            marker = f"__F_{i}_"
            pattern = r'\b' + re.escape(func) + r'\s*\('
            eq_text = re.sub(pattern, marker + '(', eq_text, flags=re.IGNORECASE)
            temp_markers[marker] = func
        
        # Manejar notación de derivadas
        eq_text = re.sub(r"([a-zA-Z])'", r"diff(\1)", eq_text)
        eq_text = re.sub(r"([a-zA-Z])''", r"diff(\1, 2)", eq_text)
        eq_text = re.sub(r"([a-zA-Z])'''", r"diff(\1, 3)", eq_text)
        
        # Manejar multiplicación implícita CORREGIDO
        # 1. Entre número y variable: 2x -> 2*x
        eq_text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', eq_text)
        # 2. Entre variable y número: x2 -> x*2
        eq_text = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', eq_text)
        # 3. Entre variable y paréntesis: x( -> x*( 
        eq_text = re.sub(r'([a-zA-Z])(\()', r'\1*\2', eq_text)
        # 4. Entre paréntesis y variable: )x -> )*x
        eq_text = re.sub(r'(\))([a-zA-Z])', r'\1*\2', eq_text)
        # 5. Entre número y paréntesis: 2( -> 2*( 
        eq_text = re.sub(r'(\d)(\()', r'\1*\2', eq_text)
        # 6. Entre paréntesis y número: )2 -> )*2
        eq_text = re.sub(r'(\))(\d)', r'\1*\2', eq_text)
        
        # Restaurar funciones
        for marker, func in temp_markers.items():
            eq_text = eq_text.replace(marker, func)
        
        # **CORRECCIÓN CRÍTICA: Transformar números en funciones trigonométricas en modo DEG**
        if self.angle_mode == "deg":
            # Identificar funciones trigonométricas directas
            trig_direct = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc']
            
            for func in trig_direct:
                # Patrón para encontrar funciones trigonométricas con números
                pattern = r'\b(' + func + r')\s*\(\s*([+-]?\d+(?:\.\d*)?|\.\d+)\s*\)'
                
                def replace_with_degrees(match):
                    """Reemplaza sin(30) por sin(30*pi/180) en modo grados"""
                    func_name = match.group(1)
                    number = match.group(2)
                    return f'{func_name}({number}*pi/180)'
                
                eq_text = re.sub(pattern, replace_with_degrees, eq_text, flags=re.IGNORECASE)
        
        # Manejar igualdad
        if '=' in eq_text:
            lhs, rhs = eq_text.split('=', 1)
            expr = f"({lhs.strip()}) - ({rhs.strip()})"
        else:
            expr = eq_text
        
        try:
            # Parsear la expresión con sympy
            return sp.sympify(expr, locals={
                'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
                'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
                'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
                'asinh': sp.asinh, 'acosh': sp.acosh, 'atanh': sp.atanh,
                'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
                'sqrt': sp.sqrt, 'abs': sp.Abs,
                'floor': sp.floor, 'ceil': sp.ceiling,
                'gamma': sp.gamma, 'factorial': sp.factorial,
                'erf': sp.erf, 'erfc': sp.erfc,
                'pi': sp.pi, 'e': sp.E, 'inf': sp.oo,
                'cot': sp.cot, 'sec': sp.sec, 'csc': sp.csc,
                'acot': sp.acot, 'asec': sp.asec, 'acsc': sp.acsc,
                'coth': sp.coth, 'sech': sp.sech, 'csch': sp.csch,
                'acoth': sp.acoth, 'asech': sp.asech, 'acsch': sp.acsch,
                'log10': lambda x: sp.log(x, 10),
                'log2': lambda x: sp.log(x, 2),
                'cbrt': lambda x: x**sp.Rational(1, 3),
                'pow': sp.Pow,
                'diff': lambda f, x, n=1: sp.diff(f, x, n) if n > 0 else f,
                'integral': lambda f, x: sp.integrate(f, x),
                'sum': lambda f, var, a, b: sp.summation(f, (var, a, b)),
                'product': lambda f, var, a, b: sp.product(f, (var, a, b)),
                'min': sp.Min, 'max': sp.Max,
                'sign': sp.sign
            })
        except Exception as e:
            raise ValueError(f"No se pudo parsear la ecuación: {expr}\nError: {e}")
    
    def generate_initial_guesses(self, n_vars: int, n_guesses: int = 10):
        """Genera puntos iniciales para métodos numéricos"""
        guesses = []
        
        # Puntos fijos comunes - ajustar según modo de ángulo
        if self.angle_mode == "rad":
            fixed_guesses = [
                [0.0] * n_vars,
                [1.0] * n_vars,
                [-1.0] * n_vars,
                [0.5] * n_vars,
                [-0.5] * n_vars,
                [np.pi] * n_vars,
                [np.pi/2] * n_vars,
            ]
            low, high = -5, 5
        else:
            # Para grados, valores típicos en grados
            fixed_guesses = [
                [0.0] * n_vars,
                [90.0] * n_vars,
                [180.0] * n_vars,
                [270.0] * n_vars,
                [360.0] * n_vars,
                [-90.0] * n_vars,
                [-180.0] * n_vars,
                [45.0] * n_vars,
                [135.0] * n_vars,
            ]
            low, high = -360, 360
        
        guesses.extend(fixed_guesses)
        
        # Puntos aleatorios
        for _ in range(n_guesses - len(fixed_guesses)):
            guess = np.random.uniform(low, high, n_vars).tolist()
            guesses.append(guess)
            
        # Puntos basados en distribuciones normales
        for scale in [0.1, 0.5, 1.0, 2.0]:
            guess = np.random.randn(n_vars) * scale * (10 if self.angle_mode == "rad" else 90)
            guesses.append(guess.tolist())
            
        return guesses
    
    def solve_system(self, equations_text: list):
        """Resuelve el sistema de ecuaciones usando múltiples métodos"""
        try:
            print(f"\n=== INICIANDO RESOLUCIÓN EN MODO {self.angle_mode.upper()} ===")
            self.equations = []
            self.variables = set()
            
            # Parsear ecuaciones
            for eq_text in equations_text:
                if eq_text.strip():
                    print(f"Parseando ecuación en modo {self.angle_mode}: {eq_text}")
                    expr = self.parse_equation(eq_text)
                    self.equations.append(expr)
                    self.variables.update({str(s) for s in expr.free_symbols})
            
            if not self.equations:
                return {"error": "No hay ecuaciones"}
            
            if not self.variables:
                return {"error": "No hay variables para resolver"}
            
            var_list = sorted(list(self.variables))
            print(f"Variables detectadas: {var_list}")
            print(f"Número de ecuaciones: {len(self.equations)}")
            
            # Preparar funciones para métodos numéricos
            var_symbols = [sp.Symbol(v) for v in var_list]
            
            # Crear funciones lambda con manejo de funciones especiales y modo ángulo
            numpy_funcs = []
            for eq in self.equations:
                try:
                    # Convertir a función numérica
                    if self.angle_mode == "rad":
                        # Modo radianes - usar funciones normales
                        func = sp.lambdify(var_symbols, eq, 
                                          modules=['numpy', 'math', 
                                                   {'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                                                    'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
                                                    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
                                                    'asinh': np.arcsinh, 'acosh': np.arccosh, 'atanh': np.arctanh,
                                                    'exp': np.exp, 'log': np.log, 'log10': np.log10,
                                                    'log2': lambda x: np.log2(x) if x > 0 else np.nan,
                                                    'sqrt': np.sqrt, 'abs': np.abs, 'pi': np.pi, 'e': np.e,
                                                    'cot': lambda x: 1/np.tan(x) if np.tan(x) != 0 else np.nan,
                                                    'sec': lambda x: 1/np.cos(x) if np.cos(x) != 0 else np.nan,
                                                    'csc': lambda x: 1/np.sin(x) if np.sin(x) != 0 else np.nan,
                                                    'coth': lambda x: 1/np.tanh(x) if np.tanh(x) != 0 else np.nan,
                                                    'sech': lambda x: 1/np.cosh(x),
                                                    'csch': lambda x: 1/np.sinh(x) if np.sinh(x) != 0 else np.nan,
                                                    'acot': lambda x: np.pi/2 - np.arctan(x) if x != 0 else np.pi/2,
                                                    'asec': lambda x: np.arccos(1/x) if x != 0 else np.nan,
                                                    'acsc': lambda x: np.arcsin(1/x) if x != 0 else np.nan,
                                                    'acoth': lambda x: 0.5 * np.log((x+1)/(x-1)) if abs(x) > 1 else np.nan,
                                                    'asech': lambda x: np.log((1+np.sqrt(1-x*x))/x) if 0 < x <= 1 else np.nan,
                                                    'acsch': lambda x: np.log((1+np.sqrt(1+x*x))/x) if x != 0 else np.nan,
                                                    'floor': np.floor, 'ceil': np.ceil, 'round': np.round,
                                                    'cbrt': lambda x: np.cbrt(x),
                                                    'gamma': lambda x: float(sp.gamma(float(x))) if x > 0 else np.nan,
                                                    'factorial': lambda x: np.math.factorial(int(x)) if x >= 0 and float(x).is_integer() else np.nan,
                                                    'min': np.minimum, 'max': np.maximum,
                                                    'sign': np.sign}])
                    else:
                        # **CORRECCIÓN: En modo grados, usar funciones que conviertan entrada de grados a radianes**
                        # Pero las constantes ya fueron convertidas en parse_equation, así que usamos funciones normales
                        func = sp.lambdify(var_symbols, eq, 
                                          modules=['numpy', 'math', 
                                                   {'sin': np.sin,  # Ya viene convertido a radianes
                                                    'cos': np.cos,
                                                    'tan': np.tan,
                                                    'asin': lambda x: np.rad2deg(np.arcsin(x)) if -1 <= x <= 1 else np.nan,
                                                    'acos': lambda x: np.rad2deg(np.arccos(x)) if -1 <= x <= 1 else np.nan,
                                                    'atan': lambda x: np.rad2deg(np.arctan(x)),
                                                    'atan2': lambda y, x: np.rad2deg(np.arctan2(y, x)),
                                                    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
                                                    'asinh': np.arcsinh, 'acosh': np.arccosh, 'atanh': np.arctanh,
                                                    'exp': np.exp, 'log': np.log, 'log10': np.log10,
                                                    'log2': lambda x: np.log2(x) if x > 0 else np.nan,
                                                    'sqrt': np.sqrt, 'abs': np.abs, 'pi': np.pi, 'e': np.e,
                                                    'cot': lambda x: 1/np.tan(x) if np.tan(x) != 0 else np.nan,
                                                    'sec': lambda x: 1/np.cos(x) if np.cos(x) != 0 else np.nan,
                                                    'csc': lambda x: 1/np.sin(x) if np.sin(x) != 0 else np.nan,
                                                    'coth': lambda x: 1/np.tanh(x) if np.tanh(x) != 0 else np.nan,
                                                    'sech': lambda x: 1/np.cosh(x),
                                                    'csch': lambda x: 1/np.sinh(x) if np.sinh(x) != 0 else np.nan,
                                                    'acot': lambda x: np.rad2deg(np.arctan(1/x)) if x != 0 else (90 if x > 0 else -90),
                                                    'asec': lambda x: np.rad2deg(np.arccos(1/x)) if abs(x) >= 1 else np.nan,
                                                    'acsc': lambda x: np.rad2deg(np.arcsin(1/x)) if abs(x) >= 1 else np.nan,
                                                    'acoth': lambda x: 0.5 * np.log((x+1)/(x-1)) if abs(x) > 1 else np.nan,
                                                    'asech': lambda x: np.log((1+np.sqrt(1-x*x))/x) if 0 < x <= 1 else np.nan,
                                                    'acsch': lambda x: np.log((1+np.sqrt(1+x*x))/x) if x != 0 else np.nan,
                                                    'floor': np.floor, 'ceil': np.ceil, 'round': np.round,
                                                    'cbrt': lambda x: np.cbrt(x),
                                                    'gamma': lambda x: float(sp.gamma(float(x))) if x > 0 else np.nan,
                                                    'factorial': lambda x: np.math.factorial(int(x)) if x >= 0 and float(x).is_integer() else np.nan,
                                                    'min': np.minimum, 'max': np.maximum,
                                                    'sign': np.sign}])
                    numpy_funcs.append(func)
                    print(f"Función creada exitosamente para ecuación en modo {self.angle_mode}")
                except Exception as e:
                    print(f"Advertencia: No se pudo crear función para {eq}: {e}")
                    def error_func(*args):
                        return float('nan')
                    numpy_funcs.append(error_func)
            
            def equations_func(vars):
                results = []
                for f in numpy_funcs:
                    try:
                        result = f(*vars)
                        if hasattr(result, '__iter__'):
                            results.append(float(result[0]) if len(result) > 0 else float(result))
                        else:
                            results.append(float(result))
                    except Exception as e:
                        print(f"Error evaluando función: {e}")
                        results.append(float('nan'))
                return results
            
            # Resultados de todos los métodos
            all_solutions = []
            methods_used = []
            
            # 1. Método más simple: Intento de solución simbólica
            if self.angle_mode == "rad":  # Solo en modo RAD
                symbolic_result = self.try_symbolic_solution()
                if symbolic_result:
                    all_solutions.extend(symbolic_result)
                    methods_used.append("simbólica")
                    if len(self.equations) == len(var_list):
                        print("Solución simbólica encontrada")
                        unique_solutions = self.filter_unique_solutions(all_solutions)
                        return {
                            "success": True, 
                            "solutions": unique_solutions,
                            "methods_used": list(set(methods_used)),
                            "total_solutions": len(unique_solutions)
                        }
            
            # 2. Método rápido: fsolve
            print("Probando método: fsolve")
            fsolve_result = self.try_fsolve(var_list, equations_func)
            if fsolve_result:
                all_solutions.extend(fsolve_result)
                methods_used.append("fsolve")
            
            # 3. Método intermedio: root con métodos simples
            print("Probando método: root-simple")
            root_result = self.try_root_methods_simple(var_list, equations_func)
            if root_result:
                all_solutions.extend(root_result)
                methods_used.append("root-simple")
            
            # 4. Método más costoso: Newton personalizado
            print("Probando método: newton")
            newton_result = self.try_newton_method(var_list, equations_func)
            if newton_result:
                all_solutions.extend(newton_result)
                methods_used.append("newton")
            
            # 5. Método costoso: least_squares
            print("Probando método: mínimos cuadrados")
            least_sq_result = self.try_least_squares(var_list, equations_func)
            if least_sq_result:
                all_solutions.extend(least_sq_result)
                methods_used.append("mínimos cuadrados")
            
            # 6. Método más costoso: root con métodos complejos (solo si no se encontraron soluciones)
            if not all_solutions:
                print("Probando método: root-complex")
                root_complex_result = self.try_root_methods_complex(var_list, equations_func)
                if root_complex_result:
                    all_solutions.extend(root_complex_result)
                    methods_used.append("root-complex")
            
            # Filtrar soluciones duplicadas
            unique_solutions = self.filter_unique_solutions(all_solutions)
            
            if unique_solutions:
                print(f"Solución encontrada. Total de soluciones únicas: {len(unique_solutions)}")
                return {
                    "success": True, 
                    "solutions": unique_solutions,
                    "methods_used": list(set(methods_used)),
                    "total_solutions": len(unique_solutions)
                }
            else:
                print("No se encontraron soluciones con ningún método")
                return {"error": "No se encontraron soluciones con ningún método"}
                    
        except Exception as e:
            print(f"Error en solve_system: {str(e)}")
            return {"error": f"Error: {str(e)}"}
    
    def try_symbolic_solution(self):
        """Intenta solución simbólica (método más liviano)"""
        try:
            var_list = sorted(list(self.variables))
            
            # Limitar el tiempo para sistemas grandes
            if len(var_list) > 5 or len(self.equations) > 5:
                print("Sistema demasiado grande para solución simbólica rápida")
                return None
                
            print("Intentando solución simbólica...")
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
                
                print("Solución simbólica encontrada")
                return result
        except Exception as e:
            print(f"Solución simbólica falló: {e}")
        return None
    
    def try_fsolve(self, var_list, equations_func):
        """Intenta solución con fsolve (método rápido)"""
        solutions = []
        guesses = self.generate_initial_guesses(len(var_list), n_guesses=5)
        
        print(f"Generando {len(guesses)} guesses iniciales para fsolve")
        
        for guess in guesses:
            try:
                def timeout_wrapper():
                    return fsolve(equations_func, guess, full_output=True, maxfev=100)
                
                future = self.executor.submit(timeout_wrapper)
                result = future.result(timeout=self.timeout)
                
                if len(result) == 4:
                    x_sol, info, ier, msg = result
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
                            print(f"fsolve encontró solución con error: {error}")
            except TimeoutError:
                print(f"fsolve timeout para guess: {guess}")
                continue
            except Exception as e:
                print(f"fsolve error para guess {guess}: {e}")
                continue
        
        return solutions
    
    def try_root_methods_simple(self, var_list, equations_func):
        """Intenta métodos root simples (hybr es el más eficiente)"""
        solutions = []
        guesses = self.generate_initial_guesses(len(var_list), n_guesses=3)
        
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
                            print(f"root-{method} encontró solución con error: {error}")
                except Exception as e:
                    print(f"root-{method} error: {e}")
                    continue
        
        return solutions
    
    def try_root_methods_complex(self, var_list, equations_func):
        """Métodos root más complejos y costosos (solo como último recurso)"""
        solutions = []
        guesses = self.generate_initial_guesses(len(var_list), n_guesses=2)
        
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
                            print(f"root-{method} encontró solución con error: {error}")
                except Exception as e:
                    print(f"root-{method} error: {e}")
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
                max_iter = 50
                tol = 1e-10
                
                for iter_count in range(max_iter):
                    f_val = np.array(equations_func(x))
                    
                    if np.linalg.norm(f_val) < tol:
                        print(f"Newton convergió en {iter_count} iteraciones")
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
                    for _ in range(5):
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
                    print(f"Newton encontró solución con error: {error}")
            except Exception as e:
                print(f"Newton error: {e}")
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
                
                if self.angle_mode == "rad":
                    bounds = (-10, 10)
                else:
                    bounds = (-360, 360)
                
                result = least_squares(residuals_func, guess, 
                                      bounds=bounds, 
                                      method='trf',
                                      max_nfev=200)
                
                if result.success:
                    error = np.linalg.norm(result.fun)
                    if error < 1e-4:
                        sol_dict = {var: float(val) for var, val in zip(var_list, result.x)}
                        solutions.append({
                            "solution": sol_dict,
                            "method": "mínimos cuadrados",
                            "error": error
                        })
                        print(f"Mínimos cuadrados encontró solución con error: {error}")
            except Exception as e:
                print(f"Mínimos cuadrados error: {e}")
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
status_queue = queue.Queue()
is_solving = False
current_method = ""
angle_mode = "rad"
angle_mode_btn = None
solving_lock = threading.Lock()
check_timer = None
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
    
    .angle-mode-btn {
        width: 60px;
        height: 32px;
        border-radius: 16px;
        background: #f8f9fa;
        color: #666;
        border: 1px solid #e0e0e0;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-right: 8px;
    }
    
    .angle-mode-btn:hover {
        background: #f0f0f0;
        border-color: #d0d0d0;
    }
    
    .angle-mode-btn.active {
        background: #000;
        color: white;
        border-color: #000;
    }
    
    .mode-indicator {
        font-size: 11px;
        color: #666;
        background: #f0f0f0;
        padding: 4px 8px;
        border-radius: 4px;
        margin-right: 8px;
        font-weight: 500;
    }
</style>
''')

# Contenedor principal
with ui.element('div').classes('app-container'):
    # Barra superior
    with ui.element('div').classes('top-bar'):
        ui.label('Advanced Equation Solver').classes('app-title')
        
        with ui.row().classes('items-center gap-2'):
            # Indicador de modo actual
            mode_indicator = ui.label('MODO: RAD').classes('mode-indicator')
            
            # Botón de modo ángulo
            angle_mode_btn = ui.button('RAD', on_click=lambda: toggle_angle_mode()).classes('angle-mode-btn')
            angle_mode_btn.props('title=Clic para cambiar entre radianes y grados')
            
            # Botón de resolver
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
                        value='sin(30) = 0.5\ncos(x) = 0.5',
                        placeholder='''UNA ECUACIÓN POR LÍNEA

IMPORTANTE: En modo DEG (grados), las constantes en funciones trigonométricas se interpretan como grados.
Ejemplo: sin(30) se interpreta como 30 grados en modo DEG.

SINTAXIS FUNCIONAL:
-------------------
Operadores: + - * / ^ **
Potencias: x^2, x**2
Trigonométricas: sin(x), cos(x), tan(x), cot(x), sec(x), csc(x)
Inversas: asin(x), acos(x), atan(x), acot(x), asec(x), acsc(x)
Hiperbólicas: sinh(x), cosh(x), tanh(x), coth(x), sech(x), csch(x)
Inversas hiperbólicas: asinh(x), acosh(x), atanh(x), acoth(x), asech(x), acsch(x)
Exponenciales: exp(x), log(x), ln(x), log10(x), log2(x)
Raíces: sqrt(x), cbrt(x)
Valor absoluto: abs(x)
Redondeo: floor(x), ceil(x), round(x)
Constantes: pi, e, inf
Funciones especiales: gamma(x), factorial(x), erf(x), erfc(x)
Mín/Max: min(x,y), max(x,y)

CÁLCULO:
Integrales: integral(x^2, x)
Integral definida: integral(x^2, (x, 0, 1))
Sumatorias: sum(i^2, (i, 1, n))
Productorias: product(i, (i, 1, n))

EJEMPLOS:
x^2 + y^2 = 25
sin(x) + cos(y) = 1
exp(x) - log(y) = 2
diff(sin(x), x) = cos(x)
min(x, y) = 5
max(x, y, z) = 10''',
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
        eq_clean = eq.lower()
        funcs = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc',
                'asin', 'acos', 'atan', 'acot', 'asec', 'acsc',
                'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
                'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch',
                'exp', 'log', 'ln', 'log10', 'log2',
                'sqrt', 'cbrt', 'abs', 'floor', 'ceil', 'round',
                'gamma', 'factorial', 'erf', 'erfc',
                'min', 'max', 'diff', 'integral', 'sum', 'product',
                'pi', 'e', 'inf']
        
        for func in funcs:
            eq_clean = eq_clean.replace(func, '')
        
        matches = re.findall(r'[a-zA-Z]', eq_clean)
        var_set.update(matches)
    
    variables = [v for v in var_set if len(v) == 1]
    
    equation_info.text = f'{len(eq_texts)} ecuaciones, {len(variables)} variables'

def toggle_angle_mode():
    """Alterna entre modo grados y radianes"""
    global angle_mode
    
    if angle_mode == "rad":
        angle_mode = "deg"
        solver.set_angle_mode("deg")
        if angle_mode_btn:
            angle_mode_btn.text = "DEG"
            angle_mode_btn.classes('active')
        if mode_indicator:
            mode_indicator.text = "MODO: DEG"
        ui.notify('Modo cambiado a GRADOS (DEG)', type='info', position='top-right', timeout=2000)
    else:
        angle_mode = "rad"
        solver.set_angle_mode("rad")
        if angle_mode_btn:
            angle_mode_btn.text = "RAD"
            angle_mode_btn.classes(remove='active')
        if mode_indicator:
            mode_indicator.text = "MODO: RAD"
        ui.notify('Modo cambiado a RADIANES (RAD)', type='info', position='top-right', timeout=2000)
    
    print(f"Modo cambiado a: {angle_mode}")
    print(f"Solver en modo: {solver.angle_mode}")

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
            ui.label(f'Probando métodos en modo {angle_mode.upper()}').classes('solving-subtext')
            
            method_counter = ui.label('Método 1 de 6').classes('method-counter')
            
            with ui.element('div').classes('method-progress'):
                progress_bar = ui.element('div').classes('method-progress-bar').style('width: 16.66%')
            
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
        for tag in method_tags:
            tag.classes(remove='active')
            tag.classes(remove='completed')
        
        for i in range(method_index):
            if i < len(method_tags):
                method_tags[i].classes('completed')
        
        method_tags[method_index].classes('active')
        
        counter_text = f'Método {method_index + 1} de {len(method_tags)}'
        if hasattr(update_solving_state, 'counter_label'):
            update_solving_state.counter_label.text = counter_text
        
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
                    
                    with ui.element('div').classes('stat-item'):
                        ui.label('Modo').classes('stat-label')
                        ui.label(angle_mode.upper()).classes('stat-value')
            
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
                    
                    with ui.element('div').classes('variables-stacked'):
                        for var, value in sorted(solution.items()):
                            with ui.element('div').classes('variable-line'):
                                ui.label(var).classes('var-name')
                                ui.label('=').classes('var-equals')
                                
                                if isinstance(value, (int, float)):
                                    formatted = f"{value:.5f}"
                                    if formatted == "-0.00000":
                                        formatted = "0.00000"
                                else:
                                    formatted = str(value)
                                
                                ui.label(formatted).classes('var-value')
                    
                    with ui.element('div').classes('solution-info'):
                        ui.label(f'Método: {method} | Error: {error:.2e} | Modo: {angle_mode.upper()}')

def solve_in_background(eq_texts):
    """Función que se ejecuta en el hilo de fondo"""
    try:
        print(f"Iniciando resolución de {len(eq_texts)} ecuaciones...")
        print(f"Modo ángulo actual del solver: {solver.angle_mode}")
        
        methods_order = [
            "simbólica",
            "fsolve", 
            "root-simple",
            "newton",
            "mínimos cuadrados",
            "root-complex"
        ]
        
        for i, method in enumerate(methods_order):
            time.sleep(0.5)
            if status_queue:
                status_queue.put(i)
        
        result = solver.solve_system(eq_texts)
        print(f"Resultado obtenido en modo {solver.angle_mode}: {result}")
        result_queue.put(result)
        
    except Exception as e:
        print(f"Error en el hilo de fondo: {e}")
        result_queue.put({"error": f"Error en el hilo de fondo: {str(e)}"})

def check_result():
    """Función que verifica si hay resultados en la cola"""
    global is_solving, check_timer, status_timer
    
    if not result_queue.empty():
        result = result_queue.get()
        
        with solving_lock:
            is_solving = False
        
        play_btn.classes(remove='loading')
        update_results(result)
        
        if check_timer:
            check_timer.deactivate()
            check_timer = None
        
        if status_timer:
            status_timer.deactivate()
            status_timer = None
        
        return False
    
    return True

def check_status():
    """Función que verifica si hay actualizaciones de estado"""
    if not status_queue.empty():
        method_index = status_queue.get()
        update_solving_state(method_index)
    
    return True

def solve_equations():
    """Función principal para resolver ecuaciones"""
    global is_solving, check_timer, status_timer, current_method_index
    
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
    
    print(f"Iniciando resolución de {len(eq_texts)} ecuaciones en modo {angle_mode}...")
    play_btn.classes('loading')
    
    current_method_index = 0
    
    show_solving_state()
    
    if results_container:
        with results_container:
            for element in results_container.default_slot.children:
                if hasattr(element, 'text') and 'Método 1 de 6' in element.text:
                    update_solving_state.counter_label = element
                elif hasattr(element, 'style') and 'method-progress-bar' in element.classes:
                    update_solving_state.progress_bar = element
    
    if check_timer:
        check_timer.deactivate()
        check_timer = None
    
    if status_timer:
        status_timer.deactivate()
        status_timer = None
    
    thread = threading.Thread(target=solve_in_background, args=(eq_texts,), daemon=True)
    thread.start()
    
    check_timer = ui.timer(0.5, check_result)
    status_timer = ui.timer(0.1, check_status)
    
    def simulate_progress():
        global current_method_index
        if current_method_index < 6:
            update_solving_state(current_method_index)
            current_method_index += 1
    
    ui.timer(1.0, lambda: simulate_progress() if is_solving else False)

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
