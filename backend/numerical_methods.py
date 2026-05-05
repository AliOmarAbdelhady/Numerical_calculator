from __future__ import annotations

import ast
import math
from typing import Any, Callable


class NumericalError(ValueError):
    """Raised when numerical input is invalid or a method cannot proceed."""


ALLOWED_FUNCTIONS = {
    name: getattr(math, name)
    for name in (
        "acos",
        "asin",
        "atan",
        "ceil",
        "cos",
        "cosh",
        "degrees",
        "exp",
        "fabs",
        "floor",
        "log",
        "log10",
        "radians",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    )
}
ALLOWED_FUNCTIONS.update({"abs": abs, "ln": math.log})
ALLOWED_CONSTANTS = {"e": math.e, "pi": math.pi, "tau": math.tau}

ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Load,
    ast.Name,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)


def _ensure_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NumericalError(f"{name} must be a number.") from exc
    if not math.isfinite(number):
        raise NumericalError(f"{name} must be finite.")
    return number


def _ensure_positive_number(value: Any, name: str) -> float:
    number = _ensure_number(value, name)
    if number <= 0:
        raise NumericalError(f"{name} must be greater than zero.")
    return number


def _ensure_iterations(value: Any) -> int:
    try:
        iterations = int(value)
    except (TypeError, ValueError) as exc:
        raise NumericalError("maxIterations must be an integer.") from exc
    if iterations <= 0:
        raise NumericalError("maxIterations must be greater than zero.")
    if iterations > 500:
        raise NumericalError("maxIterations cannot exceed 500.")
    return iterations


def _finite_value(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise NumericalError(f"{label} produced a non-finite value.")
    return value


class _ExpressionValidator(ast.NodeVisitor):
    def __init__(self, variable: str) -> None:
        self.variable = variable

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, ALLOWED_NODES):
            raise NumericalError(f"Unsupported expression element: {type(node).__name__}.")
        super().generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, (int, float)):
            raise NumericalError("Expressions may only contain numeric constants.")

    def visit_Name(self, node: ast.Name) -> None:
        allowed_names = {self.variable, *ALLOWED_FUNCTIONS.keys(), *ALLOWED_CONSTANTS.keys()}
        if node.id not in allowed_names:
            raise NumericalError(f"Unknown symbol '{node.id}'.")

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise NumericalError("Only supported math functions can be called.")
        if node.keywords:
            raise NumericalError("Function keyword arguments are not supported.")
        for arg in node.args:
            self.visit(arg)


def compile_expression(expression: str, variable: str = "x") -> Callable[[float], float]:
    if not isinstance(expression, str) or not expression.strip():
        raise NumericalError("Expression is required.")

    normalized = expression.replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise NumericalError("Expression syntax is invalid.") from exc

    _ExpressionValidator(variable).visit(tree)
    compiled = compile(tree, "<expression>", "eval")

    def evaluate(value: float) -> float:
        scope = {**ALLOWED_FUNCTIONS, **ALLOWED_CONSTANTS, variable: value}
        try:
            result = eval(compiled, {"__builtins__": {}}, scope)
        except ZeroDivisionError as exc:
            raise NumericalError("Expression divided by zero during evaluation.") from exc
        except ValueError as exc:
            raise NumericalError("Expression is outside its mathematical domain.") from exc
        except OverflowError as exc:
            raise NumericalError("Expression overflowed during evaluation.") from exc
        return _finite_value(float(result), "Expression")

    return evaluate


def _evaluate_with_derivative(node: ast.AST, value: float, variable: str) -> tuple[float, float]:
    if isinstance(node, ast.Expression):
        return _evaluate_with_derivative(node.body, value, variable)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise NumericalError("Expressions may only contain numeric constants.")
        return float(node.value), 0.0

    if isinstance(node, ast.Name):
        if node.id == variable:
            return value, 1.0
        if node.id in ALLOWED_CONSTANTS:
            return ALLOWED_CONSTANTS[node.id], 0.0
        raise NumericalError(f"Unknown symbol '{node.id}'.")

    if isinstance(node, ast.UnaryOp):
        operand_value, operand_derivative = _evaluate_with_derivative(node.operand, value, variable)
        if isinstance(node.op, ast.USub):
            return -operand_value, -operand_derivative
        if isinstance(node.op, ast.UAdd):
            return operand_value, operand_derivative
        raise NumericalError("Unsupported unary operation.")

    if isinstance(node, ast.BinOp):
        left_value, left_derivative = _evaluate_with_derivative(node.left, value, variable)
        right_value, right_derivative = _evaluate_with_derivative(node.right, value, variable)

        if isinstance(node.op, ast.Add):
            return left_value + right_value, left_derivative + right_derivative
        if isinstance(node.op, ast.Sub):
            return left_value - right_value, left_derivative - right_derivative
        if isinstance(node.op, ast.Mult):
            return left_value * right_value, left_derivative * right_value + left_value * right_derivative
        if isinstance(node.op, ast.Div):
            return left_value / right_value, (left_derivative * right_value - left_value * right_derivative) / (right_value**2)
        if isinstance(node.op, ast.Pow):
            result = left_value**right_value
            if right_derivative == 0:
                return result, right_value * (left_value ** (right_value - 1)) * left_derivative
            if left_value <= 0:
                raise NumericalError("Automatic derivative for variable exponents requires a positive base.")
            return result, result * (right_derivative * math.log(left_value) + right_value * left_derivative / left_value)
        if isinstance(node.op, ast.Mod):
            raise NumericalError("Modulo is not supported by automatic differentiation.")
        raise NumericalError("Unsupported binary operation.")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise NumericalError("Only supported math functions can be called.")
        if len(node.args) != 1:
            raise NumericalError("Automatic derivative supports one-argument math functions.")

        function_name = node.func.id
        argument_value, argument_derivative = _evaluate_with_derivative(node.args[0], value, variable)
        function_value = float(ALLOWED_FUNCTIONS[function_name](argument_value))

        derivative_rules: dict[str, Callable[[float], float]] = {
            "sin": math.cos,
            "cos": lambda argument: -math.sin(argument),
            "tan": lambda argument: 1 / (math.cos(argument) ** 2),
            "asin": lambda argument: 1 / math.sqrt(1 - argument**2),
            "acos": lambda argument: -1 / math.sqrt(1 - argument**2),
            "atan": lambda argument: 1 / (1 + argument**2),
            "sinh": math.cosh,
            "cosh": math.sinh,
            "tanh": lambda argument: 1 / (math.cosh(argument) ** 2),
            "exp": math.exp,
            "ln": lambda argument: 1 / argument,
            "log": lambda argument: 1 / argument,
            "log10": lambda argument: 1 / (argument * math.log(10)),
            "sqrt": lambda argument: 1 / (2 * math.sqrt(argument)),
            "degrees": lambda _argument: 180 / math.pi,
            "radians": lambda _argument: math.pi / 180,
        }

        if function_name in ("abs", "fabs"):
            if argument_value == 0:
                raise NumericalError("Automatic derivative of abs is undefined at zero.")
            return function_value, (1 if argument_value > 0 else -1) * argument_derivative

        derivative_rule = derivative_rules.get(function_name)
        if derivative_rule is None:
            raise NumericalError(f"Automatic derivative does not support {function_name}().")

        return function_value, derivative_rule(argument_value) * argument_derivative

    raise NumericalError(f"Unsupported expression element: {type(node).__name__}.")


def compile_derivative(expression: str, variable: str = "x") -> Callable[[float], float]:
    if not isinstance(expression, str) or not expression.strip():
        raise NumericalError("Expression is required.")

    normalized = expression.replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise NumericalError("Expression syntax is invalid.") from exc

    _ExpressionValidator(variable).visit(tree)

    def evaluate(value: float) -> float:
        try:
            _, derivative = _evaluate_with_derivative(tree, value, variable)
        except ZeroDivisionError as exc:
            raise NumericalError("Derivative divided by zero during evaluation.") from exc
        except ValueError as exc:
            raise NumericalError("Derivative is outside its mathematical domain.") from exc
        except OverflowError as exc:
            raise NumericalError("Derivative overflowed during evaluation.") from exc
        return _finite_value(float(derivative), "Derivative")

    return evaluate


def bisection_method(params: dict[str, Any]) -> dict[str, Any]:
    f = compile_expression(params.get("expression", ""))
    a = _ensure_number(params.get("a"), "a")
    b = _ensure_number(params.get("b"), "b")
    tolerance = _ensure_positive_number(params.get("tolerance", 1e-6), "tolerance")
    max_iterations = _ensure_iterations(params.get("maxIterations", 50))

    if a >= b:
        raise NumericalError("a must be less than b.")

    fa = f(a)
    fb = f(b)

    if abs(fa) <= tolerance:
        return {
            "method": "Bisection Method",
            "root": a,
            "value": fa,
            "iterations": 0,
            "converged": True,
            "steps": [],
        }
    if abs(fb) <= tolerance:
        return {
            "method": "Bisection Method",
            "root": b,
            "value": fb,
            "iterations": 0,
            "converged": True,
            "steps": [],
        }
    if fa * fb > 0:
        raise NumericalError("f(a) and f(b) must have opposite signs.")

    steps: list[dict[str, float | int]] = []
    root = a
    value = fa
    converged = False

    for iteration in range(1, max_iterations + 1):
        c = (a + b) / 2
        fc = f(c)
        error = abs(b - a) / 2
        steps.append(
            {
                "iteration": iteration,
                "a": a,
                "b": b,
                "c": c,
                "fA": fa,
                "fB": fb,
                "fC": fc,
                "error": error,
            }
        )
        root = c
        value = fc

        if abs(fc) <= tolerance or error <= tolerance:
            converged = True
            break
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return {
        "method": "Bisection Method",
        "root": root,
        "value": value,
        "iterations": len(steps),
        "converged": converged,
        "tolerance": tolerance,
        "steps": steps,
    }


def secant_method(params: dict[str, Any]) -> dict[str, Any]:
    f = compile_expression(params.get("expression", ""))
    x_prev = _ensure_number(params.get("x0"), "x0")
    x_curr = _ensure_number(params.get("x1"), "x1")
    tolerance = _ensure_positive_number(params.get("tolerance", 1e-6), "tolerance")
    max_iterations = _ensure_iterations(params.get("maxIterations", 50))

    steps: list[dict[str, float | int]] = []
    root = x_curr
    value = f(x_curr)
    converged = False

    for iteration in range(1, max_iterations + 1):
        f_prev = f(x_prev)
        f_curr = f(x_curr)
        denominator = f_curr - f_prev
        if abs(denominator) < 1e-15:
            raise NumericalError("Secant denominator is too close to zero.")

        x_next = x_curr - f_curr * (x_curr - x_prev) / denominator
        f_next = f(x_next)
        error = abs(x_next - x_curr)
        steps.append(
            {
                "iteration": iteration,
                "xPrevious": x_prev,
                "xCurrent": x_curr,
                "fPrevious": f_prev,
                "fCurrent": f_curr,
                "xNext": x_next,
                "fNext": f_next,
                "error": error,
            }
        )
        root = x_next
        value = f_next

        if abs(f_next) <= tolerance or error <= tolerance:
            converged = True
            break
        x_prev, x_curr = x_curr, x_next

    return {
        "method": "Secant Method",
        "root": root,
        "value": value,
        "iterations": len(steps),
        "converged": converged,
        "tolerance": tolerance,
        "steps": steps,
    }


def simple_iteration_method(params: dict[str, Any]) -> dict[str, Any]:
    g = compile_expression(params.get("gExpression", ""))
    x_curr = _ensure_number(params.get("x0"), "x0")
    tolerance = _ensure_positive_number(params.get("tolerance", 1e-6), "tolerance")
    max_iterations = _ensure_iterations(params.get("maxIterations", 50))

    steps: list[dict[str, float | int]] = []
    fixed_point = x_curr
    converged = False

    for iteration in range(1, max_iterations + 1):
        x_next = g(x_curr)
        error = abs(x_next - x_curr)
        steps.append(
            {
                "iteration": iteration,
                "xCurrent": x_curr,
                "xNext": x_next,
                "gValue": x_next,
                "error": error,
            }
        )
        fixed_point = x_next
        if error <= tolerance:
            converged = True
            break
        x_curr = x_next

    return {
        "method": "Method of Simple Iteration",
        "fixedPoint": fixed_point,
        "iterations": len(steps),
        "converged": converged,
        "tolerance": tolerance,
        "steps": steps,
    }


def newton_raphson_method(params: dict[str, Any]) -> dict[str, Any]:
    expression = params.get("expression", "")
    f = compile_expression(expression)
    derivative_expression = params.get("derivativeExpression")
    has_manual_derivative = isinstance(derivative_expression, str) and derivative_expression.strip()
    derivative = compile_expression(derivative_expression) if has_manual_derivative else compile_derivative(expression)
    x_curr = _ensure_number(params.get("x0"), "x0")
    tolerance = _ensure_positive_number(params.get("tolerance", 1e-6), "tolerance")
    max_iterations = _ensure_iterations(params.get("maxIterations", 50))

    steps: list[dict[str, float | int]] = []
    root = x_curr
    value = f(x_curr)
    converged = False

    if abs(value) <= tolerance:
        return {
            "method": "The Newton-Raphson Method",
            "root": root,
            "value": value,
            "iterations": 0,
            "converged": True,
            "tolerance": tolerance,
            "derivativeMode": "manual" if has_manual_derivative else "auto",
            "steps": steps,
        }

    for iteration in range(max_iterations):
        fx = f(x_curr)
        dfx = derivative(x_curr)
        if abs(dfx) < 1e-15:
            raise NumericalError("Derivative is too close to zero.")

        x_next = x_curr - fx / dfx
        f_next = f(x_next)
        delta = x_next - x_curr
        error = abs(delta)
        steps.append(
            {
                "iteration": iteration,
                "xCurrent": x_curr,
                "fCurrent": fx,
                "derivative": dfx,
                "xNext": x_next,
                "delta": delta,
                "fNext": f_next,
                "error": error,
            }
        )
        root = x_next
        value = f_next
        if abs(f_next) <= tolerance or error <= tolerance:
            converged = True
            break
        x_curr = x_next

    return {
        "method": "The Newton-Raphson Method",
        "root": root,
        "value": value,
        "iterations": len(steps),
        "converged": converged,
        "tolerance": tolerance,
        "derivativeMode": "manual" if has_manual_derivative else "auto",
        "steps": steps,
    }


def _validate_linear_system(params: dict[str, Any]) -> tuple[list[list[float]], list[float], list[float], float, int]:
    raw_matrix = params.get("matrix")
    raw_vector = params.get("vector")
    if not isinstance(raw_matrix, list) or not raw_matrix:
        raise NumericalError("matrix is required.")
    if not isinstance(raw_vector, list):
        raise NumericalError("vector is required.")

    matrix: list[list[float]] = []
    size = len(raw_matrix)
    for i, row in enumerate(raw_matrix):
        if not isinstance(row, list) or len(row) != size:
            raise NumericalError("matrix must be square.")
        matrix.append([_ensure_number(value, f"A[{i + 1},{j + 1}]") for j, value in enumerate(row)])

    if len(raw_vector) != size:
        raise NumericalError("vector length must match matrix size.")
    vector = [_ensure_number(value, f"b[{i + 1}]") for i, value in enumerate(raw_vector)]

    raw_initial = params.get("initial")
    if raw_initial is None or raw_initial == []:
        initial = [0.0] * size
    elif isinstance(raw_initial, list) and len(raw_initial) == size:
        initial = [_ensure_number(value, f"x0[{i + 1}]") for i, value in enumerate(raw_initial)]
    else:
        raise NumericalError("initial vector length must match matrix size.")

    for i in range(size):
        if abs(matrix[i][i]) < 1e-15:
            raise NumericalError(f"Diagonal value A[{i + 1},{i + 1}] cannot be zero.")

    tolerance = _ensure_positive_number(params.get("tolerance", 1e-6), "tolerance")
    max_iterations = _ensure_iterations(params.get("maxIterations", 50))
    return matrix, vector, initial, tolerance, max_iterations


def _residual_norm(matrix: list[list[float]], vector: list[float], x_values: list[float]) -> float:
    residuals = [
        abs(sum(row[j] * x_values[j] for j in range(len(x_values))) - vector[i])
        for i, row in enumerate(matrix)
    ]
    return max(residuals)


def _diagonal_dominance_warning(matrix: list[list[float]]) -> str | None:
    weakly_dominant = True
    strictly_dominant_row = False
    for i, row in enumerate(matrix):
        diagonal = abs(row[i])
        off_diagonal = sum(abs(value) for j, value in enumerate(row) if j != i)
        if diagonal < off_diagonal:
            weakly_dominant = False
        if diagonal > off_diagonal:
            strictly_dominant_row = True
    if not weakly_dominant or not strictly_dominant_row:
        return "The matrix is not strictly diagonally dominant; convergence is not guaranteed."
    return None


def jacobi_method(params: dict[str, Any]) -> dict[str, Any]:
    matrix, vector, x_old, tolerance, max_iterations = _validate_linear_system(params)
    steps: list[dict[str, Any]] = []
    solution = x_old[:]
    converged = False

    for iteration in range(1, max_iterations + 1):
        x_new = []
        for i, row in enumerate(matrix):
            sigma = sum(row[j] * x_old[j] for j in range(len(row)) if j != i)
            x_new.append((vector[i] - sigma) / row[i])

        error = max(abs(x_new[i] - x_old[i]) for i in range(len(x_new)))
        residual = _residual_norm(matrix, vector, x_new)
        steps.append(
            {
                "iteration": iteration,
                "previous": x_old[:],
                "current": x_new[:],
                "error": error,
                "residual": residual,
            }
        )
        solution = x_new
        if error <= tolerance or residual <= tolerance:
            converged = True
            break
        x_old = x_new

    return {
        "method": "Jacobi Method",
        "solution": solution,
        "iterations": len(steps),
        "converged": converged,
        "tolerance": tolerance,
        "residual": _residual_norm(matrix, vector, solution),
        "warning": _diagonal_dominance_warning(matrix),
        "steps": steps,
    }


def gauss_seidel_method(params: dict[str, Any]) -> dict[str, Any]:
    matrix, vector, x_old, tolerance, max_iterations = _validate_linear_system(params)
    steps: list[dict[str, Any]] = []
    solution = x_old[:]
    converged = False

    for iteration in range(1, max_iterations + 1):
        x_new = x_old[:]
        for i, row in enumerate(matrix):
            before = sum(row[j] * x_new[j] for j in range(i))
            after = sum(row[j] * x_old[j] for j in range(i + 1, len(row)))
            x_new[i] = (vector[i] - before - after) / row[i]

        error = max(abs(x_new[i] - x_old[i]) for i in range(len(x_new)))
        residual = _residual_norm(matrix, vector, x_new)
        steps.append(
            {
                "iteration": iteration,
                "previous": x_old[:],
                "current": x_new[:],
                "error": error,
                "residual": residual,
            }
        )
        solution = x_new
        if error <= tolerance or residual <= tolerance:
            converged = True
            break
        x_old = x_new

    return {
        "method": "Gauss-Seidel Method",
        "solution": solution,
        "iterations": len(steps),
        "converged": converged,
        "tolerance": tolerance,
        "residual": _residual_norm(matrix, vector, solution),
        "warning": _diagonal_dominance_warning(matrix),
        "steps": steps,
    }


def _poly_multiply(left: list[float], right: list[float]) -> list[float]:
    result = [0.0] * (len(left) + len(right) - 1)
    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            result[i + j] += left_value * right_value
    return result


def _format_polynomial(coefficients: list[float]) -> str:
    terms: list[str] = []
    for power in range(len(coefficients) - 1, -1, -1):
        coefficient = coefficients[power]
        if abs(coefficient) < 1e-12:
            continue
        sign = "-" if coefficient < 0 else "+"
        magnitude = abs(coefficient)
        if power == 0:
            body = f"{magnitude:.10g}"
        elif power == 1:
            body = "x" if abs(magnitude - 1) < 1e-12 else f"{magnitude:.10g}x"
        else:
            body = f"x^{power}" if abs(magnitude - 1) < 1e-12 else f"{magnitude:.10g}x^{power}"
        terms.append(f"{sign} {body}")
    if not terms:
        return "0"
    polynomial = " ".join(terms)
    return polynomial[2:] if polynomial.startswith("+ ") else polynomial


def lagrange_interpolation(params: dict[str, Any]) -> dict[str, Any]:
    raw_points = params.get("points")
    target = _ensure_number(params.get("target"), "target")
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        raise NumericalError("At least two interpolation points are required.")

    points: list[tuple[float, float]] = []
    seen_x: set[float] = set()
    for index, point in enumerate(raw_points):
        if not isinstance(point, dict):
            raise NumericalError("Each point must contain x and y.")
        x_value = _ensure_number(point.get("x"), f"x[{index + 1}]")
        y_value = _ensure_number(point.get("y"), f"y[{index + 1}]")
        if x_value in seen_x:
            raise NumericalError("Interpolation x values must be distinct.")
        seen_x.add(x_value)
        points.append((x_value, y_value))

    total = 0.0
    steps: list[dict[str, Any]] = []
    coefficients = [0.0] * len(points)

    for i, (xi, yi) in enumerate(points):
        basis_at_target = 1.0
        denominator = 1.0
        basis_coefficients = [1.0]
        factors: list[dict[str, float | int]] = []

        for j, (xj, _) in enumerate(points):
            if i == j:
                continue
            numerator = target - xj
            divisor = xi - xj
            factor = numerator / divisor
            basis_at_target *= factor
            denominator *= divisor
            basis_coefficients = _poly_multiply(basis_coefficients, [-xj, 1.0])
            factors.append(
                {
                    "point": j + 1,
                    "numerator": numerator,
                    "denominator": divisor,
                    "factor": factor,
                }
            )

        term = yi * basis_at_target
        total += term
        scale = yi / denominator
        for power, coefficient in enumerate(basis_coefficients):
            coefficients[power] += scale * coefficient

        steps.append(
            {
                "iteration": i + 1,
                "x": xi,
                "y": yi,
                "basis": basis_at_target,
                "term": term,
                "factors": factors,
            }
        )

    return {
        "method": "Lagrange Interpolation",
        "target": target,
        "value": total,
        "polynomial": _format_polynomial(coefficients),
        "steps": steps,
    }


METHODS = {
    "bisection": bisection_method,
    "secant": secant_method,
    "iteration": simple_iteration_method,
    "newton": newton_raphson_method,
    "jacobi": jacobi_method,
    "gauss_seidel": gauss_seidel_method,
    "lagrange": lagrange_interpolation,
}


def calculate(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method not in METHODS:
        raise NumericalError("Unknown numerical method.")
    return METHODS[method](params)
