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

PIVOT_TOLERANCE = 1e-12
NEWTON_DERIVATIVE_TOLERANCE = 1e-15
NEWTON_FORMULA = "x(i+1) = x(i) - f(x(i)) / f'(x(i))"


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


def _evaluate_with_derivatives(node: ast.AST, value: float, variable: str) -> tuple[float, float, float]:
    if isinstance(node, ast.Expression):
        return _evaluate_with_derivatives(node.body, value, variable)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise NumericalError("Expressions may only contain numeric constants.")
        return float(node.value), 0.0, 0.0

    if isinstance(node, ast.Name):
        if node.id == variable:
            return value, 1.0, 0.0
        if node.id in ALLOWED_CONSTANTS:
            return ALLOWED_CONSTANTS[node.id], 0.0, 0.0
        raise NumericalError(f"Unknown symbol '{node.id}'.")

    if isinstance(node, ast.UnaryOp):
        operand_value, operand_derivative, operand_second_derivative = _evaluate_with_derivatives(node.operand, value, variable)
        if isinstance(node.op, ast.USub):
            return -operand_value, -operand_derivative, -operand_second_derivative
        if isinstance(node.op, ast.UAdd):
            return operand_value, operand_derivative, operand_second_derivative
        raise NumericalError("Unsupported unary operation.")

    if isinstance(node, ast.BinOp):
        left_value, left_derivative, left_second_derivative = _evaluate_with_derivatives(node.left, value, variable)
        right_value, right_derivative, right_second_derivative = _evaluate_with_derivatives(node.right, value, variable)

        if isinstance(node.op, ast.Add):
            return left_value + right_value, left_derivative + right_derivative, left_second_derivative + right_second_derivative
        if isinstance(node.op, ast.Sub):
            return left_value - right_value, left_derivative - right_derivative, left_second_derivative - right_second_derivative
        if isinstance(node.op, ast.Mult):
            return (
                left_value * right_value,
                left_derivative * right_value + left_value * right_derivative,
                left_second_derivative * right_value + 2 * left_derivative * right_derivative + left_value * right_second_derivative,
            )
        if isinstance(node.op, ast.Div):
            reciprocal = 1 / right_value
            reciprocal_derivative = -right_derivative / (right_value**2)
            reciprocal_second_derivative = (2 * right_derivative**2) / (right_value**3) - right_second_derivative / (right_value**2)
            return (
                left_value * reciprocal,
                left_derivative * reciprocal + left_value * reciprocal_derivative,
                left_second_derivative * reciprocal + 2 * left_derivative * reciprocal_derivative + left_value * reciprocal_second_derivative,
            )
        if isinstance(node.op, ast.Pow):
            result = left_value**right_value
            if right_derivative == 0 and right_second_derivative == 0:
                if right_value == 0:
                    return result, 0.0, 0.0
                if right_value == 1:
                    return result, left_derivative, left_second_derivative
                return (
                    result,
                    right_value * (left_value ** (right_value - 1)) * left_derivative,
                    right_value * (right_value - 1) * (left_value ** (right_value - 2)) * (left_derivative**2)
                    + right_value * (left_value ** (right_value - 1)) * left_second_derivative,
                )
            if left_value <= 0:
                raise NumericalError("Automatic derivative for variable exponents requires a positive base.")
            log_left = math.log(left_value)
            exponent_derivative = right_derivative * log_left + right_value * left_derivative / left_value
            exponent_second_derivative = (
                right_second_derivative * log_left
                + 2 * right_derivative * left_derivative / left_value
                + right_value * (left_second_derivative / left_value - (left_derivative / left_value) ** 2)
            )
            return result, result * exponent_derivative, result * (exponent_derivative**2 + exponent_second_derivative)
        if isinstance(node.op, ast.Mod):
            raise NumericalError("Modulo is not supported by automatic differentiation.")
        raise NumericalError("Unsupported binary operation.")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise NumericalError("Only supported math functions can be called.")
        if len(node.args) != 1:
            raise NumericalError("Automatic derivative supports one-argument math functions.")

        function_name = node.func.id
        argument_value, argument_derivative, argument_second_derivative = _evaluate_with_derivatives(node.args[0], value, variable)
        function_value = float(ALLOWED_FUNCTIONS[function_name](argument_value))

        derivative_rules: dict[str, tuple[Callable[[float], float], Callable[[float], float]]] = {
            "sin": (math.cos, lambda argument: -math.sin(argument)),
            "cos": (lambda argument: -math.sin(argument), lambda argument: -math.cos(argument)),
            "tan": (
                lambda argument: 1 / (math.cos(argument) ** 2),
                lambda argument: 2 * math.tan(argument) / (math.cos(argument) ** 2),
            ),
            "asin": (
                lambda argument: 1 / math.sqrt(1 - argument**2),
                lambda argument: argument / ((1 - argument**2) ** 1.5),
            ),
            "acos": (
                lambda argument: -1 / math.sqrt(1 - argument**2),
                lambda argument: -argument / ((1 - argument**2) ** 1.5),
            ),
            "atan": (
                lambda argument: 1 / (1 + argument**2),
                lambda argument: -2 * argument / ((1 + argument**2) ** 2),
            ),
            "sinh": (math.cosh, math.sinh),
            "cosh": (math.sinh, math.cosh),
            "tanh": (
                lambda argument: 1 / (math.cosh(argument) ** 2),
                lambda argument: -2 * math.tanh(argument) / (math.cosh(argument) ** 2),
            ),
            "exp": (math.exp, math.exp),
            "ln": (lambda argument: 1 / argument, lambda argument: -1 / (argument**2)),
            "log": (lambda argument: 1 / argument, lambda argument: -1 / (argument**2)),
            "log10": (lambda argument: 1 / (argument * math.log(10)), lambda argument: -1 / ((argument**2) * math.log(10))),
            "sqrt": (lambda argument: 1 / (2 * math.sqrt(argument)), lambda argument: -1 / (4 * (argument**1.5))),
            "degrees": (lambda _argument: 180 / math.pi, lambda _argument: 0.0),
            "radians": (lambda _argument: math.pi / 180, lambda _argument: 0.0),
        }

        if function_name in ("abs", "fabs"):
            if argument_value == 0:
                raise NumericalError("Automatic derivative of abs is undefined at zero.")
            sign = 1 if argument_value > 0 else -1
            return function_value, sign * argument_derivative, sign * argument_second_derivative

        derivative_rule_pair = derivative_rules.get(function_name)
        if derivative_rule_pair is None:
            raise NumericalError(f"Automatic derivative does not support {function_name}().")

        first_rule, second_rule = derivative_rule_pair
        return (
            function_value,
            first_rule(argument_value) * argument_derivative,
            second_rule(argument_value) * (argument_derivative**2) + first_rule(argument_value) * argument_second_derivative,
        )

    raise NumericalError(f"Unsupported expression element: {type(node).__name__}.")


def _evaluate_with_derivative(node: ast.AST, value: float, variable: str) -> tuple[float, float]:
    function_value, derivative, _second_derivative = _evaluate_with_derivatives(node, value, variable)
    return function_value, derivative


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


def compile_second_derivative(expression: str, variable: str = "x") -> Callable[[float], float]:
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
            _, _, second_derivative = _evaluate_with_derivatives(tree, value, variable)
        except ZeroDivisionError as exc:
            raise NumericalError("Second derivative divided by zero during evaluation.") from exc
        except ValueError as exc:
            raise NumericalError("Second derivative is outside its mathematical domain.") from exc
        except OverflowError as exc:
            raise NumericalError("Second derivative overflowed during evaluation.") from exc
        return _finite_value(float(second_derivative), "Second derivative")

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


def _is_wrapped_expression(expression: str) -> bool:
    if len(expression) < 2:
        return False

    pairs = {"(": ")", "[": "]"}
    opening = expression[0]
    if opening not in pairs or expression[-1] != pairs[opening]:
        return False

    stack: list[str] = []
    for index, char in enumerate(expression):
        if char in pairs:
            stack.append(char)
            continue
        if char in pairs.values():
            if not stack or pairs[stack[-1]] != char:
                return False
            stack.pop()
            if not stack and index != len(expression) - 1:
                return False

    return not stack


def _split_top_level_expressions(raw_expression: str) -> list[str]:
    expression = raw_expression.strip()
    while _is_wrapped_expression(expression):
        expression = expression[1:-1].strip()

    parts: list[str] = []
    start = 0
    depth = 0
    pairs = {"(": ")", "[": "]"}
    closing = set(pairs.values())

    for index, char in enumerate(expression):
        if char in pairs:
            depth += 1
            continue
        if char in closing:
            depth -= 1
            if depth < 0:
                raise NumericalError("Expression syntax is invalid.")
            continue
        if depth == 0 and char in {",", ";", "\n"}:
            part = expression[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1

    if depth != 0:
        raise NumericalError("Expression syntax is invalid.")

    tail = expression[start:].strip()
    if tail:
        parts.append(tail)

    return parts


def _split_top_level_equation(expression: str) -> tuple[str, str] | None:
    depth = 0
    equals_index: int | None = None
    pairs = {"(": ")", "[": "]"}
    closing = set(pairs.values())

    for index, char in enumerate(expression):
        if char in pairs:
            depth += 1
            continue
        if char in closing:
            depth -= 1
            if depth < 0:
                raise NumericalError("Expression syntax is invalid.")
            continue
        if depth == 0 and char == "=":
            if equals_index is not None:
                raise NumericalError("Only one equals sign is allowed in an equation.")
            equals_index = index

    if depth != 0:
        raise NumericalError("Expression syntax is invalid.")
    if equals_index is None:
        return None

    left = expression[:equals_index].strip()
    right = expression[equals_index + 1 :].strip()
    if not left or not right:
        raise NumericalError("Equation must have expressions on both sides of '='.")
    return left, right


def _root_expression(expression: Any) -> str:
    if not isinstance(expression, str):
        raise NumericalError("Expression is required.")

    equation = _split_top_level_equation(expression.strip())
    if equation is None:
        return expression

    left, right = equation
    return f"({left}) - ({right})"


def _parse_expression_tree(expression: str) -> ast.Expression:
    normalized = expression.replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise NumericalError("Expression syntax is invalid.") from exc

    _ExpressionValidator("x").visit(tree)
    return tree


def _is_variable_node(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "x"


def _multiplication_factors(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return [*_multiplication_factors(node.left), *_multiplication_factors(node.right)]
    return [node]


def _multiply_factor_expressions(factors: list[ast.AST]) -> str:
    if not factors:
        return "1"
    return " * ".join(f"({ast.unparse(factor)})" for factor in factors)


def _expression_without_one_x_factor(node: ast.AST) -> str | None:
    factors = _multiplication_factors(node)
    for index, factor in enumerate(factors):
        if _is_variable_node(factor):
            return _multiply_factor_expressions([*factors[:index], *factors[index + 1 :]])
    return None


def _unique_expressions(expressions: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for expression in expressions:
        key = "".join(expression.split())
        if key in seen:
            continue
        seen.add(key)
        unique.append(expression)
    return unique


def _fixed_point_expressions(expression: str) -> list[str]:
    equation = _split_top_level_equation(expression)
    if equation is None:
        return [expression]

    left, right = equation
    left_tree = _parse_expression_tree(left)
    right_tree = _parse_expression_tree(right)
    candidates: list[str] = []

    if _is_variable_node(left_tree.body):
        candidates.append(right)
    if _is_variable_node(right_tree.body):
        candidates.append(left)

    left_denominator = _expression_without_one_x_factor(left_tree.body)
    if left_denominator is not None and not _is_variable_node(left_tree.body):
        candidates.append(f"({right}) / ({left_denominator})")

    right_denominator = _expression_without_one_x_factor(right_tree.body)
    if right_denominator is not None and not _is_variable_node(right_tree.body):
        candidates.append(f"({left}) / ({right_denominator})")

    candidates = _unique_expressions(candidates)
    if not candidates:
        raise NumericalError("Equation could not be rearranged automatically. Enter one or more g(x) expressions instead.")
    return candidates


def _simple_iteration_expressions(params: dict[str, Any]) -> list[str]:
    raw_expressions = params.get("gExpressions")
    if raw_expressions is None:
        raw_expression = params.get("gExpression", "")
        if not isinstance(raw_expression, str):
            raise NumericalError("gExpression must be an expression.")
        raw_parts = _split_top_level_expressions(raw_expression)
    else:
        if not isinstance(raw_expressions, list):
            raise NumericalError("gExpressions must be a list of expressions.")
        raw_parts = []
        for index, raw_expression in enumerate(raw_expressions):
            if not isinstance(raw_expression, str):
                raise NumericalError(f"gExpressions[{index + 1}] must be an expression.")
            raw_parts.extend(_split_top_level_expressions(raw_expression))

    expressions: list[str] = []
    for raw_part in raw_parts:
        expressions.extend(_fixed_point_expressions(raw_part))

    if not expressions:
        raise NumericalError("At least one g(x) expression is required.")
    if len(expressions) > 12:
        raise NumericalError("Simple iteration can try at most 12 g(x) expressions.")

    return expressions


def _run_simple_iteration_branch(
    expression: str,
    branch: int,
    x0: float,
    tolerance: float,
    max_iterations: int,
) -> dict[str, Any]:
    g = compile_expression(expression)
    x_curr = x0
    steps: list[dict[str, float | int | str]] = []
    fixed_point = x_curr
    converged = False
    error: float | None = None

    for iteration in range(1, max_iterations + 1):
        try:
            x_next = g(x_curr)
        except NumericalError as exc:
            return {
                "branch": branch,
                "gExpression": expression,
                "fixedPoint": fixed_point,
                "iterations": len(steps),
                "converged": False,
                "error": error,
                "failure": str(exc),
                "steps": steps,
            }

        error = abs(x_next - x_curr)
        steps.append(
            {
                "iteration": iteration,
                "branch": branch,
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
        "branch": branch,
        "gExpression": expression,
        "fixedPoint": fixed_point,
        "iterations": len(steps),
        "converged": converged,
        "error": error,
        "steps": steps,
    }


def _iteration_branch_rank(branch: dict[str, Any]) -> tuple[int, int, float, int, int]:
    error = branch.get("error")
    error_value = error if isinstance(error, (int, float)) and math.isfinite(error) else math.inf
    return (
        1 if branch.get("failure") else 0,
        0 if branch.get("converged") else 1,
        float(error_value),
        int(branch.get("iterations", 0)),
        int(branch.get("branch", 0)),
    )


def simple_iteration_method(params: dict[str, Any]) -> dict[str, Any]:
    expressions = _simple_iteration_expressions(params)
    x_curr = _ensure_number(params.get("x0"), "x0")
    tolerance = _ensure_positive_number(params.get("tolerance", 1e-6), "tolerance")
    max_iterations = _ensure_iterations(params.get("maxIterations", 50))

    branches: list[dict[str, Any]] = []
    for index, expression in enumerate(expressions, start=1):
        try:
            branch = _run_simple_iteration_branch(expression, index, x_curr, tolerance, max_iterations)
        except NumericalError as exc:
            branch = {
                "branch": index,
                "gExpression": expression,
                "fixedPoint": x_curr,
                "iterations": 0,
                "converged": False,
                "error": None,
                "failure": str(exc),
                "steps": [],
            }
        branches.append(branch)

    selectable_branches = [branch for branch in branches if not branch.get("failure")]
    if not selectable_branches:
        failures = "; ".join(f"branch {branch['branch']}: {branch.get('failure', 'failed')}" for branch in branches)
        raise NumericalError(f"All g(x) expressions failed: {failures}")

    selected = min(selectable_branches, key=_iteration_branch_rank)
    branch_summaries = [{key: value for key, value in branch.items() if key != "steps"} for branch in branches]

    return {
        "method": "Method of Simple Iteration",
        "fixedPoint": selected["fixedPoint"],
        "iterations": selected["iterations"],
        "converged": selected["converged"],
        "tolerance": tolerance,
        "residual": selected["error"],
        "selectedBranch": selected["branch"],
        "selectedGExpression": selected["gExpression"],
        "branches": branch_summaries,
        "steps": selected["steps"],
    }


def newton_raphson_method(params: dict[str, Any]) -> dict[str, Any]:
    expression = _root_expression(params.get("expression", ""))
    f = compile_expression(expression)
    derivative_expression = params.get("derivativeExpression")
    has_manual_derivative = isinstance(derivative_expression, str) and derivative_expression.strip()
    derivative = compile_expression(derivative_expression) if has_manual_derivative else compile_derivative(expression)
    second_derivative = compile_second_derivative(expression)
    x_curr = _ensure_number(params.get("x0"), "x0")
    tolerance = _ensure_positive_number(params.get("tolerance", 1e-6), "tolerance")
    max_iterations = _ensure_iterations(params.get("maxIterations", 50))

    steps: list[dict[str, float | int]] = []
    root = x_curr
    value = f(x_curr)
    initial_derivative = derivative(x_curr)
    initial_second_derivative = second_derivative(x_curr)
    initial_convergence_product = value * initial_second_derivative
    residual = abs(value)
    converged = False
    convergence_status = "max_iterations"
    convergence_message = "Maximum iterations reached before |f(x)| was within tolerance."
    warnings: list[str] = []
    if initial_convergence_product <= 0:
        warnings.append("Initial check f(x0) * f''(x0) > 0 is not satisfied; Newton-Raphson may still converge, but the starting point is not ideal.")

    if residual <= tolerance:
        return {
            "method": "The Newton-Raphson Method",
            "normalizedExpression": expression,
            "root": root,
            "value": value,
            "residual": residual,
            "iterations": 0,
            "converged": True,
            "tolerance": tolerance,
            "formula": NEWTON_FORMULA,
            "convergenceStatus": "converged",
            "convergenceMessage": "Initial guess validates convergence because |f(x0)| is within tolerance.",
            "initialDerivative": initial_derivative,
            "initialSecondDerivative": initial_second_derivative,
            "initialConvergenceProduct": initial_convergence_product,
            "derivativeMode": "manual" if has_manual_derivative else "auto",
            "secondDerivativeMode": "auto",
            "warning": " ".join(warnings) if warnings else None,
            "steps": steps,
        }

    for iteration in range(max_iterations):
        fx = f(x_curr)
        dfx = derivative(x_curr)
        ddfx = second_derivative(x_curr)
        if abs(dfx) < NEWTON_DERIVATIVE_TOLERANCE:
            raise NumericalError("Derivative is too close to zero.")

        newton_ratio = fx / dfx
        x_next = _finite_value(x_curr - newton_ratio, "Newton update")
        f_next = f(x_next)
        delta = x_next - x_curr
        error = abs(delta)
        residual = abs(f_next)
        steps.append(
            {
                "iteration": iteration,
                "xCurrent": x_curr,
                "fCurrent": fx,
                "derivative": dfx,
                "secondDerivative": ddfx,
                "newtonRatio": newton_ratio,
                "xNext": x_next,
                "delta": delta,
                "fNext": f_next,
                "error": error,
                "residual": residual,
            }
        )
        root = x_next
        value = f_next
        if residual <= tolerance:
            converged = True
            convergence_status = "converged"
            convergence_message = "Validated convergence because |f(x(i+1))| is within tolerance."
            break
        if error <= tolerance:
            convergence_status = "stalled"
            convergence_message = "Newton step is within tolerance, but |f(x(i+1))| is still too large."
            warnings.append("Convergence was not validated: the Newton step became tiny before the residual met tolerance.")
            break
        x_curr = x_next

    return {
        "method": "The Newton-Raphson Method",
        "normalizedExpression": expression,
        "root": root,
        "value": value,
        "residual": residual,
        "iterations": len(steps),
        "converged": converged,
        "tolerance": tolerance,
        "formula": NEWTON_FORMULA,
        "convergenceStatus": convergence_status,
        "convergenceMessage": convergence_message,
        "initialDerivative": initial_derivative,
        "initialSecondDerivative": initial_second_derivative,
        "initialConvergenceProduct": initial_convergence_product,
        "warning": " ".join(warnings) if warnings else None,
        "derivativeMode": "manual" if has_manual_derivative else "auto",
        "secondDerivativeMode": "auto",
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

    tolerance = _ensure_positive_number(params.get("tolerance", 1e-6), "tolerance")
    max_iterations = _ensure_iterations(params.get("maxIterations", 50))
    return matrix, vector, initial, tolerance, max_iterations


def _residual_norm(matrix: list[list[float]], vector: list[float], x_values: list[float]) -> float:
    residuals = [
        abs(sum(row[j] * x_values[j] for j in range(len(x_values))) - vector[i])
        for i, row in enumerate(matrix)
    ]
    return max(residuals)


def _row_off_diagonal_sum(row: list[float], diagonal_index: int) -> float:
    return sum(abs(value) for j, value in enumerate(row) if j != diagonal_index)


def _is_row_strictly_dominant(row: list[float], diagonal_index: int) -> bool:
    diagonal = abs(row[diagonal_index])
    return diagonal > PIVOT_TOLERANCE and diagonal > _row_off_diagonal_sum(row, diagonal_index) + PIVOT_TOLERANCE


def _is_row_weakly_dominant(row: list[float], diagonal_index: int) -> bool:
    diagonal = abs(row[diagonal_index])
    return diagonal > PIVOT_TOLERANCE and diagonal + PIVOT_TOLERANCE >= _row_off_diagonal_sum(row, diagonal_index)


def _is_diagonally_dominant(matrix: list[list[float]]) -> bool:
    if not matrix:
        return False
    has_strict_row = False
    for i, row in enumerate(matrix):
        if not _is_row_weakly_dominant(row, i):
            return False
        if _is_row_strictly_dominant(row, i):
            has_strict_row = True
    return has_strict_row


def _diagonal_dominance_warning(matrix: list[list[float]]) -> str | None:
    if not _is_diagonally_dominant(matrix):
        return "The matrix is not strictly diagonally dominant; convergence is not guaranteed."
    return None


def _find_row_order(
    matrix: list[list[float]],
    is_usable_row: Callable[[list[float], int], bool],
) -> list[int] | None:
    size = len(matrix)
    row_order = [-1] * size
    used_rows: set[int] = set()
    columns = sorted(
        range(size),
        key=lambda column: sum(1 for row in matrix if is_usable_row(row, column)),
    )

    def assign(column_index: int) -> bool:
        if column_index == size:
            return True

        column = columns[column_index]
        candidates = [
            row_index
            for row_index, row in enumerate(matrix)
            if row_index not in used_rows and is_usable_row(row, column)
        ]
        candidates.sort(key=lambda row_index: abs(matrix[row_index][column]), reverse=True)

        for row_index in candidates:
            row_order[column] = row_index
            used_rows.add(row_index)
            if assign(column_index + 1):
                return True
            used_rows.remove(row_index)
            row_order[column] = -1

        return False

    if assign(0):
        return row_order
    return None


def _find_diagonally_dominant_row_order(matrix: list[list[float]]) -> list[int] | None:
    strict_order = _find_row_order(matrix, _is_row_strictly_dominant)
    if strict_order is not None:
        return strict_order

    weak_order = _find_row_order(matrix, _is_row_weakly_dominant)
    if weak_order is None:
        return None
    if any(_is_row_strictly_dominant(matrix[row_index], column) for column, row_index in enumerate(weak_order)):
        return weak_order
    return None


def _find_nonzero_diagonal_row_order(matrix: list[list[float]]) -> list[int] | None:
    return _find_row_order(matrix, lambda row, column: abs(row[column]) > PIVOT_TOLERANCE)


def _reorder_linear_system(
    matrix: list[list[float]],
    vector: list[float],
    row_order: list[int],
) -> tuple[list[list[float]], list[float]]:
    return [matrix[row_index][:] for row_index in row_order], [vector[row_index] for row_index in row_order]


def _prepare_linear_iteration_system(
    matrix: list[list[float]],
    vector: list[float],
) -> tuple[list[list[float]], list[float], list[int], str | None, bool, bool]:
    identity_order = list(range(len(matrix)))
    dominant_order = _find_diagonally_dominant_row_order(matrix)

    if dominant_order is not None:
        prepared_matrix, prepared_vector = _reorder_linear_system(matrix, vector, dominant_order)
        return prepared_matrix, prepared_vector, dominant_order, None, True, True

    if all(abs(row[index]) > PIVOT_TOLERANCE for index, row in enumerate(matrix)):
        return matrix, vector, identity_order, _diagonal_dominance_warning(matrix), False, True

    nonzero_order = _find_nonzero_diagonal_row_order(matrix)
    if nonzero_order is None:
        return matrix, vector, identity_order, _diagonal_dominance_warning(matrix), False, False

    prepared_matrix, prepared_vector = _reorder_linear_system(matrix, vector, nonzero_order)
    return prepared_matrix, prepared_vector, nonzero_order, _diagonal_dominance_warning(prepared_matrix), False, True


def _solve_linear_system_direct(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(matrix)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]

    for pivot_column in range(size):
        pivot_row = max(range(pivot_column, size), key=lambda row_index: abs(augmented[row_index][pivot_column]))
        if abs(augmented[pivot_row][pivot_column]) <= PIVOT_TOLERANCE:
            raise NumericalError("Linear system is singular; a unique solution could not be found.")

        if pivot_row != pivot_column:
            augmented[pivot_column], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_column]

        pivot = augmented[pivot_column][pivot_column]
        for row_index in range(pivot_column + 1, size):
            factor = augmented[row_index][pivot_column] / pivot
            if abs(factor) <= PIVOT_TOLERANCE:
                continue
            for column_index in range(pivot_column, size + 1):
                augmented[row_index][column_index] -= factor * augmented[pivot_column][column_index]

    solution = [0.0] * size
    for row_index in range(size - 1, -1, -1):
        pivot = augmented[row_index][row_index]
        if abs(pivot) <= PIVOT_TOLERANCE:
            raise NumericalError("Linear system is singular; a unique solution could not be found.")
        known_sum = sum(augmented[row_index][column_index] * solution[column_index] for column_index in range(row_index + 1, size))
        solution[row_index] = (augmented[row_index][size] - known_sum) / pivot

    return solution


def _linear_result(
    method_name: str,
    original_matrix: list[list[float]],
    original_vector: list[float],
    solution: list[float],
    steps: list[dict[str, Any]],
    converged: bool,
    tolerance: float,
    warning: str | None,
    row_order: list[int],
    solver_mode: str,
) -> dict[str, Any]:
    return {
        "method": method_name,
        "solution": solution,
        "iterations": len(steps),
        "converged": converged,
        "tolerance": tolerance,
        "residual": _residual_norm(original_matrix, original_vector, solution),
        "warning": warning,
        "solverMode": solver_mode,
        "rowOrder": [row_index + 1 for row_index in row_order],
        "steps": steps,
    }


def jacobi_method(params: dict[str, Any]) -> dict[str, Any]:
    matrix, vector, x_old, tolerance, max_iterations = _validate_linear_system(params)
    original_matrix = [row[:] for row in matrix]
    original_vector = vector[:]
    matrix, vector, row_order, warning, is_dominant, can_iterate = _prepare_linear_iteration_system(matrix, vector)
    steps: list[dict[str, Any]] = []
    solution = x_old[:]
    converged = False

    if can_iterate:
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

    solver_mode = "reordered_iterative" if row_order != list(range(len(matrix))) and is_dominant else "iterative"
    if not converged and not is_dominant:
        solution = _solve_linear_system_direct(original_matrix, original_vector)
        solver_mode = "direct_fallback"
        warning = "Could not make the matrix diagonally dominant by swapping rows; returned the direct linear-system solution."

    return _linear_result("Jacobi Method", original_matrix, original_vector, solution, steps, converged, tolerance, warning, row_order, solver_mode)


def gauss_seidel_method(params: dict[str, Any]) -> dict[str, Any]:
    matrix, vector, x_old, tolerance, max_iterations = _validate_linear_system(params)
    original_matrix = [row[:] for row in matrix]
    original_vector = vector[:]
    matrix, vector, row_order, warning, is_dominant, can_iterate = _prepare_linear_iteration_system(matrix, vector)
    steps: list[dict[str, Any]] = []
    solution = x_old[:]
    converged = False

    if can_iterate:
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

    solver_mode = "reordered_iterative" if row_order != list(range(len(matrix))) and is_dominant else "iterative"
    if not converged and not is_dominant:
        solution = _solve_linear_system_direct(original_matrix, original_vector)
        solver_mode = "direct_fallback"
        warning = "Could not make the matrix diagonally dominant by swapping rows; returned the direct linear-system solution."

    return _linear_result("Gauss-Seidel Method", original_matrix, original_vector, solution, steps, converged, tolerance, warning, row_order, solver_mode)


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


def _format_step_number(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.10g}"


def _format_lagrange_factor(variable_value: str, xj: float, denominator: float) -> str:
    denominator_text = _format_step_number(denominator)
    xj_magnitude_text = _format_step_number(abs(xj))
    operator = "-" if xj >= 0 else "+"
    return f"(({variable_value} {operator} {xj_magnitude_text}) / {denominator_text})"


def _is_missing_point_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _parse_interpolation_points(
    raw_points: Any,
    *,
    allow_missing: bool = True,
) -> tuple[list[tuple[float, float | None]], int | None]:
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        raise NumericalError("At least two interpolation points are required.")

    points: list[tuple[float, float | None]] = []
    missing_indices: list[int] = []
    seen_x: set[float] = set()

    for index, point in enumerate(raw_points):
        if not isinstance(point, dict):
            raise NumericalError("Each point must contain x and y.")
        x_value = _ensure_number(point.get("x"), f"x[{index + 1}]")
        if x_value in seen_x:
            raise NumericalError("Interpolation x values must be distinct.")
        seen_x.add(x_value)

        raw_y = point.get("y")
        if _is_missing_point_value(raw_y):
            if not allow_missing:
                raise NumericalError(f"y[{index + 1}] must be a number.")
            missing_indices.append(index)
            y_value = None
        else:
            y_value = _ensure_number(raw_y, f"y[{index + 1}]")

        points.append((x_value, y_value))

    if len(missing_indices) > 1:
        raise NumericalError("Only one missing f(x) value can be recovered at a time.")

    known_count = sum(1 for _x, y_value in points if y_value is not None)
    if known_count < 2:
        raise NumericalError("At least two known f(x) values are required.")

    return points, missing_indices[0] if missing_indices else None


def _prepare_interpolation_data(params: dict[str, Any]) -> tuple[list[tuple[float, float]], float, int | None, list[tuple[float, float | None]]]:
    parsed_points, missing_index = _parse_interpolation_points(params.get("points"))
    if missing_index is None:
        target = _ensure_number(params.get("target"), "target")
    else:
        target = parsed_points[missing_index][0]

    known_points = [(x_value, y_value) for x_value, y_value in parsed_points if y_value is not None]
    return known_points, target, missing_index, parsed_points


def _sorted_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return sorted(points, key=lambda point: point[0])


def _result_points(
    parsed_points: list[tuple[float, float | None]],
    missing_index: int | None,
    recovered_value: float,
) -> list[dict[str, float | str]]:
    result_points: list[dict[str, float | str]] = []
    for index, (x_value, y_value) in enumerate(parsed_points):
        is_missing = index == missing_index
        result_points.append(
            {
                "x": x_value,
                "y": recovered_value if is_missing else float(y_value),
                "source": "recovered" if is_missing else "given",
            }
        )
    return result_points


def _newton_standard_coefficients(points: list[tuple[float, float]], newton_coefficients: list[float]) -> list[float]:
    coefficients = [0.0] * len(newton_coefficients)
    basis = [1.0]

    for order, coefficient in enumerate(newton_coefficients):
        for power, basis_coefficient in enumerate(basis):
            coefficients[power] += coefficient * basis_coefficient
        if order < len(newton_coefficients) - 1:
            basis = _poly_multiply(basis, [-points[order][0], 1.0])

    return coefficients


def _finite_difference_table(points: list[tuple[float, float]]) -> list[list[float | None]]:
    table: list[list[float | None]] = [[None for _column in points] for _row in points]
    for row, (_x_value, y_value) in enumerate(points):
        table[row][0] = y_value

    for order in range(1, len(points)):
        for row in range(len(points) - order):
            previous_next = table[row + 1][order - 1]
            previous_current = table[row][order - 1]
            if previous_next is None or previous_current is None:
                raise NumericalError("Difference table could not be constructed.")
            table[row][order] = previous_next - previous_current

    return table


def _divided_difference_table(points: list[tuple[float, float]]) -> list[list[float | None]]:
    table: list[list[float | None]] = [[None for _column in points] for _row in points]
    for row, (_x_value, y_value) in enumerate(points):
        table[row][0] = y_value

    for order in range(1, len(points)):
        for row in range(len(points) - order):
            denominator = points[row + order][0] - points[row][0]
            if abs(denominator) <= PIVOT_TOLERANCE:
                raise NumericalError("Interpolation x values must be distinct.")
            previous_next = table[row + 1][order - 1]
            previous_current = table[row][order - 1]
            if previous_next is None or previous_current is None:
                raise NumericalError("Divided difference table could not be constructed.")
            table[row][order] = (previous_next - previous_current) / denominator

    return table


def _table_rows(points: list[tuple[float, float]], table: list[list[float | None]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (x_value, _y_value) in enumerate(points):
        rows.append(
            {
                "index": index + 1,
                "x": x_value,
                "values": [value for value in table[index] if value is not None],
            }
        )
    return rows


def _validate_equal_spacing(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        raise NumericalError("At least two interpolation points are required.")

    h = points[1][0] - points[0][0]
    if abs(h) <= PIVOT_TOLERANCE:
        raise NumericalError("Interpolation x values must be distinct.")

    tolerance = max(1.0, abs(h)) * 1e-9
    for index in range(2, len(points)):
        spacing = points[index][0] - points[index - 1][0]
        if abs(spacing - h) > tolerance:
            raise NumericalError("NFDF and NBDF require equally spaced known x values. Use NFDDF or NBDDF for non-uniform points.")

    return h


def _interpolation_result(
    *,
    method_name: str,
    target: float,
    value: float,
    known_points: list[tuple[float, float]],
    parsed_points: list[tuple[float, float | None]],
    missing_index: int | None,
    polynomial: str,
    steps: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
    table_kind: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": method_name,
        "target": target,
        "value": value,
        "polynomial": polynomial,
        "mode": "recover_missing" if missing_index is not None else "evaluate",
        "points": _result_points(parsed_points, missing_index, value),
        "knownPoints": [{"x": x_value, "y": y_value} for x_value, y_value in known_points],
        "table": table_rows,
        "tableKind": table_kind,
        "steps": steps,
    }
    if missing_index is not None:
        result["missing"] = {
            "index": missing_index + 1,
            "x": target,
            "value": value,
        }
    if extra:
        result.update(extra)
    return result


def lagrange_interpolation(params: dict[str, Any]) -> dict[str, Any]:
    points, target, missing_index, parsed_points = _prepare_interpolation_data(params)

    total = 0.0
    steps: list[dict[str, Any]] = []
    coefficients = [0.0] * len(points)

    for i, (xi, yi) in enumerate(points):
        basis_at_target = 1.0
        denominator = 1.0
        basis_coefficients = [1.0]
        factors: list[dict[str, float | int]] = []
        equation_factors: list[str] = []
        substituted_factors: list[str] = []

        for j, (xj, _) in enumerate(points):
            if i == j:
                continue
            numerator = target - xj
            divisor = xi - xj
            factor = numerator / divisor
            basis_at_target *= factor
            denominator *= divisor
            basis_coefficients = _poly_multiply(basis_coefficients, [-xj, 1.0])
            equation_factors.append(_format_lagrange_factor("x", xj, divisor))
            substituted_factors.append(_format_lagrange_factor(_format_step_number(target), xj, divisor))
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

        basis_name = f"L_{i}(x)"
        basis_at_target_name = f"L_{i}({_format_step_number(target)})"
        basis_expression = " * ".join(equation_factors) if equation_factors else "1"
        substituted_expression = " * ".join(substituted_factors) if substituted_factors else "1"
        term_expression = (
            f"{_format_step_number(yi)} * {_format_step_number(basis_at_target)}"
            f" = {_format_step_number(term)}"
        )

        steps.append(
            {
                "iteration": i,
                "x": xi,
                "y": yi,
                "basisName": basis_name,
                "basisEquation": f"{basis_name} = {basis_expression}",
                "basisValueEquation": f"{basis_at_target_name} = {substituted_expression} = {_format_step_number(basis_at_target)}",
                "basis": basis_at_target,
                "termEquation": term_expression,
                "term": term,
                "partial": total,
                "factors": factors,
            }
        )

    return {
        "method": "Lagrange Interpolation",
        "target": target,
        "value": total,
        "polynomial": _format_polynomial(coefficients),
        "mode": "recover_missing" if missing_index is not None else "evaluate",
        "points": _result_points(parsed_points, missing_index, total),
        "knownPoints": [{"x": x_value, "y": y_value} for x_value, y_value in points],
        "table": [],
        "tableKind": "basis",
        "missing": {
            "index": missing_index + 1,
            "x": target,
            "value": total,
        } if missing_index is not None else None,
        "steps": steps,
    }


def newton_forward_difference_formula(params: dict[str, Any]) -> dict[str, Any]:
    known_points, target, missing_index, parsed_points = _prepare_interpolation_data(params)
    points = _sorted_points(known_points)
    h = _validate_equal_spacing(points)
    table = _finite_difference_table(points)
    p = (target - points[0][0]) / h
    steps: list[dict[str, Any]] = []
    total = 0.0
    p_term = 1.0
    factorial = 1.0

    for order in range(len(points)):
        difference = table[0][order]
        if difference is None:
            raise NumericalError("Forward difference table could not be evaluated.")
        if order > 0:
            p_term *= p - (order - 1)
            factorial *= order
        multiplier = p_term / factorial
        term = multiplier * difference
        total += term
        steps.append(
            {
                "iteration": order,
                "order": order,
                "difference": difference,
                "pTerm": p_term,
                "factorial": factorial,
                "coefficient": multiplier,
                "term": term,
                "partial": total,
            }
        )

    divided_table = _divided_difference_table(points)
    newton_coefficients = [float(divided_table[0][order]) for order in range(len(points))]
    coefficients = _newton_standard_coefficients(points, newton_coefficients)

    return _interpolation_result(
        method_name="Newton Forward Difference Formula",
        target=target,
        value=total,
        known_points=points,
        parsed_points=parsed_points,
        missing_index=missing_index,
        polynomial=_format_polynomial(coefficients),
        steps=steps,
        table_rows=_table_rows(points, table),
        table_kind="finite_difference",
        extra={"h": h, "p": p},
    )


def newton_backward_difference_formula(params: dict[str, Any]) -> dict[str, Any]:
    known_points, target, missing_index, parsed_points = _prepare_interpolation_data(params)
    points = _sorted_points(known_points)
    h = _validate_equal_spacing(points)
    table = _finite_difference_table(points)
    p = (target - points[-1][0]) / h
    steps: list[dict[str, Any]] = []
    total = 0.0
    p_term = 1.0
    factorial = 1.0
    last_index = len(points) - 1

    for order in range(len(points)):
        difference = table[last_index - order][order]
        if difference is None:
            raise NumericalError("Backward difference table could not be evaluated.")
        if order > 0:
            p_term *= p + (order - 1)
            factorial *= order
        multiplier = p_term / factorial
        term = multiplier * difference
        total += term
        steps.append(
            {
                "iteration": order,
                "order": order,
                "difference": difference,
                "pTerm": p_term,
                "factorial": factorial,
                "coefficient": multiplier,
                "term": term,
                "partial": total,
            }
        )

    divided_table = _divided_difference_table(points)
    newton_coefficients = [float(divided_table[0][order]) for order in range(len(points))]
    coefficients = _newton_standard_coefficients(points, newton_coefficients)

    return _interpolation_result(
        method_name="Newton Backward Difference Formula",
        target=target,
        value=total,
        known_points=points,
        parsed_points=parsed_points,
        missing_index=missing_index,
        polynomial=_format_polynomial(coefficients),
        steps=steps,
        table_rows=_table_rows(points, table),
        table_kind="finite_difference",
        extra={"h": h, "p": p},
    )


def _newton_divided_difference_result(
    params: dict[str, Any],
    *,
    method_name: str,
    reverse: bool,
) -> dict[str, Any]:
    known_points, target, missing_index, parsed_points = _prepare_interpolation_data(params)
    sorted_known_points = _sorted_points(known_points)
    calculation_points = list(reversed(sorted_known_points)) if reverse else sorted_known_points
    table = _divided_difference_table(calculation_points)
    newton_coefficients = [float(table[0][order]) for order in range(len(calculation_points))]
    steps: list[dict[str, Any]] = []
    product = 1.0
    total = 0.0

    for order, coefficient in enumerate(newton_coefficients):
        if order > 0:
            product *= target - calculation_points[order - 1][0]
        term = coefficient * product
        total += term
        steps.append(
            {
                "iteration": order,
                "order": order,
                "xAnchor": calculation_points[order][0],
                "dividedDifference": coefficient,
                "product": product,
                "term": term,
                "partial": total,
            }
        )

    coefficients = _newton_standard_coefficients(calculation_points, newton_coefficients)

    return _interpolation_result(
        method_name=method_name,
        target=target,
        value=total,
        known_points=sorted_known_points,
        parsed_points=parsed_points,
        missing_index=missing_index,
        polynomial=_format_polynomial(coefficients),
        steps=steps,
        table_rows=_table_rows(calculation_points, table),
        table_kind="divided_difference",
    )


def newton_forward_divided_difference_formula(params: dict[str, Any]) -> dict[str, Any]:
    return _newton_divided_difference_result(
        params,
        method_name="Newton Forward Divided Difference Formula",
        reverse=False,
    )


def newton_backward_divided_difference_formula(params: dict[str, Any]) -> dict[str, Any]:
    return _newton_divided_difference_result(
        params,
        method_name="Newton Backward Divided Difference Formula",
        reverse=True,
    )


METHODS = {
    "bisection": bisection_method,
    "secant": secant_method,
    "iteration": simple_iteration_method,
    "newton": newton_raphson_method,
    "jacobi": jacobi_method,
    "gauss_seidel": gauss_seidel_method,
    "lagrange": lagrange_interpolation,
    "nfdf": newton_forward_difference_formula,
    "nbdf": newton_backward_difference_formula,
    "nfddf": newton_forward_divided_difference_formula,
    "nbddf": newton_backward_divided_difference_formula,
}


def calculate(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method not in METHODS:
        raise NumericalError("Unknown numerical method.")
    return METHODS[method](params)
