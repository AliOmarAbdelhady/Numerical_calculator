import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numerical_methods import (  # noqa: E402
    bisection_method,
    gauss_seidel_method,
    jacobi_method,
    lagrange_interpolation,
    newton_raphson_method,
    newton_backward_difference_formula,
    newton_backward_divided_difference_formula,
    newton_forward_difference_formula,
    newton_forward_divided_difference_formula,
    secant_method,
    simple_iteration_method,
)


class NumericalMethodTests(unittest.TestCase):
    def test_bisection_method(self):
        result = bisection_method(
            {
                "expression": "x^3 - x - 2",
                "a": 1,
                "b": 2,
                "tolerance": 1e-8,
                "maxIterations": 100,
            }
        )
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["root"], 1.5213797068, places=6)

    def test_secant_method(self):
        result = secant_method(
            {
                "expression": "x^3 - x - 2",
                "x0": 1,
                "x1": 2,
                "tolerance": 1e-8,
                "maxIterations": 100,
            }
        )
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["root"], 1.5213797068, places=6)

    def test_simple_iteration_method(self):
        result = simple_iteration_method(
            {
                "gExpression": "cos(x)",
                "x0": 0.5,
                "tolerance": 1e-8,
                "maxIterations": 100,
            }
        )
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["fixedPoint"], 0.7390851332, places=6)

    def test_simple_iteration_method_tries_multiple_candidates(self):
        result = simple_iteration_method(
            {
                "gExpression": "-sqrt(x - 2), sqrt(x + 2)",
                "x0": 0,
                "tolerance": 1e-8,
                "maxIterations": 100,
            }
        )
        self.assertTrue(result["converged"])
        self.assertEqual(result["selectedBranch"], 2)
        self.assertAlmostEqual(result["fixedPoint"], 2, places=6)
        self.assertEqual(len(result["branches"]), 2)
        self.assertIn("domain", result["branches"][0]["failure"])

    def test_simple_iteration_method_rearranges_product_equation(self):
        result = simple_iteration_method(
            {
                "gExpression": "3*x*e^(x) = 1",
                "x0": 1,
                "tolerance": 1e-8,
                "maxIterations": 100,
            }
        )
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["fixedPoint"], 0.2576276530, places=6)
        self.assertIn("e ** x", result["selectedGExpression"])

    def test_newton_raphson_method(self):
        result = newton_raphson_method(
            {
                "expression": "x^2 - 2",
                "x0": 1,
                "tolerance": 1e-10,
                "maxIterations": 100,
            }
        )
        self.assertTrue(result["converged"])
        self.assertEqual(result["derivativeMode"], "auto")
        self.assertEqual(result["formula"], "x(i+1) = x(i) - f(x(i)) / f'(x(i))")
        self.assertEqual(result["convergenceStatus"], "converged")
        self.assertEqual(result["steps"][0]["iteration"], 0)
        self.assertAlmostEqual(result["steps"][0]["xCurrent"], 1)
        self.assertAlmostEqual(result["steps"][0]["derivative"], 2)
        self.assertAlmostEqual(result["steps"][0]["newtonRatio"], -0.5)
        self.assertAlmostEqual(result["steps"][0]["xNext"], 1.5)
        self.assertAlmostEqual(result["steps"][0]["delta"], 0.5)
        self.assertLessEqual(result["residual"], result["tolerance"])
        self.assertAlmostEqual(result["root"], math.sqrt(2), places=8)

    def test_newton_raphson_method_requires_residual_convergence(self):
        result = newton_raphson_method(
            {
                "expression": "atan(1e12*x) + 2",
                "x0": 0,
                "tolerance": 1e-8,
                "maxIterations": 100,
            }
        )
        self.assertFalse(result["converged"])
        self.assertEqual(result["convergenceStatus"], "stalled")
        self.assertGreater(result["residual"], result["tolerance"])
        self.assertLessEqual(result["steps"][0]["error"], result["tolerance"])

    def test_newton_raphson_method_solves_equation_with_second_derivative(self):
        result = newton_raphson_method(
            {
                "expression": "x^3 - 4*x^(2) - 10 = 0",
                "x0": 1,
                "tolerance": 1e-8,
                "maxIterations": 100,
            }
        )
        self.assertTrue(result["converged"])
        self.assertEqual(result["normalizedExpression"], "(x^3 - 4*x^(2) - 10) - (0)")
        self.assertEqual(result["derivativeMode"], "auto")
        self.assertEqual(result["secondDerivativeMode"], "auto")
        self.assertAlmostEqual(result["initialDerivative"], -5)
        self.assertAlmostEqual(result["initialSecondDerivative"], -2)
        self.assertAlmostEqual(result["initialConvergenceProduct"], 26)
        self.assertAlmostEqual(result["steps"][0]["derivative"], -5)
        self.assertAlmostEqual(result["steps"][0]["secondDerivative"], -2)
        self.assertAlmostEqual(result["root"], 4.49493967126358, places=8)

    def test_jacobi_method(self):
        result = jacobi_method(
            {
                "matrix": [[10, -1, 2], [-1, 11, -1], [2, -1, 10]],
                "vector": [6, 25, -11],
                "initial": [0, 0, 0],
                "tolerance": 1e-8,
                "maxIterations": 100,
            }
        )
        self.assertTrue(result["converged"])
        for value, expected in zip(result["solution"], [1.0432692308, 2.2692307692, -1.0817307692]):
            self.assertAlmostEqual(value, expected, places=5)

    def test_gauss_seidel_method(self):
        result = gauss_seidel_method(
            {
                "matrix": [[10, -1, 2], [-1, 11, -1], [2, -1, 10]],
                "vector": [6, 25, -11],
                "initial": [0, 0, 0],
                "tolerance": 1e-8,
                "maxIterations": 100,
            }
        )
        self.assertTrue(result["converged"])
        for value, expected in zip(result["solution"], [1.0432692308, 2.2692307692, -1.0817307692]):
            self.assertAlmostEqual(value, expected, places=5)

    def test_linear_methods_reorder_to_diagonal_dominance(self):
        params = {
            "matrix": [[1, 10, 2], [10, 1, 2], [1, 2, 10]],
            "vector": [27, 18, 35],
            "initial": [0, 0, 0],
            "tolerance": 1e-8,
            "maxIterations": 100,
        }

        for method in (jacobi_method, gauss_seidel_method):
            with self.subTest(method=method.__name__):
                result = method(params)
                self.assertTrue(result["converged"])
                self.assertEqual(result["solverMode"], "reordered_iterative")
                self.assertEqual(result["rowOrder"], [2, 1, 3])
                for value, expected in zip(result["solution"], [1, 2, 3]):
                    self.assertAlmostEqual(value, expected, places=5)

    def test_linear_methods_use_direct_fallback_when_dominance_is_impossible(self):
        params = {
            "matrix": [[1, -1, 2], [-1, 11, -1], [2, -1, 10]],
            "vector": [6, 25, -11],
            "initial": [0, 0, 0],
            "tolerance": 1e-8,
            "maxIterations": 1,
        }

        expected_solution = [1085 / 59, 209 / 59, -261 / 59]
        for method in (jacobi_method, gauss_seidel_method):
            with self.subTest(method=method.__name__):
                result = method(params)
                self.assertFalse(result["converged"])
                self.assertEqual(result["solverMode"], "direct_fallback")
                self.assertIn("direct linear-system solution", result["warning"])
                self.assertLessEqual(result["residual"], 1e-8)
                for value, expected in zip(result["solution"], expected_solution):
                    self.assertAlmostEqual(value, expected, places=8)

    def test_lagrange_interpolation(self):
        result = lagrange_interpolation(
            {
                "points": [{"x": 0, "y": 1}, {"x": 1, "y": 3}, {"x": 2, "y": 2}],
                "target": 1.5,
            }
        )
        self.assertAlmostEqual(result["value"], 2.875, places=8)
        self.assertEqual(result["steps"][0]["iteration"], 0)
        self.assertEqual(result["steps"][0]["basisEquation"], "L_0(x) = ((x - 1) / -1) * ((x - 2) / -2)")
        self.assertEqual(result["steps"][0]["basisValueEquation"], "L_0(1.5) = ((1.5 - 1) / -1) * ((1.5 - 2) / -2) = -0.125")
        self.assertEqual(result["steps"][0]["termEquation"], "1 * -0.125 = -0.125")
        self.assertAlmostEqual(result["steps"][0]["basis"], -0.125)
        self.assertAlmostEqual(result["steps"][0]["term"], -0.125)
        self.assertAlmostEqual(result["steps"][-1]["partial"], 2.875)

    def test_newton_finite_difference_formulas(self):
        params = {
            "points": [{"x": 0, "y": 1}, {"x": 1, "y": 3}, {"x": 2, "y": 2}],
            "target": 1.5,
        }

        for method in (newton_forward_difference_formula, newton_backward_difference_formula):
            with self.subTest(method=method.__name__):
                result = method(params)
                self.assertAlmostEqual(result["value"], 2.875, places=8)
                self.assertEqual(result["tableKind"], "finite_difference")
                self.assertEqual(len(result["steps"]), 3)

    def test_newton_divided_difference_formulas_with_non_uniform_points(self):
        params = {
            "points": [{"x": 0, "y": 1}, {"x": 1, "y": 2}, {"x": 3, "y": 10}],
            "target": 2,
        }

        for method in (newton_forward_divided_difference_formula, newton_backward_divided_difference_formula):
            with self.subTest(method=method.__name__):
                result = method(params)
                self.assertAlmostEqual(result["value"], 5, places=8)
                self.assertEqual(result["tableKind"], "divided_difference")
                self.assertEqual(len(result["steps"]), 3)

    def test_newton_divided_difference_recovers_one_missing_value(self):
        params = {
            "points": [
                {"x": 0, "y": 1},
                {"x": 1, "y": 3},
                {"x": 2, "y": None},
                {"x": 3, "y": 13},
            ],
        }

        for method in (newton_forward_divided_difference_formula, newton_backward_divided_difference_formula):
            with self.subTest(method=method.__name__):
                result = method(params)
                self.assertEqual(result["mode"], "recover_missing")
                self.assertEqual(result["missing"]["index"], 3)
                self.assertAlmostEqual(result["missing"]["value"], 7, places=8)


if __name__ == "__main__":
    unittest.main()
