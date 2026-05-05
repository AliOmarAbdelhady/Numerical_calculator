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
        self.assertEqual(result["steps"][0]["iteration"], 0)
        self.assertAlmostEqual(result["steps"][0]["xCurrent"], 1)
        self.assertAlmostEqual(result["steps"][0]["derivative"], 2)
        self.assertAlmostEqual(result["steps"][0]["delta"], 0.5)
        self.assertAlmostEqual(result["root"], math.sqrt(2), places=8)

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

    def test_lagrange_interpolation(self):
        result = lagrange_interpolation(
            {
                "points": [{"x": 0, "y": 1}, {"x": 1, "y": 3}, {"x": 2, "y": 2}],
                "target": 1.5,
            }
        )
        self.assertAlmostEqual(result["value"], 2.875, places=8)


if __name__ == "__main__":
    unittest.main()
