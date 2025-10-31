import importlib
import math
import unittest
import numpy as np


# Load the module from try.py without using the reserved keyword as an identifier
mod = importlib.import_module("try")


class TestIntersectionsReflections(unittest.TestCase):
    def test_intersect_plane_simple(self):
        # плоскость z = 5 
        surface = mod.Surface(lambda x, y, z: z - 5)
        # луч паралелен OZ
        ray = mod.Ray(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        pt = mod.intersect_ray_surface(ray, surface)
        self.assertIsNotNone(pt)
        x, y, z = pt
        self.assertTrue(math.isclose(x, 0.0, abs_tol=1e-3))
        self.assertTrue(math.isclose(y, 0.0, abs_tol=1e-3))
        # пересечение ~ 5+-1e-2
        self.assertTrue(math.isclose(z, 5.0, abs_tol=1e-2))


    def test_intersect_sphere_simple(self):
        # сфера радиуса 2: x^2 + y^2 + z^2 - 4 = 0
        surface = mod.Surface(lambda x, y, z: x*x + y*y + z*z - 4.0)
        # луч из точки (0,0,5) противоположно z
        ray = mod.Ray(0.0, 0.0, 0.0, 0.0, 5.0, -1.0)
        pt = mod.intersect_ray_surface(ray, surface)
        self.assertIsNotNone(pt)
        x, y, z = pt
        self.assertTrue(math.isclose(x, 0.0, abs_tol=1e-3))
        self.assertTrue(math.isclose(y, 0.0, abs_tol=1e-3))
        # пересечение ~ 2+-1e-2
        self.assertTrue(math.isclose(z, 2.0, abs_tol=1e-2))


    def test_intersect_no_hit(self):
        # плоскость z = -1, луч  из (0,0,0) и вдоль +z,
        # поэтому для всех t >= 0 он имеет z = t >= 0 > -1 => нет пересечения
        surface = mod.Surface(lambda x, y, z: z + 1.0)
        ray = mod.Ray(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self.assertIsNone(mod.intersect_ray_surface(ray, surface))


    def test_reflection_on_plate_normal_incidence(self):
        # плоскость с нормалью (0,0,1)
        # луч падает вертикально вниз, отражается вверх
        plate = mod.Plate(0.0, 0.0, 1.0, 0.0)
        ray = mod.Ray(0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
        v_ref = mod.get_reflected_ray(ray, plate)
        self.assertTrue(np.allclose(v_ref, np.array([0.0, 0.0, 1.0]), atol=1e-12))


    def test_reflection_on_plate_oblique(self):
        # плоскость с нормалью (0,0,1)
        # луч падает под углом, отражается с инверсией "z"
        plate = mod.Plate(0.0, 0.0, 1.0, 0.0)
        ray = mod.Ray(0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
        v_ref = mod.get_reflected_ray(ray, plate)
        self.assertTrue(np.allclose(v_ref, np.array([1.0, 0.0, 1.0]), atol=1e-12))


    def test_reflection_on_surface_plane(self):
        # поверхность z = 0 (нормаль (0,0,1))
        # луч из z=1 направлен вниз, отражается вверх
        surface = mod.Surface(lambda x, y, z: z)
        ray = mod.Ray(0.0, 0.0, 0.0, 0.0, 1.0, -1.0)
        result = mod.get_reflected_ray_from_surface(ray, surface)
        self.assertIsNotNone(result)
        point, v_ref = result
        x, y, z = point
        # пересечение ~ (0,0,0)
        self.assertTrue(math.isclose(x, 0.0, abs_tol=1e-3))
        self.assertTrue(math.isclose(y, 0.0, abs_tol=1e-3))
        self.assertTrue(math.isclose(z, 0.0, abs_tol=1e-2))
        # отражённое направление вверх
        self.assertTrue(np.allclose(v_ref, np.array([0.0, 0.0, 1.0]), atol=1e-6))

if __name__ == "__main__":
    unittest.main()
