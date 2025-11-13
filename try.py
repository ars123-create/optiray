import numpy as np


class Vector(np.ndarray):
    def __new__(cls, x, y, z):
        arr = np.asarray([x, y, z], dtype=float).view(cls)
        return arr

    @property
    def x(self) -> float:
        return float(self[0])

    @property
    def y(self) -> float:
        return float(self[1])

    @property
    def z(self) -> float:
        return float(self[2])

    def __repr__(self):
        return f"vector({self.x}, {self.y}, {self.z})"

    def __abs__(self):
        return np.linalg.norm(self)


class Ray:
    def __init__(self, x0, a, y0, b, z0, c, frec=0):
        self.x0 = x0
        self.a = a
        self.y0 = y0
        self.b = b
        self.z0 = z0
        self.c = c
        self.frec = frec

    # можно переписать через точку и вектор, но я не уверен надо ли

    def __repr__(self):
        return (
            f"ray((x-{self.x0})/{self.a} = (y-{self.y0})/{self.b} = "
            f"(z-{self.z0})/{self.c}, frec={self.frec})"
        )

    def get_line_v(self):
        return Vector(self.a, self.b, self.c)

    @property
    def origin(self):
        return Vector(self.x0, self.y0, self.z0)

    @property
    def direction(self):
        return self.get_line_v()

    def direction_normalized(self):
        v = self.get_line_v()
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("направление луча не может быть нулевым")
        return v / norm


class Surface:
    """Общая кривая поверхность F(x, y, z) = 0"""

    def __init__(self, func, gradient=None):
        self.func = func  # передаём сюда lambda x, y, z: выражение F(x,y,z)
        self._gradient = gradient

    def F(self, x, y, z):
        return self.func(x, y, z)

    def get_norm(self, x, y, z):  # TODO: мб надо умнее (буквенно)
        if self._gradient is not None:
            gx, gy, gz = self._gradient(x, y, z)
        else:
            eps = 1e-6
            gx = (self.func(x + eps, y, z) - self.func(x - eps, y, z)) / (2 * eps)
            gy = (self.func(x, y + eps, z) - self.func(x, y - eps, z)) / (2 * eps)
            gz = (self.func(x, y, z + eps) - self.func(x, y, z - eps)) / (2 * eps)
        n = Vector(gx, gy, gz)
        return n / np.linalg.norm(n)

    def intersect(self, ray, t_max=100.0, steps=10000):
        x0, y0, z0 = ray.x0, ray.y0, ray.z0
        a, b, c = ray.a, ray.b, ray.c

        def F_of_t(t):
            return self.F(x0 + a * t, y0 + b * t, z0 + c * t)

        t_vals = np.linspace(0, t_max, steps + 1)
        F_vals = [F_of_t(t) for t in t_vals]
        sign_change = np.where(np.sign(F_vals[:-1]) != np.sign(F_vals[1:]))[0]
        if len(sign_change) == 0:
            return None  # не пересекает
        i = sign_change[0]
        t_hit = (t_vals[i] + t_vals[i + 1]) / 2
        x, y, z = x0 + a * t_hit, y0 + b * t_hit, z0 + c * t_hit
        return x, y, z


class Mirror(Surface):
    def __init__(self, func, gradient=None):
        super().__init__(func, gradient=gradient)

    def reflect(self, ray, with_point=False):
        intersection = self.intersect(ray)
        if intersection is None:
            return None
        x, y, z = intersection
        normal = self.get_norm(x, y, z)
        direction = ray.get_line_v()
        reflected = direction - 2 * np.dot(direction, normal) * normal
        if with_point:
            return intersection, reflected
        return reflected


class Glass(Surface):
    def __init__(self, func, n_inside, n_outside=1.0, gradient=None):
        super().__init__(func, gradient=gradient)
        self.n_inside = n_inside
        self.n_outside = n_outside

    def refract(self, ray, n1=None, n2=None, with_point=False):
        intersection = self.intersect(ray)
        if intersection is None:
            return None
        x, y, z = intersection
        normal = self.get_norm(x, y, z)

        if n1 is None:
            n1 = self.n_outside
        if n2 is None:
            n2 = self.n_inside

        direction = ray.direction_normalized()
        cosi = -np.dot(normal, direction)
        if cosi < 0:
            normal = -normal
            cosi = -np.dot(normal, direction)
            n1, n2 = n2, n1

        ratio = n1 / n2
        k = 1.0 - ratio ** 2 * (1.0 - cosi ** 2)
        if k < 0:
            return None
        refracted = ratio * direction + (ratio * cosi - np.sqrt(k)) * normal
        if with_point:
            return intersection, refracted
        return refracted


class Plate(Mirror):
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        def func(x, y, z):
            return A * x + B * y + C * z + D

        gradient = lambda x, y, z: (A, B, C)
        super().__init__(func, gradient=gradient)

        normal = Vector(A, B, C)
        self._normal = normal / np.linalg.norm(normal)

    def __repr__(self):
        return f"plate({self.A}, {self.B}, {self.C}, {self.D})"

    def get_norm(self, *args, **kwargs):
        return self._normal


def get_angle_from_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if not (norm1 and norm2):
        raise ValueError("ошибка")
    cos_alpha = dot_product / (norm1 * norm2)
    alpha = np.arccos(cos_alpha)
    return alpha


def get_reflected_ray(ray, plate):
    if isinstance(plate, Mirror):
        result = plate.reflect(ray)
        if result is None:
            return None
        return result
    v = ray.get_line_v()
    n = plate.get_norm()
    v_reflected = v - 2 * np.dot(v, n) * n
    return v_reflected


def get_refrected_ray(ray, plate, n1, n2):
    if isinstance(plate, Glass):
        result = plate.refract(ray, n1, n2)
        if result is None:
            return None
        return result
    v = ray.get_line_v()
    v = v / np.linalg.norm(v)
    n = plate.get_norm()

    cosi = -np.dot(n, v)
    if cosi < 0:
        n = -n
        cosi = -np.dot(n, v)

    N = n1 / n2
    k = 1.0 - N**2 * (1.0 - cosi**2)
    if k < 0:
        return None

    t = N * v + (N * cosi - np.sqrt(k)) * n
    return t

def intersect_ray_surface(ray, surface):
    if hasattr(surface, "intersect"):
        return surface.intersect(ray)
    x0, y0, z0 = ray.x0, ray.y0, ray.z0
    a, b, c = ray.a, ray.b, ray.c

    def F_of_t(t):
        return surface.F(x0 + a*t, y0 + b*t, z0 + c*t)

    #TODO: доказать разрешимость (?)
    t_vals = np.linspace(0, 100, 10001)
    F_vals = [F_of_t(t) for t in t_vals]
    sign_change = np.where(np.sign(F_vals[:-1]) != np.sign(F_vals[1:]))[0]
    if len(sign_change) == 0:
        return None  # не пересекает
    i = sign_change[0]
    t_hit = (t_vals[i] + t_vals[i+1]) / 2
    x, y, z = x0 + a*t_hit, y0 + b*t_hit, z0 + c*t_hit
    return x, y, z

def get_reflected_ray_from_surface(ray, surface):
    if isinstance(surface, Mirror):
        return surface.reflect(ray, with_point=True)
    point = intersect_ray_surface(ray, surface)
    if point is None:
        return None
    x, y, z = point
    n = surface.get_norm(x, y, z)
    v = ray.get_line_v() / np.linalg.norm(ray.get_line_v())
    v_reflected = v - 2 * np.dot(v, n) * n
    return point, v_reflected



ray = Ray(0, 1, 0, 1, 0, 1)
plate = Plate(0, 0, 1, 0)

print("Отраженный луч от зеркальной пластины:", get_reflected_ray(ray, plate))

glass_surface = Glass(lambda x, y, z: z, n_inside=1.5, n_outside=1.0, gradient=lambda x, y, z: (0.0, 0.0, 1.0))
print("Преломленный луч при переходе воздух->стекло:", get_refrected_ray(ray, glass_surface, 1.0, 1.5))

surface = Surface(lambda x, y, z: 2*x**2 + 2*y**2 - z**2 - 4)
ray = Ray(-5, 1, -2, 3, 3, 1)  # пример

result = get_reflected_ray_from_surface(ray, surface)
if result is None:
    print("Луч не пересекает поверхность")
else:
    point, reflected_dir = result
    print("Точка падения:", point)
    print("Направление отражённого луча:", reflected_dir)

