import numpy as np
from PIL import Image
import os

os.makedirs("images", exist_ok=True)

# ==================================================
# Math Utilities
# ==================================================

def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def dot(a, b):
    return np.dot(a, b)

def reflect(I, N):
    return I - 2 * dot(I, N) * N

# ==================================================
# Scene Objects
# ==================================================

class Sphere:
    def __init__(self, center, radius, color, specular=50, reflective=0.5):
        self.center     = np.array(center, dtype=float)
        self.radius     = radius
        self.color      = np.array(color,  dtype=float)
        self.specular   = specular
        self.reflective = reflective

    def intersect(self, origin, direction):
        oc   = origin - self.center
        a    = dot(direction, direction)
        b    = 2 * dot(oc, direction)
        c    = dot(oc, oc) - self.radius ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return np.inf
        sq = np.sqrt(disc)
        t1 = (-b - sq) / (2 * a)
        t2 = (-b + sq) / (2 * a)
        ts = [t for t in (t1, t2) if t > 1e-4]
        return min(ts) if ts else np.inf

    def normal_at(self, p):
        return normalize(p - self.center)

    def get_color(self, point):
        return self.color


class Plane:
    def __init__(self, point, normal, color, specular=10, reflective=0.2,
                 checkerboard=False):
        self.point        = np.array(point,  dtype=float)
        self.normal       = normalize(np.array(normal, dtype=float))
        self.color        = np.array(color,  dtype=float)
        self.specular     = specular
        self.reflective   = reflective
        self.checkerboard = checkerboard

    def intersect(self, origin, direction):
        denom = dot(self.normal, direction)
        if abs(denom) < 1e-6:
            return np.inf
        t = dot(self.point - origin, self.normal) / denom
        return t if t > 1e-4 else np.inf

    def normal_at(self, _):
        return self.normal

    def get_color(self, point):
        if not self.checkerboard:
            return self.color
        x, z  = point[0], point[2]
        check = (int(np.floor(x)) + int(np.floor(z))) % 2
        return self.color if check == 0 else np.array([30.0, 30.0, 30.0])


# ==================================================
# Lights
# ==================================================

class Light:
    def __init__(self, position, intensity):
        self.position  = np.array(position, dtype=float)
        self.intensity = intensity

# ==================================================
# Blinn-Phong Lighting
# ==================================================

def compute_lighting(point, normal, view_dir, specular, objects, lights):
    intensity = 0.05   # ambient
    shadow_origin = point + normal * 1e-3

    for light in lights:
        to_light   = normalize(light.position - point)
        light_dist = np.linalg.norm(light.position - point)

        # Shadow check
        in_shadow = any(
            obj.intersect(shadow_origin, to_light) < light_dist
            for obj in objects
        )
        if in_shadow:
            continue

        # Diffuse
        diff = max(dot(normal, to_light), 0.0)
        intensity += light.intensity * diff

        # Specular (Blinn-Phong half-vector)
        if specular > 0:
            half = normalize(to_light + view_dir)
            spec = max(dot(normal, half), 0.0) ** specular
            intensity += light.intensity * spec * 0.3

    return intensity

# ==================================================
# Recursive Ray Tracing
# ==================================================

def trace_ray(origin, direction, objects, lights, depth=3):
    closest_t = np.inf
    hit_obj   = None

    for obj in objects:
        t = obj.intersect(origin, direction)
        if t < closest_t:
            closest_t = t
            hit_obj   = obj

    if hit_obj is None:
        return np.array([30.0, 30.0, 40.0])   # background

    hit_point  = origin + closest_t * direction
    normal     = hit_obj.normal_at(hit_point)
    view_dir   = normalize(-direction)
    base_color = hit_obj.get_color(hit_point)

    lighting    = compute_lighting(hit_point, normal, view_dir,
                                   hit_obj.specular, objects, lights)
    local_color = base_color * np.clip(lighting, 0.0, 2.0)

    if depth <= 0 or hit_obj.reflective <= 0:
        return np.clip(local_color, 0, 255)

    # Recursive reflection
    refl_dir    = normalize(reflect(direction, normal))
    refl_origin = hit_point + normal * 1e-3
    refl_color  = trace_ray(refl_origin, refl_dir, objects, lights, depth - 1)

    final = (1 - hit_obj.reflective) * local_color + hit_obj.reflective * refl_color
    return np.clip(final, 0, 255)

# ==================================================
# Scene & Render
# ==================================================

def build_scene():
    objects = [
        Sphere([ 0.0, -0.2, 3.0], 0.8, [220,  60,  60], specular=100, reflective=0.4),
        Sphere([ 1.5,  0.2, 4.0], 0.9, [ 60, 120, 240], specular= 50, reflective=0.3),
        Sphere([-1.8,  0.0, 4.5], 1.0, [ 60, 200, 120], specular= 30, reflective=0.2),
        Plane([0, -1, 0], [0, 1, 0], [210, 200, 180],
              specular=5, reflective=0.25, checkerboard=True),
    ]
    lights = [
        Light([ 2,  3, -1], 1.4),
        Light([-3,  5, -2], 0.8),
    ]
    return objects, lights


def render(width=640, height=480, samples=3):
    objects, lights = build_scene()
    camera   = np.array([0.0, 0.0, -1.0])
    viewport = 1.0

    image  = Image.new("RGB", (width, height))
    pixels = image.load()

    for x in range(-width // 2, width // 2):
        for y in range(-height // 2, height // 2):
            color = np.zeros(3)
            for i in range(samples):
                for j in range(samples):
                    vx = (x + (i + 0.5) / samples) * viewport / width
                    vy = (y + (j + 0.5) / samples) * viewport / height
                    direction = normalize(np.array([vx, vy, 1.0]))
                    color += trace_ray(camera, direction, objects, lights, depth=3)
            color /= samples ** 2
            pixels[x + width // 2, height // 2 - y - 1] = tuple(color.astype(np.uint8))

    path = "images/classic_ray_tracer.png"
    image.save(path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    render()