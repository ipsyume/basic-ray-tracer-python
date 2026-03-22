"""
Hybrid Neural Ray Tracer
========================
Classical Blinn-Phong is used as the "teacher".
A small MLP (NeuralShader) learns to predict per-point lighting intensity
from surface geometry.  At render time both are blended:

    intensity = (1 - blend) * phong + blend * neural

This preserves material base colours while letting the network take over
the lighting computation.
"""

import numpy as np
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.optim as optim

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
    def __init__(self, center, radius, color, specular=50, reflective=0.4):
        self.center     = np.array(center, float)
        self.radius     = radius
        self.color      = np.array(color,  float)
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
        self.point        = np.array(point,  float)
        self.normal       = normalize(np.array(normal, float))
        self.color        = np.array(color,  float)
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
        self.position  = np.array(position, float)
        self.intensity = intensity

# ==================================================
# Classical Blinn-Phong (Teacher)
# ==================================================

def compute_lighting(point, normal, view_dir, specular, objects, lights):
    """Returns a scalar lighting intensity using Blinn-Phong."""
    intensity     = 0.05                       # ambient
    shadow_origin = point + normal * 1e-3

    for light in lights:
        to_light   = normalize(light.position - point)
        light_dist = np.linalg.norm(light.position - point)

        in_shadow = any(
            obj.intersect(shadow_origin, to_light) < light_dist
            for obj in objects
        )
        if in_shadow:
            continue

        intensity += light.intensity * max(dot(normal, to_light), 0.0)
        if specular > 0:
            half = normalize(to_light + view_dir)
            intensity += light.intensity * (max(dot(normal, half), 0.0) ** specular) * 0.3

    return float(np.clip(intensity, 0.0, 2.0))

# ==================================================
# Neural Shader
# ==================================================

class NeuralShader(nn.Module):
    """
    Input  : 9-D feature [surface_normal(3), view_dir(3), primary_light_dir(3)]
    Output : scalar lighting intensity in [0, 1]

    Keeping the output as a scalar means the network learns pure shading
    logic — base colour is always multiplied in separately, so material
    identity is never lost.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64,  32), nn.ReLU(),
            nn.Linear(32,   1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def generate_training_data(lights, n_samples=10_000):
    """
    Generate (geometry → phong_intensity) pairs.
    Samples random normals / view directions and computes exact Phong
    intensity — the network's job is to approximate this mapping.
    """
    primary_light = lights[0]
    X, Y = [], []

    for _ in range(n_samples):
        # Random unit normal biased toward upper hemisphere
        normal = normalize(np.random.randn(3))
        if normal[1] < 0:
            normal[1] = abs(normal[1])
        normal = normalize(normal)

        view      = normalize(np.random.randn(3))
        ref_point = np.random.uniform(-2, 2, 3)
        light_dir = normalize(primary_light.position - ref_point)

        # Exact Blinn-Phong (no shadow, just the lighting equation)
        intensity = 0.05
        diff = max(dot(normal, light_dir), 0.0)
        intensity += primary_light.intensity * diff
        half = normalize(light_dir + view)
        intensity += primary_light.intensity * (max(dot(normal, half), 0.0) ** 50) * 0.3
        intensity = float(np.clip(intensity, 0.0, 1.0))

        X.append(np.concatenate([normal, view, light_dir]))
        Y.append([intensity])

    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32))


def train_neural_shader(model, lights, epochs=600):
    print("  Generating training data …")
    X, Y = generate_training_data(lights)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.MSELoss()

    for e in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if e % 100 == 0:
            print(f"  Epoch {e:4d} | Loss: {loss.item():.6f}")

    print(f"  Training done. Final loss: {loss.item():.8f}\n")


def neural_shade(point, normal, view_dir, lights, model):
    """Query the neural shader — returns a scalar intensity."""
    light_dir = normalize(lights[0].position - point)
    feat = torch.tensor(
        np.concatenate([normal, view_dir, light_dir]).astype(np.float32)
    )
    with torch.no_grad():
        return model(feat).item()   # scalar in [0, 1]

# ==================================================
# Hybrid Ray Tracing
# ==================================================

def trace_ray(origin, direction, objects, lights, model, depth=3, blend=0.5):
    """
    blend=0.0  →  pure classical Phong
    blend=1.0  →  pure neural shading
    blend=0.5  →  equal mix (default)
    """
    closest_t = np.inf
    hit_obj   = None

    for obj in objects:
        t = obj.intersect(origin, direction)
        if t < closest_t:
            closest_t = t
            hit_obj   = obj

    if hit_obj is None:
        return np.array([30.0, 30.0, 40.0])   # sky background

    hit_point  = origin + closest_t * direction
    normal     = hit_obj.normal_at(hit_point)
    view_dir   = normalize(-direction)
    base_color = hit_obj.get_color(hit_point)

    # ── Two intensity estimates ──────────────────────────────────────────────
    phong  = compute_lighting(hit_point, normal, view_dir,
                              hit_obj.specular, objects, lights)   # scalar
    neural = neural_shade(hit_point, normal, view_dir, lights, model)  # scalar

    # Hybrid blend: BOTH are scalars → multiply by base_color ONCE
    intensity   = (1 - blend) * phong + blend * neural
    local_color = base_color * np.clip(intensity, 0.0, 2.0)
    # ─────────────────────────────────────────────────────────────────────────

    if depth <= 0 or hit_obj.reflective <= 0:
        return np.clip(local_color, 0, 255)

    refl_dir   = normalize(reflect(direction, normal))
    refl_color = trace_ray(hit_point + normal * 1e-3, refl_dir,
                           objects, lights, model, depth - 1, blend)

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


def render(width=640, height=480, blend=0.5):
    objects, lights = build_scene()
    camera = np.array([0.0, 0.0, -1.0])

    print("Training neural shader …")
    model = NeuralShader()
    train_neural_shader(model, lights)

    img    = Image.new("RGB", (width, height))
    pixels = img.load()

    print("Rendering …")
    hw, hh = width // 2, height // 2

    for x in range(-hw, hw):
        for y in range(-hh, hh):
            direction = normalize(np.array([x / width, y / height, 1.0]))
            color     = trace_ray(camera, direction, objects, lights,
                                  model, depth=3, blend=blend)
            pixels[x + hw, hh - y - 1] = tuple(color.astype(np.uint8))

    path = "images/hybrid_neural_render.png"
    img.save(path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    render(blend=0.5)   # 50 % classical, 50 % neural