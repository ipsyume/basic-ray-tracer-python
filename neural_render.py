"""
Pure Neural Renderer
====================
Step 1 – Use the classical ray tracer to generate pixel-level ground truth
         (ray direction → RGB colour).
Step 2 – Train an MLP on that data.
Step 3 – Render the full image using only the network (no geometry traversal).

This illustrates how a learned model can approximate a renderer and is the
simplest form of neural rendering — a precursor to NeRF-style approaches.
"""

import numpy as np
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.optim as optim

os.makedirs("images", exist_ok=True)

# ==================================================
# Mini Classical Ray Tracer (Ground-Truth Generator)
# ==================================================

def _normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def _dot(a, b):
    return np.dot(a, b)

def _reflect(I, N):
    return I - 2 * _dot(I, N) * N


class _Sphere:
    def __init__(self, center, radius, color, specular, reflective):
        self.center, self.radius = np.array(center, float), radius
        self.color = np.array(color, float)
        self.specular, self.reflective = specular, reflective

    def intersect(self, o, d):
        oc = o - self.center
        a, b = _dot(d, d), 2 * _dot(oc, d)
        c = _dot(oc, oc) - self.radius ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return np.inf
        sq = np.sqrt(disc)
        ts = [t for t in ((-b - sq) / (2*a), (-b + sq) / (2*a)) if t > 1e-4]
        return min(ts) if ts else np.inf

    def normal_at(self, p):
        return _normalize(p - self.center)

    def get_color(self, _):
        return self.color


class _Plane:
    def __init__(self, point, normal, color, specular, reflective, checker):
        self.point = np.array(point, float)
        self.normal = _normalize(np.array(normal, float))
        self.color = np.array(color, float)
        self.specular, self.reflective, self.checker = specular, reflective, checker

    def intersect(self, o, d):
        denom = _dot(self.normal, d)
        if abs(denom) < 1e-6:
            return np.inf
        t = _dot(self.point - o, self.normal) / denom
        return t if t > 1e-4 else np.inf

    def normal_at(self, _):
        return self.normal

    def get_color(self, point):
        if not self.checker:
            return self.color
        check = (int(np.floor(point[0])) + int(np.floor(point[2]))) % 2
        return self.color if check == 0 else np.array([30.0, 30.0, 30.0])


class _Light:
    def __init__(self, pos, intensity):
        self.position, self.intensity = np.array(pos, float), intensity


def _lighting(pt, normal, view, spec, objects, lights):
    I = 0.05
    so = pt + normal * 1e-3
    for L in lights:
        tl = _normalize(L.position - pt)
        ld = np.linalg.norm(L.position - pt)
        if any(o.intersect(so, tl) < ld for o in objects):
            continue
        I += L.intensity * max(_dot(normal, tl), 0)
        if spec > 0:
            h = _normalize(tl + view)
            I += L.intensity * (max(_dot(normal, h), 0) ** spec) * 0.3
    return float(np.clip(I, 0, 2))


def _trace(o, d, objects, lights, depth=3):
    best_t, hit = np.inf, None
    for obj in objects:
        t = obj.intersect(o, d)
        if t < best_t:
            best_t, hit = t, obj
    if hit is None:
        return np.array([30.0, 30.0, 40.0])
    pt = o + best_t * d
    n  = hit.normal_at(pt)
    v  = _normalize(-d)
    c  = hit.get_color(pt) * _lighting(pt, n, v, hit.specular, objects, lights)
    if depth <= 0 or hit.reflective <= 0:
        return np.clip(c, 0, 255)
    rc = _trace(pt + n * 1e-3, _normalize(_reflect(d, n)), objects, lights, depth-1)
    return np.clip((1 - hit.reflective) * c + hit.reflective * rc, 0, 255)


def _build_scene():
    objects = [
        _Sphere([ 0.0, -0.2, 3.0], 0.8, [220,  60,  60], 100, 0.4),
        _Sphere([ 1.5,  0.2, 4.0], 0.9, [ 60, 120, 240],  50, 0.3),
        _Sphere([-1.8,  0.0, 4.5], 1.0, [ 60, 200, 120],  30, 0.2),
        _Plane([0, -1, 0], [0, 1, 0], [210, 200, 180], 5, 0.25, True),
    ]
    lights = [_Light([2, 3, -1], 1.4), _Light([-3, 5, -2], 0.8)]
    return objects, lights

# ==================================================
# Dataset: sample random rays → ground-truth colours
# ==================================================

def generate_ray_color_dataset(n_samples=60_000, width=640, height=480):
    """
    Randomly sample (x, y) pixel coordinates, compute their ray direction,
    trace the classical ray, and record the resulting RGB colour.

    Returns:
        X : (N, 3) tensor of unit ray directions
        Y : (N, 3) tensor of normalised RGB colours in [0, 1]
    """
    objects, lights = _build_scene()
    camera = np.array([0.0, 0.0, -1.0])
    hw, hh = width // 2, height // 2

    xs = np.random.randint(0, width,  size=n_samples)
    ys = np.random.randint(0, height, size=n_samples)

    X_list, Y_list = [], []
    for xi, yi in zip(xs, ys):
        cx = (xi - hw) / width
        cy = (hh - yi) / height
        d  = _normalize(np.array([cx, cy, 1.0]))
        c  = _trace(camera, d, objects, lights, depth=3) / 255.0
        X_list.append(d.astype(np.float32))
        Y_list.append(c.astype(np.float32))

    return (torch.tensor(np.array(X_list), dtype=torch.float32),
            torch.tensor(np.array(Y_list), dtype=torch.float32))

# ==================================================
# Neural Renderer — MLP mapping ray_dir → RGB
# ==================================================

class NeuralRenderer(nn.Module):
    """
    Approximates the entire rendering pipeline as a function:
        f(ray_direction) → (R, G, B)

    Architecture: positional encoding + 5-layer MLP.
    Positional encoding lifts the 3-D direction into a higher-frequency
    space so the network can represent fine colour boundaries.
    """
    def __init__(self, freq_bands=6):
        super().__init__()
        self.freq_bands = freq_bands
        in_dim = 3 * (1 + 2 * freq_bands)   # raw + sin/cos pairs per band

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),    nn.ReLU(),
            nn.Linear(256, 128),    nn.ReLU(),
            nn.Linear(128,  64),    nn.ReLU(),
            nn.Linear( 64,   3),    nn.Sigmoid(),
        )

    def positional_encode(self, x):
        """Fourier feature encoding for input directions."""
        parts = [x]
        for k in range(self.freq_bands):
            freq = 2.0 ** k
            parts.append(torch.sin(freq * x))
            parts.append(torch.cos(freq * x))
        return torch.cat(parts, dim=-1)

    def forward(self, x):
        return self.net(self.positional_encode(x))


def train(model, X, Y, epochs=800, batch_size=4096):
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.MSELoss()
    N         = X.shape[0]

    for e in range(epochs):
        idx  = torch.randperm(N)[:batch_size]
        pred = model(X[idx])
        loss = loss_fn(pred, Y[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if e % 100 == 0:
            print(f"  Epoch {e:4d} | Loss: {loss.item():.6f}")

    print(f"  Training done. Final loss: {loss.item():.8f}\n")

# ==================================================
# Render Purely via the Network
# ==================================================

def render(width=640, height=480):
    print("Step 1 — Generating ground-truth ray-colour dataset …")
    X, Y = generate_ray_color_dataset(n_samples=60_000, width=width, height=height)
    print(f"  Dataset: {X.shape[0]:,} samples\n")

    print("Step 2 — Training neural renderer …")
    model = NeuralRenderer(freq_bands=6)
    train(model, X, Y)

    print("Step 3 — Rendering with neural model (no geometry) …")
    hw, hh = width // 2, height // 2
    img    = Image.new("RGB", (width, height))
    pixels = img.load()

    model.eval()
    with torch.no_grad():
        # Build full grid of ray directions
        xs = np.arange(-hw, hw)
        ys = np.arange(-hh, hh)
        gx, gy = np.meshgrid(xs, ys, indexing='ij')    # (W, H)
        dx = (gx / width).ravel().astype(np.float32)
        dy = (gy / height).ravel().astype(np.float32)
        dz = np.ones_like(dx)
        dirs = np.stack([dx, dy, dz], axis=1)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs  = (dirs / norms).astype(np.float32)

        rays   = torch.tensor(dirs)
        colors = model(rays).numpy() * 255.0  # (W*H, 3)

    colors = colors.reshape(width, height, 3)
    for xi in range(width):
        for yi in range(height):
            y_pixel = hh - (yi - hh) - 1
            if 0 <= y_pixel < height:
                pixels[xi, y_pixel] = tuple(colors[xi, yi].astype(np.uint8))

    path = "images/neural_render.png"
    img.save(path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    render()