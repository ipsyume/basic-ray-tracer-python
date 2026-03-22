# Recursive Ray Tracer with Neural Rendering Extensions

A physically-based renderer built from scratch in Python — no rendering libraries, pure NumPy math — extended with two neural rendering approaches bridging classical graphics and learned pipelines.

---

## Renders

| Classic Ray Tracer | Hybrid Neural | Pure Neural |
|---|---|---|
| Blinn–Phong + reflections | 50% classical + 50% neural lighting | 100% network, no geometry |

> Images saved to `images/` after running each script.

---

## Project Structure

```
├── classic_ray_tracer.py   # Fully classical: ray tracing + Blinn-Phong + reflections
├── main.py                 # Hybrid: classical scene + neural lighting MLP blended
├── neural_render.py        # Pure neural: MLP trained on ray→RGB, renders with no geometry
└── requirements.txt
```

---

## What's Implemented

### `classic_ray_tracer.py` — Classical Pipeline
- Ray–sphere and ray–plane intersection (analytical)
- Blinn–Phong shading (ambient + diffuse + specular)
- Recursive reflections (configurable depth)
- Hard shadows via shadow ray casting
- Checkerboard procedural texture on ground plane
- Multi-sample anti-aliasing (3×3 grid supersampling)

### `main.py` — Hybrid Neural Ray Tracer
- Same scene geometry and shadow logic as classical
- Small MLP (`NeuralShader`) trained to predict **scalar lighting intensity** from:
  - Surface normal
  - View direction  
  - Primary light direction
- At render time: `intensity = (1 - blend) * phong + blend * neural`
- Base colour multiplied **after** blending — material identity preserved
- Training: 10,000 Phong-computed samples, 600 epochs, cosine LR schedule

### `neural_render.py` — Pure Neural Renderer
- Classical tracer used only to generate 60,000 `(ray_direction → RGB)` training pairs
- MLP with **Fourier positional encoding** (NeRF-inspired) maps directions to colours
- Final render: entire image produced by network inference — no intersection tests, no lights, no geometry
- Architecture: 5-layer MLP, 256 hidden units, positional encoding with 6 frequency bands

---

## Setup

```bash
# Clone or download files, then:
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Run

```bash
# Fastest (~1-2 min, CPU)
python classic_ray_tracer.py

# Hybrid neural (~5-10 min)
python main.py

# Pure neural renderer (~10-15 min)
python neural_render.py
```

All outputs saved to `images/`.

---

## Tech Stack

| Library | Role |
|---|---|
| `numpy` | All ray math, vector operations, scene geometry |
| `torch` | NeuralShader and NeuralRenderer MLP training |
| `Pillow` | Image construction and PNG export |

---

## Key Design Decisions

**Why scalar output for the neural shader?**  
The MLP predicts lighting *intensity* (a single float), not colour. Base colour is multiplied after blending. This prevents the network from washing out material colours — a common failure mode when networks output full RGB directly.

**Why positional encoding in the pure neural renderer?**  
Raw 3D ray directions are too low-frequency for an MLP to learn sharp colour boundaries (the sphere edges, shadow lines). Fourier encoding lifts inputs into higher-frequency space, the same insight that makes NeRF work.

---

## Context

Built as a portfolio project for the Computer Graphics Laboratory at National Cheng Kung University (NCKU), Taiwan. Demonstrates the transition from classical physically-based rendering to neural/learned rendering pipelines — a core theme in modern computer graphics research.

---

## License

MIT
