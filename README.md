# Basic Ray Tracer in Python

This project implements a ray tracer from scratch in Python to render 3D scenes using analytical ray–object intersections and physically inspired lighting models. The primary goal is to translate core computer graphics theory; such as geometry, illumination, and reflection, into a functioning rendering pipeline.
---

## Overview
Ray tracing is a rendering technique that simulates the behavior of light by tracing rays from a virtual camera into a scene and evaluating their interactions with objects. Each ray contributes to the final pixel color based on surface geometry, material properties, lighting conditions, and recursive reflection.

This implementation renders scenes composed of:

Spherical objects

A ground plane

Point light sources

Recursive reflections and shadows

---

## Features
- Ray–sphere and ray–plane intersection

- Surface normal computation and reflection vectors

- Ambient, diffuse, and specular shading using the Blinn–Phong model

- Hard shadows using shadow rays

- Recursive reflection for reflective surfaces

- Supersampling-based anti-aliasing

- Image generation using Python and Pillow

---

## Scene Description
- Multiple colored spheres with varying material properties

- A ground plane with optional checkerboard patterning

- Multiple point light sources

- A fixed pinhole camera model

---

## Rendered Output
Below is an example output produced by the ray tracer:

![Rendered Output](final_ray_tracer.png) ![Rendered Output](neural_render.png)


---
