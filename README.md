# Basic Ray Tracer in Python

This project implements a minimal ray tracer from scratch in Python to render simple 3D scenes using geometric ray-object intersections and lighting calculations. The goal is to demonstrate the fundamentals of computer graphics by converting mathematical concepts into a rendered image.

---

## Overview
Ray tracing is a rendering technique that simulates the physical behavior of light by tracing rays from a virtual camera into a scene. Each ray interacts with objects to determine color based on geometry, surface properties, and lighting.

This implementation renders:
- Spheres
- A ground plane
- Basic lighting and shading
- Shadows and specular highlights

---

## Features
- Ray–sphere intersection
- Ray–plane intersection
- Surface normals and reflection vectors
- Ambient, diffuse, and specular lighting (Blinn-Phong model)
- Hard shadows using shadow rays
- Image generation using Python and Pillow

---

## Scene Description
The rendered scene consists of:
- Multiple colored spheres with different material properties
- A ground plane
- A single point light source
- A fixed camera viewing the scene

---

## Rendered Output
Below is an example output produced by the ray tracer:

![Rendered Output](screenshots/rendered_output.png)

---

## Project Structure
