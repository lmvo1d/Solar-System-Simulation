# 🌌 Solar System Simulation (Python + OpenGL)

A real-time **3D solar system simulation** built from scratch using **Python, Modern OpenGL, and GLFW**.  
This project demonstrates core **graphics programming concepts** such as camera systems, lighting models, procedural geometry, and shader-based rendering.

> 🚀 Everything is implemented manually using OpenGL fundamentals , no game engines or high-level frameworks.

---

## ✨ Features

### 🌞 Astronomical Simulation
- Sun placed at the center as a **real light source**
- All **8 planets** orbiting with independent speeds
- Scaled planetary sizes and orbital distances
- Smooth time-based animation using delta time

### 💡 Lighting (Phong Shading)
- Ambient, Diffuse, and Specular lighting
- Sun acts as a **point light source**
- Correct normal transformation using model matrices
- Emissive Sun (not affected by lighting)

### 🎮 Camera System
- FPS-style camera
- Mouse look (yaw & pitch)
- WASD movement
- Perspective projection
- ESC key to exit

### 🌀 Visual Enhancements
- Procedurally generated orbit rings
- Infinite starfield background
- Depth-correct rendering order
- Separate shaders for:
  - Planets (Phong lighting)
  - Sun (emissive texture)
  - Orbits (line rendering)
  - Stars (point rendering)

---

## 🧠 Concepts Demonstrated

- Modern OpenGL (VAOs, VBOs, shaders)
- Model / View / Projection (MVP) pipeline
- Phong lighting model
- Normal vectors & inverse-transpose matrices
- Procedural mesh generation (UV sphere)
- Multiple shader programs
- Real-time keyboard & mouse input
- Depth testing & render state management

---

## 🛠️ Technologies Used

- **Python 3**
- **PyOpenGL**
- **GLFW**
- **PyGLM**
- **Pillow (PIL)**
- **NumPy**

---

## 📁 Project Structure

```
solar-system/
│
├── main.py
├── README.md
└── textures/
    ├── sun.jpg
    ├── mercury.jpg
    ├── venus.jpg
    ├── earth.jpg
    ├── mars.jpg
    ├── jupiter.jpg
    ├── saturn.jpg
    ├── uranus.jpg
    └── neptune.jpg
```

---

## ▶️ How to Run

### Install dependencies
```bash
pip install glfw PyOpenGL PyGLM Pillow numpy
```

### Run
```bash
python main.py
```

---

## 🎮 Controls

| Input | Action |
|------|--------|
| Mouse | Look around |
| W / A / S / D | Move camera |
| ESC | Exit application |

---

## 🚀 Future Improvements

- Light attenuation
- Earth–Moon hierarchy
- Saturn rings
- Atmosphere scattering
- Shadow mapping
- Skybox cube-map

---

⭐ If you found this project interesting, consider starring the repository.
