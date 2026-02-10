import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math
import glm
import ctypes
from PIL import Image
import random

# =========================================================
# SHADERS
# =========================================================

# ---------- SUN (emissive, textured) ----------
sun_vertex = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTex;
uniform mat4 MVP;
out vec2 TexCoord;
void main()
{
    TexCoord = aTex;
    gl_Position = MVP * vec4(aPos, 1.0);
}
"""

sun_fragment = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D tex;
void main()
{
    FragColor = texture(tex, TexCoord);
}
"""

# ---------- PLANETS (Phong lighting) ----------
planet_vertex = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTex;

uniform mat4 MVP;
uniform mat4 model;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTex;
    gl_Position = MVP * vec4(aPos, 1.0);
}
"""

planet_fragment = """
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D tex;
uniform vec3 lightPos;
uniform vec3 viewPos;

void main()
{
    vec3 color = texture(tex, TexCoord).rgb;

    // Ambient
    vec3 ambient = 0.15 * color;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * color;

    // Specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = vec3(0.5) * spec;

    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}
"""

# ---------- ORBITS ----------
orbit_vertex = """
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 MVP;
void main()
{
    gl_Position = MVP * vec4(aPos, 1.0);
}
"""

orbit_fragment = """
#version 330 core
out vec4 FragColor;
uniform vec3 color;
void main()
{
    FragColor = vec4(color, 1.0);
}
"""

# ---------- STARS ----------
star_vertex = """
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 MVP;
void main()
{
    gl_Position = MVP * vec4(aPos, 1.0);
    gl_PointSize = 2.0;
}
"""

star_fragment = """
#version 330 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(1.0);
}
"""

# =========================================================
# GEOMETRY
# =========================================================

def generate_sphere(sectors=36, stacks=18):
    v, i = [], []

    for s in range(stacks + 1):
        sa = math.pi/2 - s*math.pi/stacks
        xy = math.cos(sa)
        y = math.sin(sa)

        for t in range(sectors + 1):
            a = t * 2 * math.pi / sectors
            x = xy * math.cos(a)
            z = xy * math.sin(a)

            # position
            v += [x, y, z]
            # normal
            v += [x, y, z]
            # uv
            v += [t/sectors, s/stacks]

    for s in range(stacks):
        for t in range(sectors):
            f = s*(sectors+1) + t
            n = f + sectors + 1
            i += [f, n, f+1, n, n+1, f+1]

    return np.array(v, np.float32), np.array(i, np.uint32)

def generate_orbit(r, seg=180):
    return np.array(
        [r*math.cos(a) for a in np.linspace(0,2*math.pi,seg)
         for _ in (0,1)], np.float32
    ).reshape(-1,2).flatten()

def generate_stars(count=4000, radius=300):
    stars = []
    for _ in range(count):
        th = random.uniform(0, 2*math.pi)
        ph = random.uniform(-math.pi/2, math.pi/2)
        r = random.uniform(radius*0.6, radius)
        stars += [
            r*math.cos(ph)*math.cos(th),
            r*math.sin(ph),
            r*math.cos(ph)*math.sin(th)
        ]
    return np.array(stars, np.float32)

# =========================================================
# TEXTURE LOADER
# =========================================================

def load_texture(path):
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGB")
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
    glGenerateMipmap(GL_TEXTURE_2D)
    return tex

# =========================================================
# CAMERA
# =========================================================

class Camera:
    def __init__(self):
        self.pos = glm.vec3(0,3,25)
        self.front = glm.vec3(0,0,-1)
        self.up = glm.vec3(0,1,0)
        self.yaw = -90
        self.pitch = 0
        self.speed = 12
        self.sens = 0.12
        self.first = True
        self.lx, self.ly = 600, 400

    def view(self):
        return glm.lookAt(self.pos, self.pos+self.front, self.up)

    def keyboard(self, w, dt):
        v = self.speed * dt
        right = glm.normalize(glm.cross(self.front, self.up))
        if glfw.get_key(w, glfw.KEY_W) == glfw.PRESS:
            self.pos += self.front * v
        if glfw.get_key(w, glfw.KEY_S) == glfw.PRESS:
            self.pos -= self.front * v
        if glfw.get_key(w, glfw.KEY_A) == glfw.PRESS:
            self.pos -= right * v
        if glfw.get_key(w, glfw.KEY_D) == glfw.PRESS:
            self.pos += right * v

    def mouse(self, x, y):
        if self.first:
            self.lx, self.ly = x, y
            self.first = False
        dx = (x - self.lx) * self.sens
        dy = (self.ly - y) * self.sens
        self.lx, self.ly = x, y
        self.yaw += dx
        self.pitch = max(-89, min(89, self.pitch + dy))
        d = glm.vec3(
            math.cos(math.radians(self.yaw))*math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw))*math.cos(math.radians(self.pitch))
        )
        self.front = glm.normalize(d)

# =========================================================
# PLANET CLASS
# =========================================================

class Planet:
    def __init__(self, r, orad, os, rs, tex):
        self.r, self.orad, self.os, self.rs, self.tex = r, orad, os, rs, tex
        self.a, self.ra = 0, 0

    def update(self, dt):
        self.a += self.os * dt
        self.ra += self.rs * dt

    def model(self):
        x = self.orad * math.cos(self.a)
        z = self.orad * math.sin(self.a)
        m = glm.translate(glm.mat4(1), glm.vec3(x,0,z))
        m = glm.rotate(m, self.ra, glm.vec3(0,1,0))
        return glm.scale(m, glm.vec3(self.r))

# =========================================================
# INIT
# =========================================================

glfw.init()
win = glfw.create_window(1200,800,"Solar System Simulation",None,None)
glfw.make_context_current(win)
glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_DISABLED)
glEnable(GL_DEPTH_TEST)

camera = Camera()
glfw.set_cursor_pos_callback(win, lambda w,x,y: camera.mouse(x,y))
projection = glm.perspective(glm.radians(45), 1200/800, 0.1, 500)

# =========================================================
# BUFFERS
# =========================================================

sphere_v, sphere_i = generate_sphere()
index_count = len(sphere_i)

sphere_VAO = glGenVertexArrays(1)
sphere_VBO = glGenBuffers(1)
sphere_EBO = glGenBuffers(1)

glBindVertexArray(sphere_VAO)
glBindBuffer(GL_ARRAY_BUFFER, sphere_VBO)
glBufferData(GL_ARRAY_BUFFER, sphere_v.nbytes, sphere_v, GL_STATIC_DRAW)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere_i.nbytes, sphere_i, GL_STATIC_DRAW)

glVertexAttribPointer(0,3,GL_FLOAT,False,32,ctypes.c_void_p(0))
glEnableVertexAttribArray(0)
glVertexAttribPointer(1,3,GL_FLOAT,False,32,ctypes.c_void_p(12))
glEnableVertexAttribArray(1)
glVertexAttribPointer(2,2,GL_FLOAT,False,32,ctypes.c_void_p(24))
glEnableVertexAttribArray(2)

# =========================================================
# SHADERS
# =========================================================

sun_shader = compileProgram(
    compileShader(sun_vertex, GL_VERTEX_SHADER),
    compileShader(sun_fragment, GL_FRAGMENT_SHADER)
)

planet_shader = compileProgram(
    compileShader(planet_vertex, GL_VERTEX_SHADER),
    compileShader(planet_fragment, GL_FRAGMENT_SHADER)
)

orbit_shader = compileProgram(
    compileShader(orbit_vertex, GL_VERTEX_SHADER),
    compileShader(orbit_fragment, GL_FRAGMENT_SHADER)
)

star_shader = compileProgram(
    compileShader(star_vertex, GL_VERTEX_SHADER),
    compileShader(star_fragment, GL_FRAGMENT_SHADER)
)

# Uniform locations
sun_mvp = glGetUniformLocation(sun_shader, "MVP")
planet_mvp = glGetUniformLocation(planet_shader, "MVP")
planet_model = glGetUniformLocation(planet_shader, "model")
light_pos = glGetUniformLocation(planet_shader, "lightPos")
view_pos = glGetUniformLocation(planet_shader, "viewPos")
orbit_mvp = glGetUniformLocation(orbit_shader, "MVP")
orbit_color = glGetUniformLocation(orbit_shader, "color")
star_mvp = glGetUniformLocation(star_shader, "MVP")

# =========================================================
# STARFIELD
# =========================================================

stars = generate_stars()
star_VAO = glGenVertexArrays(1)
star_VBO = glGenBuffers(1)
glBindVertexArray(star_VAO)
glBindBuffer(GL_ARRAY_BUFFER, star_VBO)
glBufferData(GL_ARRAY_BUFFER, stars.nbytes, stars, GL_STATIC_DRAW)
glVertexAttribPointer(0,3,GL_FLOAT,False,12,ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

# =========================================================
# ORBITS
# =========================================================

orbit_radii = [4,6,8,10,13,16,19,22]
orbit_vaos = []

for r in orbit_radii:
    verts = np.array([[r*math.cos(a),0,r*math.sin(a)]
                      for a in np.linspace(0,2*math.pi,180)],
                      np.float32).flatten()
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    glVertexAttribPointer(0,3,GL_FLOAT,False,12,ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    orbit_vaos.append((vao, len(verts)//3))

# =========================================================
# OBJECTS
# =========================================================

sun = Planet(2.2, 0, 0, 0.3, load_texture("textures/sun.jpg"))

names = ["mercury","venus","earth","mars","jupiter","saturn","uranus","neptune"]
sizes = [0.3,0.5,0.55,0.4,1.2,1.0,0.8,0.75]
speeds = [2.0,1.6,1.2,1.0,0.7,0.6,0.5,0.4]

planets = [
    Planet(sizes[i], orbit_radii[i], speeds[i], 1.5,
           load_texture(f"textures/{names[i]}.jpg"))
    for i in range(8)
]

# =========================================================
# MAIN LOOP
# =========================================================

last = glfw.get_time()

while not glfw.window_should_close(win):
    glfw.poll_events()
    if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
        break

    now = glfw.get_time()
    dt = now - last
    last = now

    camera.keyboard(win, dt)
    view = camera.view()

    glClearColor(0.02,0.02,0.05,1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # -------- STARS --------
    glDisable(GL_DEPTH_TEST)
    glUseProgram(star_shader)
    MVP = projection * view * glm.translate(glm.mat4(1), camera.pos)
    glUniformMatrix4fv(star_mvp, 1, GL_FALSE, glm.value_ptr(MVP))
    glBindVertexArray(star_VAO)
    glDrawArrays(GL_POINTS, 0, len(stars)//3)
    glEnable(GL_DEPTH_TEST)

    glUseProgram(orbit_shader)
    glLineWidth(1.5)
    
    for vao, cnt in orbit_vaos:
        MVP = projection * view * glm.mat4(1)
        glUniformMatrix4fv(orbit_mvp, 1, GL_FALSE, glm.value_ptr(MVP))
        glUniform3f(orbit_color, 0.7, 0.7, 0.7)
        glBindVertexArray(vao)
        glDrawArrays(GL_LINE_LOOP, 0, cnt)

    glUseProgram(sun_shader)
    glBindVertexArray(sphere_VAO)
    glBindTexture(GL_TEXTURE_2D, sun.tex)
    sun.update(dt)

    MVP = projection * view * sun.model()
    glUniformMatrix4fv(sun_mvp, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, None)

    glUseProgram(planet_shader)
    glUniform3fv(light_pos, 1, glm.value_ptr(glm.vec3(0,0,0)))
    glUniform3fv(view_pos, 1, glm.value_ptr(camera.pos))

    for p in planets:
        p.update(dt)
        model = p.model()
        MVP = projection * view * model
        glUniformMatrix4fv(planet_mvp, 1, GL_FALSE, glm.value_ptr(MVP))
        glUniformMatrix4fv(planet_model, 1, GL_FALSE, glm.value_ptr(model))
        glBindTexture(GL_TEXTURE_2D, p.tex)
        glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, None)

    glfw.swap_buffers(win)

glfw.terminate()
