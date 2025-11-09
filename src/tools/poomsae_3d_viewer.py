# src/tools/poomsae_3d_viewer.py
import argparse
from pathlib import Path
import numpy as np
import math

import pyglet
from pyglet.window import key, mouse

# usamos PyOpenGL porque pyglet 2.x no trae el fixed pipeline completo
from OpenGL.GL import *
from OpenGL.GLU import *

from src.segmentation import move_capture as mc

# usamos las mismas conexiones que tu sistema
LMK_CONNECTIONS = mc.CONNECTIONS

# ids "por lado" para inventar profundidad
LEFT_IDS = {11, 13, 15, 23, 25, 27, 29, 31}
RIGHT_IDS = {12, 14, 16, 24, 26, 28, 30, 32}
CENTER_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}


class OrbitCamera:
    def __init__(self):
        self.distance = 3.5
        self.yaw = 35.0
        self.pitch = -20.0
        self.center = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def apply(self, width, height):
        aspect = float(width) / float(height if height > 0 else 1)
        glViewport(0, 0, width, height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, aspect, 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)

        eye_x = self.center[0] + self.distance * math.cos(rad_pitch) * math.cos(rad_yaw)
        eye_y = self.center[1] + self.distance * math.sin(rad_pitch)
        eye_z = self.center[2] + self.distance * math.cos(rad_pitch) * math.sin(rad_yaw)

        gluLookAt(
            eye_x, eye_y, eye_z,
            self.center[0], self.center[1], self.center[2],
            0.0, 1.0, 0.0
        )

    def orbit(self, dx, dy):
        self.yaw += dx * 0.3
        self.pitch += dy * 0.3
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def dolly(self, delta):
        self.distance *= (1.0 - delta * 0.1)
        self.distance = max(1.0, min(15.0, self.distance))


class Poomsae3DWindow(pyglet.window.Window):
    def __init__(self, capture_result, csv_path: Path, video_path: Path | None):
        config = pyglet.gl.Config(double_buffer=True, depth_size=24)
        super().__init__(1280, 720, "Poomsae 3D Viewer", resizable=True, config=config)

        self.capture = capture_result
        self.csv_path = csv_path
        self.video_path = video_path
        self.cam = OrbitCamera()

        # CSV completo
        self.landmarks_df = mc.load_landmarks_csv(csv_path)
        self.nframes = int(self.landmarks_df["frame"].max()) + 1

        # rearmamos (frame -> [33 x 3])
        self.frames_xyz = self._build_xyz_from_csv(self.landmarks_df, self.nframes)

        self.curr_frame = 0
        self.fps = capture_result.fps if capture_result.fps > 0 else 30.0
        pyglet.clock.schedule_interval(self.update_frame, 1.0 / self.fps)

        self.mode_avatar = True  # True = cuerpo, False = esqueleto
        self._mouse_last = None

        self._init_gl()

    # ------------------- OPENGL SETUP -------------------
    def _init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.04, 0.04, 0.06, 1.0)

        # iluminación básica
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(2.5, 4.0, 2.0, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 4)(1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat * 4)(0.25, 0.25, 0.25, 1.0))

        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)

    # ------------------- CSV -> FRAMES 3D -------------------
    def _build_xyz_from_csv(self, df, nframes: int):
        """
        Inventamos z según lado, así cuando rotas se ve volumen.
        Además, metemos un pequeño "sway" de z por frame.
        """
        frames = []
        for f in range(nframes):
            sub = df[df["frame"] == f]
            arr = np.zeros((33, 3), dtype=np.float32)
            sway = math.sin(f * 0.03) * 0.015  # pequeño movimiento
            for _, row in sub.iterrows():
                lmk = int(row["lmk_id"])
                x = float(row["x"])
                y = float(row["y"])

                # normalizamos similar a antes
                X = (x - 0.5) * 1.6
                Y = (1.0 - y) * 2.0

                # profundidad sintética por lado
                if lmk in LEFT_IDS:
                    Z = 0.14 + sway
                elif lmk in RIGHT_IDS:
                    Z = -0.14 + sway
                else:
                    Z = 0.0 + sway

                arr[lmk, 0] = X
                arr[lmk, 1] = Y
                arr[lmk, 2] = Z
            frames.append(arr)
        return frames

    # ------------------- UPDATE -------------------
    def update_frame(self, dt):
        self.curr_frame = (self.curr_frame + 1) % self.nframes

    # ------------------- DRAW -------------------
    def on_draw(self):
        self.clear()
        width, height = self.get_size()
        self.cam.apply(width, height)

        self._draw_ground()
        if self.mode_avatar:
            self._draw_avatar(self.curr_frame)
        else:
            self._draw_skeleton(self.curr_frame)

        self._draw_info_overlay()

    def _draw_ground(self):
        glDisable(GL_TEXTURE_2D)
        glColor3f(0.15, 0.15, 0.15)
        glBegin(GL_QUADS)
        s = 5.5
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(-s, 0.0, -s)
        glVertex3f(s, 0.0, -s)
        glVertex3f(s, 0.0, s)
        glVertex3f(-s, 0.0, s)
        glEnd()

        # ejes
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # X
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0); glVertex3f(1.0, 0.0, 0.0)
        # Y
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0); glVertex3f(0.0, 1.0, 0.0)
        # Z
        glColor3f(0.0, 0.4, 1.0)
        glVertex3f(0.0, 0.0, 0.0); glVertex3f(0.0, 0.0, 1.0)
        glEnd()

    # ------------------- SKELETON MODE -------------------
    def _draw_skeleton(self, frame_idx: int):
        if frame_idx >= len(self.frames_xyz):
            return
        pts = self.frames_xyz[frame_idx]

        # huesos
        glLineWidth(4.0)
        glColor3f(1.0, 0.9, 0.4)
        glBegin(GL_LINES)
        for i, j in LMK_CONNECTIONS:
            p1 = pts[i]; p2 = pts[j]
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])
        glEnd()

        # articulaciones
        glPointSize(7.0)
        glBegin(GL_POINTS)
        glColor3f(1.0, 0.3, 0.4)
        for p in pts:
            glVertex3f(p[0], p[1], p[2])
        glEnd()

    # ------------------- AVATAR MODE -------------------
    def _draw_avatar(self, frame_idx: int):
        if frame_idx >= len(self.frames_xyz):
            return
        pts = self.frames_xyz[frame_idx]

        # partes clave
        L_SH = pts[mc.LMK["L_SH"]]
        R_SH = pts[mc.LMK["R_SH"]]
        L_HIP = pts[mc.LMK["L_HIP"]]
        R_HIP = pts[mc.LMK["R_HIP"]]
        NECK = (L_SH + R_SH) * 0.5
        PELVIS = (L_HIP + R_HIP) * 0.5
        CHEST = NECK + np.array([0.0, 0.18, 0.0], dtype=np.float32)

        # 1) torso (caja)
        self._draw_box(center=(PELVIS + CHEST) * 0.5,
                       sx=0.35, sy=abs(CHEST[1] - PELVIS[1]) * 0.8 + 0.15,
                       sz=0.25,
                       color=(0.9, 0.9, 0.95))

        # 2) pelvis (caja más pequeña)
        self._draw_box(center=PELVIS + np.array([0.0, -0.10, 0.0]),
                       sx=0.30, sy=0.20, sz=0.22,
                       color=(0.85, 0.85, 0.9))

        # 3) cabeza (esferita)
        HEAD = NECK + np.array([0.0, 0.28, 0.0], dtype=np.float32)
        self._draw_sphere(HEAD, 0.11, color=(0.95, 0.85, 0.75))

        # 4) brazos
        self._draw_limb(pts, "L_SH", "L_ELB", "L_WRIST", radius=0.05, color=(0.9, 0.9, 1.0))
        self._draw_limb(pts, "R_SH", "R_ELB", "R_WRIST", radius=0.05, color=(0.9, 0.9, 1.0))

        # 5) piernas
        self._draw_limb(pts, "L_HIP", "L_KNEE", "L_ANK", radius=0.06, color=(0.9, 0.9, 1.0))
        self._draw_limb(pts, "R_HIP", "R_KNEE", "R_ANK", radius=0.06, color=(0.9, 0.9, 1.0))

    # -------------- helpers avatar --------------
    def _draw_box(self, center, sx, sy, sz, color=(1, 1, 1)):
        cx, cy, cz = center
        hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5

        glColor3f(*color)
        glBegin(GL_QUADS)

        # front
        glNormal3f(0, 0, 1)
        glVertex3f(cx - hx, cy - hy, cz + hz)
        glVertex3f(cx + hx, cy - hy, cz + hz)
        glVertex3f(cx + hx, cy + hy, cz + hz)
        glVertex3f(cx - hx, cy + hy, cz + hz)

        # back
        glNormal3f(0, 0, -1)
        glVertex3f(cx - hx, cy - hy, cz - hz)
        glVertex3f(cx + hx, cy - hy, cz - hz)
        glVertex3f(cx + hx, cy + hy, cz - hz)
        glVertex3f(cx - hx, cy + hy, cz - hz)

        # left
        glNormal3f(-1, 0, 0)
        glVertex3f(cx - hx, cy - hy, cz - hz)
        glVertex3f(cx - hx, cy - hy, cz + hz)
        glVertex3f(cx - hx, cy + hy, cz + hz)
        glVertex3f(cx - hx, cy + hy, cz - hz)

        # right
        glNormal3f(1, 0, 0)
        glVertex3f(cx + hx, cy - hy, cz - hz)
        glVertex3f(cx + hx, cy - hy, cz + hz)
        glVertex3f(cx + hx, cy + hy, cz + hz)
        glVertex3f(cx + hx, cy + hy, cz - hz)

        # top
        glNormal3f(0, 1, 0)
        glVertex3f(cx - hx, cy + hy, cz - hz)
        glVertex3f(cx + hx, cy + hy, cz - hz)
        glVertex3f(cx + hx, cy + hy, cz + hz)
        glVertex3f(cx - hx, cy + hy, cz + hz)

        # bottom
        glNormal3f(0, -1, 0)
        glVertex3f(cx - hx, cy - hy, cz - hz)
        glVertex3f(cx + hx, cy - hy, cz - hz)
        glVertex3f(cx + hx, cy - hy, cz + hz)
        glVertex3f(cx - hx, cy - hy, cz + hz)

        glEnd()

    def _draw_sphere(self, center, radius, color=(1, 1, 1), slices=14, stacks=10):
        cx, cy, cz = center
        glPushMatrix()
        glTranslatef(cx, cy, cz)
        glColor3f(*color)
        quad = gluNewQuadric()
        gluSphere(quad, radius, slices, stacks)
        gluDeleteQuadric(quad)
        glPopMatrix()

    def _draw_limb(self, pts, root_name, mid_name, end_name, radius=0.05, color=(1, 1, 1)):
        """
        Dibuja brazo o pierna como 2 "huesos" gordos: root->mid y mid->end
        """
        root = pts[mc.LMK[root_name]]
        mid = pts[mc.LMK[mid_name]]
        end = pts[mc.LMK[end_name]]

        self._draw_bone(root, mid, radius, color)
        self._draw_bone(mid, end, radius * 0.85, color)

    def _draw_bone(self, p1, p2, radius, color):
        """
        Cilindro orientado de p1 a p2
        """
        dir_vec = p2 - p1
        length = np.linalg.norm(dir_vec)
        if length < 1e-5:
            return

        # base orientation: cylinder along +Z
        # we need rotation from (0,0,1) to dir_vec
        vx, vy, vz = dir_vec / length

        # axis = cross((0,0,1), dir)
        axis = np.cross([0, 0, 1], [vx, vy, vz])
        axis_len = np.linalg.norm(axis)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, vz))))  # between z and dir

        glPushMatrix()
        glTranslatef(p1[0], p1[1], p1[2])
        glColor3f(*color)

        if axis_len > 1e-5:
            glRotatef(angle, axis[0]/axis_len, axis[1]/axis_len, axis[2]/axis_len)

        quad = gluNewQuadric()
        gluCylinder(quad, radius, radius * 0.6, length, 10, 1)
        gluDeleteQuadric(quad)

        # esfera en el extremo
        glPushMatrix()
        glTranslatef(0, 0, length)
        quad2 = gluNewQuadric()
        gluSphere(quad2, radius * 0.7, 10, 8)
        gluDeleteQuadric(quad2)
        glPopMatrix()

        glPopMatrix()

    # ------------------- 2D OVERLAY -------------------
    def _draw_info_overlay(self):
        width, height = self.get_size()
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, width, 0, height)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        text1 = f"Frame: {self.curr_frame+1}/{self.nframes}  FPS:{self.capture.fps:.1f}"
        text2 = f"Moves detectados: {len(self.capture.moves)}"
        text3 = f"Modo: {'AVATAR' if self.mode_avatar else 'SKELETON'} (TAB)"

        lab1 = pyglet.text.Label(text1, x=10, y=height-20, anchor_x='left',
                                 anchor_y='top', font_name='Consolas', font_size=11,
                                 color=(255, 255, 255, 255))
        lab1.draw()
        lab2 = pyglet.text.Label(text2, x=10, y=height-40, anchor_x='left',
                                 anchor_y='top', font_name='Consolas', font_size=11,
                                 color=(210, 220, 255, 255))
        lab2.draw()
        lab3 = pyglet.text.Label(text3, x=10, y=height-60, anchor_x='left',
                                 anchor_y='top', font_name='Consolas', font_size=11,
                                 color=(255, 210, 160, 255))
        lab3.draw()

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

    # ------------------- INPUT -------------------
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & mouse.LEFT:
            self.cam.orbit(dx, -dy)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.cam.dolly(-scroll_y)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.close()
        elif symbol == key.TAB:
            self.mode_avatar = not self.mode_avatar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="ruta CSV de landmarks")
    parser.add_argument("--spec", required=False, default="config/patterns/8yang_spec.json")
    parser.add_argument("--video", required=False, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    video_path = Path(args.video) if args.video else None

    # usamos tu move_capture que ahora redirige al sistema con especificaciones
    capture = mc.capture_moves_from_csv(csv_path, video_path=video_path)

    win = Poomsae3DWindow(capture, csv_path, video_path)
    pyglet.app.run()


if __name__ == "__main__":
    main()
