"""
Feder–Masse–Kette mit pymunk/pygame

Anforderungen:
- N=100 Massen (m=0.1 kg), horizontal angeordnet.
- Jede Masse ist mit Vorgänger/Nachfolger über Federn (Ruhelänge l=0.1 m) verbunden.
- Die letzte Feder ist an einer vertikalen Wand (rechter Rand) befestigt.
- Die erste Feder wird kurz nach dem Start schlagartig um 0.5 m nach oben bewegt.
- Gravitation g und Reibung (Dämpfung) sind einstellbar; initial g=0.

Hinweis zu Einheiten: Wir verwenden einen Skalierungsfaktor Meter→Pixel, so dass die
physikalischen Längenangaben in Metern als Pixel auf dem Bildschirm dargestellt werden.
"""

from __future__ import annotations

import math
import pygame
import pymunk
import pymunk.pygame_util


# -----------------------------
# Konfigurierbare Parameter
# -----------------------------

# Grundlegendes
N_MASSES = 150              # Anzahl Massen
MASS_KG = 0.05               # Masse je Körper [kg]

# Federlängen
L0_M = 0.1                  # entspannte Federlänge (Ruhelänge) l0 [m]
LV_M = 0.12                 # Vorspannungslänge lv (Start-Abstand pro Segment) [m]

PIXELS_PER_METER = 50.0     # Skalierung [px/m]

# Federparameter
SPRING_STIFFNESS = 1200.0   # Federsteifigkeit [N/m]
SPRING_DAMPING = 8.0       # Federdämpfung [N·s/m]

# Körpergeometrie (SI)
MASS_RADIUS_M = 0.04       # Radius der Massen [m]

# Gravitation und Reibung (Dämpfung)
GRAVITY = (0.0, 0.0)      # initial g=0; SI: (0, -9.81) für ~1g nach unten
SPACE_DAMPING = 1.0       # =1.0: keine Dämpfung, <1.0: Dämpfung
                          # space.damping ist ein Multiplikationsfaktor 
                          # für die Geschwindigkeit pro Integrationsschritt

# Impuls der ersten Feder (linkes Ende)
IMPULSE_TIME_S = 0.25       # Zeitpunkt nach Start [s]
ANCHOR_LIFT_M = 0.5         # Sprung nach oben [m]

# Fenster / Darstellung
WIDTH, HEIGHT = 1200, 600
FPS_LIMIT = 60
MASS_RADIUS_PX = 2          # (veraltet) Darstellungsradius [px] – wird dynamisch aus MASS_RADIUS_M berechnet
LINE_COLOR = (30, 144, 255) # Farbe der Federlinien (DodgerBlue)
MASS_COLOR = (10, 10, 10)
BG_COLOR = (245, 245, 245)
WALL_COLOR = (180, 180, 180)


def m_to_px(x_m: float) -> float:
    return x_m * PIXELS_PER_METER

# Welt→Bildschirm Abbildung (nur für Darstellung)
OFFSET_LEFT_PX = 100.0
BASELINE_PX = HEIGHT * 0.5

def world_to_screen(p: pymunk.Vec2d | tuple[float, float]) -> tuple[float, float]:
    x, y = (p.x, p.y) if isinstance(p, pymunk.Vec2d) else p
    return (OFFSET_LEFT_PX + x * PIXELS_PER_METER, BASELINE_PX - y * PIXELS_PER_METER)


class MassSpringChain:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.font = pygame.font.SysFont(None, 18)

        # Physikraum
        self.space = pymunk.Space()

        # Laufzeit-Parameter (veränderbar per Tastatur)
        self.gravity_presets = [
            (0.0, 0.0),
            (0.0, -9.81),
        ]
        self.gravity_index = 0
        self.space.gravity = self.gravity_presets[self.gravity_index]

        self.damping = SPACE_DAMPING
        self.space.damping = self.damping

        self.impulse_time_s = IMPULSE_TIME_S
        self.anchor_lift_m = ANCHOR_LIFT_M

        # Darstellungsradius aus SI-Radius ableiten
        self.mass_radius_px = max(1, int(round(MASS_RADIUS_M * PIXELS_PER_METER)))

        # Zeitschritt
        self.dt = 1.0 / 120.0
        self.physics_steps_per_frame = 2

        # Aufbau der Kette
        self.bodies: list[pymunk.Body] = []
        self.shapes: list[pymunk.Shape] = []
        self.springs: list[pymunk.DampedSpring] = []

        self._build_chain()

        # Steuerung des Federimpulses
        self.running = True
        self.time_accum = 0.0
        self.impulse_applied = False

    def _build_chain(self) -> None:
        # Geometrie der Kette (SI)
        rest_m = L0_M
        span_m = LV_M
        start_x = 100.0  # nur Darstellung (linke Einrückung in px)
        baseline_y = HEIGHT * 0.5  # nur Darstellung (y=0 in px)

        # Linker Anker (Kinematischer Körper, per Sprung bewegbar)
        self.left_anchor = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.left_anchor_base = (0.0, 0.0)
        self.left_anchor.position = self.left_anchor_base

        # Rechter Anker (Wand) als statischer Körper an rechter Seite
        wall_x_m = (N_MASSES + 1) * span_m
        self.right_anchor = self.space.static_body
        self.right_anchor_pos = (wall_x_m, 0.0)

        # Optionale Darstellung der Wand (als dünnes Segment in SI)
        wall_half_h_m = max(0.1, (HEIGHT - 80) / PIXELS_PER_METER * 0.5)
        self.wall_shape = pymunk.Segment(self.right_anchor, (wall_x_m, -wall_half_h_m), (wall_x_m, wall_half_h_m), 0.01)
        self.wall_shape.color = (*WALL_COLOR, 255)
        self.wall_shape.elasticity = 0.0
        self.wall_shape.friction = 0.0
        self.space.add(self.wall_shape)

        # Massen erzeugen (Startpositionen mit Vorspannungsabstand, SI)
        for i in range(N_MASSES):
            x_m = (i + 1) * span_m
            y_m = 0.0
            moment = pymunk.moment_for_circle(MASS_KG, 0, MASS_RADIUS_M)
            body = pymunk.Body(MASS_KG, moment)
            body.position = (x_m, y_m)
            shape = pymunk.Circle(body, MASS_RADIUS_M)
            shape.color = (*MASS_COLOR, 255)
            shape.elasticity = 0.0
            shape.friction = 0.0
            self.bodies.append(body)
            self.shapes.append(shape)

        self.space.add(*self.bodies, *self.shapes)

        # Federn anbringen:
        # 1) Erste Feder: linker Anker ↔ erste Masse
        s0 = pymunk.DampedSpring(
            self.left_anchor,
            self.bodies[0],
            (0, 0),
            (0, 0),
            rest_m,
            SPRING_STIFFNESS,
            SPRING_DAMPING,
        )
        self.springs.append(s0)

        # 2) Zwischen benachbarten Massen
        for i in range(1, N_MASSES):
            s = pymunk.DampedSpring(
                self.bodies[i - 1],
                self.bodies[i],
                (0, 0),
                (0, 0),
                rest_m,
                SPRING_STIFFNESS,
                SPRING_DAMPING,
            )
            self.springs.append(s)

        # 3) Letzte Feder: letzte Masse ↔ rechte Wand (statischer Körper)
        s_last = pymunk.DampedSpring(
            self.bodies[-1],
            self.right_anchor,
            (0, 0),
            self.right_anchor_pos,
            rest_m,
            SPRING_STIFFNESS,
            SPRING_DAMPING,
        )
        self.springs.append(s_last)

        self.space.add(*self.springs)

    def _apply_first_spring_impulse(self) -> None:
        # Einmaliger Sprung des linken Ankers nach oben (SI)
        x, y = self.left_anchor.position
        self.left_anchor.position = (x, y + self.anchor_lift_m)
        self.impulse_applied = True

    def _process_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_g:
                    # Toggle Gravitation zwischen 0 und ~1g
                    self.gravity_index = (self.gravity_index + 1) % len(self.gravity_presets)
                    self.space.gravity = self.gravity_presets[self.gravity_index]
                elif k == pygame.K_1:
                    # Dämpfung verringern
                    self.damping = max(0.0, self.damping - 0.01)
                    self.space.damping = self.damping
                elif k == pygame.K_2:
                    # Dämpfung erhöhen
                    self.damping = min(0.999, self.damping + 0.01)
                    self.space.damping = self.damping
                elif k == pygame.K_3:
                    # Impuls-Höhe verringern
                    self.anchor_lift_m = max(0.0, self.anchor_lift_m - 0.05)
                elif k == pygame.K_4:
                    # Impuls-Höhe erhöhen
                    self.anchor_lift_m = min(2.0, self.anchor_lift_m + 0.05)
                elif k == pygame.K_5:
                    # Impulszeit früher
                    self.impulse_time_s = max(0.0, self.impulse_time_s - 0.05)
                elif k == pygame.K_6:
                    # Impulszeit später
                    self.impulse_time_s = min(5.0, self.impulse_time_s + 0.05)
                elif k == pygame.K_i:
                    # Impuls sofort auslösen, falls noch nicht erfolgt
                    if not self.impulse_applied:
                        self._apply_first_spring_impulse()
                elif k == pygame.K_r:
                    # Simulation zurücksetzen
                    self._reset_sim()

    def _reset_sim(self) -> None:
        # Raum neu erstellen (einfachste saubere Variante)
        self.space = pymunk.Space()
        self.space.gravity = self.gravity_presets[self.gravity_index]
        self.space.damping = self.damping
        self.bodies = []
        self.shapes = []
        self.springs = []
        self._build_chain()
        self.time_accum = 0.0
        self.impulse_applied = False

    def _draw(self) -> None:
        self.screen.fill(BG_COLOR)

        # Federn als Linien zeichnen (inkl. Enden)
        # Linker Anker → erste Masse
        p0 = world_to_screen(self.left_anchor.position)
        p1 = world_to_screen(self.bodies[0].position)
        pygame.draw.line(self.screen, LINE_COLOR, p0, p1, 1)

        # Zwischen Massen
        for i in range(1, N_MASSES):
            a = world_to_screen(self.bodies[i - 1].position)
            b = world_to_screen(self.bodies[i].position)
            pygame.draw.line(self.screen, LINE_COLOR, a, b, 1)

        # Letzte Masse → rechte Wand
        last = world_to_screen(self.bodies[-1].position)
        wall = world_to_screen(self.right_anchor_pos)
        pygame.draw.line(self.screen, LINE_COLOR, last, wall, 1)

        # Massen als kleine Kreise (Darstellung)
        for body in self.bodies:
            sx, sy = world_to_screen(body.position)
            pygame.draw.circle(self.screen, MASS_COLOR, (int(sx), int(sy)), self.mass_radius_px)

        # Optionale Debug-Zeichnung der Shapes (kleine Kreise) und Wand
        # self.space.debug_draw(self.draw_options)

        # HUD
        lines = [
            f"g: {self.space.gravity[1]:0.2f} m/s^2  (g toggeln: G)",
            f"damping: {self.space.damping:0.3f}  (1/2 -/+)",
            f"impulse_time: {self.impulse_time_s:0.2f}s  (5/6 -/+)",
            f"impulse_lift: {self.anchor_lift_m:0.2f} m  (3/4 -/+)",
            "i: Impuls jetzt | r: Reset | Esc: Quit",
        ]
        y = 8
        for txt in lines:
            surf = self.font.render(txt, True, (0, 0, 0))
            self.screen.blit(surf, (8, y))
            y += 18

        pygame.display.flip()

    def run(self) -> None:
        while self.running:
            self._process_events()

            # Impuls der ersten Feder nach IMPULSE_TIME_S einmalig ausführen
            if not self.impulse_applied:
                self.time_accum += self.dt * self.physics_steps_per_frame
                if self.time_accum >= self.impulse_time_s:
                    self._apply_first_spring_impulse()

            # Physik-Integration
            for _ in range(self.physics_steps_per_frame):
                self.space.step(self.dt)

            self._draw()
            self.clock.tick(FPS_LIMIT)
            pygame.display.set_caption(f"fps: {self.clock.get_fps():.1f}")


def main() -> None:
    app = MassSpringChain()
    app.run()


if __name__ == "__main__":
    main()
