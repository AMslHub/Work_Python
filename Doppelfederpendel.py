"""
Doppelfederpendel (zwei Massen, zwei Federn) mit pymunk + pygame.

Anforderungen:
- Massen: m1 = m2 = 1 kg
- Federkonstante: c = 200 N/m (beide Federn)
- Erdbeschleunigung: g = 9.81 m/s^2
- Visualisierung und Videoausgabe (5s)

Hinweis zur Skalierung der Einheiten:
- Positionen werden in Pixeln simuliert, mit PPM = 100 Pixel pro Meter.
- Schwerkraft wird als g * PPM gesetzt (Pixel/s^2).
- Federsteifigkeit (N/m) wird in N/Pixel umgerechnet: k_px = k_m / PPM.
- Dämpfung wird klein gewählt für Stabilität (ebenfalls pro Pixel skaliert).

Videoausgabe:
- Versucht, mit imageio ein MP4 zu schreiben (output.mp4).
- Falls imageio/ffmpeg nicht verfügbar ist, werden einzelne PNG-Frames
  nach ./frames gespeichert (als Fallback) und ein Hinweis ausgegeben.

Abhängigkeiten:
  pip install pygame pymunk imageio
  (Für MP4-Export ggf. ffmpeg im PATH, oder imageio-ffmpeg installiert.)
"""

from __future__ import annotations

import os
import math
import time
from pathlib import Path

import pygame
import pymunk
from pymunk import Vec2d

try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None  # Fallback auf PNG-Frames


# ---------- Simulationseinstellungen ----------
WIDTH, HEIGHT = 800, 600
PPM = 100.0  # Pixel pro Meter
FPS = 60
DURATION_S = 10.0 
NUM_FRAMES = int(DURATION_S * FPS)

GRAVITY_M_S2 = 9.81
GRAVITY = (0.0, GRAVITY_M_S2 * PPM)  # Pixel/s^2

# Feder- und Massenparameter
MASS_1 = 2.0  # kg
MASS_2 = 1.0  # kg
K_N_PER_M = 4000.0  # N/m
K_N_PER_PIXEL = K_N_PER_M / PPM  # N/Pixel

# Leichte Dämpfung für numerische Stabilität (N*s/m -> N*s/pixel)
DAMPING_Ns_PER_M = 1.0
DAMPING_Ns_PER_PIXEL = DAMPING_Ns_PER_M / PPM

# Ruhe-Längen der Federn (in Metern)
L1_M = 1.0
L2_M = 1.0
L1_PX = L1_M * PPM
L2_PX = L2_M * PPM

# Visualisierung
BG_COLOR = (15, 18, 25)
SPRING_COLOR = (200, 200, 220)
MASS1_COLOR = (80, 180, 255)
MASS2_COLOR = (255, 140, 80)
CEIL_COLOR = (220, 220, 220)

MASS_RADIUS = 12  # px

# Zeichnungsradius abhängig von der Masse (Fläche ~ Masse ⇒ r ~ sqrt(m))
def mass_to_draw_radius(mass: float) -> int:
    base = MASS_RADIUS  # Basisradius für 1 kg
    if mass <= 0:
        return max(2, int(base))
    r = base * math.sqrt(mass / 1.0)
    return max(2, int(round(r)))

# Linke Wand: 0.2 m links vom Aufhängepunkt, Darstellung und Physik-Parameter
WALL_COLOR = (180, 220, 180)
WALL_OFFSET_M = 0.25
WALL_ELASTICITY = 0.98
WALL_FRICTION = 0.4


def setup_space() -> tuple[pymunk.Space, pymunk.Body, pymunk.Body, Vec2d]:
    space = pymunk.Space()
    space.gravity = GRAVITY

    top_anchor = Vec2d(WIDTH // 2, 100)  # "Decke"-Aufhängepunkt

    # Körper 1 (Masse 1 kg)
    moment1 = pymunk.moment_for_circle(MASS_1, 0, MASS_RADIUS)
    body1 = pymunk.Body(MASS_1, moment1)

    # Körper 2 (Masse 1 kg)
    moment2 = pymunk.moment_for_circle(MASS_2, 0, MASS_RADIUS)
    body2 = pymunk.Body(MASS_2, moment2)

    # Anfangsbedingungen: leichte horizontale Auslenkung, damit Bewegung sichtbar ist
    # Gleichgewichtslage ca. top_anchor.y + L1_PX + L2_PX
    body1.position = top_anchor + (120, L1_PX + 10)  # leicht gedehnt und seitlich versetzt
    body2.position = body1.position + (80, L2_PX + 10)

    shape1 = pymunk.Circle(body1, MASS_RADIUS)
    shape1.friction = 0.4
    shape1.elasticity = 0.9
    shape2 = pymunk.Circle(body2, MASS_RADIUS)
    shape2.friction = 0.4
    shape2.elasticity = 0.9

    space.add(body1, shape1, body2, shape2)

    # Linke, vertikale Wand als statisches Segment bei x = top_anchor.x - 0.2 m
    wall_x = top_anchor.x - WALL_OFFSET_M * PPM
    wall = pymunk.Segment(space.static_body, (wall_x, 0), (wall_x, HEIGHT), 2.0)
    wall.friction = WALL_FRICTION
    wall.elasticity = WALL_ELASTICITY
    space.add(wall)

    # Federn (DampedSpring):
    # Static body hat Weltkoordinaten als lokale Koordinaten (Transform = I), daher anchor_a = top_anchor
    spring1 = pymunk.DampedSpring(
        space.static_body,
        body1,
        top_anchor,  # local zu static_body -> gleich Weltkoordinate
        (0, 0),
        L1_PX,
        K_N_PER_PIXEL,
        DAMPING_Ns_PER_PIXEL,
    )

    spring2 = pymunk.DampedSpring(
        body1,
        body2,
        (0, 0),
        (0, 0),
        L2_PX,
        K_N_PER_PIXEL,
        DAMPING_Ns_PER_PIXEL,
    )

    space.add(spring1, spring2)

    return space, body1, body2, top_anchor


def draw_scene(surface: pygame.Surface, body1: pymunk.Body, body2: pymunk.Body, top_anchor: Vec2d) -> None:
    surface.fill(BG_COLOR)

    p1 = body1.position
    p2 = body2.position

    # Linke Wand zeichnen
    wx = int(top_anchor.x - WALL_OFFSET_M * PPM)
    pygame.draw.line(surface, WALL_COLOR, (wx, 0), (wx, HEIGHT), 3)

    # Deckenpunkt markieren
    pygame.draw.circle(surface, CEIL_COLOR, (int(top_anchor.x), int(top_anchor.y)), 4)

    # Federn als Linien zeichnen
    pygame.draw.line(surface, SPRING_COLOR, (int(top_anchor.x), int(top_anchor.y)), (int(p1.x), int(p1.y)), 2)
    pygame.draw.line(surface, SPRING_COLOR, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 2)

    # Massen (Zeichnungsradius skaliert mit Masse)
    r1 = mass_to_draw_radius(MASS_1)
    r2 = mass_to_draw_radius(MASS_2)
    pygame.draw.circle(surface, MASS1_COLOR, (int(p1.x), int(p1.y)), r1)
    pygame.draw.circle(surface, MASS2_COLOR, (int(p2.x), int(p2.y)), r2)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    pygame.init()
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
    except Exception:
        # Fallback headless Surface, wenn kein Display (z.B. CI)
        screen = pygame.Surface((WIDTH, HEIGHT))

    clock = pygame.time.Clock()
    space, body1, body2, top_anchor = setup_space()

    # Videoausgabe vorbereiten
    output_path = Path("output.mp4")
    writer = None
    frame_dir = Path("frames")
    use_video = False

    if imageio is not None:
        try:
            writer = imageio.get_writer(str(output_path), fps=FPS, codec="libx264", quality=7)
            use_video = True
        except Exception:
            writer = None
            use_video = False

    if not use_video:
        ensure_dir(frame_dir)
        print("imageio/ffmpeg nicht verfügbar – speichere PNG-Frames nach ./frames")

    # Feste Zeitschritt-Integration mit Substeps für Stabilität
    dt = 1.0 / FPS
    substeps = 4
    sub_dt = dt / substeps

    for frame in range(NUM_FRAMES):
        # Event-Pump (Fenster reaktiv halten)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        # Physik-Update
        for _ in range(substeps):
            space.step(sub_dt)

        # Rendern
        draw_scene(screen, body1, body2, top_anchor)

        # Frame erfassen
        # Pygame-Flächen sind (W,H); surfarray.array3d liefert Array (W,H,3) -> transponieren auf (H,W,3)
        try:
            import numpy as np  # lazy import

            frame_rgb = pygame.surfarray.array3d(screen)
            frame_rgb = np.transpose(frame_rgb, (1, 0, 2))  # (H, W, 3)
        except Exception:
            frame_rgb = None

        if use_video and writer is not None and frame_rgb is not None:
            writer.append_data(frame_rgb)
        else:
            # Fallback: PNG-Frame sichern
            frame_file = frame_dir / f"frame_{frame:05d}.png"
            pygame.image.save(screen, str(frame_file))

        # Display aktualisieren (falls sichtbar)
        try:
            pygame.display.flip()
        except Exception:
            pass

        clock.tick(FPS)

    # Writer schließen
    if writer is not None:
        writer.close()
        print(f"Video gespeichert: {output_path}")
    else:
        print("PNG-Frames gespeichert. Erstelle ein Video z.B. mit:")
        print("ffmpeg -r 60 -i frames/frame_%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4")

    pygame.quit()


if __name__ == "__main__":
    main()
