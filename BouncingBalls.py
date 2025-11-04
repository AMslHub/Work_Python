"""This example spawns (bouncing) balls randomly on a L-shape constructed of
two segment shapes. Not interactive.
"""

__docformat__ = "reStructuredText"

# Python imports
import random

# Library imports
import pygame 

# pymunk imports
import pymunk
import pymunk.pygame_util


# -----------------------------
# Konfigurierbare Parameter
# -----------------------------

# Anzeige / Fenster (zum Bezug für Geometrien)
WIDTH, HEIGHT = 600, 600
WINDOW_SIZE = (WIDTH, HEIGHT)

# Zeit / Integration
DT = 1.0 / 60.0
PHYSICS_STEPS_PER_FRAME = 1
FPS_LIMIT = 50  # Rendering-Deckel (nicht Physik, aber hilfreich)

# Gravitation
GRAVITY = (0.0, 900.0)   #(0.0, 900.0)

# Statische Geometrie (L-Form)
# Einträge: ((x1, y1), (x2, y2), radius)
STATIC_SEGMENTS = [
    ((111.0, HEIGHT - 280), (407.0, HEIGHT - 246), 0.0),
    ((407.0, HEIGHT - 246), (407.0, HEIGHT - 343), 0.0),
]
STATIC_ELASTICITY = 0.95
STATIC_FRICTION = 0.9

# Bälle
BALL_MASS = 10.0
BALL_RADIUS_BASE = 25.0
BALL_SCALE_MIN = 0.5
BALL_SCALE_MAX = 2.0
BALL_START_X_MIN = 115
BALL_START_X_MAX = 350
BALL_START_Y = 80
BALL_ELASTICITY = 0.98
BALL_FRICTION = 0.9

# Lebenszyklus der Bälle
INITIAL_TICKS_TO_NEXT_BALL = 10
TICKS_BETWEEN_BALLS = 100
BALL_REMOVE_Y = 500  # y-Schwelle zum Entfernen

# Nachbarschaftsanziehung (optionale Zusatzphysik)
ATTR_MAX_NEIGHBORS = 0  # 0 = keine Anziehung
ATTR_G = 2.0e6        # Stärke (angepasst an Pixel + Massen ~10)
ATTR_SOFTENING = 100.0  # Pixel; verhindert Singularität bei kleinem r
ATTR_MAX_FORCE = 2.0e3  # Begrenzung der Kraft pro Paar


class BouncyBalls(object):
    """
    This class implements a simple scene in which there is a static platform (made up of a couple of lines)
    that don't move. Balls appear occasionally and drop onto the platform. They bounce around.
    """

    def __init__(self) -> None:
        # Space
        self._space = pymunk.Space()
        self._space.gravity = GRAVITY

        # Physics
        # Time step
        self._dt = DT
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = PHYSICS_STEPS_PER_FRAME

        # pygame
        pygame.init()
        self._screen = pygame.display.set_mode(WINDOW_SIZE)
        self._clock = pygame.time.Clock()

        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        # Balls that exist in the world
        self._balls: list[pymunk.Circle] = []

        # Execution control and time until the next ball spawns
        self._running = True
        self._ticks_to_next_ball = INITIAL_TICKS_TO_NEXT_BALL

    def run(self) -> None:
        """
        The main loop of the game.
        :return: None
        """
        # Main loop
        while self._running:
            self._process_events()
            self._apply_neighbor_attraction()
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)
            self._update_balls()
            self._clear_screen()
            self._draw_objects()
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(FPS_LIMIT)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

    def _add_static_scenery(self) -> None:
        """
        Create the static bodies.
        :return: None
        """
        static_body = self._space.static_body
        static_lines = [
            pymunk.Segment(static_body, seg[0], seg[1], seg[2]) for seg in STATIC_SEGMENTS
        ]
        for line in static_lines:
            line.elasticity = STATIC_ELASTICITY
            line.friction = STATIC_FRICTION
        self._space.add(*static_lines)

    def _process_events(self) -> None:
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(self._screen, "bouncing_balls.png")

    def _update_balls(self) -> None:
        """
        Create/remove balls as necessary. Call once per frame only.
        :return: None
        """
        self._ticks_to_next_ball -= 1
        if self._ticks_to_next_ball <= 0:
            self._create_ball()
            self._ticks_to_next_ball = TICKS_BETWEEN_BALLS
        # Remove balls that pass the configured y-threshold
        balls_to_remove = [ball for ball in self._balls if ball.body.position.y > BALL_REMOVE_Y]
        for ball in balls_to_remove:
            self._space.remove(ball, ball.body)
            self._balls.remove(ball)

    def _create_ball(self) -> None:
        """
        Create a ball.
        :return:
        """
        mass = float(BALL_MASS)
        # Größenvariation: Faktor im Bereich 0.5 .. 2.0
        scale = random.uniform(BALL_SCALE_MIN, BALL_SCALE_MAX)
        radius = int(round(BALL_RADIUS_BASE * scale))
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        x = random.randint(BALL_START_X_MIN, BALL_START_X_MAX)
        # Fallhöhe verdoppeln: höhere Startposition (näher am oberen Bildschirmrand)
        body.position = x, BALL_START_Y
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = BALL_ELASTICITY
        shape.friction = BALL_FRICTION
        self._space.add(body, shape)
        self._balls.append(shape)

    # --- Mutual attraction among nearest neighbors ---

    def _apply_neighbor_attraction(self) -> None:
        # Apply symmetric, softened inverse-square attraction between each ball
        # and up to its 5 nearest neighbors. Momentum-conserving: equal/opposite.
        balls = self._balls
        n = len(balls)
        if n < 2:
            return

        # Build positions list for speed
        positions = [b.body.position for b in balls]

        # Helper to clamp vector magnitude
        def clamp_force(vx: float, vy: float, max_f: float) -> tuple[float, float]:
            mag2 = vx * vx + vy * vy
            if mag2 <= 0:
                return (0.0, 0.0)
            if mag2 <= max_f * max_f:
                return (vx, vy)
            s = max_f / (mag2 ** 0.5)
            return (vx * s, vy * s)

        # For each ball, find indices of up to 5 nearest others
        processed_pairs: set[tuple[int, int]] = set()
        soft2 = ATTR_SOFTENING * ATTR_SOFTENING
        for i in range(n):
            pi = positions[i]
            # Compute squared distances to all others
            dists: list[tuple[float, int]] = []
            for j in range(n):
                if i == j:
                    continue
                pj = positions[j]
                dx = pj.x - pi.x
                dy = pj.y - pi.y
                d2 = dx * dx + dy * dy
                dists.append((d2, j))

            # Take up to k nearest
            dists.sort(key=lambda x: x[0])
            for _, j in dists[: ATTR_MAX_NEIGHBORS]:
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in processed_pairs:
                    continue
                processed_pairs.add((a, b))

                bi = balls[i].body
                bj = balls[j].body
                pi = bi.position
                pj = bj.position

                dx = pj.x - pi.x
                dy = pj.y - pi.y
                r2 = dx * dx + dy * dy
                if r2 <= 0:
                    continue
                rs2 = r2 + soft2
                inv_r = rs2 ** -0.5
                inv_r3 = inv_r * inv_r * inv_r

                # Force magnitude based on softened inverse-square (accel ~ 1/r^2)
                # Convert to force with masses mi,mj
                mi = bi.mass
                mj = bj.mass
                scale = ATTR_G * mi * mj * inv_r3
                fx = dx * scale
                fy = dy * scale
                fx, fy = clamp_force(fx, fy, ATTR_MAX_FORCE)

                # Apply equal and opposite forces at body centers
                bi.apply_force_at_world_point((fx, fy), pi)
                bj.apply_force_at_world_point((-fx, -fy), pj)

    def _clear_screen(self) -> None:
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(pygame.Color("white"))

    def _draw_objects(self) -> None:
        """
        Draw the objects.
        :return: None
        """
        self._space.debug_draw(self._draw_options)


def main():
    game = BouncyBalls()
    game.run()


if __name__ == "__main__":
    main()
