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


class BouncyBalls(object):
    """
    This class implements a simple scene in which there is a static platform (made up of a couple of lines)
    that don't move. Balls appear occasionally and drop onto the platform. They bounce around.
    """

    def __init__(self) -> None:
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 900.0)

        # Physics
        # Time step
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        pygame.init()
        self._screen = pygame.display.set_mode((600, 600))
        self._clock = pygame.time.Clock()

        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        # Balls that exist in the world
        self._balls: list[pymunk.Circle] = []

        # Execution control and time until the next ball spawns
        self._running = True
        self._ticks_to_next_ball = 10

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
            self._clock.tick(50)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

    def _add_static_scenery(self) -> None:
        """
        Create the static bodies.
        :return: None
        """
        static_body = self._space.static_body
        static_lines = [
            pymunk.Segment(static_body, (111.0, 600 - 280), (407.0, 600 - 246), 0.0),
            pymunk.Segment(static_body, (407.0, 600 - 246), (407.0, 600 - 343), 0.0),
        ]
        for line in static_lines:
            line.elasticity = 0.95
            line.friction = 0.9
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
            self._ticks_to_next_ball = 100
        # Remove balls that fall below 100 vertically
        balls_to_remove = [ball for ball in self._balls if ball.body.position.y > 500]
        for ball in balls_to_remove:
            self._space.remove(ball, ball.body)
            self._balls.remove(ball)

    def _create_ball(self) -> None:
        """
        Create a ball.
        :return:
        """
        mass = 10
        # Größenvariation: Faktor im Bereich 0.5 .. 2.0
        scale = random.uniform(0.5, 2.0)
        radius = int(round(25 * scale))
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        x = random.randint(115, 350)
        # Fallhöhe verdoppeln: höhere Startposition (näher am oberen Bildschirmrand)
        body.position = x, 80
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0.95
        shape.friction = 0.9
        self._space.add(body, shape)
        self._balls.append(shape)

    # --- Mutual attraction among nearest neighbors ---
    _ATTR_MAX_NEIGHBORS = 5
    _ATTR_G = 2.0e6  # strength factor; tuned for pixels + masses ~10
    _ATTR_SOFTENING = 100.0  # pixels; avoids singularity at small r
    _ATTR_MAX_FORCE = 2.0e3  # clamp force magnitude per pair (optional safety)

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
        soft2 = self._ATTR_SOFTENING * self._ATTR_SOFTENING
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
            for _, j in dists[: self._ATTR_MAX_NEIGHBORS]:
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
                scale = self._ATTR_G * mi * mj * inv_r3
                fx = dx * scale
                fy = dy * scale
                fx, fy = clamp_force(fx, fy, self._ATTR_MAX_FORCE)

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
