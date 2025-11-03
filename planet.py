"""
Showcase of planets/satellites (small boxes) orbiting around a star. 

Uses a custom velocity function to manually calculate the gravity, assuming 
the star is in the middle and is massive enough that the satellites does not 
affect it.

This is also a demonstration of the performance boost from the batch api in 
pymunk.batch. It uses both the batch api to get positions and velocitites, 
and to update the velocitites.

The speedup of when both batching apis are enabled is huge!

(This is a modified port of the Planet demo included in Chipmunk.)
"""

import math 
import random

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    np = None  # type: ignore
    HAS_NUMPY = False
import pygame

import pymunk
import pymunk.batch
import pymunk.pygame_util

random.seed(1)

gravityStrength = 5.0e6 #5.0e6
planet_radius = 3
center = pymunk.Vec2d(450, 450)
screen_size = (900, 900)
starting_planets = 400

# Fraction of planets initialized with elliptical (non-circular) orbits
ELLIPTICAL_FRACTION = 0.01
# Speed factor range for elliptical orbits (fraction of circular speed)
# Values < 1 give bound elliptical orbits when velocity is tangential
ELLIPTICAL_SPEED_RANGE = (0.6, 0.95)

dt = 1 / 60.0

# Collision tagging for planets
PLANET_COLLISION_TYPE = 1

# Near-neighbor mutual gravity (simple approximation)
MUTUAL_NEIGHBOR_GRAVITY = True
NEIGHBOR_RADIUS = 100.0  # pixels (legacy; not used when NEIGHBOR_K is set)
NEIGHBOR_G = gravityStrength * 0.001  # relative to central field 0.02
NEIGHBOR_SOFTENING = 25.0  # pixels, avoids singularities
NEIGHBOR_K = 15  # apply attraction with up to K nearest neighbors always


def planet_gravity(body, gravity, damping, dt):
    # Gravitational acceleration is proportional to the inverse square of
    # distance, and directed toward the origin. The central planet is assumed
    # to be massive enough that it affects the satellites but not vice versa.
    p = body.position
    sq_dist = p.get_distance_squared(center)
    g = (p - center) * -gravityStrength / (sq_dist * math.sqrt(sq_dist))

    # body.velocity += g * dt # setting velocity directly like would be slower
    pymunk.Body.update_velocity(body, g, damping, dt)


def batched_planet_gravity(draw_buffer, dt, update_buffer):
    if not HAS_NUMPY:
        return
    # get current position and velocity
    arr = np.frombuffer(draw_buffer.float_buf())
    # pick every 4th item to position.x etc.
    p_x = arr[::4]
    p_y = arr[1::4]
    v_x = arr[2::4]
    v_y = arr[3::4]

    sq_dist = (p_x - center.x) ** 2 + (p_y - center.y) ** 2

    scaled_dist_sq_dist = -gravityStrength / (sq_dist * np.sqrt(sq_dist))
    g_x = (p_x - center.x) * scaled_dist_sq_dist
    g_y = (p_y - center.y) * scaled_dist_sq_dist
    # at this point we have calculated 'g' as in planet_graivity(...)

    # This is the simpliced update_velocity function from planet_gravity(...)
    # (since space.gravity == 0 and space.damping == 1)
    new_v_x = v_x + g_x * dt
    new_v_y = v_y + g_y * dt

    # make resulting array by altering x and y values for the velocity
    v_arr = np.ravel([new_v_x, new_v_y], "F")
    update_buffer.set_float_buf(v_arr.tobytes())


def add_planet(space):
    body = pymunk.Body()
    while True:
        # Loop to filter out planets too close to the center star
        body.position = pymunk.Vec2d(
            # random.randint(-150, 750), random.randint(-150, 750)
            random.randint(0, 900), random.randint(0, 900)
        )
        r = body.position.get_distance(center)
        if r > 40:
            break

    body.velocity_func = planet_gravity

    # Set the planets's velocity to put it into a circular orbit from its
    # starting position.
    v_circ = math.sqrt(gravityStrength / r) / r

    # With probability ELLIPTICAL_FRACTION, reduce speed to get an ellipse.
    # Purely tangential speed < circular creates an ellipse with current
    # position near apocenter; keep it comfortably below escape.
    if random.random() < ELLIPTICAL_FRACTION:
        speed_factor = random.uniform(*ELLIPTICAL_SPEED_RANGE)
        v = v_circ * speed_factor
    else:
        v = v_circ

    body.velocity = (body.position - center).perpendicular() * v
    # Set the planets's angular velocity to match its orbital period and
    # align its initial angle with its position.
    body.angular_velocity = v
    body.angle = math.atan2(body.position.y, body.position.x)

    circle = pymunk.Circle(body, planet_radius)
    circle.mass = 1
    circle.friction = 0.7
    circle.elasticity = 0
    circle.collision_type = PLANET_COLLISION_TYPE
    space.add(body, circle)


## runtime moved into main()


def _merge_pair(space: pymunk.Space, s1: pymunk.Shape, s2: pymunk.Shape) -> None:
    # Guard if already removed
    if s1.space is None or s2.space is None:
        return
    b1, b2 = s1.body, s2.body
    # Only handle circles
    if not isinstance(s1, pymunk.Circle) or not isinstance(s2, pymunk.Circle):
        return

    m1 = getattr(s1, "mass", 1.0)
    m2 = getattr(s2, "mass", 1.0)
    M = m1 + m2
    if M <= 0:
        return

    # Conserve linear momentum
    pos = (b1.position * m1 + b2.position * m2) / M
    vel = (b1.velocity * m1 + b2.velocity * m2) / M

    # Constant density in 3D: volumes add -> r_new^3 = r1^3 + r2^3
    r_new = (s1.radius ** 3 + s2.radius ** 3) ** (1.0 / 3.0)

    moment = pymunk.moment_for_circle(M, 0, r_new)
    nb = pymunk.Body(M, moment)
    nb.position = pos
    nb.velocity = vel
    # Keep velocity update mode consistent with existing planets
    if b1.velocity_func is pymunk.Body.update_velocity and b2.velocity_func is pymunk.Body.update_velocity:
        nb.velocity_func = pymunk.Body.update_velocity
    else:
        nb.velocity_func = planet_gravity

    ns = pymunk.Circle(nb, r_new)
    ns.mass = M
    ns.friction = max(getattr(s1, "friction", 0.7), getattr(s2, "friction", 0.7))
    ns.elasticity = 0
    ns.collision_type = PLANET_COLLISION_TYPE

    # Replace the two planets with the merged one
    space.add(nb, ns)
    space.remove(s1, b1, s2, b2)


def _merge_overlaps_once(space: pymunk.Space) -> bool:
    # Find and merge first overlapping pair of circle shapes
    shapes = [s for s in space.shapes if isinstance(s, pymunk.Circle)]
    n = len(shapes)
    for i in range(n):
        s1 = shapes[i]
        if s1.space is None:
            continue
        p1 = s1.body.position
        r1 = s1.radius
        for j in range(i + 1, n):
            s2 = shapes[j]
            if s2.space is None:
                continue
            # Quick AABB check
            rsum = r1 + s2.radius
            if abs(p1.x - s2.body.position.x) > rsum or abs(p1.y - s2.body.position.y) > rsum:
                continue
            # Precise circle overlap check
            if p1.get_distance(s2.body.position) <= rsum:
                _merge_pair(space, s1, s2)
                return True
    return False


def _apply_neighbor_gravity(space: pymunk.Space, dt: float) -> None:
    if not MUTUAL_NEIGHBOR_GRAVITY:
        return
    circles = [s for s in space.shapes if isinstance(s, pymunk.Circle)]
    n = len(circles)
    if n < 2:
        return
    # Pre-extract positions and masses
    pos = [c.body.position for c in circles]
    mass = [getattr(c, "mass", 1.0) for c in circles]

    # Build set of unique neighbor pairs based on K nearest neighbors per planet
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        # distances to others
        dists = []
        pi = pos[i]
        for j in range(n):
            if i == j:
                continue
            pj = pos[j]
            dx = pj.x - pi.x
            dy = pj.y - pi.y
            d2 = dx * dx + dy * dy
            dists.append((d2, j))
        # pick up to K nearest
        dists.sort(key=lambda t: t[0])
        for _, j in dists[: max(0, min(NEIGHBOR_K, n - 1))]:
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))

    # Accumulate symmetric accelerations for all chosen pairs
    ax = [0.0] * n
    ay = [0.0] * n
    soft2 = NEIGHBOR_SOFTENING * NEIGHBOR_SOFTENING
    for (i, j) in pairs:
        dx = pos[j].x - pos[i].x
        dy = pos[j].y - pos[i].y
        r2 = dx * dx + dy * dy
        if r2 <= 0:
            continue
        rs2 = r2 + soft2
        inv_r3 = 1.0 / (rs2 * math.sqrt(rs2))
        s_ij = NEIGHBOR_G * mass[j] * inv_r3
        s_ji = NEIGHBOR_G * mass[i] * inv_r3
        ax[i] += dx * s_ij
        ay[i] += dy * s_ij
        ax[j] -= dx * s_ji
        ay[j] -= dy * s_ji

    # Apply accumulated accelerations
    for i in range(n):
        if ax[i] != 0.0 or ay[i] != 0.0:
            a = pymunk.Vec2d(ax[i], ay[i])
            pymunk.Body.update_velocity(circles[i].body, a, 1.0, dt)


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode(screen_size)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 20)

    space = pymunk.Space()

    for x in range(starting_planets):
        add_planet(space)

    use_batch_draw = False
    use_batch_update = False
    draw_buffer = pymunk.batch.Buffer()
    update_buffer = pymunk.batch.Buffer()
    planet_color = pygame.Color("white")

    print("Planet demo starting. Press A to add, D toggle batch draw, U toggle batch update, ESC to quit.")

    while True:
        for event in pygame.event.get():
            if (
                event.type == pygame.QUIT
                or event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                for x in range(100):
                    add_planet(space)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(screen, "planet_batch.png")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                if HAS_NUMPY:
                    use_batch_draw = not use_batch_draw
                else:
                    use_batch_draw = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_u:
                if HAS_NUMPY:
                    use_batch_update = not use_batch_update
                    if use_batch_update:
                        for b in space.bodies:
                            b.velocity_func = pymunk.Body.update_velocity
                    else:
                        for b in space.bodies:
                            b.velocity_func = planet_gravity
                else:
                    use_batch_update = False

        screen.fill(pygame.Color("black"))

        if HAS_NUMPY and (use_batch_draw or use_batch_update):
            # Reuse the position / velocity buffer for both drawing and calculating velocity
            draw_buffer.clear()
            pymunk.batch.get_space_bodies(
                space,
                pymunk.batch.BodyFields.POSITION | pymunk.batch.BodyFields.VELOCITY,
                draw_buffer,
            )

        # If all planets have the same radius, batch draw is fine; otherwise draw per-shape to use actual radii
        variable_radii = any(
            isinstance(s, pymunk.Circle) and abs(s.radius - planet_radius) > 1e-9 for s in space.shapes
        )

        if HAS_NUMPY and use_batch_draw and not variable_radii:
            ps = list(memoryview(draw_buffer.float_buf()).cast("d"))
            for idx in range(0, len(ps), 4):
                cx = int(ps[idx])
                cy = int(ps[idx + 1])
                pygame.draw.circle(screen, planet_color, (cx, cy), int(planet_radius))
        else:
            for s in space.shapes:
                if isinstance(s, pymunk.Circle):
                    cx = int(s.body.position.x)
                    cy = int(s.body.position.y)
                    rr = max(1, int(round(s.radius)))
                    pygame.draw.circle(screen, planet_color, (cx, cy), rr)

        # 'Star' in the center of screen
        pygame.draw.circle(screen, pygame.Color("yellow"), center, 10)

        if HAS_NUMPY and use_batch_update:
            batched_planet_gravity(draw_buffer, dt, update_buffer)
            pymunk.batch.set_space_bodies(
                space, pymunk.batch.BodyFields.VELOCITY, update_buffer
            )

        # Apply simple near-neighbor mutual gravity before stepping
        _apply_neighbor_gravity(space, dt)

        space.step(dt)

        # Merge overlapping planets (perform a few per frame for stability)
        for _ in range(5):
            if not _merge_overlaps_once(space):
                break

        help = "Press a to add planets, d to toggle batched drawing and u to toggle batched updates."
        draw_mode = "batch" if use_batch_draw else "loop"
        update_mode = "batch" if use_batch_update else "callback"
        status = (
            f"Planets: {len(space.bodies)}. Draw mode: {draw_mode}. Update: {update_mode}"
        )

        screen.blit(font.render(status, True, pygame.Color("orange")), (5, 25))
        screen.blit(font.render(help, True, pygame.Color("orange")), (5, 5))

        pygame.display.flip()
        clock.tick(1 / dt)
        pygame.display.set_caption(f"fps: {clock.get_fps():.2f} {status}")


if __name__ == "__main__":
    main()

