# ast_ai.py
# Autopilot for Asteroids. Given the tracked objects (ast_server.Tracker), the
# ship's heading and velocity from RAM, decides which controls to press.
#
# Each frame:
#   1. Threat check: will the ship, drifting as it is, hit anything within HORIZON?
#   2. If a hit is coming within DANGER_T, simulate turning to each of
#      EVADE_HEADINGS and thrusting; thrust along the safest one.
#   3. Otherwise (and opportunistically while evading) aim at the highest
#      priority target - saucers, then small, medium, large rocks - and fire.
#
# Units: game units (x 0-8191, y 0-6143, y up) and frames. Heading 0-255,
# 0 = +x, counterclockwise. All constants were measured from RAM in MAME.

import math
from collections import Counter

import numpy as np

W, H = 8192, 6144

# Ship and bullet physics
TURN_RATE = 3            # heading units per frame while Left/Right is held
THRUST_ACCEL = 0.48      # units/frame^2 along heading
DRAG = 0.9962            # velocity multiplier per frame
MAX_SPEED = 40.0         # assumed cap, units/frame
BULLET_SPEED = 63.6      # units/frame along heading, added to ship velocity
BULLET_LIFE = 72         # frames; the bullet's status counts down 18..1, one step per 4 frames
MAX_BULLETS = 4
FIRE_DELAY = 2           # frames from the fire decision until the bullet appears in RAM
# Inputs also take 2 frames to show up in RAM: a turn decided at frame n changes the
# heading read at n+2. So at frame n the heading read is missing the turn decided at
# n-1, and the bullet leaves along that newer heading (3 units = 4 deg off otherwise).
MUZZLE = 159             # the bullet spawns this far ahead of the ship along the heading

# Collision radii
RADIUS = {"L": 260, "M": 135, "S": 70, "U": 180, "u": 90, "R": 20}
SHIP_RADIUS = 80
MARGIN = 60              # extra clearance wanted around the ship

# Decision tuning
HORIZON = 150            # frames to look ahead for collisions
DANGER_T = 90            # evade when a collision is predicted sooner than this (the ship
                         # accelerates slowly: clearing a rock takes ~60 frames)
HYPER_T = 6              # hyperspace when every option collides sooner than this
EVADE_HEADINGS = 16      # candidate escape directions
EVADE_THRUST = 25        # frames of thrust simulated for each escape candidate
THRUST_ALIGN = 12        # thrust only when within this many heading units
PRIORITY = (("U", "u"), ("S",), ("M",), ("L",))
HIT_FRAC = 0.7           # a shot counts as on target if it passes within this fraction of the radius
MAX_TURN = 42            # heading steps considered each way when aiming (42*3 = half a turn)
THREAT_KINDS = "LMSUuR"

# Game start / restart: with no lives left (boot or game over), press Coin then
# Start after this many frames without a ship
RESTART_AFTER = 180


def wrap(d, size):
    """Shortest signed distance on a wrapping axis (works on numpy arrays)."""
    return (d + size / 2) % size - size / 2


def heading_vec(h):
    a = h * 2 * math.pi / 256
    return math.cos(a), math.sin(a)


def heading_of(dx, dy):
    return (math.atan2(dy, dx) * 256 / (2 * math.pi)) % 256


def heading_diff(target, h):
    """Signed heading units to turn from h to target; positive = Left."""
    return (target - h + 128) % 256 - 128


def simulate_ship(pos, vel, heading, target_h, thrust_frames, horizon=HORIZON):
    """Ship positions for frames 0..horizon: turn toward target_h, then thrust.

    target_h None means coast without thrusting.
    """
    x, y = pos
    vx, vy = vel
    h = heading
    path = np.empty((horizon + 1, 2))
    path[0] = x, y
    for t in range(1, horizon + 1):
        if target_h is not None:
            d = heading_diff(target_h, h)
            if d > 1.5:
                h += TURN_RATE
            elif d < -1.5:
                h -= TURN_RATE
            elif thrust_frames > 0:
                cx, cy = heading_vec(h)
                vx += THRUST_ACCEL * cx
                vy += THRUST_ACCEL * cy
                thrust_frames -= 1
        vx *= DRAG
        vy *= DRAG
        speed = math.hypot(vx, vy)
        if speed > MAX_SPEED:
            vx, vy = vx * MAX_SPEED / speed, vy * MAX_SPEED / speed
        x += vx
        y += vy
        path[t] = x, y
    return path


def first_hit(path, threats):
    """(first colliding frame or len(path), smallest clearance) for a ship path.

    threats: (p, v, r) numpy arrays of shape (n,2), (n,2), (n,).
    """
    p, v, r = threats
    if len(r) == 0:
        return len(path), float("inf")
    t = np.arange(len(path))
    obj = p[:, None, :] + v[:, None, :] * t[None, :, None]
    d = obj - path[None, :, :]
    d[..., 0] = wrap(d[..., 0], W)
    d[..., 1] = wrap(d[..., 1], H)
    clearance = np.hypot(d[..., 0], d[..., 1]) - (r[:, None] + SHIP_RADIUS + MARGIN)
    hit = (clearance < 0).any(axis=0)
    t_hit = int(np.argmax(hit)) if hit.any() else len(path)
    return t_hit, float(clearance.min())


def default_bullet_v(h):
    """Bullet velocity relative to the ship for heading h, before any is learned."""
    c, sn = heading_vec(h)
    return round(BULLET_SPEED * c), round(BULLET_SPEED * sn)


def covered(target, bullets):
    """True if a player bullet in flight is predicted to hit target.

    Bullets use the game's exact velocity bytes and their remaining life.
    """
    tx, ty = target.pos
    r = RADIUS[target.kind]
    for b in bullets:
        t = np.arange(max(b.status, 1) * 4)
        bx, by = b.pos
        dx = wrap((tx - bx) + (target.vx - b.ram_v[0]) * t, W)
        dy = wrap((ty - by) + (target.vy - b.ram_v[1]) * t, H)
        if (np.hypot(dx, dy) < r).any():
            return True
    return False


def bcd(b):
    return (b >> 4) * 10 + (b & 0x0F)


class Pilot:
    """Stateful autopilot; call step() once per frame."""

    def __init__(self):
        self.fired_last = False
        self.last_turn = 0        # +1 Left / -1 Right / 0, decided last frame (not yet in RAM)
        self.target_uid = None
        self.evade_h = None
        self.no_ship_frames = 0
        self.restart_t = None
        self.lives = None
        self.bullet_v = {}        # heading -> learned bullet velocity relative to ship
        self.pending_shots = {}   # spawn frame -> (heading, predicted pos, predicted vel)
        self.heading_hist = []    # (frame, heading) for the last few frames
        # stats
        self.spawn_err_sum = 0.0
        self.spawn_n = 0
        self.modes = Counter()
        self.shots = 0
        self.hits = 0
        self.misses = 0
        self.deaths = 0
        self.score = 0

    def step(self, frame, tracker, ship_slot, zp, ship_vel):
        """Return (inputs, overlay_tokens, mode)."""
        tracks = tracker.tracks
        # A player bullet that vanishes with life left (status >= 2) hit something.
        for t in tracker.ended:
            if t.kind == "B":
                if t.status >= 2:
                    self.hits += 1
                else:
                    self.misses += 1
        self.score = (bcd(zp[0x53]) * 100 + bcd(zp[0x52])) * 10
        lives = zp[0x57]
        if self.lives is not None and lives < self.lives:
            self.deaths += 1
        self.lives = lives

        ship = tracks.get(ship_slot)
        if ship is None or ship.kind != "H":
            self.fired_last = False
            self.last_turn = 0
            return self._restart(lives), [], "WAIT"
        self.no_ship_frames = 0
        self.restart_t = None

        # Effective heading: RAM heading plus the turn still in the input pipeline.
        heading = (zp[0x61] + TURN_RATE * self.last_turn) % 256
        pos = ship.pos
        self.heading_hist = (self.heading_hist + [(frame, heading)])[-4:]
        self._check_spawn(frame, tracks, heading, ship_vel)
        objs = list(tracks.values())
        threats = [t for t in objs if t.kind in THREAT_KINDS]
        tarr = (np.array([t.pos for t in threats], float).reshape(-1, 2),
                np.array([(t.vx, t.vy) for t in threats], float).reshape(-1, 2),
                np.array([RADIUS[t.kind] for t in threats], float))

        inputs = set()
        overlay = []

        # 1-2. Threat check and evasion plan.
        coast = first_hit(simulate_ship(pos, ship_vel, heading, None, 0), tarr)
        evade = None
        if coast[0] < DANGER_T:
            evade = self._plan_evade(pos, ship_vel, heading, tarr, coast)
            if evade == "H":
                inputs.add("H")

        # 3. Aim: the (target, reachable heading) whose simulated shot hits soonest.
        plan = self._pick_target(objs, pos, ship_vel, heading)
        tgt = plan["track"] if plan else None
        if plan is not None:
            overlay.append(f"A,{plan['aim'][0]},{plan['aim'][1]}")

        turn = None
        if isinstance(evade, float):
            mode = "EVADE"
            d = heading_diff(evade, heading)
            turn = "L" if d > 1.5 else "R" if d < -1.5 else None
            if abs(d) <= THRUST_ALIGN:
                inputs.add("T")
        else:
            mode = "ATTACK-" + tgt.kind if tgt else "IDLE"
            if plan is not None and plan["k"] != 0:
                turn = "L" if plan["k"] > 0 else "R"

        # 4. Fire when a shot from the current heading would hit an allowed target.
        n_bullets = sum(1 for t in objs if t.kind == "B")
        if (not self.fired_last and n_bullets < MAX_BULLETS
                and self._hits_now(objs, pos, ship_vel, heading, plan)):
            inputs.add("F")
            self.shots += 1
            turn = None  # hold the heading so the bullet leaves along it
            self._expect_spawn(frame, pos, ship_vel, heading)
        if turn:
            inputs.add(turn)
        self.last_turn = 1 if turn == "L" else -1 if turn == "R" else 0
        self.fired_last = "F" in inputs

        self.modes[mode.split("-")[0]] += 1
        overlay.append(f"MODE:{mode}")
        return "".join(sorted(inputs)) or "-", overlay, mode

    def _plan_evade(self, pos, vel, heading, tarr, coast):
        """Best escape heading (float), "H" for hyperspace, or None to coast.

        coast: first_hit() score of drifting without thrust.
        """
        best_h, best_score = None, coast
        prev_score = None
        for k in range(EVADE_HEADINGS):
            h = k * 256 / EVADE_HEADINGS
            score = first_hit(simulate_ship(pos, vel, heading, h, EVADE_THRUST), tarr)
            if self.evade_h is not None and h == self.evade_h:
                prev_score = score
            if score > best_score:
                best_h, best_score = h, score
        # Hysteresis: stay with the previous escape heading if it is nearly as good.
        if prev_score is not None and prev_score[0] >= best_score[0] and prev_score[1] > best_score[1] - 50:
            best_h = self.evade_h
        self.evade_h = best_h
        if best_score[0] <= HYPER_T:
            return "H"
        return best_h

    def bv(self, h):
        h = int(h) % 256
        return self.bullet_v.get(h) or default_bullet_v(h)

    def _shot_hits(self, targets, pos, vel, heading, ks):
        """Simulate firing after turning k steps, for each target and each k in ks.

        Returns (hit, first): boolean (targets, ks) and the frame of flight at
        which each shot first comes within HIT_FRAC of the target's radius.
        """
        hs = (heading + TURN_RATE * ks) % 256
        bo = np.array([self.bv(h) for h in hs], float)                    # (nk, 2)
        ang = hs * 2 * np.pi / 256
        muz = MUZZLE * np.stack([np.cos(ang), np.sin(ang)], axis=1)       # (nk, 2)
        delay = (np.abs(ks) + FIRE_DELAY)[None, :, None]                  # (1, nk, 1)
        p = np.array([t.pos for t in targets], float)[:, None, :]         # (m, 1, 2)
        v = np.array([(t.vx, t.vy) for t in targets], float)[:, None, :]
        sv = np.asarray(vel, float)
        # target - bullet, at frame t of the bullet's flight
        d0 = p - np.asarray(pos, float) - muz[None] + (v - sv) * delay    # (m, nk, 2)
        rv = v - sv - bo[None]                                            # (m, nk, 2)
        t = np.arange(BULLET_LIFE, dtype=float)[None, None, :]
        dx = wrap(d0[..., 0:1] + rv[..., 0:1] * t, W)                     # (m, nk, L)
        dy = wrap(d0[..., 1:2] + rv[..., 1:2] * t, H)
        r = np.array([RADIUS[tr.kind] for tr in targets], float)[:, None, None] * HIT_FRAC
        inside = np.hypot(dx, dy) < r
        return inside.any(axis=2), np.argmax(inside, axis=2)

    def _pick_target(self, objs, pos, vel, heading):
        """Highest-priority uncovered target that a shot can hit, choosing the
        heading step that hits soonest. Returns a plan dict or None."""
        bullets = [t for t in objs if t.kind == "B"]
        ks = np.arange(-MAX_TURN, MAX_TURN + 1)
        for ci, kinds in enumerate(PRIORITY):
            cands = [t for t in objs if t.kind in kinds and not covered(t, bullets)]
            if not cands:
                continue
            hit, first = self._shot_hits(cands, pos, vel, heading, ks)
            if not hit.any():
                continue
            cost = np.where(hit, np.abs(ks)[None, :] + first, np.inf)
            for i, t in enumerate(cands):
                if t.uid == self.target_uid:
                    cost[i] -= 10  # stick with the current target unless another is clearly better
            i, j = np.unravel_index(np.argmin(cost), cost.shape)
            t, k = cands[i], int(ks[j])
            n = abs(k) + FIRE_DELAY + int(first[i, j])
            aim = (int(t.pos[0] + t.vx * n) % W, int(t.pos[1] + t.vy * n) % H)
            self.target_uid = t.uid
            return {"track": t, "k": k, "aim": aim, "class": ci}
        self.target_uid = None
        return None

    def _hits_now(self, objs, pos, vel, heading, plan):
        """Would a shot from the current heading hit a target of the planned
        priority class or higher?"""
        if plan is None:
            return False
        kinds = "".join("".join(k) for k in PRIORITY[:plan["class"] + 1])
        bullets = [t for t in objs if t.kind == "B"]
        targets = [t for t in objs if t.kind in kinds and not covered(t, bullets)]
        if not targets:
            return False
        hit, _ = self._shot_hits(targets, pos, vel, heading, np.array([0]))
        return bool(hit.any())

    def _expect_spawn(self, frame, pos, vel, heading):
        """Remember where this shot's bullet should appear, to check the model."""
        c, s = heading_vec(heading)
        bo = self.bv(heading)
        p = (pos[0] + vel[0] * FIRE_DELAY + MUZZLE * c, pos[1] + vel[1] * FIRE_DELAY + MUZZLE * s)
        self.pending_shots[frame + FIRE_DELAY] = (heading, p)

    def _check_spawn(self, frame, tracks, heading, ship_vel):
        """Compare a predicted bullet spawn with RAM; learn the exact bullet velocity."""
        for f in [f for f in self.pending_shots if f < frame]:
            del self.pending_shots[f]
        exp = self.pending_shots.pop(frame, None)
        if exp is None:
            return
        h, (px, py) = exp
        new = [t for t in tracks.values() if t.kind == "B" and len(t.hist) == 1]
        if not new:
            return
        b = min(new, key=lambda t: math.hypot(wrap(t.pos[0] - px, W), wrap(t.pos[1] - py, H)))
        self.spawn_err_sum += math.hypot(wrap(b.pos[0] - px, W), wrap(b.pos[1] - py, H))
        self.spawn_n += 1
        # Learn the velocity table only when the heading was steady around the shot.
        if all(hh == h for _, hh in self.heading_hist):
            self.bullet_v[h] = (b.ram_v[0] - ship_vel[0], b.ram_v[1] - ship_vel[1])

    def _restart(self, lives):
        """After game over, press Coin then Start."""
        if lives > 0:
            self.no_ship_frames = 0
            return "-"
        self.no_ship_frames += 1
        if self.no_ship_frames < RESTART_AFTER:
            return "-"
        k = self.no_ship_frames - RESTART_AFTER
        if k < 5:
            return "C"
        if 60 <= k < 65:
            return "S"
        if k > 120:
            self.no_ship_frames = 0
        return "-"

    def stats(self):
        total = sum(self.modes.values()) or 1
        modes = " ".join(f"{m}={100 * c // total}%" for m, c in self.modes.most_common())
        done = self.hits + self.misses
        acc = f"{100 * self.hits / done:.0f}%" if done else "-"
        spawn = f"{self.spawn_err_sum / self.spawn_n:.0f}u" if self.spawn_n else "-"
        return (f"score={self.score} lives={self.lives} deaths={self.deaths} "
                f"shots={self.shots} acc={acc} spawn_err={spawn} learned={len(self.bullet_v)} {modes}")
