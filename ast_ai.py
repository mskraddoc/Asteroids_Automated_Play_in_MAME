# ast_ai.py
# Autopilot for Asteroids. Given the tracked objects (ast_server.Tracker), the
# ship's heading and velocity from RAM, decides which controls to press.
#
# Each frame:
#   1. Threat check: will the ship, drifting as it is, hit anything within HORIZON?
#   2. If a hit is coming within DANGER_T (per kind), simulate turning to each of
#      EVADE_HEADINGS and thrusting; thrust along the safest one. If the first
#      thing that would hit is a rock or saucer that a shot can reach in time,
#      turn to it and shoot it (defensive fire), thrusting when the escape
#      heading lines up; hyperspace only when neither works.
#   3. Otherwise (and opportunistically while evading) aim at the highest
#      priority target - saucers, then small, medium, large rocks - and fire.
#
# Play styles change what the ship does when it is not evading:
#   classic  the above; returns to the middle of the screen between waves
#   flyer    thrusts with a throttle that drifts between 0 and full, but only
#            along paths predicted to stay clear of everything; returns to the
#            middle of the screen between waves
#   shadow   spares one large rock per level and flies close to it, until two
#            saucers have been killed in that level
#
# Units: game units (x 0-8191, y 0-6143, y up) and frames. Heading 0-255,
# 0 = +x, counterclockwise. All constants were measured from RAM in MAME.

import math
import random
from collections import Counter, deque

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
# Evade when a collision with this kind is predicted sooner than this (the ship
# accelerates slowly: clearing a rock takes ~60 frames). Benchmarked: warning
# earlier for medium/large rocks (130) and evading rather than shooting them
# both gave more hyperspace jumps than shooting everything at 90.
DANGER_T = {"S": 90, "M": 90, "L": 90, "R": 90, "U": 90, "u": 90}
DEFEND_KINDS = "LMSUu"   # shoot these when they are about to hit (saucer bullets can't be)
HYPER_T = 6              # hyperspace when every option collides sooner than this
DEFEND_MARGIN = 3        # defensive shot must land this many frames before the predicted hit
EVADE_HEADINGS = 16      # candidate escape directions
EVADE_THRUST = 25        # frames of thrust simulated for each escape candidate
THRUST_ALIGN = 12        # thrust only when within this many heading units
PRIORITY = (("U", "u"), ("S",), ("M",), ("L",))
HIT_FRAC = 0.7           # a shot counts as on target if it passes within this fraction of the radius
MAX_TURN = 42            # heading steps considered each way when aiming (42*3 = half a turn)
THREAT_KINDS = "LMSUuR"

# Play styles
STYLES = ("classic", "flyer", "shadow")
STEER_DV = 3.0           # steer when the ship's velocity is off by more than this (units/frame)
FLYER_HOLD = (180, 480)  # frames between new throttle levels
FLYER_RAMP = 60          # frames to ramp to a new level
FLY_LOOKAHEAD = 20       # flyer thrusts only if thrusting this many frames, then coasting
                         # to HORIZON, is predicted to stay clear of everything
FLY_CLEAR = 0            # extra clearance wanted beyond the collision distance (which
                         # already includes MARGIN)
FLY_MAX_SPEED = 20.0     # no flyer thrust that adds speed above this (units/frame)
STEER_BRAKE = 0.2        # units/frame^2 assumed for slowing down (thrust, less the turnaround)
HOME_MAX_V = 30.0        # top speed returning to the centre
HOME_DV = 1.5            # tighter tolerance so the ship settles near the centre
HOME_DONE = (400, 2.0)   # home when this close (units) and this slow (units/frame)
HOME_GRACE = 180         # frames into a new wave before giving up on getting home
SHADOW_DIST = 700        # centre-to-centre distance kept from the boulder (~300 units clear)
SHADOW_BAND = 150        # no closing speed within this many units of SHADOW_DIST
SHADOW_MAX_V = 10.0
SHADOW_SAUCERS = 2       # saucer kills in a level that release the boulder
SHADOW_REACQUIRE = 400   # a large rock this close to the boulder's predicted position is it
PET_VETO_FRAC = 1.2      # don't fire a shot passing within this fraction of the boulder's radius

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


def hit_times(path, threats):
    """First colliding frame of each threat against a ship path (len(path) if none)."""
    p, v, r = threats
    if len(r) == 0:
        return np.empty(0)
    t = np.arange(len(path))
    d = p[:, None, :] + v[:, None, :] * t[None, :, None] - path[None, :, :]
    d[..., 0] = wrap(d[..., 0], W)
    d[..., 1] = wrap(d[..., 1], H)
    hit = np.hypot(d[..., 0], d[..., 1]) < (r[:, None] + SHIP_RADIUS + MARGIN)
    return np.where(hit.any(axis=1), np.argmax(hit, axis=1), len(path))


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

    def __init__(self, style="classic", seed=None):
        self.rng = random.Random(seed)
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
        self.steering = False     # hysteresis for _steer
        # saucer kills this level (shadow)
        self.level = None
        self.saucer_kills = 0
        self.saucer_boom = None   # frame a saucer exploded, waiting for its points
        self.gains = deque(maxlen=8)  # (frame, score gained)
        # stats
        self.spawn_err_sum = 0.0
        self.spawn_n = 0
        self.modes = Counter()
        self.shots = 0
        self.hits = 0
        self.misses = 0
        self.deaths = 0
        self.score = 0
        self.set_style(style)

    def set_style(self, style):
        """Switch play style and clear the per-style state."""
        assert style in STYLES, style
        self.style = style
        self.throttle = self.fly_target = self.fly_acc = 0.0
        self.fly_next = 0         # frame to pick the next throttle level
        self.home_until = None    # frame to stop heading home (None = not homing)
        self.pet_uid = None       # shadow's boulder
        self.pet_last = None      # (frame, x, y, vx, vy) last seen
        self.steering = False

    def step(self, frame, tracker, ship_slot, zp, ship_vel, saucer_status=0, level=0):
        """Return (inputs, overlay_tokens, mode).

        saucer_status: the saucer slot's RAM status byte (>= 0x80 while exploding).
        level: the wave number, for resetting the shadow style's saucer count.
        """
        tracks = tracker.tracks
        # A player bullet that vanishes with life left (status >= 2) hit something.
        for t in tracker.ended:
            if t.kind == "B":
                if t.status >= 2:
                    self.hits += 1
                else:
                    self.misses += 1
        score = (bcd(zp[0x53]) * 100 + bcd(zp[0x52])) * 10
        gain = (score - self.score) % 100_000
        self.gains.append((frame, gain if gain < 50_000 else 0))
        self.score = score
        self._count_saucer_kills(frame, tracker.ended, saucer_status, level)
        lives = zp[0x57]
        if self.lives is not None and lives < self.lives:
            self.deaths += 1
        self.lives = lives

        ship = tracks.get(ship_slot)
        if ship is None or ship.kind != "H":
            self.fired_last = False
            self.last_turn = 0
            mode = self._mode_name("WAIT")
            return self._restart(lives), [f"MODE:{mode}"], mode
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

        # Shadow: the boulder is spared, so leave it out of the target lists.
        pet = self._pet(frame, objs, pos) if self.style == "shadow" else None
        if pet is not None:
            overlay.append(f"PET,{next(s for s, t in tracks.items() if t is pet)}")
        targets = [t for t in objs if t is not pet]

        # 1-2. Threat check, evasion plan, and defensive fire: shoot the rock or
        # saucer about to hit us when a shot can land in time.
        coast_path = simulate_ship(pos, ship_vel, heading, None, 0)
        coast = first_hit(coast_path, tarr)
        evade = defend = None
        times = hit_times(coast_path, tarr)
        if any(t_hit < DANGER_T[t.kind] for t_hit, t in zip(times, threats)):
            evade = self._plan_evade(pos, ship_vel, heading, tarr, coast)
            defend = self._plan_defend(threats, tarr, coast_path, objs, pos, ship_vel, heading)
            if evade == "H" and defend is None:
                inputs.add("H")

        # 3. Aim: the (target, reachable heading) whose simulated shot hits soonest.
        plan = self._pick_target(targets, pos, ship_vel, heading)
        tgt = plan["track"] if plan else None
        if plan is not None:
            overlay.append(f"A,{plan['aim'][0]},{plan['aim'][1]}")

        turn = None
        detail = ""
        if defend is not None:
            action = "DEFEND-" + defend["track"].kind
            if defend["k"] != 0:
                turn = "L" if defend["k"] > 0 else "R"
            if isinstance(evade, float) and abs(heading_diff(evade, heading)) <= THRUST_ALIGN:
                inputs.add("T")
        elif isinstance(evade, float):
            action = "EVADE"
            d = heading_diff(evade, heading)
            turn = "L" if d > 1.5 else "R" if d < -1.5 else None
            if abs(d) <= THRUST_ALIGN:
                inputs.add("T")
        else:
            action = "ATTACK-" + tgt.kind if tgt else "IDLE"
            if plan is not None and plan["k"] != 0:
                turn = "L" if plan["k"] > 0 else "R"
            if evade is None:
                turn, action, detail = self._style_move(frame, objs, pos, ship_vel, heading,
                                                        pet, turn, action, inputs, tarr)

        # 4. Fire when a shot from the current heading would hit an allowed target.
        n_bullets = sum(1 for t in objs if t.kind == "B")
        if defend is not None:
            shoot = defend["k"] == 0 and not defend["covered"]  # survival first, even the pet
        else:
            shoot = (self._hits_now(targets, pos, ship_vel, heading, plan)
                     and not (pet is not None and self._would_hit(pet, pos, ship_vel, heading)))
        if not self.fired_last and n_bullets < MAX_BULLETS and shoot:
            inputs.add("F")
            self.shots += 1
            turn = None  # hold the heading so the bullet leaves along it
            self._expect_spawn(frame, pos, ship_vel, heading)
        if turn:
            inputs.add(turn)
        self.last_turn = 1 if turn == "L" else -1 if turn == "R" else 0
        self.fired_last = "F" in inputs

        self.modes[action.split("-")[0]] += 1
        mode = self._mode_name(action, detail)
        overlay.append(f"MODE:{mode}")
        return "".join(sorted(inputs)) or "-", overlay, mode

    def _mode_name(self, action, detail=""):
        """e.g. SHADOW_1/2_FOLLOW; tokens carry no spaces, Lua shows "_" as a space."""
        return "_".join(w for w in (self.style.upper(), detail, action) if w)

    def _style_move(self, frame, objs, pos, vel, heading, pet, turn, action, inputs, tarr):
        """Style-specific steering and thrust while not evading or defending.

        tarr: the threat arrays used for the evasion check.
        Returns (turn, action, detail); may add "T" to inputs.
        """
        detail = ""
        if self.style == "shadow":
            detail = f"{self.saucer_kills}/{SHADOW_SAUCERS}"
            if pet is not None:
                s = self._steer(pos, vel, heading, pet.pos, (pet.vx, pet.vy), SHADOW_DIST,
                                SHADOW_BAND, SHADOW_MAX_V, STEER_DV)
                if s:
                    turn = s[0]
                    if s[1]:
                        inputs.add("T")
                    return turn, "FOLLOW", detail
        else:
            home = self._go_home(frame, objs, pos, vel, heading, tarr, inputs)
            if home is not False:
                return home, "HOME", detail
        if self.style == "flyer":
            if frame >= self.fly_next:
                self.fly_target = self.rng.random()
                self.fly_next = frame + self.rng.randint(*FLYER_HOLD)
            step = 1 / FLYER_RAMP
            self.throttle += max(-step, min(step, self.fly_target - self.throttle))
            # Thrust credit builds with the throttle; it is spent only on a safe
            # thrust, so a refused frame thrusts later instead of being lost.
            self.fly_acc = min(self.fly_acc + self.throttle, 2.0)
            if self.fly_acc >= 1:
                if self._thrust_safe(pos, vel, heading, tarr):
                    self.fly_acc -= 1
                    inputs.add("T")
                else:
                    action = "HOLD"
            detail = f"{round(self.throttle * 100)}%"
        return turn, action, detail

    def _go_home(self, frame, objs, pos, vel, heading, tarr, inputs):
        """classic and flyer: between waves (no rocks) fly back to the middle and stop there; keep at it
        for up to HOME_GRACE frames into the next wave. Returns the turn (or None)
        while homing, False otherwise; may add "T" to inputs."""
        if not any(t.kind in "LMS" for t in objs):
            self.home_until = math.inf
        elif self.home_until == math.inf:
            self.home_until = frame + HOME_GRACE
        if self.home_until is not None:
            d = math.hypot(wrap(W / 2 - pos[0], W), wrap(H / 2 - pos[1], H))
            if frame > self.home_until or (self.home_until != math.inf and d < HOME_DONE[0]
                                           and math.hypot(*vel) < HOME_DONE[1]):
                self.home_until = None
        if self.home_until is None:
            return False
        s = self._steer(pos, vel, heading, (W / 2, H / 2), (0.0, 0.0), 0, 0, HOME_MAX_V, HOME_DV)
        if not s:
            return None
        if s[1] and self._thrust_safe(pos, vel, heading, tarr, cap=False):
            inputs.add("T")
        return s[0]

    def _thrust_safe(self, pos, vel, heading, tarr, cap=True, start=None):
        """Is thrusting along `heading` safe? Simulates turning there from `start`
        (default: already on it), FLY_LOOKAHEAD frames of thrust, then coasting out
        to HORIZON, against every threat's predicted motion. With cap, also refuses
        thrust that adds speed above FLY_MAX_SPEED."""
        if cap:
            c, sn = heading_vec(heading)
            if math.hypot(*vel) >= FLY_MAX_SPEED and vel[0] * c + vel[1] * sn > 0:
                return False
        path = simulate_ship(pos, vel, heading if start is None else start, heading, FLY_LOOKAHEAD)
        t_hit, clearance = first_hit(path, tarr)
        return t_hit >= HORIZON + 1 and clearance >= FLY_CLEAR

    def _steer(self, pos, vel, heading, tpos, tvel, standoff, band, max_v, tol):
        """Match velocity with a target point moving at tvel, closing to within
        `band` of `standoff` units away, at up to max_v and slow enough to stop
        in time. Returns (turn or None, thrust) or None when close enough."""
        dx, dy = wrap(tpos[0] - pos[0], W), wrap(tpos[1] - pos[1], H)
        dist = math.hypot(dx, dy)
        err = max(0.0, abs(dist - standoff) - band)
        closing = math.copysign(min(max_v, math.sqrt(2 * STEER_BRAKE * err)), dist - standoff)
        ux, uy = (dx / dist, dy / dist) if dist > 1 else (0.0, 0.0)
        dvx = tvel[0] + closing * ux - vel[0]
        dvy = tvel[1] + closing * uy - vel[1]
        # Hysteresis: once steering, keep going until well inside the tolerance.
        if math.hypot(dvx, dvy) <= (tol / 2 if self.steering else tol):
            self.steering = False
            return None
        self.steering = True
        d = heading_diff(heading_of(dvx, dvy), heading)
        return ("L" if d > 1.5 else "R" if d < -1.5 else None), abs(d) <= THRUST_ALIGN

    def _pet(self, frame, objs, pos):
        """Shadow's boulder track, choosing one if needed; None once released."""
        if self.saucer_kills >= SHADOW_SAUCERS:
            self.pet_uid = None
            return None
        pet = next((t for t in objs if t.uid == self.pet_uid), None)
        if pet is None:
            bigs = [t for t in objs if t.kind == "L"]
            if not bigs:
                self.pet_uid = None
                return None
            pet = None
            if self.pet_last is not None:
                # The game may move a rock to another RAM slot (a new track); find it again.
                f0, x0, y0, vx, vy = self.pet_last
                px, py = x0 + vx * (frame - f0), y0 + vy * (frame - f0)
                near = min(bigs, key=lambda t: math.hypot(wrap(t.pos[0] - px, W), wrap(t.pos[1] - py, H)))
                if math.hypot(wrap(near.pos[0] - px, W), wrap(near.pos[1] - py, H)) < SHADOW_REACQUIRE:
                    pet = near
            if pet is None:
                pet = min(bigs, key=lambda t: math.hypot(wrap(t.pos[0] - pos[0], W), wrap(t.pos[1] - pos[1], H)))
            self.pet_uid = pet.uid
        self.pet_last = (frame, pet.pos[0], pet.pos[1], pet.vx, pet.vy)
        return pet

    def _count_saucer_kills(self, frame, ended, saucer_status, level):
        """Count saucers the player destroyed this level: the saucer explodes
        (status >= 0x80, rather than flying off) and at least 200 points arrive."""
        if level != self.level:
            self.level = level
            self.saucer_kills = 0
            self.pet_uid = self.pet_last = None
        if any(t.kind in "Uu" for t in ended) and saucer_status >= 0x80:
            self.saucer_boom = frame
        if self.saucer_boom is not None:
            if any(g >= 200 and abs(f - self.saucer_boom) <= 3 for f, g in self.gains):
                self.saucer_kills += 1
                self.saucer_boom = None
            elif frame > self.saucer_boom + 3:
                self.saucer_boom = None

    def _would_hit(self, track, pos, vel, heading):
        """Would a shot from the current heading pass near this track?"""
        hit, _ = self._shot_hits([track], pos, vel, heading, np.array([0]), PET_VETO_FRAC)
        return bool(hit.any())

    def _plan_defend(self, threats, tarr, coast_path, objs, pos, vel, heading):
        """Shoot the threat that will hit first, if it is a rock or saucer and a shot
        (after turning k steps) lands DEFEND_MARGIN frames before the predicted hit.
        Returns {"track", "k", "covered"} or None; covered means a bullet already in
        flight will hit it, so just hold."""
        times = hit_times(coast_path, tarr)
        if len(times) == 0:
            return None
        i = int(np.argmin(times))
        t_hit, threat = int(times[i]), threats[i]
        if threat.kind not in DEFEND_KINDS:
            return None
        if covered(threat, [t for t in objs if t.kind == "B"]):
            return {"track": threat, "k": 0, "covered": True}
        ks = np.arange(-MAX_TURN, MAX_TURN + 1)
        hit, first = self._shot_hits([threat], pos, vel, heading, ks)
        land = np.abs(ks) + FIRE_DELAY + first[0]
        ok = hit[0] & (land < t_hit - DEFEND_MARGIN)
        if not ok.any():
            return None
        j = int(np.argmin(np.where(ok, land, np.inf)))
        return {"track": threat, "k": int(ks[j]), "covered": False}

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

    def _shot_hits(self, targets, pos, vel, heading, ks, frac=HIT_FRAC):
        """Simulate firing after turning k steps, for each target and each k in ks.

        Returns (hit, first): boolean (targets, ks) and the frame of flight at
        which each shot first comes within frac of the target's radius.
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
        r = np.array([RADIUS[tr.kind] for tr in targets], float)[:, None, None] * frac
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
