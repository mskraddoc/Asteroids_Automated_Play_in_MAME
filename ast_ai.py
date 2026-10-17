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
#      heading lines up; hyperspace only when neither works and the game's own
#      collision test (ROM_REACH) agrees that every option still hits.
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

# Ship and bullet physics, as the game does them (ship control $703F, movement $6FC7,
# fire $6CD7, bullet life $738D); checked frame by frame against RAM in MAME.
TURN_RATE = 3            # heading units per frame while Left/Right is held (every frame)
# Thrust and drag act only on frames when the frame counter (zero page $5C) is even.
# Thrust adds 2 * sine / 256 per axis (0.99 units/frame per step, 0.5 on average);
# without thrust, drag takes (2 * int + 1) / 256 off a positive velocity, or
# 2 * int / 256 off a negative one (about x0.9961 per frame). Velocity is 16-bit (the
# integer byte plus a fraction byte, $64 / $65), each axis is clamped to +-63.996, and
# the position moves by the integer byte only.
SIN_TABLE = (0, 3, 6, 9, 12, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 51, 54, 57,
             60, 63, 65, 68, 71, 73, 76, 78, 81, 83, 85, 88, 90, 92, 94, 96, 98, 100, 102,
             104, 106, 107, 109, 111, 112, 113, 115, 116, 117, 118, 120, 121, 122, 122, 123,
             124, 125, 125, 126, 126, 126, 127, 127, 127, 127)  # $57B9: a quarter turn
MAX_V16 = 0x3FFF         # per-axis velocity clamp, in 1/256 units/frame
# A shot's velocity is sine / 2 per axis (at most 63) plus the ship's integer velocity,
# clamped to +-BULLET_V_MAX. It starts 1.5x its relative velocity ahead of where the ship
# was and moves once on the same frame (about 157 units ahead in all). Its life counter
# starts at 18 and drops on frames when the frame counter is a multiple of 4, so it is
# in RAM for 68-71 frames.
BULLET_V_MAX = 111
MAX_BULLETS = 4
FIRE_DELAY = 2           # frames from the fire decision until the bullet appears in RAM
# Inputs also take 2 frames to show up in RAM: a turn decided at frame n changes the
# heading read at n+2. So at frame n the heading read is missing the turn decided at
# n-1, and the bullet leaves along that newer heading (3 units = 4 deg off otherwise).

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
DEFEND_MARGIN = 1        # defensive shot must land this many frames before the predicted hit
# The game's collision test ($6A55): an object hits the ship when |dx| < A, |dy| < A
# and |dx| + |dy| < 1.5 A, with A in game units below. RADIUS + SHIP_RADIUS + MARGIN
# is much larger, which is right for evading but called for hyperspace about twice
# as often as needed (half the jumps were near misses), so a jump is re-checked
# against this before it is taken.
ROM_REACH = {"R": 84, "S": 140, "M": 200, "L": 320, "u": 122, "U": 158}
HYPER_PAD = 8            # units added to ROM_REACH for that re-check
# The same test for a player bullet (a point) against each target: reach A in game units
ROM_SHOT_REACH = {"S": 84, "M": 144, "L": 264, "u": 84, "U": 144}
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

# Ships (lives byte) kept at most; extra ones are spent on purpose. The game draws
# every reserve ship each frame, and with ~150+ the main loop falls behind and the
# watchdog resets the machine.
LIVES_CAP = 25


def wrap(d, size):
    """Shortest signed distance on a wrapping axis (works on numpy arrays)."""
    return (d + size / 2) % size - size / 2


def wrap_arr(d, size):
    """wrap() for numpy arrays, ~10x faster (float % is slow in numpy). Only the
    sign at exactly +-size/2 can differ, which abs/hypot/octagon_hit ignore."""
    return d - size * np.rint(d / size)


def heading_vec(h):
    a = h * 2 * math.pi / 256
    return math.cos(a), math.sin(a)


def heading_of(dx, dy):
    return (math.atan2(dy, dx) * 256 / (2 * math.pi)) % 256


def heading_diff(target, h):
    """Signed heading units to turn from h to target; positive = Left."""
    return (target - h + 128) % 256 - 128


def sin_rom(h):
    """The game's sine of heading h (0-255), -127..127 ($77D5)."""
    h = int(h) & 0xFF
    a = h & 0x7F
    v = SIN_TABLE[0x80 - a if a > 0x40 else a]
    return -v if h >= 0x80 else v


def cos_rom(h):
    return sin_rom(int(h) + 0x40)


SIN256 = tuple(sin_rom(h) for h in range(256))
COS256 = tuple(cos_rom(h) for h in range(256))


def drag16(v):
    """One drag step on a 16-bit velocity (1/256 units/frame), as at $70E6."""
    if v == 0:
        return 0
    i = v >> 8
    return v - (2 * i + 1) if i >= 0 else v - 2 * i


def clamp16(v):
    """The per-axis velocity clamp at $7125."""
    if v >= 0x4000:
        return MAX_V16
    if v < -0x4000:
        return -MAX_V16
    return v


def simulate_ship(pos, vel16, heading, target_h, thrust_frames, counter, pending_thrust=False,
                  horizon=HORIZON):
    """Ship positions for frames 0..horizon, stepped exactly as the game does.

    vel16: (vx, vy) in 1/256 units/frame. counter: the frame counter ($5C) at frame 0.
    heading already includes the turn still in the input pipeline, and pending_thrust
    says whether thrust is pressed for frame 1 (inputs act 2 frames late). From frame 2
    the ship turns toward target_h, then thrusts for thrust_frames frames; target_h None
    coasts.
    """
    # Hot loop: heading_diff, clamp16 and drag16 are inlined, sine/cosine come from
    # the tables, and the path is built from lists at the end.
    x, y = pos
    vx, vy = vel16
    h = int(heading) % 256
    xs, ys = [x], [y]
    for t in range(1, horizon + 1):
        thrust = t == 1 and pending_thrust
        if t > 1 and target_h is not None:
            d = (target_h - h + 128) % 256 - 128
            if d > 1.5:
                h = (h + TURN_RATE) % 256
            elif d < -1.5:
                h = (h - TURN_RATE) % 256
            elif thrust_frames > 0:
                thrust = True
                thrust_frames -= 1
        if (counter + t) % 2 == 0:
            if thrust:
                vx += 2 * COS256[h]
                vy += 2 * SIN256[h]
                vx = MAX_V16 if vx >= 0x4000 else -MAX_V16 if vx < -0x4000 else vx
                vy = MAX_V16 if vy >= 0x4000 else -MAX_V16 if vy < -0x4000 else vy
            else:
                if vx:
                    i = vx >> 8
                    vx -= 2 * i + 1 if i >= 0 else 2 * i
                if vy:
                    i = vy >> 8
                    vy -= 2 * i + 1 if i >= 0 else 2 * i
        x += vx >> 8
        y += vy >> 8
        xs.append(x)
        ys.append(y)
    return np.column_stack((xs, ys)).astype(float)


def octagon_hit(dx, dy, a):
    """The game's collision test ($6A55) on offsets dx, dy with reach a."""
    dx, dy = np.abs(dx), np.abs(dy)
    return (dx < a) & (dy < a) & (dx + dy < 1.5 * a)


def bullet_rel(h):
    """A shot's velocity relative to the ship for heading h (fire $6CD7)."""
    return cos_rom(h) >> 1, sin_rom(h) >> 1


BULLET_REL = np.array([bullet_rel(h) for h in range(256)], float)  # (256, 2)


def bullet_life(counter):
    """Frames a shot that appears on a frame with this counter value stays in RAM."""
    return 68 + (4 - counter % 4) % 4


def frames_left(status, counter):
    """Further frames a bullet with this life status stays in RAM after this one."""
    return (4 - counter % 4) + 4 * (status - 1) - 1


def path_offsets(paths, p, v):
    """Wrapped threat-minus-ship offsets (dx, dy), each of shape (n, k, T), for k
    ship paths (k, T, 2) against n threats moving from p at v ((n,2) arrays)."""
    t = np.arange(paths.shape[1])
    obj = p[:, None, :] + v[:, None, :] * t[None, :, None]       # (n, T, 2)
    d = obj[:, None] - paths[None]                                # (n, k, T, 2)
    return wrap_arr(d[..., 0], W), wrap_arr(d[..., 1], H)


def clearances(paths, threats):
    """Clearance (n, k, T) of n threats from k ship paths (k, T, 2); < 0 is a hit."""
    p, v, r = threats
    dx, dy = path_offsets(paths, p, v)
    return np.hypot(dx, dy) - (r[:, None, None] + SHIP_RADIUS + MARGIN)


def first_hits(paths, threats):
    """first_hit() for k ship paths (k, T, 2) at once: (t_hit, clearance) arrays (k,)."""
    k, n_t = paths.shape[:2]
    if len(threats[2]) == 0:
        return np.full(k, n_t), np.full(k, np.inf)
    c = clearances(paths, threats)
    hit = (c < 0).any(axis=0)                                     # (k, T)
    return np.where(hit.any(axis=1), np.argmax(hit, axis=1), n_t), c.min(axis=(0, 2))


def first_hit(path, threats):
    """(first colliding frame or len(path), smallest clearance) for a ship path.

    threats: (p, v, r) numpy arrays of shape (n,2), (n,2), (n,).
    """
    t_hit, clearance = first_hits(path[None], threats)
    return int(t_hit[0]), float(clearance[0])


def coast_hits(path, threats):
    """first_hit() plus each threat's first colliding frame (len(path) if none) for
    one ship path, from a single clearance computation."""
    if len(threats[2]) == 0:
        return (len(path), float("inf")), np.empty(0)
    c = clearances(path[None], threats)[:, 0]                     # (n, T)
    hit = c < 0
    times = np.where(hit.any(axis=1), np.argmax(hit, axis=1), len(path))
    return (int(times.min()), float(c.min())), times


def rom_first_hits(paths, p, v, reach):
    """First frame the game's own collision test would fire for each of k ship
    paths (k, T, 2), or T. p, v: (n,2) threat positions and velocities; reach: (n,)
    ROM_REACH plus padding."""
    k, n_t = paths.shape[:2]
    if len(reach) == 0:
        return np.full(k, n_t)
    dx, dy = path_offsets(paths, p, v)
    hit = octagon_hit(dx, dy, reach[:, None, None]).any(axis=0)   # (k, T)
    return np.where(hit.any(axis=1), np.argmax(hit, axis=1), n_t)


def covered(target, bullets, counter):
    """True if a player bullet in flight is predicted to hit target.

    Bullets use the game's exact velocity bytes, remaining life and hit test.
    counter: the frame counter ($5C) now.
    """
    if not bullets:
        return False
    tx, ty = target.pos
    a = ROM_SHOT_REACH[target.kind]
    left = np.array([frames_left(max(b.status, 1), counter) for b in bullets])
    bp = np.array([b.pos for b in bullets], float)
    bv = np.array([b.ram_v for b in bullets], float)
    t = np.arange(left.max() + 1)[None, :]
    dx = wrap_arr((tx - bp[:, 0:1]) + (target.vx - bv[:, 0:1]) * t, W)
    dy = wrap_arr((ty - bp[:, 1:2]) + (target.vy - bv[:, 1:2]) * t, H)
    return bool((octagon_hit(dx, dy, a) & (t <= left[:, None])).any())


def bcd(b):
    return (b >> 4) * 10 + (b & 0x0F)


class Pilot:
    """Stateful autopilot; call step() once per frame."""

    def __init__(self, style="classic", seed=None):
        self.rng = random.Random(seed)
        self.fired_last = False
        self.last_turn = 0        # +1 Left / -1 Right / 0, decided last frame (not yet in RAM)
        self.last_thrust = False  # thrust decided last frame (not yet in RAM)
        self.ship_sim = ((0, 0), 0)  # this frame's 16-bit ship velocity and frame counter
        self.target_uid = None
        self.evade_h = None
        self.no_ship_frames = 0
        self.restart_t = None
        self.lives = None
        self.pending_shots = {}   # spawn frame -> (heading, predicted pos, predicted vel)
        self._covered = {}        # covered() per track uid, rebuilt each frame
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
        self.spent = 0            # ships spent on purpose over LIVES_CAP
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
            if self.lives > LIVES_CAP:
                self.spent += 1
            else:
                self.deaths += 1
        self.lives = lives

        ship = tracks.get(ship_slot)
        if ship is None or ship.kind != "H":
            self.fired_last = False
            self.last_turn = 0
            self.last_thrust = False
            mode = self._mode_name("WAIT")
            return self._restart(lives), [f"MODE:{mode}"], mode
        self.no_ship_frames = 0
        self.restart_t = None

        # Effective heading: RAM heading plus the turn still in the input pipeline.
        heading = (zp[0x61] + TURN_RATE * self.last_turn) % 256
        pos = ship.pos
        self.ship_sim = ((ship_vel[0] * 256 + zp[0x64], ship_vel[1] * 256 + zp[0x65]), zp[0x5C])
        self._check_spawn(frame, tracks)
        objs = list(tracks.values())
        threats = [t for t in objs if t.kind in THREAT_KINDS]
        tarr = (np.array([t.pos for t in threats], float).reshape(-1, 2),
                np.array([(t.vx, t.vy) for t in threats], float).reshape(-1, 2),
                np.array([RADIUS[t.kind] for t in threats], float))

        inputs = set()
        overlay = []

        # Over the ship cap: spend ships the way marathon players did, by letting
        # them die. Too many reserve ships slow the game and trip its watchdog.
        if lives > LIVES_CAP:
            return self._spend_ship(objs, pos, heading, overlay)

        # Shadow: the boulder is spared, so leave it out of the target lists.
        pet = self._pet(frame, objs, pos) if self.style == "shadow" else None
        if pet is not None:
            overlay.append(f"PET,{next(s for s, t in tracks.items() if t is pet)}")
        targets = [t for t in objs if t is not pet]

        # 1-2. Threat check, evasion plan, and defensive fire: shoot the rock or
        # saucer about to hit us when a shot can land in time.
        self._covered = {}  # covered() per track uid, this frame
        coast_path = self._sim(pos, heading, None, 0)
        coast, times = coast_hits(coast_path, tarr)
        evade = defend = None
        if any(t_hit < DANGER_T[t.kind] for t_hit, t in zip(times, threats)):
            reach = np.array([ROM_REACH[t.kind] + HYPER_PAD for t in threats], float)
            evade = self._plan_evade(pos, heading, tarr, coast, coast_path, reach)
            defend = self._plan_defend(threats, times, objs, pos, ship_vel, heading)
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
            shoot = (plan is not None and plan["now"]
                     and not (pet is not None and self._would_hit(pet, pos, ship_vel, heading)))
        if not self.fired_last and n_bullets < MAX_BULLETS and shoot:
            inputs.add("F")
            self.shots += 1
            turn = None  # hold the heading so the bullet leaves along it
            self._expect_spawn(frame, pos, ship_vel, heading)
        if turn:
            inputs.add(turn)
        self.last_turn = 1 if turn == "L" else -1 if turn == "R" else 0
        self.last_thrust = "T" in inputs
        self.fired_last = "F" in inputs

        self.modes[action.split("-")[0]] += 1
        mode = self._mode_name(action, detail)
        overlay.append(f"MODE:{mode}")
        return "".join(sorted(inputs)) or "-", overlay, mode

    def _spend_ship(self, objs, pos, heading, overlay):
        """Ram the nearest rock: turn to it and thrust, no evasion, no fire."""
        inputs = set()
        rocks = [t for t in objs if t.kind in "LMS"]
        if rocks:
            dx, dy = min(((wrap(t.pos[0] - pos[0], W), wrap(t.pos[1] - pos[1], H)) for t in rocks),
                         key=lambda d: math.hypot(*d))
            d = heading_diff(heading_of(dx, dy), heading)
            if d > 1.5:
                inputs.add("L")
            elif d < -1.5:
                inputs.add("R")
            if abs(d) <= THRUST_ALIGN:
                inputs.add("T")
        self.last_turn = 1 if "L" in inputs else -1 if "R" in inputs else 0
        self.last_thrust = "T" in inputs
        self.fired_last = False
        self.modes["SPEND"] += 1
        mode = self._mode_name("SPEND")
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
        path = self._sim(pos, heading if start is None else start, heading, FLY_LOOKAHEAD)
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

    def _is_covered(self, target, bullets):
        """covered() for this frame, computed once per track."""
        c = self._covered.get(target.uid)
        if c is None:
            c = self._covered[target.uid] = covered(target, bullets, self.ship_sim[1])
        return c

    def _plan_defend(self, threats, times, objs, pos, vel, heading):
        """Shoot the threat that will hit first, if it is a rock or saucer and a shot
        (after turning k steps) lands DEFEND_MARGIN frames before the predicted hit.
        times: each threat's first colliding frame while coasting (coast_hits).
        Returns {"track", "k", "covered"} or None; covered means a bullet already in
        flight will hit it, so just hold."""
        if len(times) == 0:
            return None
        i = int(np.argmin(times))
        t_hit, threat = int(times[i]), threats[i]
        if threat.kind not in DEFEND_KINDS:
            return None
        if self._is_covered(threat, [t for t in objs if t.kind == "B"]):
            return {"track": threat, "k": 0, "covered": True}
        ks = np.arange(-MAX_TURN, MAX_TURN + 1)
        hit, first = self._shot_hits([threat], pos, vel, heading, ks)
        land = np.abs(ks) + FIRE_DELAY + first[0]
        ok = hit[0] & (land < t_hit - DEFEND_MARGIN)
        if not ok.any():
            return None
        j = int(np.argmin(np.where(ok, land, np.inf)))
        return {"track": threat, "k": int(ks[j]), "covered": False}

    def _plan_evade(self, pos, heading, tarr, coast, coast_path, reach):
        """Best escape heading (float), "H" for hyperspace, or None to coast.

        coast, coast_path: first_hit() score and path of drifting without thrust;
        reach: each threat's ROM_REACH plus HYPER_PAD, for the check before a jump.
        """
        best_h, best_score = None, coast
        prev_score = None
        hs = [k * 256 / EVADE_HEADINGS for k in range(EVADE_HEADINGS)]
        paths = np.stack([self._sim(pos, heading, h, EVADE_THRUST) for h in hs])
        t_hits, clears = first_hits(paths, tarr)
        for h, t_hit, clear in zip(hs, t_hits, clears):
            score = int(t_hit), float(clear)
            if self.evade_h is not None and h == self.evade_h:
                prev_score = score
            if score > best_score:
                best_h, best_score = h, score
        # Hysteresis: stay with the previous escape heading if it is nearly as good.
        if prev_score is not None and prev_score[0] >= best_score[0] and prev_score[1] > best_score[1] - 50:
            best_h = self.evade_h
        self.evade_h = best_h
        if best_score[0] <= HYPER_T:
            # Jump only if the game's own collision test agrees nothing escapes;
            # otherwise take the option that it says clears longest.
            p, v, _ = tarr
            ts = rom_first_hits(np.concatenate((coast_path[None], paths)), p, v, reach)
            options = list(zip(ts.tolist(), [None] + hs))
            t_best, h_best = max(options, key=lambda o: o[0])
            if t_best <= HYPER_T:
                return "H"
            self.evade_h = h_best
            return h_best
        return best_h

    def _sim(self, pos, heading, target_h, thrust_frames):
        """simulate_ship from this frame's exact ship state."""
        vel16, counter = self.ship_sim
        return simulate_ship(pos, vel16, heading, target_h, thrust_frames, counter, self.last_thrust)

    def _shot_hits(self, targets, pos, vel, heading, ks, frac=HIT_FRAC):
        """Simulate firing after turning k steps, for each target and each k in ks.

        The shot appears delay = |k| + FIRE_DELAY frames from now, ahead of where the
        ship was the frame before, with the game's exact velocity and life. Returns
        (hit, first): boolean (targets, ks) and the frame of flight at which each shot
        first comes within frac of the game's hit reach for the target.
        """
        counter = self.ship_sim[1]
        hs = (heading + TURN_RATE * ks) % 256
        rel = BULLET_REL[hs.astype(int)]                                   # (nk, 2)
        sv = np.asarray(vel, float)
        bvel = np.clip(sv + rel, -BULLET_V_MAX, BULLET_V_MAX)               # (nk, 2)
        off = np.floor(rel / 2) + rel                                      # spawn offset
        delay = np.abs(ks) + FIRE_DELAY                                     # (nk,)
        p = np.array([t.pos for t in targets], float)[:, None, :]          # (m, 1, 2)
        v = np.array([(t.vx, t.vy) for t in targets], float)[:, None, :]
        # target - bullet at frame t of the bullet's flight (t = 0: it appears)
        d0 = (p - np.asarray(pos, float) - off - bvel + v * delay[:, None]
              - sv * (delay[:, None] - 1))                                  # (m, nk, 2)
        rv = v - bvel[None]                                                 # (m, nk, 2)
        life = bullet_life(counter + delay)                                 # (nk,)
        t = np.arange(life.max(), dtype=float)[None, None, :]
        dx = wrap_arr(d0[..., 0:1] + rv[..., 0:1] * t, W)                  # (m, nk, L)
        dy = wrap_arr(d0[..., 1:2] + rv[..., 1:2] * t, H)
        a = np.array([ROM_SHOT_REACH[tr.kind] for tr in targets], float)[:, None, None] * frac
        inside = octagon_hit(dx, dy, a) & (t < life[None, :, None])
        return inside.any(axis=2), np.argmax(inside, axis=2)

    def _pick_target(self, objs, pos, vel, heading):
        """Highest-priority uncovered target that a shot can hit, choosing the
        heading step that hits soonest. Returns a plan dict or None; its "now" says
        whether a shot from the current heading (k = 0) would hit a target of that
        class (higher classes have no hit at any k, so none at k = 0 either)."""
        bullets = [t for t in objs if t.kind == "B"]
        ks = np.arange(-MAX_TURN, MAX_TURN + 1)
        for ci, kinds in enumerate(PRIORITY):
            cands = [t for t in objs if t.kind in kinds and not self._is_covered(t, bullets)]
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
            return {"track": t, "k": k, "aim": aim, "class": ci,
                    "now": bool(hit[:, MAX_TURN].any())}
        self.target_uid = None
        return None

    def _expect_spawn(self, frame, pos, vel, heading):
        """Remember where this shot's bullet should appear, to check the model."""
        rel = bullet_rel(heading)
        p = tuple(pos[i] + vel[i] * (FIRE_DELAY - 1) + (rel[i] >> 1) + rel[i]
                  + max(-BULLET_V_MAX, min(BULLET_V_MAX, vel[i] + rel[i])) for i in (0, 1))
        self.pending_shots[frame + FIRE_DELAY] = (heading, p)

    def _check_spawn(self, frame, tracks):
        """Compare a predicted bullet spawn with RAM (the spawn_err stat)."""
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
        return (f"score={self.score} lives={self.lives} deaths={self.deaths} spent={self.spent} "
                f"shots={self.shots} acc={acc} spawn_err={spawn} {modes}")
