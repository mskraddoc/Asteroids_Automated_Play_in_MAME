# ast_server.py
# Python side of the lock-step link with MAME (ast_link.lua).
# Receives "FRAME <n> <nram> <projN> [<style_step>]\n" + RAM bytes each emulated frame (zero
# page 0x00-0xFF, then object page 0x200-0x2FF), decodes the game's object table,
# tracks objects between frames to estimate velocity, lets the autopilot
# (ast_ai.Pilot) choose inputs, and replies
# "ACK <n> <inputs> K,slot,x,y[,px,py] ... A,x,y MODE:<mode> G,...\n".
#   inputs: letters L R T F H (left right thrust fire hyperspace), C S (coin,
#           start), or "-"
#   px,py:  position projected projN frames ahead, only when projN > 0
#   A,x,y:  the autopilot's aim point
#   G,score,ships,level,deaths,hyper,coll,shot,kami,jumps: total score (no
#           rollover at 99,990), ships left, wave number, ships lost this game, the
#           losses by cause (failed hyperspace, collision with a rock or saucer,
#           saucer shot, spent over the ship cap), hyperspace jumps this game
#   style_step: 1 on the frame the style key is pressed (next play style), else 0
#   PET,slot: the shadow style's spared boulder
#   SEED,lo,hi: sent once at each game start; Lua writes it into the game's
#           random-number generator (zero page 0x5F/0x60)
#
# Start this first:  python3 ast_server.py [--random on|off|<seed>] [--style <style>]
#   on (default): pick a fresh seed from the clock and OS entropy, printed so
#                 the session can be replayed; off: never touch the game's RNG.

import argparse
import hashlib
import os
import random
import socket
import time
from collections import Counter, deque

from ast_ai import LIVES_CAP, STYLES, Pilot, bcd

HOST, PORT = "127.0.0.1", 9999
STATUS_EVERY = 60  # log a status line every N frames

# Asteroids object table (RAM page 0x0200), indexed by slot.
ROCK_SLOTS = range(0, 27)
SHIP_SLOT = 27
SAUCER_SLOT = 28
SAUCER_BULLET_SLOTS = range(29, 31)
PLAYER_BULLET_SLOTS = range(31, 35)
X_HI, Y_HI, X_LO, Y_LO = 0x69, 0x8C, 0xAF, 0xD2
VX, VY = 0x23, 0x46  # signed velocity bytes, units/frame
RNG_LO, RNG_HI = 0x5F, 0x60  # the game's 16-bit random-number generator (never 0)
HYPER = 0x59  # hyperspace: 0 none, 0x01 jump in progress, 0x80 jump that will kill the ship
ROCK_SIZE = {4: "L", 2: "M", 0: "S"}
SAUCER_SIZE = {2: "U", 1: "u"}

# Playfield size in game units; it wraps at the edges.
W, H = 8192, 6144

# Ways a ship is lost, in overlay order: failed hyperspace, collision with a rock
# or saucer, saucer shot, spent over the ship cap (kamikaze)
CAUSES = ("hyper", "coll", "shot", "kami")

HISTORY = 8       # frames of position history kept per object
MAX_JUMP = 400    # per-frame move larger than this restarts the track


def decode_objects(ram):
    """Return [(slot, kind, x, y, status, vx, vy)] for live objects.

    x is 0-8191, y is 0-6143 (up); vx, vy are the game's own signed velocity
    bytes (units/frame). For bullets, status is the remaining-life countdown.

    kind: L/M/S rock, H ship, U/u large/small saucer, R saucer bullet, B player bullet.
    Empty (status 0) and exploding (status >= 0x80) slots are skipped.
    """
    objs = []
    for i in range(PLAYER_BULLET_SLOTS.stop):
        st = ram[i]
        if st == 0 or st >= 0x80:
            continue
        if i in ROCK_SLOTS:
            kind = ROCK_SIZE[st & 0x06]
        elif i == SHIP_SLOT:
            kind = "H"
        elif i == SAUCER_SLOT:
            kind = SAUCER_SIZE.get(st & 0x03)
            if kind is None:
                continue
        elif i in SAUCER_BULLET_SLOTS:
            kind = "R"
        else:
            kind = "B"
        x = ram[X_HI + i] * 256 + ram[X_LO + i]
        y = ram[Y_HI + i] * 256 + ram[Y_LO + i]
        objs.append((i, kind, x, y, st, signed(ram[VX + i]), signed(ram[VY + i])))
    return objs


def signed(b):
    return b - 256 if b >= 128 else b


def wrap_delta(d, size):
    """Shortest signed distance on a wrapping axis."""
    return (d + size // 2) % size - size // 2


class Track:
    """Recent history of one RAM slot's object, with estimated velocity."""

    _next_uid = 0

    def __init__(self, kind):
        Track._next_uid += 1
        self.uid = Track._next_uid  # distinguishes successive objects in one slot
        self.kind = kind
        self.hist = deque(maxlen=HISTORY)  # (frame, x, y)
        self.vx = self.vy = 0.0
        self.status = 0
        self.ram_v = (0, 0)  # the game's velocity bytes

    def add(self, frame, x, y, status=0, ram_v=(0, 0)):
        self.status = status
        self.ram_v = ram_v
        self.hist.append((frame, x, y))
        # The game's own velocity bytes match the measured motion exactly (rocks,
        # bullets; saucers to within 0.3 units/frame as they change course) and,
        # unlike a position average, are right from an object's first frame.
        self.vx, self.vy = float(ram_v[0]), float(ram_v[1])

    @property
    def pos(self):
        return self.hist[-1][1], self.hist[-1][2]


def project(track, n):
    """Position n frames ahead assuming constant velocity, wrapped to the playfield."""
    x, y = track.pos
    return round(x + track.vx * n) % W, round(y + track.vy * n) % H


class Tracker:
    """Keeps one Track per RAM slot across frames, and scores its own predictions."""

    def __init__(self):
        self.tracks = {}
        self.ended = []    # tracks that disappeared this frame
        self.pending = {}  # target frame -> [(slot, uid, px, py)]
        self.err_sum = 0.0
        self.err_n = 0

    def update(self, frame, objs):
        seen = set()
        self.ended = []
        for slot, kind, x, y, status, rvx, rvy in objs:
            seen.add(slot)
            old = t = self.tracks.get(slot)
            if t is not None and t.kind == kind:
                px, py = t.pos
                if max(abs(wrap_delta(x - px, W)), abs(wrap_delta(y - py, H))) > MAX_JUMP:
                    t = None
            else:
                t = None
            if t is None:
                if old is not None:
                    self.ended.append(old)  # slot reused by a new object
                t = self.tracks[slot] = Track(kind)
            t.add(frame, x, y, status, (rvx, rvy))
        for slot in list(self.tracks):
            if slot not in seen:
                self.ended.append(self.tracks.pop(slot))
        self._score(frame)

    def predict(self, frame, n):
        """Projected positions n frames ahead; remembered so they can be scored later."""
        out = {}
        for slot, t in self.tracks.items():
            out[slot] = project(t, n)
        if n > 0:
            self.pending[frame + n] = [(s, self.tracks[s].uid, *p) for s, p in out.items()
                                       if len(self.tracks[s].hist) == HISTORY]
        return out

    def _score(self, frame):
        for slot, uid, px, py in self.pending.pop(frame, []):
            t = self.tracks.get(slot)
            if t is None or t.uid != uid:
                continue  # object destroyed or replaced; prediction not comparable
            x, y = t.pos
            self.err_sum += max(abs(wrap_delta(x - px, W)), abs(wrap_delta(y - py, H)))
            self.err_n += 1
        for f in [f for f in self.pending if f < frame]:
            del self.pending[f]  # projection length changed; drop stale entries

    def take_error(self):
        """Mean prediction error (game units) since the last call, or None."""
        if not self.err_n:
            return None
        e = self.err_sum / self.err_n
        self.err_sum, self.err_n = 0.0, 0
        return e


class Scorekeeper:
    """Game total score (without the arcade's rollover), ships left, level, and
    how each ship was lost.

    The level counts waves: rocks reach zero between waves, so each change from
    no rocks to some rocks during a game is a new wave. At game start the attract
    rocks are cleared before the first wave appears.
    """

    MAX_SCORE = 9_999_999_999

    def __init__(self):
        self.total = 0
        self.ships = 0
        self.level = 0
        self.prev_score = 0
        self.had_rocks = True
        self.started = False  # True on the frame a game starts
        self.ship_start = 0    # total when the current ship began
        self.ship_points = []  # points earned by each ship lost this game
        self.causes = dict.fromkeys(CAUSES, 0)  # ships lost this game, by cause
        self.jumps = 0         # hyperspace jumps this game
        self.prev_hyper = 0

    def update(self, zp, objs, ended):
        """ended: tracks that disappeared this frame (Tracker.ended)."""
        score = (bcd(zp[0x53]) * 100 + bcd(zp[0x52])) * 10
        ships = zp[0x57]
        rocks = any(o[0] in ROCK_SLOTS for o in objs)
        self.started = bool(ships and not self.ships)
        if self.started:
            self.total = self.level = 0
            self.ship_start = 0
            self.ship_points = []
            self.causes = dict.fromkeys(CAUSES, 0)
            self.jumps = 0
            self.prev_score = score
        if ships or self.ships:
            # The arcade score wraps at 100,000; a large "drop" is a reset, not points.
            delta = (score - self.prev_score) % 100_000
            if delta < 50_000:
                self.total = min(self.total + delta, self.MAX_SCORE)
            if rocks and not self.had_rocks:
                self.level += 1
            if ships < self.ships:  # a ship lost (an extra ship earned doesn't count)
                self.ship_points.append(self.total - self.ship_start)
                self.ship_start = self.total
                self.causes[self._cause(ended)] += 1
            if zp[HYPER] and not self.prev_hyper:
                self.jumps += 1
        self.prev_hyper = zp[HYPER]
        self.prev_score = score
        self.ships = ships
        self.had_rocks = rocks

    def _cause(self, ended):
        """Why the ship was just lost. A saucer bullet that hits the ship is removed
        on that frame; a rock or saucer it hits starts exploding (also gone from the
        object list). A failed jump kills the ship on return, the frame after the
        hyperspace flag was 0x80."""
        if self.ships > LIVES_CAP:
            return "kami"
        if self.prev_hyper == 0x80:
            return "hyper"
        if any(t.kind == "R" for t in ended):
            return "shot"
        return "coll"

    @property
    def current_ship_points(self):
        """Points the ship in play has earned; after game over, the last ship's."""
        if not self.ships and self.ship_points:
            return self.ship_points[-1]
        return self.total - self.ship_start

    def ship_stats(self):
        """Status-line text: this ship's points, then avg and max per ship once a
        ship has been lost, then the last-3 average once three have been lost."""
        s = (f"causes={'/'.join(str(n) for n in self.causes.values())}"
             f" jumps={self.jumps} ship={self.current_ship_points:,}")
        lost = self.ship_points
        if lost:
            s += (f" avg/ship={round(sum(lost) / len(lost)):,}"
                  f" max/ship={max(lost + [self.current_ship_points]):,}")
        if len(lost) >= 3:
            s += f" last3={round(sum(lost[-3:]) / 3):,}"
        return s

    def token(self):
        causes = ",".join(str(n) for n in self.causes.values())
        return f"G,{self.total},{self.ships},{self.level},{len(self.ship_points)},{causes},{self.jumps}"


def summary(n, tracker, proj_n, pilot, mode):
    tracks = tracker.tracks.values()
    counts = Counter(t.kind for t in tracks)
    ship = tracker.tracks.get(SHIP_SLOT)
    if ship:
        ship_s = f"ship({ship.pos[0]},{ship.pos[1]} v={ship.vx:+.1f},{ship.vy:+.1f})"
    else:
        ship_s = "ship(-)"
    rocks = [t for t in tracks if t.kind in "LMS"]
    rock_v = sum((t.vx ** 2 + t.vy ** 2) ** 0.5 for t in rocks) / len(rocks) if rocks else 0.0
    rest = " ".join(f"{k}{counts[k]}" for k in "UuLMSRB")
    s = f"f{n} {ship_s} {rest} rock_v={rock_v:.1f} {mode} | {pilot.stats()}"
    if proj_n:
        err = tracker.take_error()
        s += f" proj={proj_n}" + (f" err={err:.0f}u" if err is not None else "")
    return s


def serve(conn, seed, style):
    """seed: None to leave the game's RNG alone, else seeds each game's RNG value."""
    rng = random.Random(seed) if seed is not None else None
    games = 0
    rx = conn.makefile("rb")
    tracker = Tracker()
    pilot = Pilot(style, seed if seed is not None else 0)
    keeper = Scorekeeper()
    t0 = time.time()
    while True:
        line = rx.readline()
        if not line:
            break  # MAME disconnected
        parts = line.decode().split()
        if len(parts) < 4 or parts[0] != "FRAME":
            print(f"unexpected message: {line!r}")
            continue
        n, nram, proj_n = int(parts[1]), int(parts[2]), int(parts[3])
        if len(parts) > 4 and parts[4] == "1":
            pilot.set_style(STYLES[(STYLES.index(pilot.style) + 1) % len(STYLES)])
            print(f"style -> {pilot.style}")
        ram = rx.read(nram)
        if len(ram) < nram:
            break

        zp, page = ram[:256], ram[256:]
        objs = decode_objects(page)
        tracker.update(n, objs)
        keeper.update(zp, objs, tracker.ended)
        ship_vel = (signed(page[VX + SHIP_SLOT]), signed(page[VY + SHIP_SLOT]))
        inputs, overlay, mode = pilot.step(n, tracker, SHIP_SLOT, zp, ship_vel,
                                           page[SAUCER_SLOT], keeper.level)
        proj = tracker.predict(n, proj_n) if proj_n > 0 else {}
        tokens = []
        for slot, kind, x, y, *_ in objs:
            if slot in proj:
                tokens.append(f"{kind},{slot},{x},{y},{proj[slot][0]},{proj[slot][1]}")
            else:
                tokens.append(f"{kind},{slot},{x},{y}")
        tokens += overlay
        tokens.append(keeper.token())
        if keeper.started:
            games += 1
            if rng:
                game_seed = rng.randrange(1, 0x10000)
                tokens.append(f"SEED,{game_seed & 0xFF},{game_seed >> 8}")
            else:
                game_seed = zp[RNG_HI] << 8 | zp[RNG_LO]
            print(f"game {games} start: RNG 0x{game_seed:04X}" + ("" if rng else " (game's own)"))
        conn.sendall(f"ACK {n} {inputs} {' '.join(tokens)}\n".encode())

        if n % STATUS_EVERY == 0:
            t1 = time.time()
            print(f"{summary(n, tracker, proj_n, pilot, mode)} total={keeper.total} level={keeper.level}"
                  f" {keeper.ship_stats()}"
                  f"  {STATUS_EVERY / (t1 - t0):.1f} fps")
            t0 = t1


def clock_seed():
    """A seed mixing the clocks, the process id and OS entropy."""
    h = hashlib.sha256()
    for v in (time.time_ns(), time.perf_counter_ns(), time.monotonic_ns(), os.getpid()):
        h.update(v.to_bytes(16, "little"))
    h.update(os.urandom(16))
    return int.from_bytes(h.digest()[:6], "little")  # 48 bits: short enough to retype


def parse_random(arg):
    """--random value -> seed int, or None for off."""
    if arg == "off":
        return None
    if arg == "on":
        return clock_seed()
    try:
        seed = int(arg, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("expected on, off or a seed number")
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be >= 0")
    return seed


def main():
    ap = argparse.ArgumentParser(description="Asteroids autopilot server for ast_link.lua")
    ap.add_argument("--random", default="on", type=parse_random, metavar="on|off|SEED",
                    help="randomize each game's starting RNG (default on); a number replays a session")
    ap.add_argument("--style", default="classic", choices=STYLES,
                    help="autopilot play style (default classic); the M key in MAME cycles it")
    args = ap.parse_args()
    seed = args.random
    print(f"style: {args.style}")
    if seed is None:
        print("random: off (the game's RNG runs as the ROM leaves it)")
    else:
        print(f"random: seed {seed} (replay with --random {seed})")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"listening on {PORT}")
        conn, addr = s.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"MAME connected from {addr}")
        with conn:
            serve(conn, seed, args.style)
        print("MAME disconnected")


if __name__ == "__main__":
    main()
