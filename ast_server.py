# ast_server.py
# Python side of the lock-step link with MAME (ast_link.lua).
# Receives "FRAME <n> <nram> <projN>\n" + RAM bytes each emulated frame (zero
# page 0x00-0xFF, then object page 0x200-0x2FF), decodes the game's object table,
# tracks objects between frames to estimate velocity, lets the autopilot
# (ast_ai.Pilot) choose inputs, and replies
# "ACK <n> <inputs> K,slot,x,y[,px,py] ... A,x,y MODE:<mode>\n".
#   inputs: letters L R T F H (left right thrust fire hyperspace), C S (coin,
#           start), or "-"
#   px,py:  position projected projN frames ahead, only when projN > 0
#   A,x,y:  the autopilot's aim point
#
# Start this first:  python3 ast_server.py

import socket
import time
from collections import Counter, deque

from ast_ai import Pilot

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
ROCK_SIZE = {4: "L", 2: "M", 0: "S"}
SAUCER_SIZE = {2: "U", 1: "u"}

# Playfield size in game units; it wraps at the edges.
W, H = 8192, 6144

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
        f0, x0, y0 = self.hist[0]
        if frame > f0:
            # Average over the window: the game does not move every object on
            # every MAME frame, so single-frame deltas are jittery.
            self.vx = wrap_delta(x - x0, W) / (frame - f0)
            self.vy = wrap_delta(y - y0, H) / (frame - f0)

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


def serve(conn):
    rx = conn.makefile("rb")
    tracker = Tracker()
    pilot = Pilot()
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
        ram = rx.read(nram)
        if len(ram) < nram:
            break

        zp, page = ram[:256], ram[256:]
        objs = decode_objects(page)
        tracker.update(n, objs)
        ship_vel = (signed(page[VX + SHIP_SLOT]), signed(page[VY + SHIP_SLOT]))
        inputs, overlay, mode = pilot.step(n, tracker, SHIP_SLOT, zp, ship_vel)
        proj = tracker.predict(n, proj_n) if proj_n > 0 else {}
        tokens = []
        for slot, kind, x, y, *_ in objs:
            if slot in proj:
                tokens.append(f"{kind},{slot},{x},{y},{proj[slot][0]},{proj[slot][1]}")
            else:
                tokens.append(f"{kind},{slot},{x},{y}")
        tokens += overlay
        conn.sendall(f"ACK {n} {inputs} {' '.join(tokens)}\n".encode())

        if n % STATUS_EVERY == 0:
            t1 = time.time()
            print(f"{summary(n, tracker, proj_n, pilot, mode)}  {STATUS_EVERY / (t1 - t0):.1f} fps")
            t0 = t1


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"listening on {PORT}")
        conn, addr = s.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"MAME connected from {addr}")
        with conn:
            serve(conn)
        print("MAME disconnected")


if __name__ == "__main__":
    main()
