# Asteroids Automated Play in MAME

An autopilot that plays Atari's *Asteroids* (1979) inside the MAME emulator.
A Lua script running in MAME sends the game's RAM to a Python server once per
emulated frame and waits for a reply, so the emulator and the AI stay in
lock-step. The server decodes the game's object table, tracks rocks, saucers and
bullets between frames to estimate their velocities, and decides which controls
to press. It avoids incoming collisions first. Otherwise it aims at the
highest-priority target (saucers, then small, medium and large rocks) and fires.
MAME presses those controls and draws an overlay showing what the AI sees.

## Tested with

- **MAME 0.270**
- **Asteroids (rev 4)**, MAME set `asteroid`
- **macOS 26.5.2** (Intel x86_64), **zsh 5.9** shell

It might work with other MAME versions, ROM revisions, operating systems or
shells, but those haven't been tested.

## Files

| File | Purpose |
|------|---------|
| `ast_link.lua` | MAME autoboot script: sends RAM each frame, applies the inputs it gets back, draws the overlay |
| `ast_server.py` | Python lock-step server (127.0.0.1:9999): decodes and tracks objects, calls the autopilot |
| `ast_ai.py` | The autopilot: threat check, evasion, target choice and aiming |
| `cfg/` | MAME configuration for `asteroid` |

## Requirements

- Python 3 with `numpy` (`pip3 install numpy`)
- A MAME build (0.270 tested), in `mame/` next to these files
- Your own Asteroids (rev 4) ROM set, `asteroid.zip`, in `roms/`.
  **The ROMs are not included.**

## Run

In one terminal, start the server first:

```zsh
python3 ast_server.py
```

In a second terminal, start MAME from `mame/`:

```zsh
cd mame
./mame asteroid -rompath ../roms -window -autoboot_script ../ast_link.lua
```

The autopilot inserts a coin, starts the game, and restarts it after a game
over. If the server doesn't reply within 2 seconds, MAME stops waiting for it.

## Keys (in MAME)

| Key | Action |
|-----|--------|
| `I` | Autopilot on/off (off = you play) |
| `B` | Object boxes and aim line on/off |
| `O` | Object list on/off |
| `V` | Projected positions on/off |
| `]` / `[` | Project further / less far ahead (5–300 frames, step 5) |
