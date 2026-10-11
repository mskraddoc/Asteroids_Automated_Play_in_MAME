# Asteroids Automated Play in MAME

LUA and Python scripts that enable automated play of Atari's *Asteroids* (1979) inside MAME.
A Lua script running in MAME sends game data to a Python server once per
emulated frame and waits for a reply. The emulator and the python server stay in
lock-step. The server decodes the game's object table, tracks rocks, saucers and
bullets between frames to estimate their velocities, and decides which controls
to press. Avoiding player ship collision is prioritized. Otherwise, the heuristics aim at the
highest-priority target (saucers, then small, medium and large rocks) and fires.
The server sends MAME keypresses presses back to the MAME Lua script to control gameplay. Information overlays showing what the AI sees can be toggled.

## Tested with

- **MAME 0.270**
- **Asteroids (rev 4)**, MAME set `asteroid`
- **macOS 26.5.2** (Intel x86_64), **zsh 5.9** shell

It seems liekly to work with other MAME versions, Asteroids ROM revisions, operating systems, and/or
shells. Let me know how it goes.

## Files

| File            | Purpose                                                                                                |
|-----------------|--------------------------------------------------------------------------------------------------------|
| `ast_link.lua`  | MAME autoboot Lua script                                                                               |
| `ast_server.py` | Python server. Handles local socket communication, decodes and tracks objects, and calls the autopilot |
| `ast_ai.py`     | The autopilot: threat check, evasion, target choice and aiming                                         |
| `cfg/`          | MAME configuration for `asteroid`                                                                      |


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
