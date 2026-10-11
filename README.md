# Asteroids Automated Play in MAME

This project contains Lua and Python scripts that enable automated play of Atari's *Asteroids* (1979) inside MAME.
A Lua script running in MAME sends game data to a Python server. The server decodes the game's object table and decides which controls
to press. Avoiding player ship collision is prioritized. Otherwise, the heuristics aim the player ship at the
highest-priority target and fire. The server sends MAME keypresses back to the MAME Lua script to control gameplay. Information overlays showing what the AI sees can be toggled.

## Tested with

- **MAME 0.270**
- **Asteroids (rev 4)**, MAME set `asteroid`
- **macOS 26.5.2** (Intel x86_64), **zsh 5.9** shell

It seems likely to work with other MAME versions, Asteroids ROM revisions, operating systems, and/or
shells. Let me know how it goes. I struggled to find working examples of MAME Lua<>Python socket communication.  The approach here works well and you are welcome to extend it as you like.  I used Claude Code to help me clear the way.

## Files

| File            | Purpose                                                                                                |
|-----------------|--------------------------------------------------------------------------------------------------------|
| `ast_link.lua`  | MAME autoboot Lua script                                                                               |
| `ast_server.py` | Python server. Handles local socket communication, decodes and tracks objects, and calls the autopilot |
| `ast_ai.py`     | The autopilot: threat check, evasion, target choice and aiming                                         |
| `cfg/`          | MAME configuration for `asteroid`                                                                      |

## Requirements

- Python 3 with `numpy` (`pip3 install numpy`)
- A MAME build (tested with version 0.270)
- Asteroids (rev 4) ROM set, `asteroid.zip`, in `roms/`. You will have to find your own ROM set.
  **The ROMs are not included.**

## Run

In a terminal window, start the Python server first:

```zsh
python3 ast_server.py
```

In a second terminal, start MAME using:

```zsh
mame asteroid -rompath ../roms -window -autoboot_script ../ast_link.lua
```

Make sure to edit `-rompath ../roms` to match your configuration.

The Python autopilot script inserts a coin, starts the game, and restarts it after a game
ends. If the Python server doesn't reply within 2 seconds, MAME stops waiting for it.

## Keys (in MAME)

| Key | Action |
|-----|--------|
| `I` | Autopilot on/off (off = you play) |
| `B` | Object boxes and aim line on/off |
| `O` | Object list on/off |
| `V` | Projected positions on/off |
| `]` / `[` | Project further / less far ahead (5–300 frames, step 5) |


## Object Detection Overlay 

<img width="1541" height="1112" style="width: 50%; height: auto;" alt="Screenshot 2026-10-11 at 11 18 47 AM" src="https://github.com/user-attachments/assets/746a8c26-4568-41e8-a89b-fb4a9f1210b1" title="Object Detection Overlay" />
<img width="1541" height="1112" alt="Screenshot 2026-10-11 at 11 17 15 AM" src="https://github.com/user-attachments/assets/36e4519a-c83f-42b8-bc36-6874749fa58a" />
<img width="1541" height="1112" alt="Screenshot 2026-10-11 at 11 15 01 AM" src="https://github.com/user-attachments/assets/65cd985e-1e08-4847-9eb9-ec8eb6bb923d" />
