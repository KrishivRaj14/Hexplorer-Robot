# Old versions of wander.py (for filming / documentation)

Each file is an earlier version of `wander.py`, saved under its own name so it
can be run without touching the current `wander.py`. All of them end in
damping mode on Ctrl+C. Run any of them the same way, for example:

    python3 ~/wander_1_first_version.py --minutes 1

| File | Version (Nov 16, 2026) | What goes wrong (what to film) |
|---|---|---|
| `wander_1_first_version.py` | First LiDAR + camera wander | Walks fine, but next to a wall it stops: "too tight to turn" (turn-room rule too strict, network cable seen as an obstacle) |
| `wander_2_cable_ignored.py` | Cable ignored, side-steps | Better near walls; can stop with "floor not found" when facing furniture after a turn |
| `wander_3_floor_not_found.py` | Keeps the last floor for 10 s | Still stops "floor not found" if little floor is visible for longer |
| `wander_4_saved_floor.py` | Floor learned once, saved to a file | Refuses to start when no floor is visible after standing up (e.g. facing a door), until a floor has been saved |
| `wander_5_coworkers_band.py` | Previous owner's fixed LiDAR height band | Starts anywhere, but stops for "random" obstacles ~1-2 m ahead (the tilted floor) and over-turns (old camera pictures) |
| `wander_6_live_view_floor_bug.py` | + live view in the browser | Same floor bug as 5, but now visible: open http://192.168.1.10:8080 and see red "obstacle" rows on the empty floor ~2 m ahead |
| `../wander.py` (current) | LiDAR straightened with the IMU, camera only vetoes | The fix: the red rows are gone |

For the 2 m straight walk use the current `walk_safe.py` (same walking logic
as on Day 2, plus the Ctrl+C fix): `python3 ~/walk_safe.py 2.0`.
