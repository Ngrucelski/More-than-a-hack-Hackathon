import os
import math
import xml.etree.ElementTree as ET
from collections import Counter
import csv as csv_module

import numpy as np

# ----------------------------------------------------------------------
# 1. PATHS  (one clear name per thing, so nothing gets overwritten)
# ----------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data", "tactical")

xml_path = os.path.join(data_dir, "BAR-ATH_tactical_atd.xml")        # metadata (header)
atd_txt_path = os.path.join(data_dir, "BAR-ATH_tactical_atd.txt")    # tracking data (body)

# csv_path = os.path.join(data_dir, "BAR-ATH_tactical_homography.csv")  # not used yet

# ----------------------------------------------------------------------
# 2. SETTINGS  (everything you might tune lives here)
# ----------------------------------------------------------------------
BALL_R = 0.11            # m, from the regulation circumference (~69 cm / pi / 2) 
PLAYER_R = 1.0668        # m, set this from the histogram "knee" (see SHOW_HISTOGRAM)
CONTEST_MODE = "freeze"  # "exclude", "freeze", or "incumbent"
INCUMBENT_MARGIN = 0.75  # m, challenger must be this much closer (incumbent mode)
MAX_GAP_SECONDS = 5      # carry the last possessor through gaps up to this long
MIN_STREAK = 3           # frames a team must hold a clean overlap before taking over
MIN_TRACKED = 0          # treat frames with fewer tracked players as unreliable (0 = off)
BALL_MAX_KPH = None      # e.g. 30 to ignore a fast ball (in flight); None = off

SHOW_HISTOGRAM = False    # distance-to-nearest-player histogram, to choose PLAYER_R
RUN_SENSITIVITY = True   # run several radii and modes and print a comparison table

# ----------------------------------------------------------------------
# 3. READ THE XML HEADER: teams, players, field size, frame rate
# ----------------------------------------------------------------------
tree = ET.parse(xml_path)
atd_root = tree.getroot()

# playerId -> teamId (players are NOT nested inside teams; they point to them)
lookup = {}
for p in atd_root.iterfind(".//Players/Player"):
    lookup[p.get("id")] = p.get("teamId")

# Home/Away from the Score element
score = atd_root.find(".//Score")
local_id = score.get("idLocalTeam")
away_id = score.get("idVisitingTeam")
team_label = {local_id: "Home", away_id: "Away"}
player_team = {pid: team_label[tid] for pid, tid in lookup.items()}

W = float(atd_root.findtext(".//FieldSize/Width"))    # x scale (pitch length)
H = float(atd_root.findtext(".//FieldSize/Height"))   # y scale (pitch width)
FPS = float(atd_root.findtext(".//FrameRate"))
MAX_GAP = int(FPS * MAX_GAP_SECONDS)

print("Players per team:", Counter(player_team.values()), "| total:", len(player_team))
print("Field size:", W, "x", H, "| FPS:", FPS)

# ----------------------------------------------------------------------
# 4. PARSE THE TRACKING FILE
#    line = frame : player1 ; player2 ; ... ; player46 : ball
#    each entry = x , y , s   (x, y normalized 0-1; s = speed in kph)
# ----------------------------------------------------------------------
def to_float(s):
    try:
        v = float(s)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def parse_frame(line):
    parts = line.strip().split(":")
    players = []   # (team, x_m, y_m)
    for i, entry in enumerate(parts[1].strip(";").split(";")):
        vals = entry.split(",")
        if len(vals) < 2:
            continue
        x, y = to_float(vals[0]), to_float(vals[1])
        if x is not None and y is not None:
            players.append((player_team[str(i + 1)], x * W, y * H))

    ball = None    # (x_m, y_m, speed_kph or None)
    if len(parts) > 2:
        bv = parts[2].strip(";").split(",")
        if len(bv) >= 2:
            bx, by = to_float(bv[0]), to_float(bv[1])
            bs = to_float(bv[2]) if len(bv) > 2 else None
            if bx is not None and by is not None:
                ball = (bx * W, by * H, bs)
    return players, ball


# Parse once and keep in memory so repeated runs are fast.
# (If memory is an issue, re-read the file inside run_possession instead.)
frames = []
with open(atd_txt_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.strip():
            frames.append(parse_frame(line))

n_frames = len(frames)
n_ball = sum(1 for _, b in frames if b is not None)
avg_players = sum(len(p) for p, _ in frames) / n_frames
print(f"Frames: {n_frames} | with ball: {n_ball} ({100*n_ball/n_frames:.1f}%) "
      f"| avg players with data: {avg_players:.1f}")
print(f"Match length: {n_frames / FPS / 60:.1f} min")

# ----------------------------------------------------------------------
# 5. OPTIONAL: HISTOGRAM OF BALL -> NEAREST PLAYER DISTANCE
#    Look for where the bars stop falling steeply: that "knee" is a
#    data-driven value for PLAYER_R + BALL_R.
# ----------------------------------------------------------------------
if SHOW_HISTOGRAM:
    step = 0.25
    bins = np.arange(0, 8 + step, step)
    hist = np.zeros(len(bins) - 1, dtype=int)
    for players, ball in frames:
        if ball and players:
            d = min(math.hypot(x - ball[0], y - ball[1]) for _, x, y in players)
            if d < bins[-1]:
                hist[int(d // step)] += 1
    print("\nDistance from ball to nearest player:")
    for lo, c in zip(bins[:-1], hist):
        bar = "#" * int(60 * c / max(1, hist.max()))
        print(f"{lo:4.2f}-{lo + step:4.2f} m  {c:7d}  {bar}")

# ----------------------------------------------------------------------
# 6. POSSESSION
# ----------------------------------------------------------------------
def classify(players, ball, radius):
    """Returns (status, info): 'none', 'clean' (team), or 'contested' (distances)."""
    near = []
    for team, x, y in players:
        d = math.hypot(x - ball[0], y - ball[1])
        if d <= radius:
            near.append((d, team))
    if not near:
        return "none", None
    teams = {t for _, t in near}
    if len(teams) == 1:
        return "clean", teams.pop()
    return "contested", near


def run_possession(player_r=PLAYER_R, mode=CONTEST_MODE, verbose=True):
    radius = player_r + BALL_R
    counts = Counter()    # frames credited to each team
    carried = Counter()   # of those, frames that were carried forward (assumed, not measured)
    stats = Counter()
    last_team, gap = None, 0
    cand, streak = None, 0

    for players, ball in frames:
        usable = (ball is not None and len(players) > 0
                  and len(players) >= MIN_TRACKED)
        if usable and BALL_MAX_KPH is not None and ball[2] is not None and ball[2] > BALL_MAX_KPH:
            usable = False   # ball moving too fast: probably a pass/shot

        if not usable:
            stats["no_ball_or_unreliable"] += 1
            gap += 1
            cand, streak = None, 0
        else:
            status, info = classify(players, ball, radius)
            stats[status] += 1

            if status == "clean":
                if info == last_team:
                    cand, streak, gap = None, 0, 0
                else:
                    streak = streak + 1 if cand == info else 1
                    cand = info
                    if last_team is None or streak >= MIN_STREAK:
                        last_team, cand, streak, gap = info, None, 0, 0

            elif status == "contested":
                cand, streak = None, 0
                if mode == "exclude":
                    continue   # counts for nobody (also left out of the denominator)
                if mode == "incumbent" and last_team:
                    best = {}
                    for d, t in info:
                        best[t] = min(best.get(t, 9e9), d)
                    others = [t for t in best if t != last_team]
                    if (others and last_team in best
                            and best[others[0]] + INCUMBENT_MARGIN < best[last_team]):
                        last_team = others[0]
                gap = 0   # freeze / incumbent: keep the ball with last_team

            else:  # "none": nobody overlaps the ball
                gap += 1
                cand, streak = None, 0

        if last_team and gap <= MAX_GAP:
            counts[last_team] += 1
            if gap > 0:
                carried[last_team] += 1

    total = sum(counts.values())
    pct = {t: 100 * c / total for t, c in counts.items()} if total else {}

    if verbose:
        ball_frames = stats["none"] + stats["clean"] + stats["contested"]
        print(f"\n--- radius {player_r:.2f}+{BALL_R:.2f} m | mode={mode} ---")
        print("Frame status:", dict(stats))
        print("Contested share of ball frames:",
              f"{100 * stats['contested'] / max(1, ball_frames):.1f}%")
        for t in ("Home", "Away"):
            c = counts[t]
            share_carried = 100 * carried[t] / c if c else 0
            print(f"{t}: {c} frames, {pct.get(t, 0):.1f}%  "
                  f"({share_carried:.0f}% of these were carried forward)")
    return pct


# ----------------------------------------------------------------------
# 7. RUN
# ----------------------------------------------------------------------
run_possession()

if RUN_SENSITIVITY:
    print("\n=== Sensitivity: Home possession % ===")
    radii = [PLAYER_R - 0.3, PLAYER_R, PLAYER_R + 0.3]
    modes = ["exclude", "freeze", "incumbent"]
    print(f"{'PLAYER_R':>9} " + " ".join(f"{m:>10}" for m in modes))
    for r in radii:
        row = [run_possession(r, m, verbose=False).get("Home", float("nan")) for m in modes]
        print(f"{r:9.2f} " + " ".join(f"{v:10.1f}" for v in row))
    print("\nReport the range across this table, not a single number.") 

MAX_GAP_FRAMES = 50      # None = every gap; e.g. 50 = only short gaps (more likely noise)
OFFSET_SECONDS = 0.0     # add/subtract to line up with the video (see notes below)

FPS = float(atd_root.findtext(".//FrameRate"))

# real frame labels from the first ':' section of each line
labels = []
with open(atd_txt_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.strip():
            labels.append(line.split(":")[0].strip())
try:
    labels = [int(x) for x in labels]
except ValueError:
    print("Frame labels aren't plain integers, using line positions instead.")
    labels = list(range(len(frames)))
assert len(labels) == len(frames), "label count doesn't match frames list"

def stamp(label):
    s = (label - labels[0]) / FPS + OFFSET_SECONDS
    sign = "-" if s < 0 else ""
    s = abs(s)
    return f"{sign}{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:05.2f}"

intervals = []
start = None
for i, (_, ball) in enumerate(frames):
    if ball is None and start is None:
        start = i
    elif ball is not None and start is not None:
        intervals.append((start, i - 1))
        start = None
if start is not None:
    intervals.append((start, len(frames) - 1))

with open(os.path.join(script_dir, "missing_ball_intervals.txt"), "w") as f:
    for s, e in intervals:
        if MAX_GAP_FRAMES is None or (e - s + 1) <= MAX_GAP_FRAMES:
            line = f"[{labels[s]}, {labels[e]}]  {stamp(labels[s])} -> {stamp(labels[e])}  ({e - s + 1} frames)"
            print(line)
            f.write(line + "\n")

# line positions in the tracking file (find them with the notes below)
H1_START, H1_END = 0, 0      # first half:  kickoff -> final whistle
H2_START, H2_END = 0, 0      # second half: kickoff -> final whistle

PAD = int(FPS * 60)          # 60 s of empty frames between the halves (longer than any gap setting)

frames_full = frames
first_half = frames_full[H1_START:H1_END + 1]
second_half = frames_full[H2_START:H2_END + 1]
frames = first_half + [([], None)] * PAD + second_half

print(f"first half: {len(first_half)} frames ({len(first_half) / FPS / 60:.1f} min)")
print(f"second half: {len(second_half)} frames ({len(second_half) / FPS / 60:.1f} min)")
print(f"dropped: {len(frames_full) - len(first_half) - len(second_half)} frames")

 