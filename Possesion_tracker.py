import os
import csv
import xml.etree.ElementTree as ET
import re

script_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(script_dir, "data", "tactical", "BAR-ATH_tactical_homography.csv")

atd_path = os.path.join(script_dir, "data", "tactical", "BAR-ATH_tactical_atd.txt")

ball_found_frames = []
total_frames = 0

with open(atd_path, "r") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        total_frames += 1
        
        parts = line.split(":")
        if len(parts) >= 3:
            ball_part = parts[2].strip()
            # If it's not all NaNs, the ball was tracked
            if "NaN,NaN,NaN" not in ball_part and ball_part != "":
                frame_id = int(parts[0].strip())
                ball_found_frames.append((frame_id, ball_part))

print(f"Total frames scanned: {total_frames}")
print(f"Frames with ball tracked: {len(ball_found_frames)}")

if ball_found_frames:
    print("\nFirst 3 frames with ball data:")
    for f_id, b_data in ball_found_frames[:3]:
        print(f"Frame {f_id}: {b_data}\n")



xml_path = os.path.join(script_dir, "data", "tactical", "BAR-ATH_tactical_atd.xml")
xml_path = "data/tactical/BAR-ATH_tactical_atd.xml"

tree = ET.parse(xml_path)
root = tree.getroot()


teams = root.findall(".//Team")
if teams:
    for id in teams:
         team_id = id.attrib.get("id")
         child_elim = id[0]
         team_name = child_elim.text.strip() if child_elim.text else child_elim.attrib.get("value")

         print(f"----------------------------------------------------------------------------------------------- \nClub Name: {team_name}\n----------------------------------------------------------------------------------------------- \n")
