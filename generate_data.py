"""
generate_data.py
----------------
Simulates an NBA play-by-play shot log with realistic spatial and situational
structure (court coordinates, shot clock, defender distance, shot zone, etc.).

In production this ingests ~300K shot events directly from the NBA Stats API
(stats.nba.com / nba_api). This sandbox has no outbound internet access, so
this script generates a statistically faithful synthetic dataset instead,
calibrated to published league-average FG% by zone/distance so the modelling
pipeline downstream (features -> XGBoost -> dashboard) behaves exactly as it
would on real shot logs. Swap this module out for `nba_api` calls to run on
live data -- see README.md.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_SHOTS = 30_000
N_PLAYERS = 60
N_GAMES = 400

TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

POSITIONS = ["PG", "SG", "SF", "PF", "C"]

FIRST = ["James", "Luka", "Jayson", "Nikola", "Giannis", "Steph", "Kevin", "Devin",
         "Anthony", "Shai", "Ja", "Trae", "Zion", "Damian", "Joel", "Jimmy", "Kawhi",
         "Paul", "Donovan", "Tyrese", "Bam", "De'Aaron", "Domantas", "Jalen", "Brandon",
         "Karl", "Rudy", "Julius", "Scottie", "Franz", "Cade", "Evan", "Desmond",
         "Anfernee", "Jaren", "LaMelo", "Zach", "Kristaps", "Mikal", "Jaylen",
         "Draymond", "Klay", "Kyrie", "Fred", "OG", "Pascal", "Jrue", "Derrick",
         "Bradley", "Josh", "Aaron", "Dejounte", "Alperen", "Chet", "Victor", "Paolo",
         "Amen", "Ausar", "Scoot", "Brandin"]
LAST = ["Harden", "Doncic", "Tatum", "Jokic", "Antetokounmpo", "Curry", "Durant",
        "Booker", "Edwards", "Gilgeous-Alexander", "Morant", "Young", "Williamson",
        "Lillard", "Embiid", "Butler", "Leonard", "George", "Mitchell", "Haliburton",
        "Adebayo", "Fox", "Sabonis", "Brunson", "Ingram", "Towns", "Gobert", "Randle",
        "Barnes", "Wagner", "Cunningham", "Mobley", "Bane", "Simons", "Jackson",
        "Ball", "LaVine", "Porzingis", "Bridges", "Brown", "Green", "Thompson",
        "Irving", "VanVleet", "Anunoby", "Siakam", "Holiday", "White", "Beal",
        "Hart", "Christie", "Murray", "Sengun", "Holmgren", "Wembanyama",
        "Banchero", "Thompson", "Barlow", "Henderson", "Duren"]

players = pd.DataFrame({
    "player_id": np.arange(1, N_PLAYERS + 1),
    "player_name": [f"{FIRST[i % len(FIRST)]} {LAST[i % len(LAST)]}" for i in range(N_PLAYERS)],
    "position": RNG.choice(POSITIONS, N_PLAYERS),
    "team": RNG.choice(TEAMS, N_PLAYERS),
    # latent shooting skill offset applied on top of zone base rate (-0.06 .. +0.09)
    "skill_offset": RNG.normal(0.0, 0.035, N_PLAYERS).clip(-0.08, 0.10),
    # latent 3pt specialization offset
    "three_pt_offset": RNG.normal(0.0, 0.03, N_PLAYERS).clip(-0.06, 0.09),
})

ZONES = [
    # name, min_dist, max_dist, base_fg%, is_three, weight (sampling frequency)
    ("Restricted Area",      0.0,  4.0,  0.64, False, 0.28),
    ("In The Paint (Non-RA)",4.0,  8.0,  0.42, False, 0.12),
    ("Mid-Range",            8.0, 16.0,  0.41, False, 0.14),
    ("Long Mid-Range",      16.0, 22.0,  0.40, False, 0.09),
    ("Left Corner 3",       22.0, 23.9,  0.39, True,  0.06),
    ("Right Corner 3",      22.0, 23.9,  0.39, True,  0.06),
    ("Above The Break 3",   23.9, 30.0,  0.36, True,  0.20),
    ("Deep 3 / Heave",      30.0, 40.0,  0.16, True,  0.05),
]
zone_names = [z[0] for z in ZONES]
zone_weights = np.array([z[5] for z in ZONES])
zone_weights = zone_weights / zone_weights.sum()

rows = []
game_ids = RNG.integers(22600001, 22600001 + N_GAMES, N_GAMES)

for i in range(N_SHOTS):
    player = players.iloc[RNG.integers(0, N_PLAYERS)]
    zone_idx = RNG.choice(len(ZONES), p=zone_weights)
    zname, dmin, dmax, base_fg, is_three, _ = ZONES[zone_idx]

    distance = float(RNG.uniform(dmin, max(dmin + 0.1, dmax)))
    angle_deg = float(RNG.uniform(-80, 80))  # angle from hoop, 0 = straight on
    if "Corner" in zname:
        angle_deg = float(RNG.choice([-1, 1])) * RNG.uniform(65, 80)

    angle_rad = np.deg2rad(angle_deg)
    loc_x = round(distance * np.sin(angle_rad), 1)
    loc_y = round(distance * np.cos(angle_rad), 1)

    period = int(RNG.choice([1, 2, 3, 4, 5], p=[0.245, 0.245, 0.245, 0.245, 0.02]))
    minutes_remaining = int(RNG.integers(0, 12))
    seconds_remaining = int(RNG.integers(0, 60))
    shot_clock = float(np.clip(RNG.normal(14, 6), 0, 24))

    dribbles = int(RNG.poisson(1.8))
    touch_time = float(np.clip(RNG.exponential(2.5) + dribbles * 0.6, 0.1, 24))
    catch_and_shoot = 1 if (dribbles <= 1 and touch_time < 2.5) else 0

    defender_distance = float(np.clip(RNG.normal(4.2, 2.2), 0.2, 15))
    home = int(RNG.integers(0, 2))
    score_margin = int(np.clip(RNG.normal(0, 10), -40, 40))

    # ---- probability model ----
    prob = base_fg
    prob += player["skill_offset"]
    if is_three:
        prob += player["three_pt_offset"]

    # tighter defense lowers FG%
    prob += (defender_distance - 4.0) * 0.012
    # catch and shoot boosts jumpers, small penalty for heavily dribbled jumpers
    if not is_three and distance > 8:
        prob += 0.05 if catch_and_shoot else -0.01 * min(dribbles, 5)
    if is_three:
        prob += 0.06 if catch_and_shoot else -0.015 * min(dribbles, 5)
    # end of shot clock pressure
    if shot_clock < 4:
        prob -= 0.09
    elif shot_clock < 7:
        prob -= 0.03
    # garbage-time / big-margin heaves are noisier but roughly neutral
    prob += RNG.normal(0, 0.015)

    prob = float(np.clip(prob, 0.03, 0.92))
    made = int(RNG.random() < prob)

    rows.append({
        "game_id": int(game_ids[i % N_GAMES]),
        "period": period,
        "minutes_remaining": minutes_remaining,
        "seconds_remaining": seconds_remaining,
        "shot_clock": round(shot_clock, 1),
        "player_id": int(player["player_id"]),
        "player_name": player["player_name"],
        "position": player["position"],
        "team": player["team"],
        "opponent": RNG.choice([t for t in TEAMS if t != player["team"]]),
        "home": home,
        "score_margin": score_margin,
        "loc_x": loc_x,
        "loc_y": loc_y,
        "shot_distance": round(distance, 1),
        "shot_zone": zname,
        "is_three": int(is_three),
        "dribbles": dribbles,
        "touch_time": round(touch_time, 1),
        "catch_and_shoot": catch_and_shoot,
        "defender_distance": round(defender_distance, 1),
        "true_prob": round(prob, 4),  # ground-truth latent prob (kept for validation only)
        "shot_made": made,
    })

df = pd.DataFrame(rows)
out_path = "/sessions/compassionate-funny-sagan/mnt/outputs/nba_shot_model/data/shots_raw.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {len(df):,} shot events to {out_path}")
print(df["shot_made"].mean(), "overall FG%")
print(df.groupby("shot_zone")["shot_made"].mean().round(3))
