import json
from pathlib import Path

from data_structures import Parameters


PROFILE_FILE = Path(__file__).parent / "config" / "catch_profiles.json"


def load_catch_profile(name):
    with PROFILE_FILE.open("r", encoding="utf-8") as file:
        profiles = json.load(file)
    if name not in profiles:
        raise ValueError(f"Unknown CaTCH profile: {name}")

    profile = profiles[name]
    return Parameters(
        MAX_PLAUSIBLE_RANGE=profile["mpr"],
        MAX_PLAUSIBLE_DIST_NEGATIVE=profile["mpdn"],
        MAX_PLAUSIBLE_SPEED=profile["mps"],
        MAX_PLAUSIBLE_ACCEL=profile["mpa"],
        MAX_PLAUSIBLE_DECEL=profile["mpd"],
        MAX_HEADING_CHANGE=profile["mhc"],
        MAX_DELTA_INTERSECTION=profile["mdi"],
        MAX_TIME_DELTA=profile["mtd"],
        POS_HEADING_TIME=profile["pht"],
        MAX_MGT_RNG_UP=profile["mmru"],
        MAX_MGT_RNG_DOWN=profile["mmrd"],
        MAX_NON_ROUTE_SPEED=profile["mnrs"],
    )
