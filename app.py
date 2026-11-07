"""
UrbanGrid | Cork City Business Location Intelligence
=====================================================

A plain-English site selection and footfall copilot for small businesses
looking to open in Cork City, Ireland.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Data sources (all fetched live, with an embedded fallback dataset so the
app never crashes when an API is slow or down):
    * Cork City pedestrian footfall counters  - data.corkcity.ie (CKAN datastore)
    * Cork City planning applications         - data.corkcity.ie (CKAN datastore)
    * NTA NaPTAN public transport stop points - data.gov.ie (CKAN datastore)
    * Existing businesses (cafés, bakeries...) - OpenStreetMap via Overpass API
"""

from __future__ import annotations

import html
import json
import re
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from functools import partial

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

try:  # st.dataframe needs pyarrow; some locked-down Windows machines block its DLLs.
    import pyarrow.lib  # noqa: F401
    ARROW_OK = True
except Exception:  # noqa: BLE001
    ARROW_OK = False

try:  # PDF area report; the app offers an HTML report instead if fpdf2 is missing.
    from fpdf import FPDF
    FPDF_OK = True
except Exception:  # noqa: BLE001
    FPDF_OK = False

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UrbanGrid | Cork Location Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CITY_CENTER = (51.8985, -8.4756)
SAMPLE_QUERY = (
    "Where is the best spot on Oliver Plunkett Street for a specialty coffee "
    "shop between 8 AM and 10 AM?"
)

HTTP_HEADERS = {"User-Agent": "UrbanGrid/1.0 (Cork Hackathon demo)"}
HTTP_TIMEOUT = (3.0, 6.0)          # (connect, read) seconds per request
LIVE_FETCH_BUDGET_S = 10.0         # wall-clock budget for all live calls together
CACHE_TTL_S = 900                  # re-try live APIs every 15 minutes

CORK_CKAN = "https://data.corkcity.ie/api/3/action"
GOV_CKAN = "https://data.gov.ie/api/3/action"
FOOTFALL_RESOURCE = "db32fcd6-acbb-458d-ac02-3e58a3c1577b"   # Cork pedestrian counts
PLANNING_RESOURCE = "8d5bbfa9-3b0c-40ac-8630-4243bed94b2d"   # Cork planning permissions
NAPTAN_RESOURCE = "61f00deb-8bcc-43f3-b984-b8255d11948a"     # NTA NaPTAN stop points
NAPTAN_AREAS = {837: "Cork City", 838: "Cork County"}
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter"]
OSM_BOXES = ["51.893,-8.490,51.904,-8.462", "51.870,-8.448,51.884,-8.428", "51.883,-8.603,51.894,-8.585"]
OSM_KIND_TAGS = {"cafe": "cafe", "coffee": "cafe", "bakery": "bakery", "pastry": "bakery", "dentist": "dentist",
                 "clothes": "clothing", "boutique": "clothing"}
OSM_LIVE_TTL_S = 6 * 3600          # refresh live OpenStreetMap data every 6 hours
OSM_RETRY_S = 600                  # after a failed refresh, retry in 10 minutes

PENDING_STATUSES = {
    "NEW APPLICATION",
    "FURTHER INFORMATION",
    "INCOMPLETED APPLICATION",
    "PRE-VALIDATION",
    "APPEALED",
}
PENDING_LOOKBACK_DAYS = 3 * 365    # ignore "pending" records older than this (stale statuses)
PLANNING_RADIUS_M = 200
TRANSIT_RADII_M = (50, 100, 150, 250)
AUTO = "auto"

# Half-hour options shown on the sidebar slider (human-readable labels are the stored values).
TIME_OPTIONS = [f"{int(h) % 12 or 12}:{int(round((h % 1) * 60)):02d} {'AM' if h < 12 else 'PM'}"
                for h in np.arange(6.0, 21.01, 0.5)]

BUSINESS_TYPES = {
    "Specialty Coffee Shop": {
        "patterns": [r"coffee", r"caf[eé]", r"espresso", r"barista", r"roaster"],
        "competitor": "food_beverage",
        "window": ("7:00 AM", "11:00 AM"),
        "osm_kind": "cafe",
        # Break-even defaults: avg spend €, gross margin %, other monthly costs €, trading
        # days/month, opening hours, typical capture rate range (% of passers-by who buy).
        "be": {"spend": 5.0, "margin": 70, "other": 9000, "days": 26, "hours": ('7:00 AM', '6:00 PM'), "capture": (1.0, 3.0)},
    },
    "Retail Boutique": {
        "patterns": [r"boutique", r"retail", r"clothing", r"fashion", r"apparel", r"\bshop\b", r"\bstore\b"],
        "competitor": "retail",
        "window": ("11:00 AM", "5:00 PM"),
        "osm_kind": "clothing",
        # Break-even defaults: avg spend €, gross margin %, other monthly costs €, trading
        # days/month, opening hours, typical capture rate range (% of passers-by who buy).
        "be": {"spend": 55.0, "margin": 55, "other": 7000, "days": 26, "hours": ('10:00 AM', '6:00 PM'), "capture": (0.05, 0.3)},
    },
    "Artisan Bakery": {
        "patterns": [r"baker(y|ies)", r"bakehouse", r"bread", r"pastr(y|ies)", r"patisserie", r"sourdough"],
        "competitor": "food_beverage",
        "window": ("7:00 AM", "12:00 PM"),
        "osm_kind": "bakery",
        # Break-even defaults: avg spend €, gross margin %, other monthly costs €, trading
        # days/month, opening hours, typical capture rate range (% of passers-by who buy).
        "be": {"spend": 7.5, "margin": 65, "other": 9000, "days": 26, "hours": ('7:00 AM', '5:00 PM'), "capture": (0.8, 2.5)},
    },
    "Dental Clinic": {
        "patterns": [r"dental", r"dentist", r"orthodont", r"clinic", r"surgery"],
        "competitor": "health",
        "window": ("9:00 AM", "5:00 PM"),
        "osm_kind": "dentist",
        # Break-even defaults: avg spend €, gross margin %, other monthly costs €, trading
        # days/month, opening hours, typical capture rate range (% of passers-by who buy).
        "be": {"spend": 120.0, "margin": 75, "other": 15000, "days": 22, "hours": ('9:00 AM', '5:30 PM'), "capture": (0.02, 0.1)},
    },
}
# When parsing free text, check specific types before the generic "shop/store" retail words,
# so "coffee shop" is understood as coffee, not retail.
BUSINESS_PARSE_ORDER = ["Specialty Coffee Shop", "Artisan Bakery", "Dental Clinic", "Retail Boutique"]

OSM_KIND_LABELS = {"cafe": ("café", "cafés"), "bakery": ("bakery", "bakeries"),
                   "dentist": ("dental practice", "dental practices"), "clothing": ("clothing shop", "clothing shops")}
EXISTING_RADIUS_M = 150

CATEGORY_LABELS = {"food_beverage": "food/beverage", "retail": "retail", "health": "healthcare", "other": "other"}
CATEGORY_PATTERNS = [  # order matters: first match wins
    ("food_beverage", r"caf[eé]|coffee|restaurant|take-?away|\bbar\b|wine bar|\bpub\b|public house|hot food|"
                      r"bakery|\bdeli\b|bistro|licensed premises|food"),
    ("health", r"dental|dentist|clinic|medical|pharmacy|surgery|healthcare|physio"),
    ("retail", r"retail|\bshop\b|\bstore\b|boutique|showroom"),
]

PRIORITIES = ["High Morning Footfall", "Low Competitor Density", "High Transit Access"]
DAY_TYPES = {"weekday": "Typical weekday", "saturday": "Saturday", "christmas": "Christmas season"}
SUBURBAN_ZONES = {"douglas", "ballincollig"}


def season_profile(profile, day_type: str, suburban: bool) -> list:
    """Adjust a typical-weekday hourly profile for Saturdays or the Christmas season.

    Estimated multipliers from general Irish/UK high-street patterns (not measured Cork data):
    Saturdays lose the commuter morning peak but gain a strong midday-afternoon shopping peak
    (bigger at suburban shopping centres); the Christmas season lifts daytime trade and late opening.
    """
    f = np.ones(24)
    if day_type == "saturday":
        f[6:10] = 0.6 if suburban else 0.45
        f[10:12] = 1.2 if suburban else 1.05
        f[12:18] = 1.45 if suburban else 1.3
        f[18:23] = 1.0 if suburban else 1.15
    elif day_type == "christmas":
        f[7:10] = 1.05
        f[10:17] = 1.4 if suburban else 1.35
        f[17:21] = 1.35 if suburban else 1.5
    return list(np.asarray(profile, dtype=float) * f)


PRIORITY_WEIGHTS = {  # footfall, competition, transit, budget
    "High Morning Footfall": (0.50, 0.20, 0.20, 0.10),
    "Low Competitor Density": (0.25, 0.45, 0.20, 0.10),
    "High Transit Access": (0.25, 0.20, 0.45, 0.10),
}

# ---------------------------------------------------------------------------
# Cork zones: geometry, analysis spots (footfall counter points) and lookups
# ---------------------------------------------------------------------------
ZONES = {
    "oliver_plunkett": {
        "name": "Oliver Plunkett Street",
        "aliases": ["oliver plunkett", "plunkett"],
        "path": [[-8.4753, 51.8976], [-8.4726, 51.8979], [-8.4700, 51.8982], [-8.4668, 51.8984]],
        "spots": [
            {"spot": "Grand Parade junction", "lat": 51.8977, "lon": -8.4746, "share": 1.25},
            {"spot": "Pembroke Street (mid-street)", "lat": 51.8980, "lon": -8.4718, "share": 0.95},
            {"spot": "Parnell Place end", "lat": 51.8983, "lon": -8.4678, "share": 0.70},
        ],
        "rent": 5500,
        "headway_est": 4,
        "planning_q": "Oliver Plunkett",
        "address_keys": ["oliver plunkett"],
        "footfall_keys": ["oliver plunkett"],
        "area_label": "Cork City",
    },
    "patrick_street": {
        "name": "St. Patrick's Street",
        "aliases": ["st. patrick's street", "st patrick's street", "patrick's street", "patricks street",
                    "patrick street", "patrick st", "pana"],
        "path": [[-8.4751, 51.8980], [-8.4738, 51.8989], [-8.4722, 51.8998], [-8.4706, 51.9006]],
        "spots": [
            {"spot": "Daunt Square end", "lat": 51.8982, "lon": -8.4747, "share": 1.10},
            {"spot": "Opera Lane / Academy Street", "lat": 51.8993, "lon": -8.4731, "share": 1.20},
            {"spot": "Patrick's Bridge end", "lat": 51.9003, "lon": -8.4712, "share": 0.85},
        ],
        "rent": 8500,
        "headway_est": 3,
        "planning_q": "Patrick Street",
        "address_keys": ["patrick street", "patrick's street", "patricks street", "patrick's st", "patrick st,"],
        "footfall_keys": ["patrick"],
        "area_label": "Cork City",
    },
    "grand_parade": {
        "name": "Grand Parade",
        "aliases": ["grand parade"],
        "path": [[-8.4752, 51.8979], [-8.4757, 51.8968], [-8.4762, 51.8955]],
        "spots": [
            {"spot": "Daunt Square junction", "lat": 51.8979, "lon": -8.4752, "share": 1.15},
            {"spot": "English Market entrance", "lat": 51.8967, "lon": -8.4757, "share": 1.10},
            {"spot": "South Mall / Berwick Fountain", "lat": 51.8957, "lon": -8.4761, "share": 0.80},
        ],
        "rent": 6200,
        "headway_est": 5,
        "planning_q": "Grand Parade",
        "address_keys": ["grand parade"],
        "footfall_keys": ["grand parade"],
        "area_label": "Cork City",
    },
    "washington_street": {
        "name": "Washington Street",
        "aliases": ["washington street", "washington st", "washington"],
        "path": [[-8.4762, 51.8969], [-8.4790, 51.8967], [-8.4820, 51.8964], [-8.4850, 51.8961]],
        "spots": [
            {"spot": "Grand Parade end", "lat": 51.8969, "lon": -8.4772, "share": 1.10},
            {"spot": "Courthouse", "lat": 51.8967, "lon": -8.4800, "share": 1.00},
            {"spot": "Lancaster Quay end", "lat": 51.8962, "lon": -8.4842, "share": 0.75},
        ],
        "rent": 3800,
        "headway_est": 7,
        "planning_q": "Washington Street",
        "address_keys": ["washington street", "washington st"],
        "footfall_keys": ["washington"],
        "area_label": "Cork City",
    },
    "douglas": {
        "name": "Douglas Village",
        "aliases": ["douglas"],
        "path": [[-8.4398, 51.8772], [-8.4375, 51.8768], [-8.4352, 51.8772], [-8.4335, 51.8777]],
        "spots": [
            {"spot": "Douglas Village Main Street", "lat": 51.8767, "lon": -8.4363, "share": 1.00},
            {"spot": "Douglas Village Shopping Centre", "lat": 51.8775, "lon": -8.4350, "share": 1.10},
            {"spot": "Douglas Court Shopping Centre", "lat": 51.8790, "lon": -8.4394, "share": 0.95},
        ],
        "rent": 4200,
        "headway_est": 10,
        "planning_q": "Douglas",
        "address_keys": ["douglas"],
        "footfall_keys": ["douglas"],
        "area_label": "Cork City",
    },
    "ballincollig": {
        "name": "Ballincollig",
        "aliases": ["ballincollig"],
        "path": [[-8.5975, 51.8885], [-8.5925, 51.8879], [-8.5875, 51.8876]],
        "spots": [
            {"spot": "Main Street (town centre)", "lat": 51.8879, "lon": -8.5925, "share": 1.00},
            {"spot": "Ballincollig Shopping Centre", "lat": 51.8890, "lon": -8.5958, "share": 1.15},
        ],
        "rent": 3300,
        "headway_est": 15,
        "planning_q": "Ballincollig",
        "address_keys": ["ballincollig"],
        "footfall_keys": ["ballincollig"],
        "area_label": "Ballincollig",
    },
}
CITY_CENTRE_ZONES = {"oliver_plunkett", "patrick_street", "grand_parade", "washington_street"}

# ---------------------------------------------------------------------------
# Embedded fallback dataset (used automatically when a live API fails)
# ---------------------------------------------------------------------------
FALLBACK_DATA = {
    # Typical weekday pedestrians per hour (hour 00..23), street average.
    # Per-spot intensity = profile x spot "share" (see ZONES).
    "footfall_hourly": {
        "oliver_plunkett":   [5, 3, 2, 2, 4, 30, 120, 420, 1180, 980, 640, 720, 1050, 1120, 820, 760, 900, 880, 640, 480, 360, 240, 120, 40],
        "patrick_street":    [10, 6, 4, 3, 5, 40, 160, 520, 1240, 1150, 980, 1250, 1720, 1810, 1540, 1500, 1620, 1580, 1180, 860, 620, 420, 210, 60],
        "grand_parade":      [6, 4, 3, 3, 5, 35, 140, 460, 1020, 920, 820, 980, 1380, 1420, 1100, 1000, 1080, 1020, 760, 560, 400, 280, 140, 45],
        "washington_street": [20, 12, 6, 4, 4, 20, 80, 260, 620, 560, 450, 520, 780, 820, 600, 560, 640, 700, 620, 560, 520, 480, 360, 160],
        "douglas":           [2, 1, 1, 1, 2, 15, 60, 210, 480, 520, 560, 640, 780, 760, 620, 600, 680, 720, 600, 420, 260, 140, 60, 15],
        "ballincollig":      [2, 1, 1, 1, 2, 12, 50, 180, 420, 460, 500, 560, 700, 680, 560, 540, 620, 660, 560, 380, 220, 120, 50, 12],
    },
    "planning": [
        {"app_no": "FB-2641187", "zone": "oliver_plunkett", "address": "24 Oliver Plunkett Street, Cork",
         "description": "Change of use of ground floor retail unit to café / specialty coffee bar with ancillary takeaway and new shopfront signage.",
         "status": "NEW APPLICATION", "received": "2026-09-14", "lat": 51.8978, "lon": -8.4739},
        {"app_no": "FB-2640932", "zone": "oliver_plunkett", "address": "3 Grand Parade (corner of Oliver Plunkett Street), Cork",
         "description": "Change of use of vacant ground floor to restaurant and wine bar, including internal alterations and extract ducting.",
         "status": "FURTHER INFORMATION", "received": "2026-08-02", "lat": 51.8975, "lon": -8.4752},
        {"app_no": "FB-2539921", "zone": "oliver_plunkett", "address": "88 Oliver Plunkett Street, Cork",
         "description": "Replacement shopfront and signage to existing retail unit.",
         "status": "APPLICATION FINALISED", "received": "2025-11-18", "lat": 51.8981, "lon": -8.4709},
        {"app_no": "FB-2642210", "zone": "oliver_plunkett", "address": "Parnell Place end, Oliver Plunkett Street, Cork",
         "description": "Refurbishment of upper floors for office use.",
         "status": "NEW APPLICATION", "received": "2026-10-03", "lat": 51.8983, "lon": -8.4674},
        {"app_no": "FB-2641675", "zone": "patrick_street", "address": "St. Patrick's Street (Opera Lane side), Cork",
         "description": "Change of use to fashion retail boutique with internal fit-out and new signage.",
         "status": "NEW APPLICATION", "received": "2026-09-28", "lat": 51.8996, "lon": -8.4726},
        {"app_no": "FB-2640418", "zone": "patrick_street", "address": "Patrick's Bridge end, St. Patrick's Street, Cork",
         "description": "Change of use to café with outdoor seating area.",
         "status": "FURTHER INFORMATION", "received": "2026-07-19", "lat": 51.9003, "lon": -8.4709},
        {"app_no": "FB-2538804", "zone": "patrick_street", "address": "St. Patrick's Street, Cork",
         "description": "Amalgamation of two retail units into a single flagship store.",
         "status": "APPLICATION FINALISED", "received": "2025-06-10", "lat": 51.8990, "lon": -8.4735},
        {"app_no": "FB-2642051", "zone": "grand_parade", "address": "Grand Parade (South Mall end), Cork",
         "description": "Change of use to bakery café with seating.",
         "status": "NEW APPLICATION", "received": "2026-10-11", "lat": 51.8955, "lon": -8.4763},
        {"app_no": "FB-2537716", "zone": "grand_parade", "address": "Bishop Lucey Park, Grand Parade, Cork",
         "description": "Public realm improvement works.",
         "status": "APPLICATION FINALISED", "received": "2025-04-22", "lat": 51.8965, "lon": -8.4760},
        {"app_no": "FB-2641330", "zone": "washington_street", "address": "Washington Street (near Courthouse), Cork",
         "description": "Change of use to hot food takeaway.",
         "status": "NEW APPLICATION", "received": "2026-09-01", "lat": 51.8968, "lon": -8.4803},
        {"app_no": "FB-2639902", "zone": "washington_street", "address": "Washington Street West, Cork",
         "description": "Change of use to dental and orthodontic clinic.",
         "status": "FURTHER INFORMATION", "received": "2026-06-22", "lat": 51.8963, "lon": -8.4830},
        {"app_no": "FB-2641019", "zone": "douglas", "address": "Douglas Court Shopping Centre, Douglas, Cork",
         "description": "Change of use of retail unit to dental clinic.",
         "status": "NEW APPLICATION", "received": "2026-08-17", "lat": 51.8792, "lon": -8.4398},
        {"app_no": "FB-2539120", "zone": "douglas", "address": "Main Street, Douglas Village, Cork",
         "description": "Change of use to coffee shop.",
         "status": "APPLICATION FINALISED", "received": "2025-12-03", "lat": 51.8766, "lon": -8.4366},
        {"app_no": "FB-2642388", "zone": "douglas", "address": "East Douglas Street, Douglas, Cork",
         "description": "Construction of three new retail units.",
         "status": "NEW APPLICATION", "received": "2026-10-20", "lat": 51.8773, "lon": -8.4342},
        {"app_no": "FB-2641554", "zone": "ballincollig", "address": "Main Street, Ballincollig, Cork",
         "description": "Change of use to artisan bakery and café.",
         "status": "NEW APPLICATION", "received": "2026-09-09", "lat": 51.8881, "lon": -8.5918},
        {"app_no": "FB-2538477", "zone": "ballincollig", "address": "Ballincollig Shopping Centre, Cork",
         "description": "Internal reconfiguration of retail units.",
         "status": "APPLICATION FINALISED", "received": "2025-05-14", "lat": 51.8891, "lon": -8.5962},
    ],
    # Bus stops with typical weekday daytime headway (minutes between arrivals).
    "transit": [
        {"stop_id": "FB-OPS-1", "name": "Grand Parade (Daunt Square)", "lat": 51.8980, "lon": -8.4750, "headway_min": 3},
        {"stop_id": "FB-OPS-2", "name": "Oliver Plunkett St (West)", "lat": 51.8979, "lon": -8.4741, "headway_min": 5},
        {"stop_id": "FB-OPS-3", "name": "Grand Parade (Oliver Plunkett St)", "lat": 51.8974, "lon": -8.4749, "headway_min": 4},
        {"stop_id": "FB-OPS-4", "name": "Oliver Plunkett St (Pembroke St)", "lat": 51.8982, "lon": -8.4714, "headway_min": 8},
        {"stop_id": "FB-OPS-5", "name": "Parnell Place Bus Station", "lat": 51.8986, "lon": -8.4664, "headway_min": 3},
        {"stop_id": "FB-PAT-1", "name": "St. Patrick's St (Opera Lane)", "lat": 51.8993, "lon": -8.4729, "headway_min": 3},
        {"stop_id": "FB-PAT-2", "name": "St. Patrick's St (Opp. Opera Lane)", "lat": 51.8995, "lon": -8.4724, "headway_min": 3},
        {"stop_id": "FB-PAT-3", "name": "Merchant's Quay", "lat": 51.9002, "lon": -8.4697, "headway_min": 4},
        {"stop_id": "FB-PAT-4", "name": "Patrick's Bridge", "lat": 51.9010, "lon": -8.4703, "headway_min": 5},
        {"stop_id": "FB-GP-1", "name": "Grand Parade (Bishop Lucey Park)", "lat": 51.8965, "lon": -8.4758, "headway_min": 5},
        {"stop_id": "FB-GP-2", "name": "South Mall (Grand Parade)", "lat": 51.8956, "lon": -8.4752, "headway_min": 6},
        {"stop_id": "FB-WAS-1", "name": "Washington St (Courthouse)", "lat": 51.8969, "lon": -8.4802, "headway_min": 7},
        {"stop_id": "FB-WAS-2", "name": "Washington St West", "lat": 51.8963, "lon": -8.4828, "headway_min": 8},
        {"stop_id": "FB-WAS-3", "name": "Lancaster Quay", "lat": 51.8960, "lon": -8.4855, "headway_min": 8},
        {"stop_id": "FB-DOU-1", "name": "Douglas Village", "lat": 51.8768, "lon": -8.4358, "headway_min": 10},
        {"stop_id": "FB-DOU-2", "name": "Douglas Court SC", "lat": 51.8789, "lon": -8.4390, "headway_min": 12},
        {"stop_id": "FB-DOU-3", "name": "Douglas West", "lat": 51.8760, "lon": -8.4395, "headway_min": 15},
        {"stop_id": "FB-BAL-1", "name": "Ballincollig Main St (Eastbound)", "lat": 51.8880, "lon": -8.5921, "headway_min": 15},
        {"stop_id": "FB-BAL-2", "name": "Ballincollig Main St (Westbound)", "lat": 51.8882, "lon": -8.5930, "headway_min": 15},
        {"stop_id": "FB-BAL-3", "name": "Ballincollig Shopping Centre", "lat": 51.8890, "lon": -8.5957, "headway_min": 20},
    ],
}


# Real snapshot of existing businesses from OpenStreetMap (refreshed live in the background).
OSM_SNAPSHOT_DATE = "2026-11-07"
OSM_SNAPSHOT = [  # (kind, name, lat, lon) - © OpenStreetMap contributors, ODbL
    ('bakery', 'Cameron Bakery', 51.89753, -8.48026),
    ('bakery', 'Cameron Bakery', 51.89862, -8.46762),
    ('bakery', 'Cameron Bakery', 51.90014, -8.47079),
    ('bakery', "Heaven's Cakes", 51.89769, -8.47427),
    ('bakery', 'Pana Bread', 51.88868, -8.59319),
    ('bakery', 'Pana bread', 51.89989, -8.4699),
    ('bakery', 'The Alternative Bread Company', 51.89772, -8.47444),
    ('bakery', 'The Grumpy Bakers', 51.89783, -8.47569),
    ('bakery', 'The Grumpy Bakers', 51.89786, -8.47565),
    ('bakery', 'The Pie Guys', 51.89789, -8.47416),
    ('cafe', '', 51.87671, -8.43897),
    ('cafe', '', 51.89964, -8.48701),
    ('cafe', '107 Barrack Street', 51.89343, -8.47862),
    ('cafe', '18CTea', 51.90016, -8.47867),
    ('cafe', '3 Little Piggies', 51.89545, -8.4685),
    ('cafe', '5 Points Deli and Cafe', 51.90144, -8.46581),
    ('cafe', 'Alchemy Coffee and Bookstore', 51.89401, -8.47755),
    ('cafe', "BB's", 51.87825, -8.43117),
    ('cafe', 'Badger & Dodo', 51.89749, -8.46863),
    ('cafe', 'Badger & Dodo Roastery Outlet', 51.89494, -8.47622),
    ('cafe', 'Bakestone Cafe', 51.90005, -8.47162),
    ('cafe', 'Barista Coffee', 51.90095, -8.46878),
    ('cafe', 'Bean and Leaf', 51.87372, -8.43866),
    ('cafe', 'Bia Blasta', 51.89313, -8.46944),
    ('cafe', 'Bitesize', 51.88784, -8.59161),
    ('cafe', 'Blend 14', 51.87835, -8.4355),
    ('cafe', 'Blondie', 51.88811, -8.58911),
    ('cafe', 'Boru Coffee shop', 51.90094, -8.46538),
    ('cafe', 'Bright Side', 51.89747, -8.48067),
    ('cafe', 'Butlers Chocolate Café', 51.89789, -8.47072),
    ('cafe', 'Butter Bubble', 51.89752, -8.47237),
    ('cafe', 'Cafe Depeche', 51.89713, -8.48261),
    ('cafe', 'Cafe Gusto', 51.89782, -8.47579),
    ('cafe', 'Cafe Gusto', 51.89869, -8.464),
    ('cafe', 'Cafe Molly', 51.89313, -8.46884),
    ('cafe', 'Cafe Moly', 51.89311, -8.46883),
    ('cafe', 'Cafespresso', 51.90149, -8.46937),
    ('cafe', 'Caffè Nero', 51.89819, -8.46927),
    ('cafe', 'Caffé Misto', 51.8888, -8.59199),
    ('cafe', 'Café Eco', 51.8983, -8.47061),
    ('cafe', 'Café Gusto', 51.89784, -8.47579),
    ('cafe', 'Café Marius', 51.89779, -8.47463),
    ('cafe', 'Café Velo', 51.89483, -8.47007),
    ('cafe', 'Chapel Gate Café', 51.88687, -8.58985),
    ('cafe', 'Chatime', 51.89769, -8.47151),
    ('cafe', 'Coffee Central', 51.89776, -8.47405),
    ('cafe', 'Coffee Scape', 51.90152, -8.47651),
    ('cafe', 'Cool Beans Coffee', 51.89874, -8.46718),
    ('cafe', 'Cork Coffee Roasters', 51.89464, -8.4652),
    ('cafe', 'Cork Coffee Roasters', 51.89939, -8.47388),
    ('cafe', 'Cork Coffee Roasters', 51.90147, -8.47003),
    ('cafe', 'Cork Opera House', 51.90016, -8.4727),
    ('cafe', 'Costa', 51.87815, -8.43209),
    ('cafe', 'Costa', 51.89974, -8.47255),
    ('cafe', 'Costa', 51.90002, -8.46923),
    ('cafe', 'Craic & Coffee', 51.89778, -8.47606),
    ('cafe', 'Cupcake Cottage', 51.89427, -8.46498),
    ('cafe', 'De Calf Cafe', 51.89948, -8.47605),
    ('cafe', 'Dukes', 51.89912, -8.47409),
    ('cafe', 'Dulce Bun House', 51.89754, -8.47248),
    ('cafe', 'EMMAUS CAFÉ', 51.89947, -8.47477),
    ('cafe', 'Fellini', 51.89861, -8.47367),
    ('cafe', 'Filter', 51.89537, -8.47195),
    ('cafe', 'Filter', 51.89848, -8.47716),
    ('cafe', 'Fly Coffee', 51.89843, -8.46496),
    ('cafe', 'Gallery Cafe', 51.89981, -8.47341),
    ('cafe', "Gloria Jean's", 51.87839, -8.43047),
    ('cafe', "Gloria Jean's", 51.89865, -8.47668),
    ('cafe', 'Good Day Deli', 51.893, -8.47258),
    ('cafe', 'Good For The Soul', 51.88758, -8.58811),
    ('cafe', 'Green Frog Cafe', 51.89573, -8.48894),
    ('cafe', 'Griolladh', 51.87753, -8.43761),
    ('cafe', 'Ground', 51.90058, -8.46307),
    ('cafe', 'Guji Coffee Bar', 51.89953, -8.47458),
    ('cafe', "Here's Health Store & Coffee", 51.89747, -8.47268),
    ('cafe', 'Insomnia', 51.87747, -8.43787),
    ('cafe', 'Joes & Bros', 51.89798, -8.47089),
    ('cafe', "L'atitude", 51.89707, -8.46662),
    ('cafe', "Lafayette's Brasserie", 51.89723, -8.46993),
    ('cafe', 'M & S Cafe', 51.87808, -8.43722),
    ('cafe', 'Mahers Pure Coffee', 51.89806, -8.4699),
    ('cafe', 'Messy Buns', 51.87755, -8.43755),
    ('cafe', 'Moodys Cafe', 51.90149, -8.46493),
    ('cafe', 'Myo', 51.90141, -8.47615),
    ('cafe', 'Nectar', 51.89877, -8.46761),
    ('cafe', 'Nosh N Coffee', 51.8885, -8.59134),
    ('cafe', "O'Brien's Sandwich Bar", 51.87803, -8.43125),
    ('cafe', 'Oakberry Açai', 51.89748, -8.47321),
    ('cafe', 'Off Quay', 51.89607, -8.46828),
    ('cafe', "On The Pig's Back", 51.87587, -8.44192),
    ('cafe', "Oriental Delight's", 51.90381, -8.47783),
    ('cafe', 'Paper Boys', 51.89744, -8.47673),
    ('cafe', 'Plugd Records', 51.90046, -8.47643),
    ('cafe', 'Plus & Minus', 51.89906, -8.4737),
    ('cafe', 'Priory Coffee', 51.8879, -8.59009),
    ('cafe', 'Priory Coffee Co.', 51.90175, -8.46994),
    ('cafe', 'Priory Coffee Company', 51.90083, -8.47946),
    ('cafe', "Puccino's", 51.89956, -8.46652),
    ('cafe', "Puccino's", 51.89959, -8.46687),
    ('cafe', 'Puccinos', 51.87766, -8.43752),
    ('cafe', 'Quay Co-op Coffee Dock', 51.89552, -8.47473),
    ('cafe', 'Rebel Coffee Cork', 51.89908, -8.4736),
    ('cafe', 'Red Bean Roastery Café', 51.89772, -8.46774),
    ('cafe', 'Red Bean Roastery Café', 51.89854, -8.46514),
    ('cafe', 'SOMA 2', 51.88228, -8.43856),
    ('cafe', 'SOMA Coffee Company', 51.89698, -8.47588),
    ('cafe', 'Sensei Cafe', 51.89776, -8.4828),
    ('cafe', 'Soma 2', 51.89555, -8.46568),
    ('cafe', 'Starbucks', 51.87787, -8.43718),
    ('cafe', 'Starbucks', 51.88822, -8.59145),
    ('cafe', 'Starbucks', 51.89945, -8.47304),
    ('cafe', 'Starbucks Princes Street', 51.89768, -8.47347),
    ('cafe', 'Swoon Dessert Bar', 51.89711, -8.47462),
    ('cafe', 'Teatpon', 51.89986, -8.46887),
    ('cafe', 'The Bookshelf', 51.89696, -8.46421),
    ('cafe', 'The Bookshelf', 51.89732, -8.46976),
    ('cafe', 'The Haven Café', 51.90058, -8.48081),
    ('cafe', 'The Hideout Café', 51.90197, -8.46846),
    ('cafe', 'The pantry', 51.89965, -8.46871),
    ('cafe', 'Three Fools Coffee', 51.89679, -8.47491),
    ('cafe', 'Tiramisu', 51.89501, -8.47883),
    ('cafe', 'Twenty', 51.89418, -8.46501),
    ('cafe', 'Union Grind Café', 51.89696, -8.46682),
    ('cafe', 'Velvet', 51.90101, -8.46979),
    ('clothing', '', 51.8871, -8.58876),
    ('clothing', '#45', 51.89759, -8.47222),
    ('clothing', '#45 Homme', 51.89749, -8.47218),
    ('clothing', 'All Dolled Up', 51.88752, -8.59104),
    ('clothing', 'Bella Bleu Bridal', 51.8971, -8.47033),
    ('clothing', 'Bluefig', 51.88811, -8.58895),
    ('clothing', 'Born', 51.87778, -8.43708),
    ('clothing', 'Born Clothing', 51.89996, -8.47025),
    ('clothing', 'Brown Thomas Menswear', 51.8988, -8.46991),
    ('clothing', "Buckley's", 51.90315, -8.47835),
    ('clothing', 'By Kate', 51.88765, -8.59208),
    ('clothing', 'Carl Scarpa', 51.89761, -8.47188),
    ('clothing', 'Casa Craft', 51.89902, -8.47619),
    ('clothing', 'Cronins', 51.89735, -8.47343),
    ('clothing', "Dani's Closet", 51.88899, -8.59129),
    ('clothing', 'Design-a-tee', 51.89861, -8.47718),
    ('clothing', 'Desmond Menswear', 51.88791, -8.59305),
    ('clothing', 'Diamond Bridal', 51.89878, -8.46826),
    ('clothing', 'Diesel', 51.89767, -8.47227),
    ('clothing', 'Douglas Fashion & Lingerie', 51.87764, -8.43769),
    ('clothing', 'Elite Femme', 51.8874, -8.59131),
    ('clothing', "Evan G's Menswear", 51.88789, -8.59385),
    ('clothing', 'FatFace', 51.89803, -8.47364),
    ('clothing', 'Fitzgerald Menswear', 51.89859, -8.47119),
    ('clothing', 'Flannels', 51.89956, -8.47102),
    ('clothing', "Gentleman's Quarters", 51.89874, -8.47205),
    ('clothing', 'H&M', 51.89927, -8.47278),
    ('clothing', 'Itso Me', 51.87753, -8.43795),
    ('clothing', 'JD Sports', 51.8994, -8.47116),
    ('clothing', 'JoJo Maman Bébé', 51.89756, -8.47329),
    ('clothing', 'Kids Care', 51.89872, -8.47736),
    ('clothing', 'Leaders', 51.89916, -8.47805),
    ('clothing', 'Louis Copeland', 51.89904, -8.47331),
    ('clothing', 'Love Lisa', 51.89757, -8.47341),
    ('clothing', 'M.F. Godsils', 51.89881, -8.47647),
    ('clothing', 'Mango', 51.89907, -8.47131),
    ('clothing', "McCarthy's", 51.89996, -8.47703),
    ('clothing', 'Memories Bridal Boutique', 51.89692, -8.47595),
    ('clothing', 'Mi Lady', 51.89722, -8.47322),
    ('clothing', 'Mick Murphy', 51.89892, -8.47752),
    ('clothing', 'Moderne', 51.90029, -8.47153),
    ('clothing', "Morley's", 51.89857, -8.47592),
    ('clothing', 'Naphisa', 51.89803, -8.47193),
    ('clothing', 'Next', 51.8991, -8.47267),
    ('clothing', 'Olori', 51.89851, -8.4681),
    ('clothing', 'P.J. Leader', 51.89931, -8.47818),
    ('clothing', 'Palmela Scott', 51.89728, -8.47378),
    ('clothing', 'Penneys', 51.89822, -8.47146),
    ('clothing', 'Phoenix_V', 51.89881, -8.46851),
    ('clothing', 'Primetime', 51.89776, -8.47635),
    ('clothing', 'Quiz', 51.87818, -8.43119),
    ('clothing', 'Red Church', 51.89986, -8.47178),
    ('clothing', 'River Island', 51.89922, -8.47263),
    ('clothing', 'Samui', 51.89975, -8.47219),
    ('clothing', "Sheena's Boutique", 51.89807, -8.46983),
    ('clothing', 'Suits.ie', 51.89957, -8.47481),
    ('clothing', 'Superdry', 51.89848, -8.47307),
    ('clothing', 'The Moderne', 51.90023, -8.4717),
    ('clothing', 'The North Face', 51.89825, -8.47267),
    ('clothing', 'The Rave Cave', 51.8979, -8.47412),
    ('clothing', 'Timberland', 51.89728, -8.47477),
    ('clothing', 'Tom Murphy Menswear', 51.89838, -8.47477),
    ('clothing', 'Tommy Hilfiger', 51.89865, -8.47236),
    ('clothing', "Tony's Menswear", 51.89782, -8.47277),
    ('clothing', 'Urban Outfitters', 51.8988, -8.47183),
    ('clothing', "Victoria's Secret", 51.89885, -8.47247),
    ('dentist', '', 51.8758, -8.43262),
    ('dentist', 'Belvedere Dental Care', 51.88099, -8.44305),
    ('dentist', 'Blarney Street Dental Surgery', 51.90229, -8.48036),
    ('dentist', 'Bridget Dineen', 51.87651, -8.43466),
    ('dentist', 'Clermont Clinic', 51.88042, -8.44166),
    ('dentist', 'Cork City Dental', 51.90027, -8.47112),
    ('dentist', 'Crotty Orthodontics', 51.88356, -8.44779),
    ('dentist', 'Dental Care Ireland Ballincollig', 51.89006, -8.59002),
    ('dentist', 'Dental Practice', 51.9016, -8.46444),
    ('dentist', 'Douglas Medical Centre', 51.87773, -8.43666),
    ('dentist', 'Elaine McCarthy', 51.88752, -8.59235),
    ('dentist', 'Fenn Dental', 51.89826, -8.48005),
    ('dentist', 'Fingerpost Dental', 51.87583, -8.4326),
    ('dentist', 'Haly Dental Care', 51.89498, -8.47267),
    ('dentist', 'Kenneally Dental Practice', 51.88791, -8.59287),
    ('dentist', 'MacCurtain Street Dental', 51.90146, -8.46568),
    ('dentist', 'McHugh House Dental Surgery', 51.89815, -8.47571),
    ('dentist', 'My New Smile', 51.89948, -8.47589),
    ('dentist', 'Smiles Dental', 51.89818, -8.47014),
    ('dentist', 'T-Dental Lab', 51.90153, -8.46479),
    ('dentist', 'Ulick Cronin', 51.87663, -8.43499),
]

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def esc(value) -> str:
    return html.escape(str(value), quote=True)


def t2h(label: str) -> float:
    """'8:30 AM' or '08:30' -> 8.5"""
    m = re.match(r"\s*(\d{1,2}):(\d{2})\s*([AaPp][Mm])?", str(label))
    if not m:
        raise ValueError(f"bad time label: {label!r}")
    hh, mm, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
    if ampm:
        hh = hh % 12 + (12 if ampm.lower() == "pm" else 0)
    return hh + mm / 60


def fmt_time(hours: float) -> str:
    hh = int(hours)
    mm = int(round((hours - hh) * 60))
    suffix = "AM" if hh < 12 else "PM"
    return f"{hh % 12 or 12}:{mm:02d} {suffix}"


def fmt_time_label(label: str) -> str:
    return fmt_time(t2h(label))


def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in metres."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6_371_000 * 2 * np.arcsin(np.sqrt(a))


def show_table(df: pd.DataFrame, column_config: dict | None = None, html_df: pd.DataFrame | None = None) -> None:
    """st.dataframe when available, otherwise a styled HTML table (never crashes the page)."""
    if ARROW_OK:
        try:
            st.dataframe(df, hide_index=True, column_config=column_config)
            return
        except Exception:  # noqa: BLE001
            pass
    out = (html_df if html_df is not None else df).copy()
    if column_config:
        out = out.rename(columns={k: v if isinstance(v, str) else v.get("label", k)
                                  for k, v in column_config.items() if k in out.columns})
    for col in out.select_dtypes(include="float").columns:
        out[col] = out[col].round(2)
    st.markdown(f"<div class='ug-table-wrap'>{out.to_html(index=False, escape=True, border=0, classes='ug-table')}</div>",
                unsafe_allow_html=True)


def classify_development(text: str) -> str:
    low = (text or "").lower()
    for category, pattern in CATEGORY_PATTERNS:
        if re.search(pattern, low):
            return category
    return "other"


def period_word(start_h: float) -> str:
    if start_h < 12:
        return "morning"
    if start_h < 14:
        return "lunchtime"
    if start_h < 17:
        return "afternoon"
    return "evening"


def point_along_path(path, fraction: float):
    """Return [lon, lat] at `fraction` (0..1) of a polyline's length."""
    pts = np.array(path, dtype=float)
    seg = np.hypot(np.diff(pts[:, 0]) * np.cos(np.radians(pts[0, 1])), np.diff(pts[:, 1]))
    target = fraction * seg.sum()
    for i, length in enumerate(seg):
        if target <= length or i == len(seg) - 1:
            t = 0 if length == 0 else min(target / length, 1)
            return (pts[i] + t * (pts[i + 1] - pts[i])).tolist()
        target -= length
    return pts[-1].tolist()


def short_error(exc: Exception) -> str:
    if isinstance(exc, requests.Timeout):
        return "timed out"
    if isinstance(exc, requests.ConnectionError):
        return "no connection"
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"HTTP {exc.response.status_code}"
    msg = str(exc) or exc.__class__.__name__
    return msg[:80]


# ---------------------------------------------------------------------------
# Live data fetchers (each raises on failure; the loader falls back per source)
# ---------------------------------------------------------------------------
def ckan_get(base: str, action: str, params: dict) -> dict:
    resp = requests.get(f"{base}/{action}", params=params, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(str(payload.get("error", "API returned success=false"))[:80])
    return payload["result"]


def fetch_live_footfall() -> dict:
    """Hourly footfall profiles per zone from the Cork City pedestrian counter datastore."""
    result = ckan_get(CORK_CKAN, "datastore_search",
                      {"resource_id": FOOTFALL_RESOURCE, "limit": 32000, "sort": "_id desc"})
    records = result.get("records") or []
    if not records:
        raise LookupError("live datastore returned 0 rows")
    df = pd.DataFrame(records)
    df["hour"] = pd.to_numeric(df["Hour"].astype(str).str[:2], errors="coerce")
    df["count"] = pd.to_numeric(df.get("ReportCount"), errors="coerce")
    if df["count"].isna().all():
        df["count"] = pd.to_numeric(df.get("InCount"), errors="coerce").fillna(0) + \
                      pd.to_numeric(df.get("OutCount"), errors="coerce").fillna(0)
    df = df.dropna(subset=["hour", "count"])
    names = df["LocationName"].astype(str).str.lower()
    profiles, counters = {}, {}
    for key, zone in ZONES.items():
        mask = names.str.contains("|".join(map(re.escape, zone["footfall_keys"])), na=False)
        if mask.any():
            hourly = df[mask].groupby("hour")["count"].mean().reindex(range(24), fill_value=0)
            if hourly.sum() > 0:
                profiles[key] = hourly.round().astype(int).tolist()
                counters[key] = sorted(df.loc[mask, "LocationName"].astype(str).unique().tolist())
    if not profiles:
        raise LookupError("no live counters matched UrbanGrid zones")
    return {"profiles": profiles, "counters": counters, "rows": len(df)}


def fetch_live_planning_zone(zone_key: str) -> list[dict]:
    """Planning applications whose development address is on a zone's street."""
    zone = ZONES[zone_key]
    result = ckan_get(CORK_CKAN, "datastore_search", {
        "resource_id": PLANNING_RESOURCE,
        "q": zone["planning_q"],
        "limit": 400,
        "sort": "ReceivedDate desc",
        "fields": "ApplicationNumber,DevelopmentAddress,DevelopmentDescription,ApplicationStatus,ReceivedDate,LinkAppDetails",
    })
    out = []
    for rec in result.get("records") or []:
        address = (rec.get("DevelopmentAddress") or "").strip()
        if not any(k in address.lower() for k in zone["address_keys"]):
            continue
        out.append({
            "app_no": str(rec.get("ApplicationNumber") or ""),
            "zone": zone_key,
            "address": address,
            "description": (rec.get("DevelopmentDescription") or "").strip(),
            "status": (rec.get("ApplicationStatus") or "").strip().upper(),
            "received": (rec.get("ReceivedDate") or "")[:10],
            # Source Latitude/Longitude are mis-projected (they land in Scotland), so we
            # deliberately ignore them and match by street address instead.
            "lat": None,
            "lon": None,
            "link": rec.get("LinkAppDetails") or "",
        })
    return out


def fetch_live_transit_area(area_ref: int) -> list[dict]:
    """Active on-street bus stops (NaPTAN BCT) for one administrative area."""
    result = ckan_get(GOV_CKAN, "datastore_search", {
        "resource_id": NAPTAN_RESOURCE,
        "filters": json.dumps({"AdministrativeAreaRef": area_ref, "StopType": "BCT"}),
        "limit": 5000,
        "fields": "AtcoCode,CommonName,Latitude,Longitude,Status",
    })
    stops = []
    for rec in result.get("records") or []:
        if str(rec.get("Status", "active")).lower() != "active":
            continue
        try:
            lat, lon = float(rec["Latitude"]), float(rec["Longitude"])
        except (TypeError, ValueError, KeyError):
            continue
        stops.append({"stop_id": rec.get("AtcoCode"), "name": rec.get("CommonName") or "Bus stop",
                      "lat": lat, "lon": lon, "headway_min": np.nan})
    if not stops:
        raise LookupError("no active stops returned")
    return stops


def fetch_live_osm() -> list[tuple]:
    """Existing cafés, bakeries, dentists and clothing shops around every zone (OpenStreetMap)."""
    parts = "".join(f'nwr["amenity"~"^(cafe|dentist)$"]({b});nwr["shop"~"^(coffee|bakery|pastry|clothes|boutique)$"]({b});'
                    for b in OSM_BOXES)
    query = f"[out:json][timeout:40];({parts});out center tags;"
    last_err = "no response"
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": query}, headers=HTTP_HEADERS, timeout=(4.0, 45.0))
            r.raise_for_status()
            rows, seen = [], set()
            for e in r.json().get("elements", []):
                t = e.get("tags", {})
                kind = OSM_KIND_TAGS.get(t.get("amenity")) or OSM_KIND_TAGS.get(t.get("shop"))
                lat = e.get("lat", e.get("center", {}).get("lat"))
                lon = e.get("lon", e.get("center", {}).get("lon"))
                if kind and lat is not None and (kind, round(lat, 5), round(lon, 5)) not in seen:
                    seen.add((kind, round(lat, 5), round(lon, 5)))
                    rows.append((kind, (t.get("name") or "").strip(), round(lat, 5), round(lon, 5)))
            if len(rows) >= 20:
                return rows
            last_err = "too few results"
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_err = short_error(exc)
    raise RuntimeError(last_err)


@st.cache_resource(show_spinner=False)
def osm_job() -> dict:
    """Process-wide holder for the background OpenStreetMap refresh (Overpass can take 20-40 s)."""
    return {"rows": None, "at": None, "error": None, "running": False, "tried": 0.0, "lock": threading.Lock()}


def ensure_osm_refresh(job: dict) -> None:
    """Start a background refresh when due. The page never waits for it."""
    now = time.time()
    with job["lock"]:
        due = OSM_LIVE_TTL_S if job["rows"] else OSM_RETRY_S
        if job["running"] or now - job["tried"] < due:
            return
        job["running"], job["tried"] = True, now

    def work():
        try:
            job["rows"], job["at"], job["error"] = fetch_live_osm(), datetime.now().strftime("%H:%M"), None
        except Exception as exc:  # noqa: BLE001
            job["error"] = short_error(exc)
        finally:
            job["running"] = False

    threading.Thread(target=work, daemon=True, name="urbangrid-osm").start()


def current_businesses() -> tuple[pd.DataFrame, tuple[str, str]]:
    job = osm_job()
    ensure_osm_refresh(job)
    if job["rows"]:
        rows, state = job["rows"], ("live", f"OpenStreetMap · {len(job['rows'])} businesses · updated {job['at']}")
    else:
        why = "live refresh running in background" if job["running"] else f"live API: {job['error'] or 'pending'}"
        snap_date = datetime.fromisoformat(OSM_SNAPSHOT_DATE).strftime("%d %b %Y")
        rows, state = OSM_SNAPSHOT, ("fallback", f"OpenStreetMap snapshot of {snap_date} · {why}")
    df = pd.DataFrame(rows, columns=["kind", "name", "lat", "lon"])
    # OSM sometimes maps one business twice (a point and a building outline): drop same-name neighbours.
    norm = df["name"].str.casefold().str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    norm = norm.str.replace(r"[^a-z0-9]", "", regex=True)
    dup = (norm != "") & pd.DataFrame({"n": norm, "k": df["kind"], "la": df["lat"].round(3),
                                       "lo": df["lon"].round(3)}).duplicated()
    df = df[~dup].reset_index(drop=True)
    df["name"] = df["name"].where(df["name"] != "", "Unnamed " + df["kind"].map(lambda k: OSM_KIND_LABELS[k][0]))
    return df, state


def recheck_live_data() -> None:
    load_all_data.clear()
    osm_job()["tried"] = 0.0


# ---------------------------------------------------------------------------
# Loader: run every live call in parallel within a fixed time budget, then
# merge with the embedded fallback dataset source-by-source.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def load_all_data() -> dict:
    tasks = {"footfall": fetch_live_footfall}
    tasks.update({f"planning:{k}": partial(fetch_live_planning_zone, k) for k in ZONES})
    tasks.update({f"transit:{a}": partial(fetch_live_transit_area, a) for a in NAPTAN_AREAS})

    results, errors = {}, {}
    pool = ThreadPoolExecutor(max_workers=len(tasks))
    futures = {name: pool.submit(fn) for name, fn in tasks.items()}
    deadline = time.monotonic() + LIVE_FETCH_BUDGET_S
    for name, fut in futures.items():
        try:
            results[name] = fut.result(timeout=max(0.05, deadline - time.monotonic()))
        except Exception as exc:  # noqa: BLE001 - any failure means "use fallback"
            errors[name] = "timed out" if exc.__class__.__name__ == "TimeoutError" else short_error(exc)
    pool.shutdown(wait=False, cancel_futures=True)

    status = {}

    # --- Footfall ---------------------------------------------------------
    profiles = {k: list(v) for k, v in FALLBACK_DATA["footfall_hourly"].items()}
    footfall_sources = {k: "Embedded Cork dataset" for k in ZONES}
    if "footfall" in results:
        live = results["footfall"]
        for k, prof in live["profiles"].items():
            profiles[k] = prof
            footfall_sources[k] = "Live Cork counters: " + ", ".join(live["counters"][k][:2])
        n = len(live["profiles"])
        status["Footfall"] = ("live" if n == len(ZONES) else "partial",
                              f"{n}/{len(ZONES)} zones from live counters")
    else:
        status["Footfall"] = ("fallback", f"Live API: {errors.get('footfall', 'unavailable')}")

    # --- Planning ---------------------------------------------------------
    planning, live_zones = [], []
    for k in ZONES:
        name = f"planning:{k}"
        if name in results:
            planning.extend(results[name])
            live_zones.append(k)
        else:
            planning.extend([dict(r, link="", source="fallback") for r in FALLBACK_DATA["planning"] if r["zone"] == k])
    for rec in planning:
        rec.setdefault("source", "live")
        rec["category"] = classify_development(rec["description"])
    if len(live_zones) == len(ZONES):
        status["Planning"] = ("live", "Cork City planning register (address-matched)")
    elif live_zones:
        status["Planning"] = ("partial", f"{len(live_zones)}/{len(ZONES)} zones live")
    else:
        first_err = next((v for k, v in errors.items() if k.startswith("planning")), "unavailable")
        status["Planning"] = ("fallback", f"Live API: {first_err}")
    live_dates = [r["received"] for r in planning if r["source"] == "live" and r["received"]]
    planning_latest = max(live_dates) if live_dates else None

    # --- Transit ----------------------------------------------------------
    live_stops = [s for a in NAPTAN_AREAS for s in results.get(f"transit:{a}", [])]
    if live_stops:
        # Keep stops within 1.5 km of any zone so the map and maths stay light.
        lat = np.array([s["lat"] for s in live_stops])
        lon = np.array([s["lon"] for s in live_stops])
        keep = np.zeros(len(live_stops), dtype=bool)
        for z in ZONES.values():
            for sp in z["spots"]:
                keep |= haversine_m(sp["lat"], sp["lon"], lat, lon) <= 1500
        transit = [dict(s, source="live") for s, k in zip(live_stops, keep) if k]
        n_areas = sum(f"transit:{a}" in results for a in NAPTAN_AREAS)
        status["Transit"] = ("live" if n_areas == len(NAPTAN_AREAS) else "partial",
                             f"NTA NaPTAN · {len(transit):,} stops nearby")
    else:
        transit = [dict(s, source="fallback") for s in FALLBACK_DATA["transit"]]
        first_err = next((v for k, v in errors.items() if k.startswith("transit")), "unavailable")
        status["Transit"] = ("fallback", f"Live API: {first_err}")

    return {
        "footfall_profiles": profiles,
        "footfall_sources": footfall_sources,
        "planning": planning,
        "planning_live_zones": live_zones,
        "planning_latest": planning_latest,
        "transit": transit,
        "status": status,
        "errors": errors,
        "fetched_at": datetime.now().strftime("%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Plain-English query parser (rule-based: no API key, cannot fail mid-demo)
# ---------------------------------------------------------------------------
_TIME_RANGE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?\s*(?:-|–|—|to|and|until|till)\s*"
    r"(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?(?![\w])",
    re.IGNORECASE,
)
_PERIODS = [
    (r"\bbreakfast\b", (7.0, 10.0)),
    (r"\bmorning(s)?\b|\bcommute(rs)?\b|\brush hour\b", (7.0, 11.0)),
    (r"\blunch(time)?\b", (12.0, 14.0)),
    (r"\bafternoon(s)?\b", (14.0, 17.0)),
    (r"\bafter work\b|\bevening(s)?\b", (17.0, 20.0)),
]


def _to_24h(hour: int, minute: int, ampm: str | None) -> float:
    if ampm:
        pm = ampm.lower().startswith("p")
        hour = hour % 12 + (12 if pm else 0)
    return hour + minute / 60


def _snap(hours: float) -> str:
    """Round to the nearest half hour inside the slider range and return its label."""
    return fmt_time(min(max(round(hours * 2) / 2, 6.0), 21.0))


def parse_query(text: str) -> dict:
    low = (text or "").lower()
    parsed: dict = {}

    # Location: earliest-mentioned zone alias wins.
    hits = []
    for key, zone in ZONES.items():
        for alias in zone["aliases"]:
            pos = low.find(alias)
            if pos >= 0:
                hits.append((pos, -len(alias), key))
    if hits:
        parsed["zone"] = sorted(hits)[0][2]

    # Business type.
    for name in BUSINESS_PARSE_ORDER:
        if any(re.search(p, low) for p in BUSINESS_TYPES[name]["patterns"]):
            parsed["business"] = name
            break

    # Time window: explicit "8 AM to 10 AM" / "7:30-9am", else period words.
    for m in _TIME_RANGE.finditer(low):
        h1, m1, ap1, h2, m2, ap2 = m.groups()
        if not (ap1 or ap2 or m1 or m2):
            continue  # "2 to 3 competitors" is not a time
        h1, h2 = int(h1), int(h2)
        if h1 > 23 or h2 > 23:
            continue
        ap1 = ap1 or ap2
        start = _to_24h(h1, int(m1 or 0), ap1)
        end = _to_24h(h2, int(m2 or 0), ap2 or ap1)
        if ap2 and not m.group(3) and start > end:   # "11 to 2pm" -> 11am-2pm
            start = _to_24h(h1, int(m1 or 0), "am")
        if end > start:
            parsed["window"] = (_snap(start), _snap(end))
            break
    if "window" not in parsed:
        for pattern, (s, e) in _PERIODS:
            if re.search(pattern, low):
                parsed["window"] = (_snap(s), _snap(e))
                break

    # Day type: "Saturday", "weekend", "Christmas", "December".
    if re.search(r"\b(christmas|xmas|december|festive season)\b", low):
        parsed["day_type"] = "christmas"
    elif re.search(r"\b(saturdays?|weekends?)\b", low):
        parsed["day_type"] = "saturday"

    # Core priority hints.
    if re.search(r"\b(bus|buses|transit|transport|commuters?|train|station|accessib\w*)\b", low):
        parsed["priority"] = "High Transit Access"
    elif re.search(r"\b(competition|competitors?|rivals?|saturat\w*|less crowded|fewer (cafes|shops|outlets))\b", low):
        parsed["priority"] = "Low Competitor Density"
    elif re.search(r"\b(footfall|foot traffic|busiest|passing trade|pedestrians?)\b", low):
        parsed["priority"] = "High Morning Footfall"

    # Rent budget: "€4,500", "4500 euro", "budget of 5k".
    m = (re.search(r"€\s*(\d[\d,]*(?:\.\d+)?)\s*(k)?\b", low)
         or re.search(r"\b(\d[\d,]*(?:\.\d+)?)\s*(k)?\s*(?:euro|eur)\b", low)
         or re.search(r"budget\D{0,15}(\d[\d,]*(?:\.\d+)?)\s*(k)?\b", low))
    if m:
        value = float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
        if 300 <= value <= 100_000:
            parsed["budget"] = int(min(max(round(value / 250) * 250, 1000), 10000))
    return parsed


# ---------------------------------------------------------------------------
# Session state & callbacks
# ---------------------------------------------------------------------------
ss = st.session_state


def apply_parsed(parsed: dict) -> None:
    """Push what the query understood into the sidebar controls."""
    if "business" in parsed:
        ss.business = parsed["business"]
        ss.prev_business = parsed["business"]
    if "priority" in parsed:
        ss.priority = parsed["priority"]
    if "budget" in parsed:
        ss.budget = parsed["budget"]
    ss.zone_focus = parsed.get("zone", AUTO)
    ss.day_type = parsed.get("day_type", "weekday")
    if "window" in parsed:
        ss.time_window = parsed["window"]
    elif "business" in parsed:
        ss.time_window = BUSINESS_TYPES[parsed["business"]]["window"]
    ss.parsed = parsed
    ss.parsed_query = ss.query


def on_analyze() -> None:
    apply_parsed(parse_query(ss.query))


def on_sample_query() -> None:
    ss.query = SAMPLE_QUERY
    on_analyze()


if "initialised" not in ss:
    ss.query = SAMPLE_QUERY
    ss.business = "Specialty Coffee Shop"
    ss.budget = 6000
    ss.priority = PRIORITIES[0]
    ss.zone_focus = AUTO
    ss.day_type = "weekday"
    ss.time_window = BUSINESS_TYPES["Specialty Coffee Shop"]["window"]
    apply_parsed(parse_query(SAMPLE_QUERY))
    ss.initialised = True

# When the user switches Business Type in the sidebar, jump to that business's typical hours.
# (Detected by comparing with the previous run rather than via on_change, which is more robust.)
if ss.business != ss.get("prev_business"):
    ss.time_window = BUSINESS_TYPES[ss.business]["window"]
    ss.prev_business = ss.business

# Defensive: the time window must always be two valid slider labels.
_tw = ss.get("time_window")
if not (isinstance(_tw, (tuple, list)) and len(_tw) == 2 and all(t in TIME_OPTIONS for t in _tw)):
    ss.time_window = BUSINESS_TYPES[ss.get("business", "Specialty Coffee Shop")]["window"]


# ---------------------------------------------------------------------------
# Analysis engine
# ---------------------------------------------------------------------------
def slot_rates(profile, start_h: float, end_h: float):
    """Half-hour slots in [start, end) with interpolated pedestrians/hour."""
    ts = np.arange(start_h, end_h, 0.5)
    rates = np.interp(ts + 0.25, np.arange(24) + 0.5, np.asarray(profile, dtype=float))
    return ts, rates


def peak_window(ts, rates, slots: int = 3):
    """Best contiguous 90-minute block (or the whole window if shorter)."""
    if len(ts) <= slots:
        return float(ts[0]), float(ts[-1] + 0.5), float(rates.mean())
    sums = np.convolve(rates, np.ones(slots), mode="valid")
    i = int(np.argmax(sums))   # earliest block wins ties
    return float(ts[i]), float(ts[i] + 0.5 * slots), float(sums[i] / slots)


def analyze(data: dict, business: str, budget: int, priority: str, zone_focus: str, window,
            businesses: pd.DataFrame) -> dict:
    start_h, end_h = t2h(window[0]), t2h(window[1])
    if end_h <= start_h:
        end_h = start_h + 0.5
    comp_cat = BUSINESS_TYPES[business]["competitor"]
    rivals = businesses[businesses["kind"] == BUSINESS_TYPES[business]["osm_kind"]].reset_index(drop=True)

    stops = pd.DataFrame(data["transit"])
    plan = pd.DataFrame(data["planning"])
    cutoff = (date.today() - timedelta(days=PENDING_LOOKBACK_DAYS)).isoformat()
    plan["pending"] = plan["status"].str.upper().isin(PENDING_STATUSES) & (plan["received"].fillna("") >= cutoff)
    plan["lat"] = pd.to_numeric(plan["lat"], errors="coerce")
    plan["lon"] = pd.to_numeric(plan["lon"], errors="coerce")
    has_coords = plan["lat"].notna().to_numpy()

    rows, near_ids = [], {}
    for zk, zone in ZONES.items():
        ts, rates = slot_rates(data["footfall_profiles"][zk], start_h, end_h)
        p_start, p_end, p_avg = peak_window(ts, rates)
        for sp in zone["spots"]:
            d_stop = haversine_m(sp["lat"], sp["lon"], stops["lat"].to_numpy(), stops["lon"].to_numpy())
            # Planning: spatial radius when coordinates are trustworthy, street match otherwise.
            d_plan = np.where(has_coords, haversine_m(sp["lat"], sp["lon"],
                                                      plan["lat"].fillna(0).to_numpy(),
                                                      plan["lon"].fillna(0).to_numpy()), np.nan)
            near = np.where(has_coords, d_plan <= PLANNING_RADIUS_M, (plan["zone"] == zk).to_numpy())
            pend = near & plan["pending"].to_numpy()
            comp = pend & (plan["category"] == comp_cat).to_numpy()
            near_ids[(zk, sp["spot"])] = plan.index[pend].tolist()
            d_rival = haversine_m(sp["lat"], sp["lon"], rivals["lat"].to_numpy(), rivals["lon"].to_numpy())
            rows.append({
                "zone_key": zk, "zone": zone["name"], "spot": sp["spot"],
                "lat": sp["lat"], "lon": sp["lon"], "share": sp["share"],
                "window_rate": float(rates.mean() * sp["share"]),
                "peak_start": p_start, "peak_end": p_end,
                "peak_rate": float(p_avg * sp["share"]),
                "stops_150m": int((d_stop <= 150).sum()),
                "nearest_stop_m": float(d_stop.min()) if len(d_stop) else np.inf,
                "pending": int(pend.sum()), "competitors": int(comp.sum()),
                "existing": int((d_rival <= EXISTING_RADIUS_M).sum()),
                "planning_mode": "street" if zk in data["planning_live_zones"] else "radius",
                "rent": zone["rent"],
            })
    df = pd.DataFrame(rows)

    wf, wc, wt, wb = PRIORITY_WEIGHTS[priority]
    df["footfall_score"] = df["window_rate"] / df["window_rate"].max()
    # Competition = 60% planning pipeline (new rivals coming) + 40% existing saturation (rivals already trading).
    pipeline_score = (1 - df["competitors"] / 4).clip(lower=0)
    existing_score = 1 - df["existing"] / max(int(df["existing"].max()), 3)
    df["competition_score"] = 0.6 * pipeline_score + 0.4 * existing_score
    df["transit_score"] = 0.6 * (df["stops_150m"].clip(upper=6) / 6) + \
        0.4 * (1 - df["nearest_stop_m"] / 400).clip(lower=0)
    df["budget_score"] = np.where(df["rent"] <= budget, 1.0, (1 - (df["rent"] - budget) / 2000).clip(lower=0))
    df["score"] = (100 * (wf * df["footfall_score"] + wc * df["competition_score"] +
                          wt * df["transit_score"] + wb * df["budget_score"])).round().astype(int)

    pool = df if zone_focus == AUTO else df[df["zone_key"] == zone_focus]
    best = df.loc[pool["score"].idxmax()]
    zone_rows = df[df["zone_key"] == best["zone_key"]]
    hottest = df.loc[zone_rows["peak_rate"].idxmax()]
    alt = df[df["zone_key"] != best["zone_key"]].sort_values("score", ascending=False).head(1)

    # Transit detail around the recommended spot.
    d_best = haversine_m(best["lat"], best["lon"], stops["lat"].to_numpy(), stops["lon"].to_numpy())
    radius = next((r for r in TRANSIT_RADII_M if (d_best <= r).sum() >= 2), TRANSIT_RADII_M[-1])
    in_r = stops[d_best <= radius].assign(distance_m=d_best[d_best <= radius].round())
    known = in_r["headway_min"].dropna() if "headway_min" in in_r else pd.Series(dtype=float)
    headway = float(known.mean()) if len(known) else float(ZONES[best["zone_key"]]["headway_est"])
    headway_estimated = len(known) == 0
    d_rival_best = haversine_m(best["lat"], best["lon"], rivals["lat"].to_numpy(), rivals["lon"].to_numpy())
    rivals = rivals.assign(distance_m=np.round(d_rival_best)).sort_values("distance_m")

    return {
        "spots": df,
        "best": best,
        "hottest": hottest,
        "alt": alt.iloc[0] if len(alt) else None,
        "window": (start_h, end_h),
        "competitor_category": comp_cat,
        "plan": plan,
        "near_plan_ids": near_ids[(best["zone_key"], best["spot"])],
        "transit_radius": radius,
        "transit_in_radius": in_r.sort_values("distance_m"),
        "transit_all_dist": d_best,
        "headway": headway,
        "headway_estimated": headway_estimated,
        "stops": stops,
        "rivals": rivals,
        "osm_kind": BUSINESS_TYPES[business]["osm_kind"],
    }


# ---------------------------------------------------------------------------
# Icons: inline SVG line icons (consistent, crisp, theme-coloured via currentColor)
# ---------------------------------------------------------------------------
_ICON_PATHS = {
    "grid": "<rect x='3' y='3' width='7' height='7' rx='1.5'/><rect x='14' y='3' width='7' height='7' rx='1.5'/>"
            "<rect x='14' y='14' width='7' height='7' rx='1.5'/><rect x='3' y='14' width='7' height='7' rx='1.5'/>",
    "pin": "<path d='M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z'/><circle cx='12' cy='10' r='3'/>",
    "store": "<path d='M3 9l1.5-5h15L21 9'/><path d='M3 9h18v1.5a3 3 0 0 1-6 0 3 3 0 0 1-6 0 3 3 0 0 1-6 0Z'/>"
             "<path d='M5 13v8h14v-8'/><path d='M10 21v-4h4v4'/>",
    "clock": "<circle cx='12' cy='12' r='10'/><path d='M12 6v6l4 2'/>",
    "target": "<circle cx='12' cy='12' r='10'/><circle cx='12' cy='12' r='6'/><circle cx='12' cy='12' r='2'/>",
    "euro": "<path d='M4 10h12'/><path d='M4 14h9'/><path d='M19 6a7.7 7.7 0 0 0-5.2-2A7.9 7.9 0 0 0 6 12c0 4.4 3.5 8 7.8 8 2 0 3.8-.8 5.2-2'/>",
    "users": "<path d='M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2'/><circle cx='9' cy='7' r='4'/>"
             "<path d='M22 21v-2a4 4 0 0 0-3-3.87'/><path d='M16 3.13a4 4 0 0 1 0 7.75'/>",
    "building": "<rect x='4' y='2' width='16' height='20' rx='2'/><path d='M9 22v-4h6v4'/>"
                "<path d='M8 6h.01M12 6h.01M16 6h.01M8 10h.01M12 10h.01M16 10h.01M8 14h.01M12 14h.01M16 14h.01'/>",
    "bus": "<path d='M8 6v6M15 6v6M2 12h19.6'/><path d='M18 18h3s.5-1.7.8-2.8c.1-.4.2-.8.2-1.2 0-.4-.1-.8-.2-1.2l-1.4-5"
           "C20.1 6.8 19.1 6 18 6H4a2 2 0 0 0-2 2v10h3'/><circle cx='7' cy='18' r='2'/><path d='M9 18h5'/>"
           "<circle cx='16' cy='18' r='2'/>",
    "check": "<circle cx='12' cy='12' r='10'/><path d='m9 12 2 2 4-4'/>",
    "alert": "<path d='m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z'/><path d='M12 9v4M12 17h.01'/>",
    "x": "<circle cx='12' cy='12' r='10'/><path d='m15 9-6 6M9 9l6 6'/>",
    "info": "<circle cx='12' cy='12' r='10'/><path d='M12 16v-4M12 8h.01'/>",
    "trend": "<path d='M22 7 13.5 15.5 8.5 10.5 2 17'/><path d='M16 7h6v6'/>",
    "minus": "<circle cx='12' cy='12' r='10'/><path d='M8 12h8'/>",
    "bulb": "<path d='M9 18h6M10 22h4'/><path d='M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.2 1 2V17h6v-.3c0-.8.4-1.5 1-2A7 7 0 0 0 12 2Z'/>",
    "calendar": "<rect x='3' y='4' width='18' height='18' rx='2'/><path d='M16 2v4M8 2v4M3 10h18'/>",
    "briefcase": "<rect x='2' y='7' width='20' height='14' rx='2'/><path d='M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16'/>",
}


def icon(name: str, size: int = 16, cls: str = "") -> str:
    return (f"<svg class='ug-ic {cls}' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' "
            f"stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' "
            f"aria-hidden='true'>{_ICON_PATHS[name]}</svg>")


def tip(text: str) -> str:
    """Small info icon that reveals `text` on hover."""
    return f"<span class='ug-tip' data-tip='{esc(text)}'>{icon('info', 15)}</span>"


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def step(n, label: str) -> None:
    html_block(f"<div class='ug-step'><span class='n'>{n}</span>{esc(label)}</div>")


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  --bg: #0a1120; --surface: #0f172a; --surface-2: #131d33; --border: #1e293f; --border-strong: #2a3854;
  --text: #e6edf7; --text-2: #a3b1c6; --text-3: #6f7f96;
  --accent: #38bdf8; --accent-soft: rgba(56,189,248,.12);
  --good: #4ade80; --good-soft: rgba(74,222,128,.12);
  --warn: #fbbf24; --warn-soft: rgba(251,191,36,.12);
  --bad: #f87171;  --bad-soft: rgba(248,113,113,.12);
  --radius: 14px;
}
html, body, .stApp, [data-testid="stSidebar"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: radial-gradient(1000px 420px at 15% -10%, rgba(56,189,248,.07), transparent 65%), var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
/* Style-only markdown blocks still take a row of vertical gap: collapse them (the CSS still applies). */
[data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] > style:only-child) { display: none; }
#MainMenu, footer, [data-testid="stDeployButton"], [data-testid="stAppDeployButton"] { display: none; }
.stAlert p, .stAlert li, [data-testid^="stAlert"] p, [data-testid^="stAlert"] li { font-size: .88rem !important; line-height: 1.55 !important; }
.block-container { padding-top: 1.8rem; padding-bottom: 2.5rem; max-width: 1240px; }
section[data-testid="stSidebar"] { background: #0c1424; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] .ug-side-h { font-size: .7rem; text-transform: uppercase; letter-spacing: .12em;
  color: var(--text-3); font-weight: 700; margin: 1.2rem 0 .4rem; }
.ug-ic { display: inline-block; vertical-align: -3px; flex-shrink: 0; }

/* Header */
.ug-brand { display: flex; gap: 14px; align-items: center; }
.ug-logo { width: 44px; height: 44px; border-radius: 12px; display: grid; place-items: center; color: var(--accent);
  background: var(--accent-soft); border: 1px solid rgba(56,189,248,.28); }
.ug-title { font-size: 1.5rem; font-weight: 700; letter-spacing: -.015em; color: var(--text); line-height: 1.2; }
.ug-title b { color: var(--accent); font-weight: 800; }
.ug-title .sep { color: var(--text-3); font-weight: 300; margin: 0 .45rem; }
.ug-sub { color: var(--text-2); font-size: .92rem; margin-top: 3px; }

/* Step labels */
.ug-step { display: flex; align-items: center; gap: 10px; font-size: .7rem; letter-spacing: .14em; text-transform: uppercase;
  color: var(--text-3); font-weight: 700; margin: 1.7rem 0 .65rem; }
.ug-step .n { width: 20px; height: 20px; border-radius: 50%; display: grid; place-items: center; letter-spacing: 0;
  background: var(--surface-2); border: 1px solid var(--border-strong); color: var(--text-2); font-size: .68rem; }
.ug-step::after { content: ""; flex: 1; height: 1px; background: var(--border); }

/* Query bar */
.stTextInput input { font-size: 1rem !important; padding: .8rem 1rem !important; }
.stTextInput div[data-baseweb="input"] { border-radius: 12px !important; border: 1px solid var(--border-strong) !important;
  background: var(--surface) !important; }
.stTextInput div[data-baseweb="input"]:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-soft); }
[data-testid="stTextInputRootElement"] { min-height: 52px; border-radius: 12px !important;
  border: 1px solid var(--border-strong) !important; background: var(--surface) !important; }
[data-testid="stTextInputRootElement"]:focus-within { border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft); }
[data-testid="stTextInputRootElement"] input { height: 50px; }
div.stButton > button[kind="primary"] { border-radius: 12px; font-weight: 600; padding: 0 1rem; min-height: 52px; border: 0;
  background: #0ea5e9; color: #fff; }
div.stButton > button[kind="primary"]:hover { background: #38bdf8; }
.ug-chips { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: .55rem; }
.ug-chips .lbl { color: var(--text-3); font-size: .78rem; margin-right: 4px; }
.ug-chip { display: inline-flex; align-items: center; gap: 6px; font-size: .8rem; padding: 4px 10px; border-radius: 8px;
  background: var(--surface-2); border: 1px solid var(--border); color: var(--text-2); }
.ug-chip.q { color: #d4f1fd; border-color: rgba(56,189,248,.35); background: var(--accent-soft); }
.ug-chip.q .ug-ic { color: var(--accent); }

/* Verdict */
.ug-verdict { display: flex; gap: 28px; align-items: center; justify-content: space-between; padding: 22px 26px;
  border-radius: var(--radius); border: 1px solid var(--border-strong);
  background: linear-gradient(135deg, rgba(56,189,248,.10) 0%, rgba(15,23,42,.9) 55%); }
.ug-verdict .main { flex: 1 1 420px; min-width: 0; }
.ug-eyebrow { font-size: .7rem; letter-spacing: .13em; text-transform: uppercase; color: var(--accent); font-weight: 700;
  display: flex; gap: 6px; align-items: center; }
.ug-verdict .where { font-size: 1.45rem; font-weight: 700; margin: .35rem 0 .85rem; color: var(--text); letter-spacing: -.01em; }
.ug-reasons { display: flex; flex-wrap: wrap; gap: 8px; }
.ug-reason { display: inline-flex; align-items: center; gap: 7px; font-size: .83rem; padding: 6px 11px; border-radius: 9px;
  border: 1px solid var(--border); background: rgba(10,17,32,.55); color: var(--text-2); }
.ug-reason.good .ug-ic { color: var(--good); } .ug-reason.warn .ug-ic { color: var(--warn); } .ug-reason.bad .ug-ic { color: var(--bad); }
.ug-note { margin-top: .75rem; font-size: .82rem; color: var(--text-2); display: flex; gap: 7px; align-items: center; }
.ug-note .ug-ic { color: var(--warn); }
.ug-ring { --p: 0; width: 116px; height: 116px; border-radius: 50%; flex-shrink: 0; display: grid; place-items: center;
  background: conic-gradient(var(--accent) calc(var(--p) * 1%), #1c2840 0); }
.ug-ring .in { width: 94px; height: 94px; border-radius: 50%; background: #0d1628; display: flex; flex-direction: column;
  align-items: center; justify-content: center; }
.ug-ring .num { font-size: 2rem; font-weight: 800; line-height: 1; color: var(--text); }
.ug-ring .lbl { font-size: .58rem; letter-spacing: .1em; text-transform: uppercase; color: var(--text-3); margin-top: 5px; text-align: center; }

/* Action cards */
.ug-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 18px 20px 20px;
  min-height: 222px; display: flex; flex-direction: column; gap: 12px; }
.ug-card-head { display: flex; justify-content: space-between; align-items: center; }
.ug-card-title { display: flex; align-items: center; gap: 8px; font-size: .74rem; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: var(--text-2); }
.ug-card-title .ug-ic { color: var(--accent); }
.ug-card-metric { font-size: 1.65rem; font-weight: 700; color: var(--text); letter-spacing: -.01em; line-height: 1.15; }
.ug-card { min-height: 236px; }
.ug-card-detail { font-size: .89rem; color: var(--text-2); line-height: 1.55; }
.ug-status { display: inline-flex; align-self: flex-start; align-items: center; gap: 6px; font-size: .78rem; font-weight: 600;
  padding: 4px 11px 4px 9px; border-radius: 999px; }
.ug-status.good { color: var(--good); background: var(--good-soft); }
.ug-status.warn { color: var(--warn); background: var(--warn-soft); }
.ug-status.bad  { color: var(--bad);  background: var(--bad-soft); }
.ug-status.neutral { color: var(--text-2); background: var(--surface-2); }

/* Hover tooltip */
.ug-tip { position: relative; color: var(--text-3); cursor: help; display: inline-flex; }
.ug-tip:hover { color: var(--text-2); }
.ug-tip:hover::after { content: attr(data-tip); position: absolute; right: -6px; top: 24px; width: 250px; padding: 9px 11px;
  border-radius: 9px; background: #1a2640; border: 1px solid var(--border-strong); color: var(--text); font-size: .75rem;
  line-height: 1.5; z-index: 50; text-transform: none; letter-spacing: 0; font-weight: 400; white-space: normal;
  box-shadow: 0 10px 28px rgba(0,0,0,.45); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { padding: 10px 14px; font-weight: 600; color: var(--text-2); }
.stTabs [aria-selected="true"] { color: var(--text); }

/* Map side panel */
.ug-panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 16px 6px; }
.ug-panel h4 { font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; color: var(--text-3); margin: 0 0 .35rem;
  padding: 0; font-weight: 700; display: flex; gap: 7px; align-items: center; }
.ug-panel h4 .ug-ic { margin-right: 2px; }
.ug-panel h4 + .ug-list, .ug-panel h4 + .ug-empty { margin-top: .2rem; }
.ug-panel .who { font-size: .95rem; font-weight: 600; color: var(--text); margin-bottom: .8rem; }
.ug-list { list-style: none; margin: 0 0 1rem; padding: 0; }
.ug-list li { display: flex; justify-content: space-between; gap: 10px; font-size: .82rem; padding: 7px 0;
  border-bottom: 1px solid var(--border); color: var(--text-2); margin: 0; }
.ug-list li:last-child { border-bottom: 0; }
.ug-list li .r { color: var(--text-3); white-space: nowrap; }
.ug-list li .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 7px; vertical-align: 1px; }
.ug-empty { font-size: .82rem; color: var(--text-3); margin: 0 0 1rem; }
/* ---------- Landing ---------- */
.ug-hero { text-align: center; padding: clamp(.4rem, 3vh, 2.6rem) 0 clamp(.4rem, 1.4vh, 1rem); }
.ug-hero .badge { display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 999px;
  border: 1px solid rgba(56,189,248,.3); background: var(--accent-soft); color: #bae6fd; font-size: .8rem; font-weight: 600; }
.ug-hero h1 { font-size: clamp(1.9rem, min(4.2vw, 6.4vh), 3.2rem) !important; line-height: 1.08 !important;
  font-weight: 800 !important; letter-spacing: -.03em; color: var(--text) !important;
  margin: clamp(.6rem, 2vh, 1.1rem) 0 clamp(.4rem, 1.4vh, .8rem) !important; padding: 0 !important; }
.ug-hero h1 span.w { color: var(--text) !important; }
.ug-hero h1 span.a { color: var(--accent) !important; }
.ug-hero p { font-size: clamp(.95rem, min(1.25vw, 2.2vh), 1.12rem); color: var(--text-2); max-width: 680px; margin: 0 auto; line-height: 1.55; }
.ug-try { text-align: center; color: var(--text-3); font-size: .76rem; letter-spacing: .12em; text-transform: uppercase;
  font-weight: 700; margin: clamp(.5rem, 2vh, 1.3rem) 0 .4rem; }
.st-key-example_pick [data-testid="stButtonGroup"] > div { justify-content: center !important; flex-wrap: wrap; }
.ug-how { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: clamp(.7rem, 2.4vh, 2rem) 0 0; }
.ug-how .h { padding: 13px 16px; border-radius: 14px; border: 1px solid var(--border); background: var(--surface); }
.ug-how .hd { display: flex; align-items: center; gap: 10px; margin-bottom: 5px; }
.ug-how .n { width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center; font-weight: 800;
  background: var(--accent-soft); color: var(--accent); font-size: .8rem; flex: none; }
.ug-how b { color: var(--text); font-size: .98rem; }
.ug-how .h > span { color: var(--text-2); font-size: .85rem; line-height: 1.45; display: block; }
.ug-checks { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.ug-check { padding: 12px 14px; border-radius: 14px; border: 1px solid var(--border); background: rgba(15,23,42,.6); }
.ug-check .t { display: flex; align-items: center; gap: 8px; font-weight: 700; color: var(--text); font-size: .95rem; }
.ug-check .t .ug-ic { color: var(--accent); }
.ug-check .d { color: var(--text-3); font-size: .8rem; margin-top: 6px; line-height: 1.45; }
.ug-check .live { display: inline-flex; align-items: center; gap: 6px; font-size: .72rem; margin-top: 8px; font-weight: 600; }
.ug-check .live i { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.ug-sec-h { text-align: center; color: var(--text-3); font-size: .72rem; letter-spacing: .14em; text-transform: uppercase;
  font-weight: 700; margin: clamp(.7rem, 2.2vh, 1.8rem) 0 .6rem; }
@media (max-width: 900px) { .ug-how, .ug-checks { grid-template-columns: 1fr 1fr; } .ug-hero h1 { font-size: 2.2rem !important; } }
@media (max-height: 820px) { .ug-hero .badge { display: none; } .ug-hero h1 { margin-top: 0 !important; } }
@media (max-height: 740px) { .ug-how .h > span { display: none; } .ug-how .hd { margin-bottom: 0; } }

/* ---------- Results ---------- */
.ug-topbar .ug-logo { width: 34px; height: 34px; border-radius: 10px; }
.ug-topbar .name { font-weight: 800; color: var(--text); font-size: 1.05rem; }
.ug-topbar .name span { color: var(--text-3); font-weight: 500; margin-left: 8px; font-size: .9rem; }
.ug-qcard { padding: 14px 18px; border-radius: 14px; border: 1px solid var(--border); background: var(--surface);
  margin: .7rem 0 .9rem; }
.ug-qcard .qt { font-size: 1.02rem; color: var(--text); font-style: italic; }
.ug-qcard .ug-chips { margin-top: .6rem; }
div[data-testid="stButtonGroup"] { margin-bottom: .2rem; }
.ug-page-h { margin: 1.3rem 0 1.1rem; }
.ug-page-h .k { font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); font-weight: 700; }
.ug-page-h h2 { font-size: 1.9rem !important; font-weight: 800 !important; letter-spacing: -.02em; color: var(--text) !important;
  margin: .2rem 0 .25rem !important; padding: 0 !important; }
.ug-page-h p { color: var(--text-2); font-size: 1rem; margin: 0; max-width: 820px; }
.ug-pager { text-align: center; color: var(--text-3); font-size: .82rem; padding-top: .55rem; }
.ug-pager i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--border-strong); margin: 0 3px; }
.ug-pager i.on { background: var(--accent); width: 22px; border-radius: 6px; }
.ug-divider { height: 1px; background: var(--border); margin: 1.6rem 0 1rem; }

/* Simple mode */
.ug-plain { color: var(--text-2); font-size: .95rem; margin-top: .45rem; line-height: 1.55; }
.ug-plain b { color: var(--text); font-weight: 600; }
.ug-verdict.simple .where { font-size: 1.5rem; line-height: 1.3; margin: .4rem 0 .6rem; }
.ug-plain-body { color: var(--text-2); font-size: 1.02rem; line-height: 1.6; }
.ug-plain-body .g { color: var(--good); font-weight: 700; }
.ug-plain-body .w { color: var(--warn); font-weight: 700; }

/* Footfall + break-even tiles */
.ug-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: .2rem 0 .8rem; }
.ug-tile { padding: 12px 14px; border-radius: 12px; border: 1px solid var(--border); background: var(--surface); }
.ug-tile .k { font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; color: var(--text-3); font-weight: 700; }
.ug-tile .v { font-size: 1.25rem; font-weight: 700; color: var(--text); margin-top: 3px; letter-spacing: -.01em; }
.ug-tile .s { font-size: .8rem; color: var(--text-2); margin-top: 2px; }
.ug-be-head { padding: 16px 18px; border-radius: var(--radius); border: 1px solid var(--border-strong); background: var(--surface);
  margin-bottom: 10px; }
.ug-be-head .big { font-size: 1.6rem; font-weight: 800; color: var(--text); letter-spacing: -.02em; }
.ug-be-head .big small { font-size: .95rem; font-weight: 500; color: var(--text-2); letter-spacing: 0; }
.ug-be-head .line { color: var(--text-2); font-size: .9rem; margin-top: 4px; line-height: 1.5; }
.ug-range { position: relative; height: 10px; border-radius: 6px; background: var(--surface-2); margin: 26px 2px 6px;
  border: 1px solid var(--border); }
.ug-range .band { position: absolute; top: 0; bottom: 0; background: rgba(74,222,128,.28); border-radius: 6px; }
.ug-range .mid { position: absolute; top: -3px; width: 2px; height: 14px; background: #4ade80; opacity: .9; }
.ug-range .mid span { position: absolute; top: 17px; left: 50%; transform: translateX(-50%); white-space: nowrap;
  font-size: .7rem; color: #86efac; }
.ug-range .mark { position: absolute; top: -7px; width: 3px; height: 22px; border-radius: 2px; background: var(--text); }
.ug-range .mark span { position: absolute; bottom: 24px; left: 50%; transform: translateX(-50%); white-space: nowrap;
  font-size: .74rem; color: var(--text); font-weight: 700; }
.ug-range-lbl { margin-top: 4px; display: flex; justify-content: space-between; font-size: .74rem; color: var(--text-3); }
.ug-report { display: flex; align-items: center; gap: 10px; color: var(--text-2); font-size: .86rem; }
.ug-report .ug-ic { color: var(--accent); }

.ug-legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: .78rem; color: var(--text-3); margin: .6rem 0 0; }
.ug-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }

/* Source status (header popover) */
.ug-src { display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 9px 0;
  border-bottom: 1px solid var(--border); font-size: .84rem; }
.ug-src .name { color: var(--text); font-weight: 600; }
.ug-src .note { color: var(--text-3); font-size: .74rem; display: block; font-weight: 400; margin-top: 2px; }
.ug-src .tag { font-size: .68rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; white-space: nowrap; letter-spacing: .04em; }
.ug-src .tag.live { color: var(--good); background: var(--good-soft); }
.ug-src .tag.partial { color: var(--warn); background: var(--warn-soft); }
.ug-src .tag.fallback { color: #a5b4fc; background: rgba(129,140,248,.14); }

/* Method tab */
.ug-method { color: var(--text-2); font-size: .9rem; line-height: 1.6; }
.ug-method h4 { color: var(--text); font-size: .95rem; margin: 1rem 0 .3rem; padding: 0; }

/* Tables (HTML fallback) */
.ug-table-wrap { max-height: 420px; overflow: auto; border: 1px solid var(--border); border-radius: 12px; }
.ug-table { width: 100%; border-collapse: collapse; font-size: .82rem; }
.ug-table th { position: sticky; top: 0; background: var(--surface-2); color: var(--text-2); text-align: left; padding: 8px 10px;
  border-bottom: 1px solid var(--border); white-space: nowrap; }
.ug-table td { padding: 7px 10px; border-bottom: 1px solid var(--border); color: var(--text); vertical-align: top; }

/* Footer */
.ug-footer { margin-top: 1.8rem; padding: 16px 20px; border-radius: var(--radius); border: 1px solid var(--border);
  background: var(--surface); display: flex; justify-content: space-between; gap: 14px; flex-wrap: wrap; align-items: center; }
.ug-footer .money { font-weight: 600; color: var(--text); display: flex; gap: 8px; align-items: center; }
.ug-footer .money .ug-ic { color: var(--accent); }
.ug-footer .money b { color: var(--accent); }
.ug-footer .credits { color: var(--text-3); font-size: .78rem; }
@media (max-width: 640px) {
  .ug-title { font-size: 1.2rem; } .ug-card { min-height: 0; } .ug-verdict { padding: 18px; }
  .ug-verdict .where { font-size: 1.15rem; } .ug-ring { width: 96px; height: 96px; } .ug-ring .in { width: 78px; height: 78px; }
}
</style>
"""
html_block(CSS)


# ---------------------------------------------------------------------------
# Area report builders (PDF via fpdf2; HTML fallback)
# ---------------------------------------------------------------------------
_PDF_SUBS = {"€": "EUR ", "–": "-", "—": "-", "≈": "~", "≤": "<=", "≥": ">=", "’": "'", "‘": "'", "“": '"', "”": '"',
             "…": "...", "★": "*", "→": "->", "−": "-", " ": " "}


def pdf_txt(text) -> str:
    out = str(text)
    for a, b in _PDF_SUBS.items():
        out = out.replace(a, b)
    out = re.sub(r"<[^>]+>", "", html.unescape(out))
    return out.encode("latin-1", "replace").decode("latin-1")


def report_payload() -> dict:
    near_plan_rows = plan_all.loc[plan_all.index.isin(R["near_plan_ids"])]
    return {
        "generated": datetime.now().strftime("%d %b %Y, %H:%M"),
        "query": ss.query,
        "settings": f"{ss.business} · {DAY_LABEL} · {ss.time_window[0]} – {ss.time_window[1]} · {ss.priority} · "
                    f"budget ≤ €{ss.budget:,}/month",
        "where": f"{best['spot']}, {best['zone']}",
        "scope": scope,
        "score": int(best["score"]),
        "reasons": [(c, t) for c, _, t in reasons],
        "cards": [
            ("Peak Hours & Traffic", f"{fmt_time(best['peak_start'])} – {fmt_time(best['peak_end'])}", ff_status, ff_cls,
             f"Highest {period} pedestrian traffic is concentrated near the {hottest['spot']}. "
             f"About {peak_rate:,.0f} pedestrians/hr at peak."),
            ("Local Planning & Competition", f"{n_pend} Pending Application{'s' if n_pend != 1 else ''}", pl_status,
             pl_cls, plan_detail),
            ("Public Transport Access", f"{n_stops} Bus Stop{'s' if n_stops != 1 else ''} (<{R['transit_radius']}m)",
             tr_status, tr_cls, tr_detail),
        ],
        "chart": {"ts": day_ts.tolist(), "rates": day_rates.tolist(), "peak": in_peak.tolist(),
                  "window": in_window.tolist()},
        "footfall_note": f"{DAY_LABEL} pedestrians/hr at {best['spot']}. Peak 90 min {fmt_time(best['peak_start'])} – "
                         f"{fmt_time(best['peak_end'])}; ≈{window_passers:,.0f} passers-by in your hours. "
                         f"Source: {DATA['footfall_sources'][best['zone_key']]}.",
        "breakeven": [
            ("Customers/day to break even", f"≈{need_day:,.0f}" if np.isfinite(need_day) else "n/a"),
            ("Share of passers-by needed", f"{capture_need:.2f}% ({be_status})" if np.isfinite(capture_need) else "n/a"),
            ("Typical range for this business", f"{cap_lo:g}–{cap_hi:g}% (average {cap_mid:g}%)"),
            ("Monthly costs covered", f"€{fixed:,.0f} (rent €{be_rent:,.0f} + other €{be_other:,.0f})"),
            ("Rent you could afford (keeping 10% profit)",
             (f"up to €{max_rent:,.0f}/month vs asking €{be_rent:,.0f} ("
              + (f"€{rent_gap:,.0f} headroom)" if rent_gap >= 0 else f"€{-rent_gap:,.0f} too high)"))
             if max_rent > 0 else "none: average takings do not cover other costs"),
            ("Monthly profit if you win an average share", f"{'-' if typical_profit < 0 else ''}€{abs(typical_profit):,.0f}"
             f" (≈{typical_cust:,.0f} customers/day, a scenario)"),
        ],
        "be_note": f"Assumes €{be_spend:,.2f} average spend, {be_margin}% gross margin, {be_days} trading days, open "
                   f"{be_hours[0]} – {be_hours[1]} (≈{passers_day:,.0f} passers-by/day). {be_status}: {be_msg}."
                   + (f" {be_appt_note}" if be_appt_note else ""),
        "stops": [(r["name"], f"{r['distance_m']:.0f}m") for _, r in R["transit_in_radius"].head(5).iterrows()],
        "applications": [(r["address"], f"{r['status'].title()} · {CATEGORY_LABELS[r['category']]}")
                         for _, r in near_plan_rows.head(5).iterrows()],
        "rivals": [(r["name"], f"{r['distance_m']:.0f}m")
                   for _, r in R["rivals"][R["rivals"]["distance_m"] <= EXISTING_RADIUS_M].head(6).iterrows()],
        "rivals_title": f"Existing {kind_many} within {EXISTING_RADIUS_M}m ({n_exist})",
        "zones": [(r["zone"], r["spot"], int(r["score"]), r["peak"], f"€{int(r['rent']):,}", r["pick"] == "★")
                  for _, r in rank.iterrows()],
        "sources": [f"{k}: {v[0].upper()} - {v[1]}" for k, v in STATUS.items()],
        "limits": [
            "Historical seasonal variance: Saturday and Christmas use estimated adjustments, not measured Cork data; "
            "summer, college terms and festivals are not modelled.",
            "Real-time rent negotiation status is unknown; rents are indicative estimates.",
            "Bus intervals, some footfall and the break-even figures are estimates.",
            "Existing businesses come from OpenStreetMap and may miss recent openings or closures.",
        ],
    }


_PDF_STATUS_RGB = {"good": (22, 163, 74), "warn": (202, 138, 4), "bad": (220, 38, 38), "neutral": (100, 116, 139)}


def build_pdf(rep_: dict) -> bytes:
    from fpdf.enums import XPos, YPos

    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_margins(14, 14, 14)
    pdf.add_page()
    W = pdf.w - 28
    ink, ink2, muted, accent = (15, 23, 42), (51, 65, 85), (100, 116, 139), (2, 132, 199)

    def txt(size, style="", color=ink):
        pdf.set_font("helvetica", style, size)
        pdf.set_text_color(*color)

    def heading(label, need=30):
        if pdf.get_y() + need > pdf.h - 16:
            pdf.add_page()
        pdf.ln(3)
        txt(8.5, "B", accent)
        pdf.cell(W, 5, pdf_txt(label.upper()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(226, 232, 240)
        pdf.line(14, pdf.get_y(), 14 + W, pdf.get_y())
        pdf.ln(2)

    # Header band
    pdf.set_fill_color(11, 18, 32)
    pdf.rect(0, 0, pdf.w, 30, style="F")
    pdf.set_xy(14, 8)
    txt(17, "B", (56, 189, 248))
    pdf.cell(32, 8, "UrbanGrid")
    txt(12, "", (226, 232, 240))
    pdf.cell(W - 32, 8, pdf_txt("|  Cork City Business Location Intelligence"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(14)
    txt(9, "", (148, 163, 184))
    pdf.cell(W, 6, pdf_txt(f"Area report · generated {rep_['generated']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(36)

    txt(9, "I", ink2)
    pdf.multi_cell(W, 4.6, pdf_txt(f'Question: "{rep_["query"]}"'), align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    txt(8.5, "", muted)
    pdf.multi_cell(W, 4.4, pdf_txt(f"Analysed as: {rep_['settings']}"), align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Verdict box
    pdf.ln(3)
    y0 = pdf.get_y()
    pdf.set_fill_color(240, 249, 255)
    pdf.set_draw_color(186, 230, 253)
    pdf.rect(14, y0, W, 30, style="DF", round_corners=True, corner_radius=2.5)
    pdf.set_xy(19, y0 + 4)
    txt(7.5, "B", accent)
    pdf.cell(W - 45, 4, pdf_txt(f"RECOMMENDED SPOT · {rep_['scope']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(19)
    txt(14, "B", ink)
    pdf.cell(W - 45, 8, pdf_txt(rep_["where"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for i, (cls, t) in enumerate(rep_["reasons"]):
        pdf.set_xy(19 + (i % 2) * ((W - 45) / 2), y0 + 17 + (i // 2) * 5.2)
        pdf.set_fill_color(*_PDF_STATUS_RGB.get(cls, _PDF_STATUS_RGB["neutral"]))
        pdf.ellipse(pdf.get_x(), pdf.get_y() + 1.3, 2, 2, style="F")
        pdf.set_x(pdf.get_x() + 3.2)
        txt(8.2, "", ink2)
        pdf.cell((W - 45) / 2 - 4, 4.6, pdf_txt(t))
    pdf.set_xy(14 + W - 32, y0 + 5)
    txt(24, "B", accent)
    pdf.cell(28, 12, str(rep_["score"]), align="C", new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.set_x(14 + W - 32)
    txt(7.5, "", muted)
    pdf.cell(28, 4, "Opportunity score /100", align="C")
    pdf.set_y(y0 + 34)

    # Action cards
    gap, cw, ch = 4, (W - 8) / 3, 40
    y0 = pdf.get_y()
    for i, (title, metric, status, cls, detail) in enumerate(rep_["cards"]):
        x = 14 + i * (cw + gap)
        pdf.set_fill_color(248, 250, 252)
        pdf.set_draw_color(226, 232, 240)
        pdf.rect(x, y0, cw, ch, style="DF", round_corners=True, corner_radius=2)
        pdf.set_xy(x + 3.5, y0 + 3.5)
        txt(7.2, "B", muted)
        pdf.cell(cw - 7, 4, pdf_txt(title.upper()), new_x=XPos.LEFT, new_y=YPos.NEXT)
        pdf.set_x(x + 3.5)
        txt(11.5, "B", ink)
        pdf.cell(cw - 7, 7, pdf_txt(metric), new_x=XPos.LEFT, new_y=YPos.NEXT)
        pdf.set_x(x + 3.5)
        txt(8, "B", _PDF_STATUS_RGB.get(cls, _PDF_STATUS_RGB["neutral"]))
        pdf.cell(cw - 7, 5, pdf_txt(status), new_x=XPos.LEFT, new_y=YPos.NEXT)
        pdf.set_x(x + 3.5)
        txt(7.4, "", ink2)
        pdf.multi_cell(cw - 7, 3.6, pdf_txt(detail), align="L", new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.set_y(y0 + ch + 2)

    # Footfall chart
    heading(f"Footfall by hour ({DAY_SHORT})")
    ch_x, ch_y, ch_w, ch_h = 22, pdf.get_y() + 2, W - 8, 34
    c = rep_["chart"]
    top = max(c["rates"]) or 1
    n = len(c["rates"])
    bw = ch_w / n
    pdf.set_draw_color(226, 232, 240)
    for frac in (0.5, 1.0):
        yy = ch_y + ch_h - ch_h * frac
        pdf.line(ch_x, yy, ch_x + ch_w, yy)
        txt(6.5, "", muted)
        pdf.set_xy(14, yy - 2)
        pdf.cell(7, 4, f"{top * frac:,.0f}", align="R")
    for i, (t, v) in enumerate(zip(c["ts"], c["rates"])):
        h_ = ch_h * v / top
        color = (2, 132, 199) if c["peak"][i] else (125, 190, 222) if c["window"][i] else (203, 213, 225)
        pdf.set_fill_color(*color)
        pdf.rect(ch_x + i * bw + bw * 0.08, ch_y + ch_h - h_, bw * 0.84, h_, style="F")
        if t % 2 == 0:
            txt(6.5, "", muted)
            pdf.set_xy(ch_x + i * bw - 4, ch_y + ch_h + 0.8)
            pdf.cell(bw + 8, 3.5, pdf_txt(fmt_time(t)), align="C")
    pdf.set_y(ch_y + ch_h + 5.5)
    for color, label in [((2, 132, 199), "Peak 90 min"), ((125, 190, 222), "Your trading hours"),
                         ((203, 213, 225), "Rest of day")]:
        pdf.set_fill_color(*color)
        pdf.rect(pdf.get_x(), pdf.get_y() + 1, 3, 3, style="F")
        pdf.set_x(pdf.get_x() + 4.5)
        txt(7.2, "", ink2)
        pdf.cell(pdf.get_string_width(label) + 6, 5, label)
    pdf.ln(5.5)
    txt(7.4, "", muted)
    pdf.multi_cell(W, 3.7, align="L", text=pdf_txt(rep_["footfall_note"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Break-even
    heading("Break-even estimate")
    for k, v in rep_["breakeven"]:
        txt(8.5, "", ink2)
        pdf.cell(70, 5.2, pdf_txt(k))
        txt(8.5, "B", ink)
        pdf.cell(W - 70, 5.2, pdf_txt(v), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    txt(7.4, "", muted)
    pdf.multi_cell(W, 3.7, align="L", text=pdf_txt(rep_["be_note"] + " A rough guide, not financial advice."),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Around the spot: three lists
    heading("Around the recommended spot")
    cols3 = [("Nearest bus stops", rep_["stops"]), ("Pending planning applications", rep_["applications"]),
             (rep_["rivals_title"], rep_["rivals"])]
    y0, colw = pdf.get_y(), (W - 8) / 3
    y_end = y0
    for i, (title, items) in enumerate(cols3):
        x = 14 + i * (colw + 4)
        pdf.set_xy(x, y0)
        txt(8, "B", ink)
        pdf.multi_cell(colw, 4.4, pdf_txt(title), align="L", new_x=XPos.LEFT, new_y=YPos.NEXT)
        if not items:
            pdf.set_x(x)
            txt(7.6, "I", muted)
            pdf.cell(colw, 4.4, "None nearby", new_x=XPos.LEFT, new_y=YPos.NEXT)
        for name, extra in items:
            pdf.set_x(x)
            txt(7.6, "", ink2)
            pdf.multi_cell(colw, 3.8, pdf_txt(f"{name}  ({extra})"), align="L", new_x=XPos.LEFT, new_y=YPos.NEXT)
        y_end = max(y_end, pdf.get_y())
    pdf.set_y(y_end + 1)

    # Zone comparison
    heading("Compare Cork zones (best spot in each)", need=45)
    widths = [W * 0.26, W * 0.33, W * 0.11, W * 0.18, W * 0.12]
    txt(7.6, "B", muted)
    for w_, h_ in zip(widths, ["Zone", "Best spot", "Score", "Peak 90 min", "Typical rent"]):
        pdf.cell(w_, 5, h_)
    pdf.ln(5)
    for zone, spot, score, peak, rent, pick in rep_["zones"]:
        txt(8, "B" if pick else "", ink if pick else ink2)
        for w_, v in zip(widths, [("* " if pick else "") + zone, spot, str(score), peak, rent]):
            pdf.cell(w_, 4.8, pdf_txt(v))
        pdf.ln(4.8)

    # Sources + limits
    heading("Data sources & honest limits")
    txt(7.4, "", ink2)
    for line in rep_["sources"]:
        pdf.multi_cell(W, 3.7, align="L", text=pdf_txt("- " + line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    for line in rep_["limits"]:
        pdf.multi_cell(W, 3.7, align="L", text=pdf_txt("- " + line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    txt(7.2, "I", muted)
    pdf.multi_cell(W, 3.6, pdf_txt("UrbanGrid · Commercial SaaS Model: €49/month for Irish SMEs & Estate Agents. "
                                   "Data: Cork City Council Open Data, NTA NaPTAN via data.gov.ie, "
                                   "© OpenStreetMap contributors (ODbL)."),
                   align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    return bytes(pdf.output())


def build_html_report(rep_: dict) -> bytes:
    li = lambda rows: "".join(f"<li>{esc(a)} <span>({esc(b)})</span></li>" for a, b in rows) or "<li>None nearby</li>"  # noqa: E731
    cards = "".join(f"<div class='c'><h4>{esc(t)}</h4><b>{esc(m)}</b><p class='{c}'>{esc(s_)}</p><p>{esc(d)}</p></div>"
                    for t, m, s_, c, d in rep_["cards"])
    zones = "".join(f"<tr{' class=pick' if p else ''}><td>{esc(z)}</td><td>{esc(sp)}</td><td>{sc}</td><td>{esc(pk)}</td>"
                    f"<td>{esc(rt)}</td></tr>" for z, sp, sc, pk, rt, p in rep_["zones"])
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>UrbanGrid area report</title><style>
body{{font-family:system-ui,sans-serif;color:#0f172a;max-width:860px;margin:24px auto;padding:0 16px;line-height:1.45}}
h1{{font-size:22px;margin:0}} h1 b{{color:#0284c7}} h3{{font-size:12px;letter-spacing:.1em;text-transform:uppercase;
color:#0284c7;border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin-top:22px}} .v{{background:#f0f9ff;border:1px solid
#bae6fd;border-radius:10px;padding:14px 18px}} .v .s{{float:right;font-size:30px;font-weight:800;color:#0284c7}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:12px}} .c{{border:1px solid #e2e8f0;
border-radius:10px;padding:10px 12px;font-size:13px}} .c h4{{margin:0;font-size:11px;color:#64748b;text-transform:uppercase}}
.good{{color:#16a34a}} .warn{{color:#ca8a04}} .bad{{color:#dc2626}} table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{text-align:left;padding:4px 6px;border-bottom:1px solid #f1f5f9}} tr.pick{{font-weight:700}} li span{{color:#64748b}}
small{{color:#64748b}} @media print{{body{{margin:0}}}}</style></head><body>
<h1><b>UrbanGrid</b> | Cork City Business Location Intelligence</h1><small>Area report · {esc(rep_['generated'])}</small>
<p><i>"{esc(rep_['query'])}"</i><br><small>Analysed as: {esc(rep_['settings'])}</small></p>
<div class="v"><span class="s">{rep_['score']}</span><small>RECOMMENDED SPOT · {esc(rep_['scope'])}</small>
<h2 style="margin:4px 0">{esc(rep_['where'])}</h2>{' · '.join(esc(t) for _, t in rep_['reasons'])}</div>
<div class="cards">{cards}</div>
<h3>Footfall</h3><p>{esc(rep_['footfall_note'])}</p>
<h3>Break-even estimate</h3><table>{''.join(f'<tr><td>{esc(k)}</td><td><b>{esc(v)}</b></td></tr>' for k, v in rep_['breakeven'])}
</table><small>{esc(rep_['be_note'])} A rough guide, not financial advice.</small>
<h3>Around the recommended spot</h3><b>Nearest bus stops</b><ul>{li(rep_['stops'])}</ul>
<b>Pending planning applications</b><ul>{li(rep_['applications'])}</ul><b>{esc(rep_['rivals_title'])}</b><ul>{li(rep_['rivals'])}</ul>
<h3>Compare Cork zones</h3><table><tr><th>Zone</th><th>Best spot</th><th>Score</th><th>Peak 90 min</th><th>Rent</th></tr>{zones}</table>
<h3>Data sources &amp; honest limits</h3><ul>{''.join(f'<li>{esc(x)}</li>' for x in rep_['sources'] + rep_['limits'])}</ul>
<small>UrbanGrid · Commercial SaaS Model: €49/month for Irish SMEs &amp; Estate Agents. © OpenStreetMap contributors.</small>
</body></html>"""
    return doc.encode("utf-8")


# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------
with st.spinner("Connecting to Cork City open data…"):
    DATA = load_all_data()
BIZ, biz_state = current_businesses()
STATUS = {**DATA["status"], "Existing businesses": biz_state}
n_live = sum(state != "fallback" for state, _ in STATUS.values())
n_src = len(STATUS)


# ---------------------------------------------------------------------------
# Navigation: landing screen -> paged results
# ---------------------------------------------------------------------------
PAGES = ["overview", "map", "footfall", "money", "compare", "data"]
PAGE_META = {  # nav label, icon, title, plain-English explainer
    "overview": ("Answer", ":material/flag:", "The answer",
                 "Our recommended spot, and the three things that decide it: foot traffic, competition and bus access."),
    "map": ("Map", ":material/map:", "Where it is",
            "The recommended spot on a map of Cork, with busy streets, competitors and bus stops around it."),
    "footfall": ("Footfall", ":material/bar_chart:", "When people walk past",
                 "How many people pass the spot at each time of day, and when the busiest 90 minutes are."),
    "money": ("Break-even", ":material/savings:", "Will it pay the rent?",
              "How many customers a day you need to cover rent and running costs, and whether that is realistic."),
    "compare": ("Compare", ":material/leaderboard:", "How the six zones compare",
                "The best spot in each Cork zone, ranked for your business and budget."),
    "data": ("Data & limits", ":material/fact_check:", "Data & honest limits",
             "Where every number comes from, how the score works, and what UrbanGrid cannot do yet."),
}
EXAMPLES = {
    "coffee": (":material/local_cafe: Coffee · Oliver Plunkett St", SAMPLE_QUERY),
    "bakery": (":material/bakery_dining: Bakery · Ballincollig",
               "Artisan bakery in Ballincollig 7:30-11am near bus routes"),
    "boutique": (":material/checkroom: Boutique · under €5,000",
                 "Where should I open a retail boutique with fewer competitors under €5,000?"),
    "dental": (":material/dentistry: Dental · Douglas",
               "Best place in Douglas for a dental clinic in the afternoon under €4,500?"),
}

ss.setdefault("view", "home")
ss.setdefault("query_in", ss.query)
# Keep per-page widget values when their page is not on screen (Streamlit drops state of unrendered widgets).
for _k in list(ss.keys()):
    if str(_k).startswith(("be_", "ff_compare", "cmp_all", "map_layers_v2", "page", "map_view", "mode",
                           "business", "budget", "priority", "zone_focus", "time_window", "day_type")):
        ss[_k] = ss[_k]
for _k, _ok, _dflt in (("mode", ("simple", "expert"), "simple"), ("day_type", DAY_TYPES, "weekday"),
                       ("map_view", ("spots", "heat"), None)):
    if _dflt is None:                    # simple mode opens the map on the clean heatmap view
        _dflt = "heat" if ss.mode == "simple" else "spots"
    if ss.get(_k) not in _ok:            # segmented controls can be clicked "off": keep the last choice
        ss[_k] = ss.get(f"last_{_k}", _dflt)
    ss[f"last_{_k}"] = ss[_k]
SIMPLE = ss.mode == "simple"
if ss.get("page") not in PAGES:          # segmented control can be clicked "off"; stay where we were
    ss.page = ss.get("last_page", "overview")
ss.last_page = ss.page


def start_analysis() -> None:
    ss.query = ss.query_in.strip() or SAMPLE_QUERY
    on_analyze()
    ss.view, ss.page = "results", "overview"


def on_example() -> None:
    pick = ss.get("example_pick")
    if pick:
        ss.query_in = EXAMPLES[pick][1]
        ss.example_pick = None
        start_analysis()


def go_home() -> None:
    ss.view = "home"
    ss.query_in = ss.query


def go_page(page: str) -> None:
    ss.page = page


def status_popover(width="stretch") -> None:
    label = f"{n_live} of {n_src} sources live" if n_live else "Offline data mode"
    with st.popover(label, icon=":material/sensors:" if n_live else ":material/cloud_off:", width=width):
        rows = []
        for source, (state, note) in STATUS.items():
            tag = {"live": "LIVE", "partial": "PARTIAL", "fallback": "FALLBACK"}[state]
            rows.append(f"<div class='ug-src'><div><span class='name'>{esc(source)}</span>"
                        f"<span class='note'>{esc(note)}</span></div><span class='tag {state}'>{tag}</span></div>")
        html_block("<div style='font-weight:700;margin-bottom:.2rem'>Data sources</div>" + "".join(rows))
        st.caption(f"Checked at {DATA['fetched_at']} · refreshes every {CACHE_TTL_S // 60} min. "
                   "If a source is down, UrbanGrid switches to embedded Cork data automatically.")
        st.button("Re-check live data", icon=":material/refresh:", on_click=recheck_live_data, width="stretch")


# ---------------------------------------------------------------------------
# Sidebar: business essentials (hidden on the landing screen)
# ---------------------------------------------------------------------------
def render_settings(compact: bool) -> None:
    """Search settings. Rendered in the sidebar (landing + Expert) or the 'Edit search' panel (Simple)."""
    st.selectbox("Business Type", list(BUSINESS_TYPES), key="business")
    st.slider("Monthly Rent Budget", 1000, 10000, step=250, format="€%d", key="budget")
    st.radio("Core Priority", PRIORITIES, key="priority")
    st.segmented_control("Day to analyse", list(DAY_TYPES), key="day_type", width="stretch",
                         format_func=lambda k: {"weekday": "Weekday", "saturday": "Saturday",
                                                "christmas": "Christmas"}[k],
                         help="Saturday and Christmas apply estimated adjustments to typical weekday footfall.")

    def where_and_when():
        st.selectbox("Location focus", [AUTO, *ZONES], key="zone_focus",
                     format_func=lambda k: "Auto: compare all Cork zones" if k == AUTO else ZONES[k]["name"])
        # value= marks it as a two-handle range slider; the live value always comes from session state.
        st.select_slider("Busiest hours to focus on", options=TIME_OPTIONS, key="time_window",
                         value=BUSINESS_TYPES[ss.business]["window"],
                         help="The hours you care about most, e.g. your morning rush. The score, cards and map "
                              "use these hours. Money calculations use your full opening hours (Break-even page).")

    if compact:
        where_and_when()
    else:
        html_block("<div class='ug-side-h'>Fine-tune</div>")
        with st.expander("Location & focus hours", icon=":material/tune:"):
            where_and_when()


if ss.view == "home" or not SIMPLE:
    with st.sidebar:
        html_block(
            f"<div class='ug-brand' style='margin:.1rem 0 .4rem'><div class='ug-logo' style='width:36px;height:36px;"
            f"border-radius:10px'>{icon('grid', 18)}</div><div><div style='font-weight:700;font-size:1rem;"
            f"color:#e6edf7'>UrbanGrid</div><div style='font-size:.74rem;color:#6f7f96'>Cork site-selection copilot"
            f"</div></div></div>"
        )
        html_block("<div class='ug-side-h'>Your business</div>")
        render_settings(compact=False)
        st.caption("Your question fills these in automatically. Adjust them here any time.")


# ---------------------------------------------------------------------------
# Landing screen
# ---------------------------------------------------------------------------
FOOTER_HTML = (
    f"<div class='ug-footer'><div class='money'>{icon('briefcase', 18)}"
    f"<span>Commercial SaaS Model: <b>€49/month</b> for Irish SMEs &amp; Estate Agents.</span></div>"
    f"<div class='credits'>Data: Cork City Council Open Data · NTA NaPTAN via data.gov.ie · © OpenStreetMap contributors · "
    f"Built for the Cork Hackathon</div></div>"
)

if ss.view == "home":
    html_block("<style>section[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'], "
               "[data-testid='stExpandSidebarButton'] { display: none !important; } "
               "[data-testid='stHeader'] { display: none !important; } "
               "[data-testid='stMainBlockContainer'] { padding-top: clamp(.8rem, 3vh, 2rem) !important; }</style>")
    top_l, top_r = st.columns([5, 1.35], vertical_alignment="center")
    with top_l:
        html_block(f"<div class='ug-brand ug-topbar'><div class='ug-logo'>{icon('grid', 18)}</div>"
                   f"<div class='name'>UrbanGrid<span>|&nbsp; Cork City Business Location Intelligence</span></div></div>")
    with top_r:
        status_popover()

    _, mid, _ = st.columns([0.55, 4, 0.55])
    with mid:
        html_block(
            f"<div class='ug-hero'><span class='badge'>{icon('pin', 14)}Built for small businesses in Cork City</span>"
            f"<h1><span class='w'>Find the best spot in Cork</span><br/><span class='a'>for your business.</span></h1>"
            f"<p>Ask a question in plain English. UrbanGrid checks foot traffic, planning applications, competitors "
            f"and bus access, then gives you one clear answer.</p></div>"
        )
        q_col, b_col = st.columns([4.2, 1.2], vertical_alignment="bottom")
        with q_col:
            st.text_input("Ask a question about a Cork location", key="query_in", on_change=start_analysis,
                          label_visibility="collapsed",
                          placeholder="e.g. Best spot in Douglas for a bakery at lunchtime under €4,500?")
        with b_col:
            st.button("Analyze Location", type="primary", icon=":material/search:", on_click=start_analysis,
                      width="stretch")
        html_block("<div class='ug-try'>or try an example</div>")
        st.pills("Examples", list(EXAMPLES), format_func=lambda k: EXAMPLES[k][0], key="example_pick",
                 on_change=on_example, label_visibility="collapsed", width="stretch")

        html_block(
            "<div class='ug-how'>"
            f"<div class='h'><div class='hd'><div class='n'>1</div><b>Ask</b></div><span>Type your business, street and hours, the way "
            f"you would ask a friend.</span></div>"
            f"<div class='h'><div class='hd'><div class='n'>2</div><b>Get one clear answer</b></div><span>The best spot, a score out of 100 "
            f"and the three facts behind it.</span></div>"
            f"<div class='h'><div class='hd'><div class='n'>3</div><b>Explore &amp; share</b></div><span>Map, footfall, break-even and a "
            f"PDF report for your bank or landlord.</span></div></div>"
        )
        checks = [
            ("users", "Foot traffic", "Cork City pedestrian counters", "Footfall"),
            ("building", "Planning", "Cork City planning applications", "Planning"),
            ("store", "Competitors", "Existing businesses on OpenStreetMap", "Existing businesses"),
            ("bus", "Bus access", "NTA national bus-stop register", "Transit"),
        ]
        tiles = []
        for ic, title, desc, key in checks:
            state = STATUS[key][0]
            colr, word = (("#4ade80", "Live data") if state == "live" else ("#fbbf24", "Partly live")
                          if state == "partial" else ("#94a3b8", "Built-in Cork data"))
            tiles.append(f"<div class='ug-check'><div class='t'>{icon(ic, 17)}{title}</div><div class='d'>{desc}</div>"
                         f"<div class='live' style='color:{colr}'><i style='background:{colr}'></i>{word}</div></div>")
        html_block("<div class='ug-sec-h'>What UrbanGrid checks for you</div><div class='ug-checks'>"
                   + "".join(tiles) + "</div>")
        html_block(FOOTER_HTML)
    st.stop()


# ---------------------------------------------------------------------------
# Results: top bar + the question being answered
# ---------------------------------------------------------------------------
top_l, top_mode, top_m, top_r = st.columns([4.2, 1.75, 1.5, 1.15], vertical_alignment="center")
with top_mode:
    st.segmented_control("View", ["simple", "expert"], key="mode", label_visibility="collapsed", width="stretch",
                         format_func=lambda k: ":material/visibility: Simple" if k == "simple" else ":material/tune: Expert",
                         help="Simple: plain-English answers. Expert: every number, table and setting.")
with top_l:
    html_block(f"<div class='ug-brand ug-topbar'><div class='ug-logo'>{icon('grid', 18)}</div>"
               f"<div class='name'>UrbanGrid<span>|&nbsp; Cork City Business Location Intelligence</span></div></div>")
with top_m:
    status_popover()
with top_r:
    st.button("New search", icon=":material/search:", on_click=go_home, width="stretch")

parsed = ss.get("parsed", {})
zone_label = "All Cork zones" if ss.zone_focus == AUTO else ZONES[ss.zone_focus]["name"]
chip_items = [
    ("pin", zone_label, "zone" in parsed and ss.zone_focus == parsed.get("zone")),
    ("store", ss.business, parsed.get("business") == ss.business),
    ("clock", f"{ss.time_window[0]} – {ss.time_window[1]}", tuple(parsed.get("window", ())) == tuple(ss.time_window)),
    ("calendar", DAY_TYPES[ss.day_type], parsed.get("day_type") == ss.day_type),
    ("target", ss.priority, parsed.get("priority") == ss.priority),
    ("euro", f"≤ €{ss.budget:,}/month", parsed.get("budget") == ss.budget),
]
chips = "".join(f"<span class='ug-chip{' q' if from_q else ''}'>{icon(ic, 14)}{esc(text)}</span>"
                for ic, text, from_q in chip_items)
PRIORITY_PLAIN = {"High Morning Footfall": "lots of people walking past",
                  "Low Competitor Density": "few competitors nearby", "High Transit Access": "good bus access"}
if SIMPLE:
    _biz = ss.business.lower()
    _where = "anywhere in Cork" if ss.zone_focus == AUTO else f"on {ZONES[ss.zone_focus]['name']}"
    _day = {"weekday": "on a typical weekday", "saturday": "on a Saturday",
            "christmas": "in the Christmas season"}[ss.day_type]
    q_l, q_r = st.columns([5, 1.2], vertical_alignment="center")
    with q_l:
        html_block(
            f"<div class='ug-qcard'><div class='qt'>“{esc(ss.query)}”</div>"
            f"<div class='ug-plain'>Finding the best spot for {'an' if _biz[0] in 'aeiou' else 'a'} <b>{esc(_biz)}</b> "
            f"{esc(_where)}, focusing on your busiest hours <b>{esc(ss.time_window[0])} – {esc(ss.time_window[1])}</b> "
            f"{_day}, with rent up to <b>€{ss.budget:,}</b> and <b>{PRIORITY_PLAIN[ss.priority]}</b> as the main "
            f"priority.{tip('Busiest hours are the hours you care about most, like a morning rush. They drive the score, cards and map. Your full opening hours are used for the money page.')}"
            f"</div></div>"
        )
    with q_r:
        with st.popover("Edit search", icon=":material/edit:", width="stretch"):
            render_settings(compact=True)
            st.caption("Changes apply instantly.")
else:
    html_block(f"<div class='ug-qcard'><div class='qt'>“{esc(ss.query)}”</div>"
               f"<div class='ug-chips'><span class='lbl'>Analysing</span>{chips}"
               f"{tip('Highlighted items were understood from your question. Change the rest in the sidebar.')}"
               f"</div></div>")


# ---------------------------------------------------------------------------
# Analysis (computed once; every page reads from it)
# ---------------------------------------------------------------------------
DAY_LABEL = DAY_TYPES[ss.day_type]
DAY_SHORT = {"weekday": "weekday", "saturday": "Saturday", "christmas": "Christmas season"}[ss.day_type]
PROFILES = {k: season_profile(v, ss.day_type, k in SUBURBAN_ZONES) for k, v in DATA["footfall_profiles"].items()}
R = analyze({**DATA, "footfall_profiles": PROFILES}, ss.business, ss.budget, ss.priority, ss.zone_focus,
            ss.time_window, BIZ)
best, hottest = R["best"], R["hottest"]
start_h, end_h = R["window"]
period = period_word(best["peak_start"])
comp_label = CATEGORY_LABELS[R["competitor_category"]]

# Peak / footfall status
peak_rate = best["peak_rate"]
if peak_rate >= 900:
    ff_status, ff_cls, ff_icon = "High Footfall Zone", "good", "trend"
elif peak_rate >= 450:
    ff_status, ff_cls, ff_icon = "Moderate Footfall", "warn", "trend"
else:
    ff_status, ff_cls, ff_icon = "Low Footfall", "neutral", "minus"

# Planning / competition status
n_pend, n_comp = int(best["pending"]), int(best["competitors"])
where_txt = (f"within {PLANNING_RADIUS_M}m" if best["planning_mode"] == "radius"
             else f"on {esc(best['zone'])} (address-matched)")
if n_comp:
    plan_detail = (f"{n_comp} competing {comp_label} outlet application{'s' if n_comp != 1 else ''} "
                   f"registered {where_txt}.")
    if n_pend - n_comp:
        plan_detail += f" Plus {n_pend - n_comp} other pending."
elif n_pend:
    plan_detail = (f"No competing {comp_label} applications {where_txt}; "
                   f"{n_pend} unrelated application{'s' if n_pend != 1 else ''} pending.")
else:
    plan_detail = f"No pending planning applications {where_txt}."
if n_comp == 0:
    pl_status, pl_cls, pl_icon = "Low Risk", "good", "check"
elif n_comp <= 2:
    pl_status, pl_cls, pl_icon = "Medium Risk", "warn", "alert"
else:
    pl_status, pl_cls, pl_icon = "High Risk", "bad", "x"
plan_src = ("Live Cork City planning register" if DATA["status"]["Planning"][0] != "fallback"
            else "Embedded Cork dataset")
n_exist = int(best["existing"])
kind_one, kind_many = OSM_KIND_LABELS[R["osm_kind"]]
exist_txt = (f"{n_exist} existing {kind_one if n_exist == 1 else kind_many}" if n_exist else f"no existing {kind_many}")
plan_detail += (f" {exist_txt} already trade{'s' if n_exist == 1 else ''} within {EXISTING_RADIUS_M}m." if n_exist else
                f" No existing {kind_many} within {EXISTING_RADIUS_M}m.")

# Transit status
n_stops = len(R["transit_in_radius"])
nearest = float(np.min(R["transit_all_dist"])) if len(R["transit_all_dist"]) else np.inf
area = ZONES[best["zone_key"]]["area_label"]
hw = f"{'~' if R['headway_estimated'] else ''}{R['headway']:.0f}-min"
if n_stops:
    tr_detail = (f"{n_stops} active {area} bus stop{'s' if n_stops != 1 else ''} within {R['transit_radius']}m "
                 f"with average {hw} arrival intervals.")
else:
    tr_detail = f"No bus stops within {R['transit_radius']}m. The nearest is about {nearest:,.0f}m away."
if n_stops >= 3 and R["transit_radius"] <= 100 and nearest <= 60:
    tr_status, tr_cls, tr_icon, tr_word = "Excellent Access", "good", "check", "Excellent"
elif n_stops >= 2 and R["transit_radius"] <= 150:
    tr_status, tr_cls, tr_icon, tr_word = "Good Access", "good", "check", "Good"
else:
    tr_status, tr_cls, tr_icon, tr_word = "Limited Access", "warn", "alert", "Limited"
transit_src = "Live NTA NaPTAN stops" if DATA["status"]["Transit"][0] != "fallback" else "Embedded Cork dataset"

reasons = [
    (ff_cls if ff_cls != "neutral" else "warn", "check" if ff_cls == "good" else "alert",
     f"{ff_status.replace(' Zone', '')} · ≈{peak_rate:,.0f} people/hr at peak"),
    ("good" if n_comp == 0 else "warn" if n_comp <= 2 else "bad", "check" if n_comp == 0 else "alert",
     ("No competing applications" if n_comp == 0 else f"{n_comp} competing application{'s' if n_comp != 1 else ''}")
     + f" · {exist_txt} nearby"),
    ("good" if tr_cls == "good" else "warn", "check" if tr_cls == "good" else "alert", f"{tr_word} bus access"),
    ("good", "check", f"≈€{best['rent']:,}/month · within budget") if best["rent"] <= ss.budget else
    ("bad", "alert", f"≈€{best['rent']:,}/month · €{best['rent'] - ss.budget:,} over budget"),
]
reasons_html = "".join(f"<span class='ug-reason {c}'>{icon(i, 15)}{esc(t)}</span>" for c, i, t in reasons)
note_html = ""
if ss.zone_focus != AUTO and R["alt"] is not None and R["alt"]["score"] > best["score"]:
    note_html = (f"<div class='ug-note'>{icon('bulb', 15)}<span><b>{esc(R['alt']['spot'])}, {esc(R['alt']['zone'])}</b> "
                 f"scores {R['alt']['score']} for your settings.</span></div>")
scope = "Across all six Cork zones" if ss.zone_focus == AUTO else f"On {ZONES[best['zone_key']]['name']}"
score_tip = (f"Weighted blend of footfall, competition, transit and budget fit for the priority "
             f"'{ss.priority}'. See Method & limits for the formula.")


def action_card(ic: str, title: str, metric: str, status: str, status_cls: str, status_icon: str,
                detail: str, source_note: str) -> str:
    return (
        f"<div class='ug-card'>"
        f"<div class='ug-card-head'><div class='ug-card-title'>{icon(ic, 16)}{esc(title)}</div>{tip(source_note)}</div>"
        f"<div class='ug-card-metric'>{esc(metric)}</div>"
        f"<div class='ug-status {status_cls}'>{icon(status_icon, 14)}{esc(status)}</div>"
        f"<div class='ug-card-detail'>{detail}</div></div>"
    )


spots = R["spots"].copy()
max_rate = spots["window_rate"].max()
DENSITY_COLORS = {"high": [248, 113, 113], "mid": [251, 191, 36], "low": [45, 212, 191]}


def density_color(rate: float, alpha: int = 230):
    ratio = rate / max_rate if max_rate else 0
    key = "high" if ratio >= 0.75 else "mid" if ratio >= 0.5 else "low"
    return DENSITY_COLORS[key] + [alpha]


# --- planning records with map positions --------------------------------------
plan_all = R["plan"].copy()
plan_all["approx"] = plan_all["lat"].isna()
for idx in plan_all.index[plan_all["approx"]]:   # live records: place along their street (source coords unreliable)
    frac = (zlib.crc32(str(plan_all.at[idx, "app_no"]).encode()) % 1000) / 1000
    lon, lat = point_along_path(ZONES[plan_all.at[idx, "zone"]]["path"], 0.1 + 0.8 * frac)
    plan_all.at[idx, "lon"], plan_all.at[idx, "lat"] = lon, lat
plan_all["is_comp"] = plan_all["pending"] & (plan_all["category"] == R["competitor_category"])

GRID_STEP_M = 100
HEAT_BINS = [(80, [125, 211, 252, 215], "80+"), (70, [56, 189, 248, 175], "70–79"), (55, [2, 132, 199, 140], "55–69"),
             (40, [7, 89, 133, 105], "40–54"), (-1, [12, 74, 110, 60], "under 40")]


def opportunity_grid() -> pd.DataFrame:
    """Score every 100 m square around the six zones with the same formula as the named spots.

    Footfall between measured spots is interpolated (inverse-distance weighting) and fades away from the
    monitored streets; competition, transit and rent use the same rules as the spot scores.
    """
    cells = []
    lat_step = GRID_STEP_M / 111_320
    for box in OSM_BOXES:
        s_, w_, n_, e_ = map(float, box.split(","))
        lon_step = GRID_STEP_M / (111_320 * np.cos(np.radians((s_ + n_) / 2)))
        for la in np.arange(s_, n_, lat_step):
            for lo in np.arange(w_, e_, lon_step):
                cells.append((la, lo, lat_step, lon_step))
    g = pd.DataFrame(cells, columns=["lat0", "lon0", "dlat", "dlon"])
    g["lat"], g["lon"] = g["lat0"] + g["dlat"] / 2, g["lon0"] + g["dlon"] / 2
    clat, clon = g["lat"].to_numpy()[:, None], g["lon"].to_numpy()[:, None]
    sp = R["spots"]

    d_sp = haversine_m(clat, clon, sp["lat"].to_numpy()[None, :], sp["lon"].to_numpy()[None, :])
    w = 1 / np.maximum(d_sp, 25) ** 2
    rate = (w * sp["window_rate"].to_numpy()).sum(1) / w.sum(1)
    rate *= np.exp(-np.maximum(d_sp.min(1) - 60, 0) / 120)      # fade away from measured streets
    near = d_sp.argmin(1)

    rv = R["rivals"]
    existing = (haversine_m(clat, clon, rv["lat"].to_numpy()[None, :], rv["lon"].to_numpy()[None, :])
                <= EXISTING_RADIUS_M).sum(1) if len(rv) else np.zeros(len(g))
    comp_apps = plan_all[plan_all["is_comp"]]
    pipeline = (haversine_m(clat, clon, comp_apps["lat"].to_numpy()[None, :], comp_apps["lon"].to_numpy()[None, :])
                <= PLANNING_RADIUS_M).sum(1) if len(comp_apps) else np.zeros(len(g))
    st_ = R["stops"]
    d_st = haversine_m(clat, clon, st_["lat"].to_numpy()[None, :], st_["lon"].to_numpy()[None, :])
    stops150, nearest_stop = (d_st <= 150).sum(1), d_st.min(1)
    rent = sp["rent"].to_numpy()[near]

    wf, wc, wt, wb = PRIORITY_WEIGHTS[ss.priority]
    f_score = np.clip(rate / sp["window_rate"].max(), 0, 1)
    c_score = 0.6 * np.clip(1 - pipeline / 4, 0, 1) + 0.4 * np.clip(1 - existing / max(int(sp["existing"].max()), 3), 0, 1)
    t_score = 0.6 * np.minimum(stops150, 6) / 6 + 0.4 * np.clip(1 - nearest_stop / 400, 0, 1)
    b_score = np.where(rent <= ss.budget, 1.0, np.clip(1 - (rent - ss.budget) / 2000, 0, 1))
    g["score"] = np.round(100 * (wf * f_score + wc * c_score + wt * t_score + wb * b_score)).astype(int)
    g["rate"], g["existing"], g["pipeline"], g["stops150"] = rate, existing, pipeline, stops150
    g["near"] = [f"{sp['spot'].iat[i]}, {sp['zone'].iat[i]}" for i in near]
    g["zone_key"] = sp["zone_key"].to_numpy()[near]
    g = g[rate >= 40].reset_index(drop=True)                     # skip squares with no meaningful foot traffic
    g["polygon"] = [[[lo, la], [lo + dlo, la], [lo + dlo, la + dla], [lo, la + dla]]
                    for la, lo, dla, dlo in zip(g["lat0"], g["lon0"], g["dlat"], g["dlon"])]
    g["color"] = [next(c for t, c, _ in HEAT_BINS if v >= t) for v in g["score"]]
    return g


def top_squares(g: pd.DataFrame, n: int = 3, min_gap_m: float = 220) -> pd.DataFrame:
    """Best n squares that are not next to each other."""
    picks = []
    for i in g.sort_values("score", ascending=False).index:
        if all(haversine_m(g.at[i, "lat"], g.at[i, "lon"], g.at[j, "lat"], g.at[j, "lon"]) > min_gap_m for j in picks):
            picks.append(i)
        if len(picks) == n:
            break
    return g.loc[picks].assign(rank=[str(k + 1) for k in range(len(picks))])


DAY_START, DAY_END = 6.0, 21.0
best_profile = PROFILES[best["zone_key"]]
day_ts, day_rates = slot_rates(best_profile, DAY_START, DAY_END)
day_rates = day_rates * best["share"]
in_window = (day_ts >= start_h) & (day_ts < end_h)
in_peak = (day_ts >= best["peak_start"]) & (day_ts < best["peak_end"])
window_passers = float(day_rates[in_window].sum() * 0.5)
busiest_i = int(np.argmax(day_rates))

# Break-even inputs live in session state so the report works from any page.
BE = BUSINESS_TYPES[ss.business]["be"]
bkey = re.sub(r"\W+", "_", ss.business.lower())
BE_KEYS = {
    "spend": (f"be_spend_{bkey}", float(BE["spend"])),
    "margin": (f"be_margin_{bkey}", int(BE["margin"])),
    "rent": (f"be_rent_{bkey}_{best['zone_key']}", int(best["rent"])),
    "other": (f"be_other_{bkey}", int(BE["other"])),
    "days": (f"be_days_{bkey}", int(BE["days"])),
    "hours": (f"be_hours_{bkey}", tuple(BE["hours"])),
}
for _key, _default in BE_KEYS.values():
    ss.setdefault(_key, _default)
_hv = ss.get(BE_KEYS["hours"][0])
if isinstance(_hv, (tuple, list)) and len(_hv) == 2 and all(isinstance(t, (int, np.integer)) for t in _hv):
    _hv = tuple(TIME_OPTIONS[int(t)] for t in _hv if 0 <= int(t) < len(TIME_OPTIONS))   # stored as slider indices
if not (isinstance(_hv, (tuple, list)) and len(_hv) == 2 and all(t in TIME_OPTIONS for t in _hv)):
    _hv = BE_KEYS["hours"][1]
ss[BE_KEYS["hours"][0]] = tuple(_hv)
be_spend, be_margin, be_rent, be_other, be_days, be_hours = (ss[BE_KEYS[k][0]] for k in
                                                             ("spend", "margin", "rent", "other", "days", "hours"))
o_h, c_h = t2h(be_hours[0]), max(t2h(be_hours[1]), t2h(be_hours[0]) + 0.5)
_, open_rates = slot_rates(best_profile, o_h, c_h)
passers_day = float(open_rates.sum() * 0.5 * best["share"])
contrib = be_spend * be_margin / 100
fixed = be_rent + be_other
need_day = fixed / (contrib * be_days) if contrib > 0 else float("inf")
rent_day = be_rent / (contrib * be_days) if contrib > 0 else float("inf")
capture_need = 100 * need_day / passers_day if passers_day > 0 else float("inf")
cap_lo, cap_hi = BE["capture"]
cap_mid = (cap_lo + cap_hi) / 2
typical_cust = passers_day * cap_mid / 100
typical_profit = typical_cust * contrib * be_days - fixed
# Rent negotiation number: the most rent an average-share business here can pay and still break even.
TARGET_PROFIT = 0.10                                     # owner keeps 10% of turnover as profit
turnover = typical_cust * be_spend * be_days
breakeven_rent = typical_cust * contrib * be_days - be_other    # rent at which profit is exactly zero
max_rent = breakeven_rent - TARGET_PROFIT * turnover
rent_gap = max_rent - be_rent
# Rate against the AVERAGE capture (midpoint), so the badge agrees with the average-share profit scenario.
if capture_need <= cap_mid:
    be_status, be_cls, be_icon = "Comfortable", "good", "check"
    be_msg = "an average business here would cover its costs"
elif capture_need <= cap_hi:
    be_status, be_cls, be_icon = "Tight", "warn", "alert"
    be_msg = "you need an above-average share of passers-by to cover costs"
else:
    be_status, be_cls, be_icon = "Stretch", "bad", "x"
    be_msg = "you need more customers than even strong businesses usually win"
# Appointment-led businesses get few customers from walk-past footfall, so flag the estimate as weaker.
APPOINTMENT_LED = {"Dental Clinic"}
be_appt_note = ("Dental patients mostly come from bookings and referrals, not from people walking past, so treat these "
                "footfall-based figures as a rough guide only." if ss.business in APPOINTMENT_LED else "")

# Zone ranking (Compare page + report).
rank = (R["spots"].sort_values("score", ascending=False).groupby("zone_key", sort=False).head(1).copy())
rank["pick"] = np.where(rank["zone_key"] == best["zone_key"], "★", "")
rank["peak"] = rank.apply(lambda r: f"{fmt_time(r['peak_start'])} – {fmt_time(r['peak_end'])}", axis=1)
rank["budget_ok"] = np.where(rank["rent"] <= ss.budget, "Within budget", "Over budget")

latest = DATA["planning_latest"]
latest_txt = (f"The live planning feed's newest record is dated {datetime.fromisoformat(latest).strftime('%d %b %Y')}."
              if latest else "Planning data is currently from the embedded Cork dataset.")
footfall_limit = (
    "The Cork counter datastore is not returning rows via API right now "
    f"({DATA['status']['Footfall'][1].replace('Live API: ', '')}), so footfall uses the embedded Cork profile."
    if DATA["status"]["Footfall"][0] == "fallback" else
    "Live counters cover specific street points; spots between counters are estimated from the nearest counter."
)


# ---------------------------------------------------------------------------
# Plain-English helpers (Simple mode)
# ---------------------------------------------------------------------------
def approx_n(n: float) -> str:
    """Friendly rounding: 7, 160, 1,300."""
    if not np.isfinite(n):
        return "n/a"
    n = float(n)
    step = 1 if n < 20 else 5 if n < 200 else 50 if n < 1000 else 100
    return f"{int(round(n / step) * step):,}"


def one_in(pct: float) -> str:
    """1.36% -> '74' (as in 'about 1 in 74 people')."""
    return f"{max(1, round(100 / pct)):,}" if pct and np.isfinite(pct) and pct > 0 else "n/a"


def join_plain(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


score_val = int(best["score"])
FIT_WORD, FIT_LEAD = (("great", "Yes") if score_val >= 80 else ("good", "Yes") if score_val >= 65 else
                      ("fair", "Possibly") if score_val >= 50 else ("weak", "Probably not"))
COMP_PLAIN = {"food_beverage": "food & drink", "retail": "shop", "health": "healthcare", "other": ""}
hw_plain = f"{R['headway']:.0f}"


# ---------------------------------------------------------------------------
# Page navigation
# ---------------------------------------------------------------------------
st.segmented_control("Section", PAGES, key="page", label_visibility="collapsed", width="stretch",
                     format_func=lambda k: f"{PAGE_META[k][1]} {PAGE_META[k][0]}")
page = ss.page
p_idx = PAGES.index(page)
_, _, p_title, p_sub = PAGE_META[page]
html_block(f"<div class='ug-page-h'><div class='k'>Step {p_idx + 1} of {len(PAGES)}</div><h2>{esc(p_title)}</h2>"
           f"<p>{esc(p_sub)}</p></div>")

if page == "overview":
    if SIMPLE:
        good, watch = [], []
        peak_txt = f"{fmt_time(best['peak_start'])}–{fmt_time(best['peak_end'])}"
        if ff_cls == "good":
            good.append(f"busy at {peak_txt} (about {approx_n(peak_rate)} people an hour)")
        else:
            watch.append(f"{'only moderate' if ff_cls == 'warn' else 'low'} foot traffic "
                         f"(about {approx_n(peak_rate)} people an hour at the busiest time)")
        if n_comp == 0:
            good.append("no new competitors planning to open nearby")
        else:
            watch.append(f"{n_comp} new competitor{'s' if n_comp != 1 else ''} planning to open nearby")
        if n_exist >= 5:
            watch.append(f"{n_exist} {kind_many} already nearby")
        elif n_exist == 0:
            good.append(f"no other {kind_many} right beside you")
        if tr_cls == "good" and n_stops:
            good.append(f"a bus about every {hw_plain} minutes close by")
        else:
            watch.append("limited bus access")
        if best["rent"] <= ss.budget:
            good.append(f"rent (about €{best['rent']:,}) is within your budget")
        else:
            watch.append(f"rent is about €{best['rent'] - ss.budget:,} over your budget")
        body = ""
        if good:
            body += f"<span class='g'>Good:</span> {esc(join_plain(good))}. "
        if watch:
            body += f"<span class='w'>Watch out:</span> {esc(join_plain(watch))}."
        alt_txt = ""
        if ss.zone_focus != AUTO and R["alt"] is not None and R["alt"]["score"] > best["score"]:
            alt_txt = (f"<div class='ug-note'>{icon('bulb', 15)}<span>Tip: <b>{esc(R['alt']['spot'])}, "
                       f"{esc(R['alt']['zone'])}</b> is an even better match ({R['alt']['score']}/100).</span></div>")
        html_block(
            f"<div class='ug-verdict simple'><div class='main'>"
            f"<div class='ug-eyebrow'>{icon('pin', 14)}Our answer</div>"
            f"<div class='where'>{FIT_LEAD}: {esc(best['spot'])}, {esc(best['zone'])} is a {FIT_WORD} fit for your "
            f"{esc(ss.business.lower())}.</div><div class='ug-plain-body'>{body}</div>{alt_txt}</div>"
            f"<div class='ug-ring' style='--p:{score_val}'><div class='in'><div class='num'>{score_val}</div>"
            f"<div class='lbl'>Match<br/>{FIT_WORD}</div></div></div></div>"
        )
        st.write("")
        plan_where = (f"within {PLANNING_RADIUS_M} m" if best["planning_mode"] == "radius" else f"on {best['zone']}")
        if n_comp:
            plan_plain = (f"{n_comp} new {COMP_PLAIN[R['competitor_category']]} business"
                          f"{'es have' if n_comp != 1 else ' has'} applied to open {plan_where}.")
        elif n_pend:
            plan_plain = f"{n_pend} planning application{'s' if n_pend != 1 else ''} {plan_where}, but none from a competitor."
        else:
            plan_plain = f"No new businesses have applied to open {plan_where}."
        plan_plain += (f" {n_exist} {kind_one if n_exist == 1 else kind_many} already trade within {EXISTING_RADIUS_M} m."
                       if n_exist else f" No {kind_many} trade within {EXISTING_RADIUS_M} m yet.")
        bus_plain = (f"{n_stops} bus stop{'s' if n_stops != 1 else ''} within {R['transit_radius']} m, with a bus about "
                     f"every {hw_plain} minutes." if n_stops else
                     f"The nearest bus stop is about {approx_n(nearest)} m away.")
        c1, c2, c3 = st.columns(3)
        c1.markdown(action_card(
            "users", "When it's busiest", f"{fmt_time(best['peak_start'])} – {fmt_time(best['peak_end'])}",
            ff_status, ff_cls, ff_icon,
            f"About {approx_n(peak_rate)} people walk past each hour at the busiest time, mostly near the "
            f"{esc(hottest['spot'])}.",
            "Peak Hours & Traffic: the busiest 90 minutes within your focus hours.",
        ), unsafe_allow_html=True)
        c2.markdown(action_card(
            "building", "New competition", f"{n_pend} Pending Application{'s' if n_pend != 1 else ''}",
            pl_status, pl_cls, pl_icon, esc(plan_plain),
            "Local Planning & Competition: a planning application is a business asking the council for permission "
            "to open or change a premises.",
        ), unsafe_allow_html=True)
        c3.markdown(action_card(
            "bus", "Getting here by bus", f"{n_stops} Bus Stop{'s' if n_stops != 1 else ''} (<{R['transit_radius']}m)",
            tr_status, tr_cls, tr_icon, esc(bus_plain),
            "Public Transport Access: bus stops close to the recommended spot.",
        ), unsafe_allow_html=True)
        st.write("")
        with st.expander("Show technical details", icon=":material/info:"):
            html_block(
                f"<div class='ug-verdict'><div class='main'>"
                f"<div class='ug-eyebrow'>{icon('pin', 14)}Recommended spot · {esc(scope)}</div>"
                f"<div class='where'>{esc(best['spot'])}, {esc(best['zone'])}</div>"
                f"<div class='ug-reasons'>{reasons_html}</div>{note_html}</div>"
                f"<div class='ug-ring' style='--p:{int(best['score'])}' title='{esc(score_tip)}'><div class='in'>"
                f"<div class='num'>{int(best['score'])}</div><div class='lbl'>Opportunity<br/>score</div></div></div></div>"
            )
            st.write("")
            c1, c2, c3 = st.columns(3)
            c1.markdown(action_card(
                "users", "Peak Hours & Traffic",
                f"{fmt_time(best['peak_start'])} – {fmt_time(best['peak_end'])}",
                ff_status, ff_cls, ff_icon,
                f"Highest {period} pedestrian traffic is concentrated near the {esc(hottest['spot'])}.",
                f"≈{peak_rate:,.0f} pedestrians/hr at peak. Source: {DATA['footfall_sources'][best['zone_key']]}.",
            ), unsafe_allow_html=True)
            c2.markdown(action_card(
                "building", "Local Planning & Competition",
                f"{n_pend} Pending Application{'s' if n_pend != 1 else ''}",
                pl_status, pl_cls, pl_icon, plan_detail,
                f"Competitor type: {comp_label}. Applications: {plan_src}. Existing {kind_many}: {biz_state[1].split(' · ')[0]}.",
            ), unsafe_allow_html=True)
            c3.markdown(action_card(
                "bus", "Public Transport Access",
                f"{n_stops} Bus Stop{'s' if n_stops != 1 else ''} (<{R['transit_radius']}m)",
                tr_status, tr_cls, tr_icon, tr_detail,
                f"Nearest stop ≈{nearest:,.0f}m. Source: {transit_src}."
                + (" Arrival intervals are timetable estimates." if R["headway_estimated"] else ""),
            ), unsafe_allow_html=True)

    else:
        html_block(
            f"<div class='ug-verdict'><div class='main'>"
            f"<div class='ug-eyebrow'>{icon('pin', 14)}Recommended spot · {esc(scope)}</div>"
            f"<div class='where'>{esc(best['spot'])}, {esc(best['zone'])}</div>"
            f"<div class='ug-reasons'>{reasons_html}</div>{note_html}</div>"
            f"<div class='ug-ring' style='--p:{int(best['score'])}' title='{esc(score_tip)}'><div class='in'>"
            f"<div class='num'>{int(best['score'])}</div><div class='lbl'>Opportunity<br/>score</div></div></div></div>"
        )
        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.markdown(action_card(
            "users", "Peak Hours & Traffic",
            f"{fmt_time(best['peak_start'])} – {fmt_time(best['peak_end'])}",
            ff_status, ff_cls, ff_icon,
            f"Highest {period} pedestrian traffic is concentrated near the {esc(hottest['spot'])}.",
            f"≈{peak_rate:,.0f} pedestrians/hr at peak. Source: {DATA['footfall_sources'][best['zone_key']]}.",
        ), unsafe_allow_html=True)
        c2.markdown(action_card(
            "building", "Local Planning & Competition",
            f"{n_pend} Pending Application{'s' if n_pend != 1 else ''}",
            pl_status, pl_cls, pl_icon, plan_detail,
            f"Competitor type: {comp_label}. Applications: {plan_src}. Existing {kind_many}: {biz_state[1].split(' · ')[0]}.",
        ), unsafe_allow_html=True)
        c3.markdown(action_card(
            "bus", "Public Transport Access",
            f"{n_stops} Bus Stop{'s' if n_stops != 1 else ''} (<{R['transit_radius']}m)",
            tr_status, tr_cls, tr_icon, tr_detail,
            f"Nearest stop ≈{nearest:,.0f}m. Source: {transit_src}."
            + (" Arrival intervals are timetable estimates." if R["headway_estimated"] else ""),
        ), unsafe_allow_html=True)

    st.write("")
    r_txt, r_btn = st.columns([3, 1.25], vertical_alignment="center")
    payload = report_payload()
    slug = re.sub(r"[^a-z0-9]+", "-", f"{best['zone']} {best['spot']}".lower()).strip("-")
    try:
        if not FPDF_OK:
            raise RuntimeError("fpdf2 not installed")
        report_bytes, r_mime, r_ext, r_label = build_pdf(payload), "application/pdf", "pdf", "Download area report (PDF)"
    except Exception:  # noqa: BLE001 - never let the report break the page
        report_bytes, r_mime, r_ext, r_label = (build_html_report(payload), "text/html", "html",
                                                "Download area report (HTML)")
    with r_txt:
        html_block(f"<div class='ug-report'>{icon('briefcase', 16)}<span>Take this analysis to your bank, landlord or "
                   f"partner: a two-page area report with the verdict, cards, footfall chart, break-even and "
                   f"nearby businesses.</span></div>")
    with r_btn:
        st.download_button(r_label, data=report_bytes, file_name=f"urbangrid-{slug}.{r_ext}", mime=r_mime,
                           icon=":material/picture_as_pdf:" if r_ext == "pdf" else ":material/description:",
                           on_click="ignore", width="stretch", key="report_dl")

    if "zone" not in ss.get("parsed", {}):
        st.caption("No Cork street was found in your question, so all six zones were compared. Try Patrick St, Grand Parade, Oliver Plunkett St, Washington St, Douglas or Ballincollig.")

elif page == "map":
    st.segmented_control("Map view", ["spots", "heat"], key="map_view", label_visibility="collapsed",
                         format_func=lambda k: {"spots": ":material/location_on: Spots & layers",
                                                "heat": ":material/grid_on: Opportunity heatmap"}[k])
    labels = pd.DataFrame([{"label": z["name"], "lon": z["path"][len(z["path"]) // 2][0],
                            "lat": z["path"][len(z["path"]) // 2][1]} for z in ZONES.values()])
    if ss.map_view == "heat":
        grid = opportunity_grid()
        tops = top_squares(grid)
        grid["name"] = "Square near " + grid["near"].map(esc)
        grid["info"] = [f"Opportunity <b>{sc}/100</b> · ≈{r:,.0f} people/hr (estimated)<br/>{int(e)} existing "
                        f"{kind_many} &lt;{EXISTING_RADIUS_M}m · {int(pl)} competitor applications · {int(b)} bus stops "
                        f"&lt;150m" for sc, r, e, pl, b in zip(grid["score"], grid["rate"], grid["existing"],
                                                               grid["pipeline"], grid["stops150"])]
        hm_col, hs_col = st.columns([3.1, 1], gap="medium")
        with hm_col:
            focus = grid[grid["zone_key"].isin(CITY_CENTRE_ZONES)] if best["zone_key"] in CITY_CENTRE_ZONES else \
                grid[grid["zone_key"] == best["zone_key"]]
            view = (pdk.ViewState(latitude=51.8978, longitude=-8.4745, zoom=15.0, pitch=0)
                    if best["zone_key"] in CITY_CENTRE_ZONES else
                    pdk.ViewState(latitude=best["lat"], longitude=best["lon"], zoom=15.4, pitch=0))
            best_df = pd.DataFrame([{"lat": best["lat"], "lon": best["lon"], "name": f"Recommended: {best['spot']}",
                                     "info": f"{best['zone']} · score {best['score']}/100"}])
            heat_layers = [
                pdk.Layer("PolygonLayer", grid, get_polygon="polygon", get_fill_color="color",
                          get_line_color=[10, 17, 32, 160], line_width_min_pixels=1, stroked=True, pickable=True),
                pdk.Layer("ScatterplotLayer", best_df, get_position="[lon, lat]", get_fill_color=[0, 0, 0, 0],
                          get_line_color=[255, 255, 255, 255], stroked=True, filled=False,
                          line_width_min_pixels=2.5, get_radius=45, radius_min_pixels=14, pickable=True),
                pdk.Layer("ScatterplotLayer", tops, get_position="[lon, lat]", get_fill_color=[251, 191, 36, 240],
                          get_line_color=[10, 17, 32, 255], stroked=True, line_width_min_pixels=2, get_radius=14,
                          radius_min_pixels=11, pickable=False),
                pdk.Layer("TextLayer", tops, get_position="[lon, lat]", get_text="rank", get_size=14,
                          get_color=[10, 17, 32, 255], font_weight=800),
                pdk.Layer("TextLayer", labels, get_position="[lon, lat]", get_text="label", get_size=13,
                          get_color=[226, 232, 240, 230], get_pixel_offset=[0, -22], font_weight=700),
            ]
            st.pydeck_chart(pdk.Deck(
                layers=heat_layers, initial_view_state=view, map_provider="carto", map_style="dark",
                tooltip={"html": "<b>{name}</b><br/>{info}",
                         "style": {"backgroundColor": "#0f172a", "color": "#e6edf7", "fontSize": "12px",
                                   "border": "1px solid #2a3854", "borderRadius": "8px", "padding": "8px 10px"}},
            ), height=470)
            bins = "".join(f"<span><i style='background:rgba({c[0]},{c[1]},{c[2]},{c[3] / 255:.2f});border-radius:3px'>"
                           f"</i>{lbl}</span>" for _, c, lbl in HEAT_BINS)
            html_block(f"<div class='ug-legend'><span style='color:var(--text-2)'>Opportunity score</span>{bins}"
                       f"<span><i style='background:#fbbf24'></i>Top 3 squares</span>"
                       f"<span><i style='background:transparent;border:2px solid #fff'></i>Recommended spot</span></div>")
            st.caption(f"Each {GRID_STEP_M} m square is scored with the same formula as the named spots, for "
                       f"{ss.business.lower()} · {DAY_SHORT} · {ss.time_window[0]}–{ss.time_window[1]}. "
                       "Foot traffic between measured spots is estimated, so treat squares far from the main streets "
                       "as indicative.")
        with hs_col:
            rows_html = "".join(
                f"<li><span><b style='color:#fbbf24'>#{r['rank']}</b> near {esc(r['near'].split(',')[0])}</span>"
                f"<span class='r'>{r['score']}</span></li>" for _, r in tops.iterrows())
            in_focus = focus.sort_values("score", ascending=False)
            html_block(
                f"<div class='ug-panel'><h4>{icon('grid', 13)}Top squares</h4><ul class='ug-list'>{rows_html}</ul>"
                f"<h4>{icon('pin', 13)}Recommended spot</h4><div class='who'>{esc(best['spot'])}</div>"
                f"<div class='ug-empty' style='margin-top:.2rem'>Spot score {best['score']}/100. Squares average a "
                f"{GRID_STEP_M} m area, so they score a little below exact spots.</div>"
                f"<h4>{icon('info', 13)}Squares scoring 70+</h4>"
                f"<div class='who'>{int((in_focus['score'] >= 70).sum())} of {len(in_focus)}</div>"
                f"<div class='ug-empty' style='margin-top:.2rem'>in this area</div></div>"
            )
    else:
        ss.setdefault("map_layers_v2", ["Footfall", "Existing competitors", "Competitor applications", "Bus stops"])
        LAYERS = ["Footfall", "Existing competitors", "Competitor applications", "Other applications", "Bus stops"]
        with (st.popover("Map layers", icon=":material/layers:") if SIMPLE else st.container()):
            shown = st.pills("Map layers", LAYERS, selection_mode="multi",
                             key="map_layers_v2", label_visibility="collapsed") or []
        map_col, side_col = st.columns([3.1, 1], gap="medium")

        with map_col:
            spots["color"] = spots["window_rate"].apply(density_color)
            spots["radius"] = 10 + 16 * spots["window_rate"] / max_rate
            spots["name"] = spots["spot"] + " · " + spots["zone"]
            spots["info"] = spots.apply(
                lambda r: (f"≈{r['window_rate']:,.0f} pedestrians/hr ({esc(fmt_time(start_h))}–{esc(fmt_time(end_h))})"
                           f"<br/>Opportunity score {r['score']}/100 · rent ≈ €{r['rent']:,}"
                           f"<br/>{r['competitors']} competing / {r['pending']} pending applications · "
                           f"{r['stops_150m']} stops &lt;150m"), axis=1)
            corridors = pd.DataFrame([
                {"path": z["path"], "color": density_color(spots.loc[spots["zone_key"] == k, "window_rate"].mean(), 140),
                 "name": z["name"], "info": f"Retail corridor · typical rent ≈ €{z['rent']:,}/month"}
                for k, z in ZONES.items()
            ])
    
            stops_df = R["stops"].assign(dist=R["transit_all_dist"])
            stops_df = stops_df[stops_df["dist"] <= 300].copy()      # only stops around the recommended spot
            stops_df["name"] = "Bus stop: " + stops_df["name"].astype(str)
            stops_df["info"] = stops_df.apply(
                lambda r: (f"{r['dist']:.0f}m from recommended spot<br/>Headway ≈ "
                           + (f"{r['headway_min']:.0f} min" if pd.notna(r.get("headway_min")) else "n/a (no timetable in NaPTAN)")),
                axis=1)

            want = []
            if "Competitor applications" in shown:
                want.append(plan_all["is_comp"])
            if "Other applications" in shown:
                want.append(plan_all["pending"] & ~plan_all["is_comp"])
            plan_df = plan_all[np.logical_or.reduce(want)] if want else plan_all.iloc[0:0]
            plan_df = plan_df.copy()
            if not plan_df.empty:
                plan_df["color"] = [[236, 72, 153, 240] if c else [167, 139, 250, 220] for c in plan_df["is_comp"]]
                plan_df["name"] = plan_df["address"].map(esc)
                plan_df["info"] = plan_df.apply(
                    lambda r: (f"{esc(r['status'].title())} · {esc(CATEGORY_LABELS[r['category']])} · received {esc(r['received'])}"
                               f"<br/>{esc(r['description'][:140])}{'…' if len(r['description']) > 140 else ''}"
                               + ("<br/><i>Position approximate (matched by street address)</i>" if r["approx"] else "")),
                    axis=1)

            rivals_df = R["rivals"][R["rivals"]["distance_m"] <= 400].copy()
            rivals_df["name"] = rivals_df["name"].map(esc)
            rivals_df["info"] = rivals_df["distance_m"].map(
                lambda d: f"Existing {kind_one} · {d:.0f}m from recommended spot<br/><i>OpenStreetMap</i>")

            best_df = pd.DataFrame([{"lat": best["lat"], "lon": best["lon"], "name": f"Recommended: {best['spot']}",
                                     "info": f"{best['zone']} · score {best['score']}/100"}])
            if best["zone_key"] in CITY_CENTRE_ZONES:
                view = pdk.ViewState(latitude=51.8978, longitude=-8.4745, zoom=15.1, pitch=0)
            else:
                view = pdk.ViewState(latitude=best["lat"], longitude=best["lon"], zoom=15.6, pitch=0)

            layers = []
            if "Footfall" in shown:
                layers.append(pdk.Layer("PathLayer", corridors, get_path="path", get_color="color", get_width=14,
                                        width_min_pixels=4, cap_rounded=True, joint_rounded=True, pickable=True))
            if "Bus stops" in shown and not stops_df.empty:
                layers.append(pdk.Layer("ScatterplotLayer", stops_df, get_position="[lon, lat]",
                                        get_fill_color=[56, 189, 248, 230], get_line_color=[10, 17, 32, 255],
                                        stroked=True, line_width_min_pixels=1.5, get_radius=6, radius_min_pixels=4,
                                        pickable=True))
            if "Existing competitors" in shown and not rivals_df.empty:
                layers.append(pdk.Layer("ScatterplotLayer", rivals_df, get_position="[lon, lat]",
                                        get_fill_color=[163, 230, 53, 220], get_line_color=[10, 17, 32, 255],
                                        stroked=True, line_width_min_pixels=1, get_radius=5, radius_min_pixels=3.5,
                                        pickable=True))
            if not plan_df.empty:
                layers.append(pdk.Layer("ScatterplotLayer", plan_df, get_position="[lon, lat]", get_fill_color="color",
                                        get_line_color=[255, 255, 255, 200], stroked=True, line_width_min_pixels=1,
                                        get_radius=8, radius_min_pixels=5, pickable=True))
            if "Footfall" in shown:
                layers.append(pdk.Layer("ScatterplotLayer", spots, get_position="[lon, lat]", get_fill_color="color",
                                        get_line_color=[255, 255, 255, 220], stroked=True, line_width_min_pixels=1,
                                        get_radius="radius", radius_min_pixels=5, pickable=True))
            layers.append(pdk.Layer("ScatterplotLayer", best_df, get_position="[lon, lat]", get_fill_color=[0, 0, 0, 0],
                                    get_line_color=[255, 255, 255, 255], stroked=True, filled=False,
                                    line_width_min_pixels=2.5, get_radius=45, radius_min_pixels=14, pickable=True))
            layers.append(pdk.Layer("TextLayer", labels, get_position="[lon, lat]", get_text="label", get_size=13,
                                    get_color=[226, 232, 240, 230], get_pixel_offset=[0, -22], font_weight=700))
            st.pydeck_chart(pdk.Deck(
                layers=layers, initial_view_state=view, map_provider="carto", map_style="dark",
                tooltip={"html": "<b>{name}</b><br/>{info}",
                         "style": {"backgroundColor": "#0f172a", "color": "#e6edf7", "fontSize": "12px",
                                   "border": "1px solid #2a3854", "borderRadius": "8px", "padding": "8px 10px"}},
            ), height=470)

            legend = ["<span><i style='background:transparent;border:2px solid #fff'></i>Recommended spot</span>"]
            if "Footfall" in shown:
                legend += ["<span><i style='background:#f87171'></i>High footfall</span>",
                           "<span><i style='background:#fbbf24'></i>Medium</span>",
                           "<span><i style='background:#2dd4bf'></i>Lower</span>"]
            if "Existing competitors" in shown:
                legend.append(f"<span><i style='background:#a3e635'></i>Existing {kind_many} (≤400m)</span>")
            if "Competitor applications" in shown:
                legend.append("<span><i style='background:#ec4899'></i>Competitor application</span>")
            if "Other applications" in shown:
                legend.append("<span><i style='background:#a78bfa'></i>Other pending application</span>")
            if "Bus stops" in shown:
                legend.append("<span><i style='background:#38bdf8'></i>Bus stop (≤300m)</span>")
            html_block(f"<div class='ug-legend'>{''.join(legend)}</div>")

        with side_col:
            near_stops = R["transit_in_radius"].head(5)
            stop_items = "".join(
                f"<li><span>{esc(r['name'])}</span><span class='r'>{r['distance_m']:.0f}m</span></li>"
                for _, r in near_stops.iterrows()
            ) or ""
            near_plan = plan_all.loc[plan_all.index.isin(R["near_plan_ids"])]
            plan_items = "".join(
                f"<li><span><span class='dot' style='background:{'#ec4899' if r['is_comp'] else '#a78bfa'}'></span>"
                f"{esc(r['address'][:42])}{'…' if len(r['address']) > 42 else ''}</span>"
                f"<span class='r'>{esc(CATEGORY_LABELS[r['category']])}</span></li>"
                for _, r in near_plan.head(6).iterrows()
            )
            more = f"<div class='ug-empty'>+ {len(near_plan) - 6} more in Raw data</div>" if len(near_plan) > 6 else ""
            rival_items = "".join(
                f"<li><span>{esc(r['name'][:34])}</span><span class='r'>{r['distance_m']:.0f}m</span></li>"
                for _, r in R["rivals"][R["rivals"]["distance_m"] <= EXISTING_RADIUS_M].head(4).iterrows()
            )
            html_block(
                f"<div class='ug-panel'><h4>{icon('pin', 13)}Around the recommended spot</h4>"
                f"<div class='who'>{esc(best['spot'])}</div>"
                f"<h4>{icon('bus', 13)}Nearest bus stops</h4>"
                + (f"<ul class='ug-list'>{stop_items}</ul>" if stop_items else "<div class='ug-empty'>None nearby</div>")
                + f"<h4>{icon('store', 13)}Existing {kind_many} (&lt;{EXISTING_RADIUS_M}m): {n_exist}</h4>"
                + (f"<ul class='ug-list'>{rival_items}</ul>" if rival_items else "<div class='ug-empty'>None nearby</div>")
                + f"<h4>{icon('building', 13)}Pending applications</h4>"
                + (f"<ul class='ug-list'>{plan_items}</ul>{more}" if plan_items else
                   "<div class='ug-empty'>None pending nearby</div>")
                + "</div>"
            )

elif page == "footfall":
    other_best = (R["spots"][R["spots"]["zone_key"] != best["zone_key"]].sort_values("score", ascending=False)
                  .groupby("zone_key", sort=False).head(1))
    cmp_opts = ["none", *other_best["zone_key"]]
    if ss.get("ff_compare") not in cmp_opts:
        ss.ff_compare = "none"
    if SIMPLE:
        f_left = st.container()
        f_right = st.expander("Compare with another street", icon=":material/compare_arrows:")
    else:
        f_left, f_right = st.columns([3, 1.3], vertical_alignment="bottom")
    with f_left:
        if SIMPLE:
            html_block(
                "<div class='ug-tiles'>"
                f"<div class='ug-tile'><div class='k'>Busiest 90 minutes</div><div class='v'>"
                f"{fmt_time(best['peak_start'])} – {fmt_time(best['peak_end'])}</div><div class='s'>about "
                f"{approx_n(peak_rate)} people an hour</div></div>"
                f"<div class='ug-tile'><div class='k'>People passing in your focus hours</div><div class='v'>about "
                f"{approx_n(window_passers)}</div><div class='s'>{fmt_time(start_h)} – {fmt_time(end_h)}, {DAY_SHORT}</div></div>"
                f"<div class='ug-tile'><div class='k'>Busiest time of the whole day</div>"
                f"<div class='v'>{fmt_time(day_ts[busiest_i])}</div><div class='s'>about {approx_n(day_rates[busiest_i])} "
                f"people an hour</div></div></div>"
            )
        else:
            html_block(
                "<div class='ug-tiles'>"
                f"<div class='ug-tile'><div class='k'>Peak 90 minutes</div><div class='v'>{fmt_time(best['peak_start'])} – "
                f"{fmt_time(best['peak_end'])}</div><div class='s'>≈{peak_rate:,.0f} people/hr</div></div>"
                f"<div class='ug-tile'><div class='k'>Passers-by in your hours</div><div class='v'>≈{window_passers:,.0f}</div>"
                f"<div class='s'>{fmt_time(start_h)} – {fmt_time(end_h)}, {DAY_SHORT}</div></div>"
                f"<div class='ug-tile'><div class='k'>Busiest half-hour of the day</div>"
                f"<div class='v'>{fmt_time(day_ts[busiest_i])}</div><div class='s'>≈{day_rates[busiest_i]:,.0f} people/hr</div></div>"
                "</div>"
            )
    with f_right:
        cmp_key = st.selectbox(
            "Compare with", cmp_opts, key="ff_compare",
            format_func=lambda k: "No comparison" if k == "none" else
            f"{ZONES[k]['name']} ({other_best.loc[other_best['zone_key'] == k, 'spot'].iloc[0]})")

    band = np.where(in_peak, "Peak 90 min", np.where(in_window, "Your trading hours", "Rest of day"))
    ff_df = pd.DataFrame({"t": day_ts, "time": [fmt_time(t) for t in day_ts],
                          "people": np.round(day_rates), "band": band, "spot": best["spot"]})
    ff_df["slot"] = ff_df["time"] + " – " + [fmt_time(t + 0.5) for t in day_ts]
    hour_ticks = [fmt_time(t) for t in day_ts if t % 2 == 0]
    x_enc = alt.X("time:O", sort=list(ff_df["time"]), title=None,
                  axis=alt.Axis(values=hour_ticks, labelAngle=0, labelColor="#6f7f96", tickColor="#1e293f",
                                domainColor="#2a3854", labelFontSize=11))
    bars = alt.Chart(ff_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width={"band": 0.86}).encode(
        x=x_enc,
        y=alt.Y("people:Q", title="Pedestrians per hour",
                axis=alt.Axis(gridColor="#1a2438", labelColor="#6f7f96", titleColor="#6f7f96", domain=False,
                              tickCount=5, titleFontWeight="normal", labelFontSize=11)),
        color=alt.Color("band:N", legend=None,
                        scale=alt.Scale(domain=["Peak 90 min", "Your trading hours", "Rest of day"],
                                        range=["#38bdf8", "#1f5f80", "#243149"])),
        tooltip=[alt.Tooltip("slot:N", title="Time"), alt.Tooltip("people:Q", title="People/hr", format=",.0f"),
                 alt.Tooltip("band:N", title="Window"), alt.Tooltip("spot:N", title="Spot")],
    )
    chart = bars
    legend_items = ["<span><i style='background:#38bdf8;border-radius:3px'></i>Peak 90 min</span>",
                    "<span><i style='background:#1f5f80;border-radius:3px'></i>Your trading hours</span>",
                    "<span><i style='background:#243149;border-radius:3px'></i>Rest of day</span>"]
    if cmp_key != "none":
        orow = other_best[other_best["zone_key"] == cmp_key].iloc[0]
        _, o_rates = slot_rates(PROFILES[cmp_key], DAY_START, DAY_END)
        o_df = pd.DataFrame({"time": ff_df["time"], "slot": ff_df["slot"], "people": np.round(o_rates * orow["share"]),
                             "spot": f"{orow['spot']}, {ZONES[cmp_key]['name']}"})
        line = alt.Chart(o_df).mark_line(color="#fbbf24", strokeWidth=2, interpolate="monotone").encode(
            x=alt.X("time:O", sort=list(ff_df["time"])), y="people:Q")
        pts = alt.Chart(o_df).mark_point(color="#fbbf24", filled=True, size=70, opacity=0).encode(
            x=alt.X("time:O", sort=list(ff_df["time"])), y="people:Q",
            tooltip=[alt.Tooltip("slot:N", title="Time"), alt.Tooltip("people:Q", title="People/hr", format=",.0f"),
                     alt.Tooltip("spot:N", title="Spot")])
        chart = alt.layer(bars, line, pts)
        legend_items.append(f"<span><i style='background:#fbbf24;height:3px;border-radius:2px;vertical-align:3px'></i>"
                            f"{esc(orow['spot'])}, {esc(ZONES[cmp_key]['name'])}</span>")
    chart = chart.properties(height=300, background="transparent").configure_view(strokeWidth=0)
    st.altair_chart(chart, width="stretch", theme=None)
    html_block(f"<div class='ug-legend' style='margin-top:.1rem'>{''.join(legend_items)}</div>")
    st.caption(f"{DAY_LABEL} pedestrians per hour at {best['spot']}, {best['zone']}, in half-hour steps. "
               + ("Saturday/Christmas figures are estimated adjustments to weekday counts. "
                  if ss.day_type != "weekday" else "")
               + 
               f"Source: {DATA['footfall_sources'][best['zone_key']]}. Hover a bar for exact values.")

elif page == "money":
    def _be_inputs():
        st.caption(f"Assumptions for a {ss.business.lower()} at **{best['spot']}**. Defaults are typical Irish figures; "
                   "change any of them.")
        i1, i2 = st.columns(2)
        i1.number_input("Average spend per customer (€)", 0.5, 2000.0, step=0.5, key=BE_KEYS["spend"][0])
        i2.number_input("Gross margin (%)", 5, 95, step=1, key=BE_KEYS["margin"][0])
        i3, i4 = st.columns(2)
        i3.number_input("Monthly rent (€)", 0, 50000, step=250, key=BE_KEYS["rent"][0],
                        help=f"Defaults to the typical rent on {best['zone']}.")
        i4.number_input("Other monthly costs (€)", 0, 200000, step=500, key=BE_KEYS["other"][0],
                        help="Staff, utilities, insurance, loan repayments.")
        i5, i6 = st.columns([1, 2])
        i5.number_input("Trading days / month", 1, 31, step=1, key=BE_KEYS["days"][0])
        i6.select_slider("Opening hours", options=TIME_OPTIONS, key=BE_KEYS["hours"][0], value=BE_KEYS["hours"][1])

    def _be_outputs():
        need_txt = f"{need_day:,.0f}" if np.isfinite(need_day) else "n/a"
        cap_txt = f"{capture_need:.2f}%" if np.isfinite(capture_need) else "n/a"
        html_block(
            f"<div class='ug-be-head'><div class='ug-eyebrow'>{icon('euro', 14)}Break-even point</div>"
            f"<div class='big'>≈{need_txt} customers <small>per day</small></div>"
            f"<div class='line'>to cover €{fixed:,.0f}/month (rent €{be_rent:,.0f} + other costs €{be_other:,.0f}). "
            f"That is <b>{cap_txt}</b> of the ≈{passers_day:,.0f} people passing during opening hours.</div>"
            f"<div class='ug-status {be_cls}' style='margin-top:.6rem;white-space:normal'>{icon(be_icon, 14)}"
            f"{be_status}: {be_msg}</div>"
            f"<div class='line' style='font-size:.82rem;margin-top:.45rem'>Typical range for this business: "
            f"{cap_lo:g}–{cap_hi:g}% of passers-by (average {cap_mid:g}%).</div></div>"
        )
        scale_max = max(cap_hi * 1.6, (capture_need if np.isfinite(capture_need) else 0) * 1.15)
        pos = lambda v: max(0.0, min(100.0, 100 * v / scale_max))  # noqa: E731
        mark_pos = pos(capture_need) if np.isfinite(capture_need) else 100
        html_block(
            f"<div class='ug-range'><div class='band' style='left:{pos(cap_lo):.1f}%;width:{pos(cap_hi) - pos(cap_lo):.1f}%'>"
            f"</div><div class='mid' style='left:calc({pos(cap_mid):.1f}% - 1px)'><span>average {cap_mid:g}%</span></div>"
            f"<div class='mark' style='left:calc({mark_pos:.1f}% - 1px)'><span>You need {cap_txt}</span></div></div>"
            f"<div class='ug-range-lbl'><span>0%</span><span></span>"
            f"<span>{scale_max:.2g}%</span></div>"
        )
        if max_rent <= 0:
            nego_big, nego_cls, nego_icon = "No rent is affordable", "bad", "x"
            nego_line = (f"At an average share (≈{typical_cust:,.0f} customers/day), takings would not cover the "
                         f"€{be_other:,.0f} of other monthly costs and still leave you a 10% profit.")
        else:
            nego_big = f"Up to €{max_rent:,.0f}<small> / month</small>"
            nego_cls, nego_icon = ("good", "check") if rent_gap >= 0 else ("bad", "alert")
            nego_line = (f"At an average share (≈{typical_cust:,.0f} customers/day, ≈€{turnover:,.0f}/month takings), "
                         f"you can pay up to €{max_rent:,.0f}/month and still keep a 10% profit "
                         f"(break-even rent, with zero profit: €{max(breakeven_rent, 0):,.0f}). "
                         + (f"The asking rent of €{be_rent:,.0f} leaves <b>€{rent_gap:,.0f}/month headroom</b>."
                            if rent_gap >= 0 else
                            f"The asking rent of €{be_rent:,.0f} is <b>€{-rent_gap:,.0f}/month too high</b>: "
                            f"negotiate down or look elsewhere."))
        html_block(
            f"<div class='ug-be-head' style='margin-top:1rem'><div class='ug-eyebrow'>{icon('briefcase', 14)}"
            f"Your rent negotiation number</div><div class='big'>{nego_big}</div>"
            f"<div class='line'>{nego_line}</div>"
            f"<div class='ug-status {nego_cls}' style='margin-top:.6rem;white-space:normal'>{icon(nego_icon, 14)}"
            + ("Asking rent is affordable" if max_rent > 0 and rent_gap >= 0 else "Asking rent is above what the "
               "trade supports") + "</div></div>"
        )
        profit_cls = "var(--good)" if typical_profit >= 0 else "var(--bad)"
        html_block(
            "<div class='ug-tiles' style='margin-top:1rem'>"
            f"<div class='ug-tile'><div class='k'>Rent alone needs</div><div class='v'>≈{rent_day:,.0f}/day</div>"
            f"<div class='s'>customers at €{contrib:,.2f} margin each</div></div>"
            f"<div class='ug-tile'><div class='k'>If you win an average share</div>"
            f"<div class='v'>≈{typical_cust:,.0f}/day</div>"
            f"<div class='s'>customers ({cap_mid:g}% of passers-by)</div></div>"
            f"<div class='ug-tile'><div class='k'>Monthly profit in that case</div>"
            f"<div class='v' style='color:{profit_cls}'>{'−' if typical_profit < 0 else ''}€{abs(typical_profit):,.0f}</div>"
            f"<div class='s'>before tax · a scenario, not a forecast</div></div></div>"
        )
        if be_appt_note:
            st.info(be_appt_note, icon=":material/event_available:")
        day_phrase = {"weekday": "a typical weekday", "saturday": "a Saturday",
                      "christmas": "a Christmas-season day"}[ss.day_type]
        st.caption(f"A rough guide, not financial advice. Assumes every trading day looks like {day_phrase} (switch "
                   "in the sidebar); capture rates are industry rules of thumb; delivery sales are not included.")

    if SIMPLE:
        verdict_word = {"Comfortable": "Yes", "Tight": "It's tight", "Stretch": "Unlikely"}[be_status]
        if max_rent > 0:
            rent_metric = f"Up to €{max_rent:,.0f} / month"
            rent_detail = (f"The asking rent here is about €{be_rent:,.0f}, so you have about €{rent_gap:,.0f} a month to "
                           f"spare while keeping a 10% profit." if rent_gap >= 0 else
                           f"The asking rent here is about €{be_rent:,.0f}, roughly €{-rent_gap:,.0f} a month more than "
                           f"the trade supports. Try to negotiate it down.")
            rent_status, rent_cls, rent_icon = (("Affordable", "good", "check") if rent_gap >= 0 else
                                                ("Too high", "bad", "alert"))
        else:
            rent_metric, rent_status, rent_cls, rent_icon = "None", "Not affordable", "bad", "x"
            rent_detail = "Average sales here would not even cover your other monthly costs."
        a1, a2, a3 = st.columns(3)
        a1.markdown(action_card(
            "users", "Customers you need", f"About {approx_n(need_day)} a day", f"Open {be_hours[0]} – {be_hours[1]}",
            "neutral", "clock",
            f"To cover €{fixed:,.0f} a month in rent and running costs, at about €{contrib:,.2f} profit per customer.",
            "Break-even point: the number of customers at which you neither make nor lose money.",
        ), unsafe_allow_html=True)
        a2.markdown(action_card(
            "target", "Is that realistic?", verdict_word, be_status, be_cls, be_icon,
            f"That is about 1 in {one_in(capture_need)} people walking past. A typical {kind_one} wins about 1 in "
            f"{one_in(cap_mid)}.",
            f"Typical share of passers-by who buy: {cap_lo:g}–{cap_hi:g}% for this business.",
        ), unsafe_allow_html=True)
        a3.markdown(action_card(
            "euro", "Rent you can afford", rent_metric, rent_status, rent_cls, rent_icon, esc(rent_detail),
            "Your rent negotiation number: the most rent an average business here can pay and keep a 10% profit.",
        ), unsafe_allow_html=True)
        if be_appt_note:
            st.info(be_appt_note, icon=":material/event_available:")
        st.write("")
        with st.expander("Change the assumptions (spend, margin, rent, costs, hours)", icon=":material/tune:"):
            _be_inputs()
        with st.expander("Show the maths", icon=":material/calculate:"):
            _be_outputs()
    else:
        be_in, be_out = st.columns([1, 1.45], gap="large")
        with be_in:
            _be_inputs()
        with be_out:
            _be_outputs()

elif page == "compare":
    show_all = st.toggle("Show all columns", key="cmp_all")
    cols = ["pick", "zone", "spot", "score", "peak", "budget_ok"]
    if show_all:
        cols += ["window_rate", "existing", "competitors", "pending", "stops_150m", "rent"]
    show_table(
        rank[cols],
        column_config={
            "pick": st.column_config.TextColumn("", width="small", help="★ = zone of the recommended spot"),
            "zone": "Zone", "spot": "Best spot in zone",
            "score": st.column_config.ProgressColumn("Opportunity score", min_value=0, max_value=100, format="%d"),
            "peak": "Peak 90 min", "budget_ok": "Budget",
            "window_rate": st.column_config.NumberColumn("Avg pedestrians/hr", format="%.0f"),
            "existing": f"Existing {kind_many} <{EXISTING_RADIUS_M}m",
            "competitors": "Competing applications", "pending": "All pending",
            "stops_150m": "Bus stops <150m",
            "rent": st.column_config.NumberColumn("Typical rent / month", format="€%d"),
        },
    )
    st.caption("Best-scoring spot in each zone for your current settings. ★ marks the recommended zone.")

else:  # data & limits
    step("i", "Honest System Limits & Edge Cases")
    lim_l, lim_r = st.columns(2)
    with lim_l:
        st.info(
            "**Handles well**\n\n"
            "- Open API footfall trends: hour-by-hour patterns for six Cork zones\n"
            "- Spatial mapping: distances to every bus stop, planning application and existing competitor\n"
            "- Live open data with automatic offline failover",
            icon=":material/verified:",
        )
    with lim_r:
        st.warning(
            "**Current limitations**\n\n"
            "- Historical seasonal variance: Saturday and Christmas are estimates; summer and festivals are not modelled\n"
            "- Real-time rent negotiation status is unknown; rents are estimates\n"
            "- Bus intervals, some footfall and break-even figures are estimates",
            icon=":material/warning:",
        )
    html_block("<div class='ug-divider'></div>")
    wf, wc, wt, wb = PRIORITY_WEIGHTS[ss.priority]
    m_left, m_right = st.columns(2, gap="large")
    with m_left:
        html_block(
            "<div class='ug-method'>"
            "<h4>How the score works</h4>"
            f"Score = 100 × ({wf} × footfall + {wc} × competition + {wt} × transit + {wb} × budget), "
            f"weighted for your priority <b>{esc(ss.priority)}</b>. Each part is scaled 0–1:"
            "<ul><li><b>Footfall:</b> average pedestrians/hr in your trading hours, relative to the busiest spot.</li>"
            "<li><b>Competition:</b> 60% planning pipeline (pending same-type applications nearby; 4 or more "
            f"scores 0) + 40% saturation (existing same-type businesses within {EXISTING_RADIUS_M}m, relative to "
            "the most crowded spot).</li>"
            "<li><b>Transit:</b> bus stops within 150m and distance to the nearest stop.</li>"
            "<li><b>Budget:</b> typical zone rent against your monthly budget.</li></ul>"
            "<h4>Where the data comes from</h4>"
            "<ul><li>Pedestrian footfall counters: Cork City Council (data.corkcity.ie)</li>"
            "<li>Planning applications: Cork City Council planning register (data.corkcity.ie)</li>"
            "<li>Bus stops: NTA NaPTAN register (data.gov.ie)</li>"
            "<li>Existing businesses: OpenStreetMap via the Overpass API (© OpenStreetMap contributors, ODbL)</li>"
            "<li>Rent levels and fallback data: embedded Cork estimates</li></ul></div>"
        )
    with m_right:
        html_block(
            "<div class='ug-method'><h4>Limitations in detail</h4><ul>"
            "<li><b>Historical seasonal variance:</b> counts are typical weekdays. The Saturday and Christmas options "
            "apply estimated multipliers (Saturday: no commuter peak, busier 12–6 PM; Christmas: busier daytime and "
            "late opening), not measured Cork data. Summer, college terms and the Jazz Festival are not modelled.</li>"
            "<li><b>Real-time rent negotiation status:</b> rents are indicative asking-rent estimates per zone, not "
            "live listings or negotiated deals.</li>"
            f"<li><b>Footfall API:</b> {esc(footfall_limit)}</li>"
            f"<li><b>Planning data:</b> live records are matched by street address (source coordinates are "
            f"mis-projected). {esc(latest_txt)}</li>"
            "<li><b>Bus frequency:</b> NaPTAN lists stop locations, not timetables, so live arrival intervals are "
            "estimates.</li>"
            "<li><b>Existing businesses:</b> OpenStreetMap is community-mapped, so recent openings or closures "
            "may be missing. The public Overpass API is often slow, so it refreshes in the background and an "
            "embedded snapshot is used meanwhile.</li>"
            "<li><b>Break-even:</b> uses the selected day's footfall for every trading day and industry capture-rate "
            "rules of thumb. It is a "
            "sense-check, not a financial forecast.</li>"
            "<li><b>Live data failover:</b> if any API times out (10-second budget), that source switches to "
            "embedded Cork data and is flagged in the header status.</li></ul></div>"
        )
    html_block("<div class='ug-divider'></div>")
    step("i", "Raw data")
    st.caption("Every number on this page comes from these tables.")
    with st.expander("Footfall: pedestrians per hour in your trading hours", icon=":material/directions_walk:",
                     expanded=True):
        ts = np.arange(start_h, end_h, 0.5)
        recs = []
        for zk, z in ZONES.items():
            _, rates = slot_rates(PROFILES[zk], start_h, end_h)
            for sp in z["spots"]:
                row = {"Zone": z["name"], "Spot": sp["spot"], "Source": DATA["footfall_sources"][zk]}
                row.update({fmt_time(t): round(r * sp["share"]) for t, r in zip(ts, rates)})
                recs.append(row)
        show_table(pd.DataFrame(recs))
    with st.expander("Planning applications", icon=":material/apartment:"):
        pdf = R["plan"][["app_no", "zone", "address", "status", "pending", "category", "received",
                         "description", "source", "lat", "lon"]].copy()
        pdf["zone"] = pdf["zone"].map(lambda k: ZONES[k]["name"])
        pdf["near_recommended_spot"] = pdf.index.isin(R["near_plan_ids"])
        st.caption("Pending = new, further information, incomplete, pre-validation or appealed, received in the last "
                   "3 years. Live records are matched by street address (source coordinates are mis-projected).")
        show_table(pdf.sort_values(["near_recommended_spot", "pending", "received"], ascending=False))
    with st.expander("Transit stops", icon=":material/directions_bus:"):
        tdf = R["stops"].copy()
        tdf["distance_to_recommended_m"] = np.round(R["transit_all_dist"])
        st.caption(f"{len(tdf):,} stops loaded, sorted by distance to the recommended spot.")
        show_table(tdf.sort_values("distance_to_recommended_m").head(300))
    with st.expander(f"Existing businesses: {kind_many} (OpenStreetMap)", icon=":material/storefront:"):
        st.caption(f"{len(R['rivals'])} {kind_many} mapped across the six zones ({biz_state[1]}), sorted by distance to "
                   f"the recommended spot. © OpenStreetMap contributors, ODbL.")
        show_table(R["rivals"][["name", "kind", "distance_m", "lat", "lon"]].head(300))
    with st.expander("Scoring model: every candidate spot", icon=":material/calculate:"):
        show_table(
            R["spots"][["zone", "spot", "score", "footfall_score", "competition_score", "transit_score",
                        "budget_score", "window_rate", "peak_rate", "pending", "competitors", "existing",
                        "stops_150m",
                        "nearest_stop_m", "rent"]].sort_values("score", ascending=False).round(3)
        )


# ---------------------------------------------------------------------------
# Back / next + footer
# ---------------------------------------------------------------------------
html_block("<div class='ug-divider'></div>")
n_l, n_m, n_r = st.columns([1.3, 2, 1.3], vertical_alignment="center")
with n_l:
    if p_idx == 0:
        st.button("New search", icon=":material/arrow_back:", on_click=go_home, width="stretch", key="nav_back")
    else:
        prev = PAGES[p_idx - 1]
        st.button(f"Back: {PAGE_META[prev][0]}", icon=":material/arrow_back:", on_click=go_page, args=(prev,),
                  width="stretch", key="nav_back")
with n_m:
    dots = "".join(f"<i class='{'on' if i == p_idx else ''}'></i>" for i in range(len(PAGES)))
    html_block(f"<div class='ug-pager'>{dots}</div>")
with n_r:
    if p_idx < len(PAGES) - 1:
        nxt = PAGES[p_idx + 1]
        st.button(f"Next: {PAGE_META[nxt][0]}", icon=":material/arrow_forward:", icon_position="right", type="primary",
                  on_click=go_page, args=(nxt,), width="stretch", key="nav_next")
    else:
        st.button("Start a new search", icon=":material/search:", type="primary", on_click=go_home, width="stretch",
                  key="nav_next")

html_block(FOOTER_HTML)
