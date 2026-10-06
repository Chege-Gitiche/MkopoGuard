"""Project-wide settings: paths and the country pool."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Source file
FINDEX_FILE = DATA_RAW / "micro_world_139countries.csv"
FINDEX_ENCODING = "latin1"  # the file is not valid UTF-8

# Country pool: Sub-Saharan countries where >= 40% have a mobile money account
POOL_REGION = "Sub-Saharan Africa (excluding high income)"
MOBILE_MONEY_THRESHOLD = 0.40
POOL_COUNTRIES = (
    "BEN",  # Benin
    "BWA",  # Botswana
    "CIV",  # Cote d'Ivoire
    "CMR",  # Cameroon
    "COG",  # Congo, Rep.
    "GAB",  # Gabon
    "GHA",  # Ghana
    "KEN",  # Kenya
    "LBR",  # Liberia
    "LSO",  # Lesotho
    "MOZ",  # Mozambique
    "NAM",  # Namibia
    "SEN",  # Senegal
    "SWZ",  # Eswatini
    "TGO",  # Togo
    "TZA",  # Tanzania
    "UGA",  # Uganda
    "ZAF",  # South Africa
    "ZMB",  # Zambia
    "ZWE",  # Zimbabwe
)
TARGET_COUNTRY = "KEN"
EXPECTED_POOL_ROWS = 20_060
