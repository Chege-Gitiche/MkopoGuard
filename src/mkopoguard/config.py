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

# Applicants: adults only (Findex surveys people aged 15+)
MIN_AGE = 18
EXPECTED_APPLICANTS = 18_718

# --- Simulation (see docs/data_card.md) ---
SIMULATION_SEED = 42
APPLICATION_START = "2025-01-01"  # application dates are drawn within this year
APPLICATION_END = "2025-12-31"
STATEMENT_DAYS = 180  # statement covers the 180 days before the application date

# Median monthly income (KES) by income quintile; all countries treated as Kenya-equivalent
INCOME_MEDIAN_BY_QUINTILE = {1: 4_000, 2: 7_000, 3: 11_000, 4: 18_000, 5: 32_000}
OUT_OF_WORKFORCE_INCOME_FACTOR = 0.5
INCOME_SIGMA = 0.5  # spread of the log-normal income draw
