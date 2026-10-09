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

# --- Loans (data card section 2.3) ---
LOAN_RATIO_MEDIAN = 1.0  # loan-to-monthly-income ratio
LOAN_RATIO_SIGMA = 0.5
LOAN_RATIO_MIN = 0.3
LOAN_RATIO_MAX = 3.0
LOAN_ROUND_TO = 500  # KES
LOAN_MIN = 1_000
LOAN_MAX = 100_000
TERM_OPTIONS = (1, 3, 6)  # months
# (largest amount in band, probabilities for 1, 3, 6 months)
TERM_PROBS_BY_AMOUNT = (
    (10_000, (0.60, 0.30, 0.10)),
    (50_000, (0.20, 0.50, 0.30)),
    (LOAN_MAX, (0.05, 0.35, 0.60)),
)

# --- Defaults (data card section 2.5) ---
DEFAULT_TARGET_RATE = 0.20
DEFAULT_RATE_BAND = (0.15, 0.25)

# --- Train / validation / test split (step 2.6) ---
SPLIT_SEED = 2026
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "applicant_split.csv"

# --- Experiment tracking (step 3.1) ---
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT = "mkopoguard"

# --- Evaluation (step 3.2, see docs/evaluation_plan.md) ---
PROVISIONAL_THRESHOLD = 0.20  # decline if p >= this, until step 3.11 picks the real threshold
LEAKAGE_ALARM = {"roc_auc": 0.86, "pr_auc": 0.65}  # above the true-probability ceiling
BOOTSTRAP_RESAMPLES = 1_000
