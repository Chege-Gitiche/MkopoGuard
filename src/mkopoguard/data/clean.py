"""Clean the Findex pool: rename columns, decode answer codes, keep adults only.

Every code mapping below comes from the Global Findex 2021 codebook
(see docs/findex_columns.md). Any code not listed in a mapping
(don't know, refused, does not apply) becomes missing (NaN).
"""

import pandas as pd

from mkopoguard import config

# Original Findex name -> readable name
COLUMN_MAP = {
    # identifiers and metadata
    "economycode": "country_code",
    "economy": "country",
    "wpid_random": "respondent_id",
    "wgt": "survey_weight",
    "year": "survey_year",
    # fairness audit only
    "female": "is_female",
    # model features
    "age": "age",
    "educ": "education_level",
    "inc_q": "income_quintile",
    "emp_in": "in_workforce",
    "account_fin": "has_bank_account",
    "account_mob": "has_mobile_money",
    "mobileowner": "owns_mobile_phone",
    "internetaccess": "has_internet_access",
    "saved": "saved_past_year",
    "fin17b": "saved_in_savings_club",
    "borrowed": "borrowed_past_year",
    "fin22a": "borrowed_from_fin_institution",
    "fin22b": "borrowed_from_family_friends",
    "fin20": "borrowed_for_medical",
    "fin24": "emergency_fund_source",
    "fin44c": "worried_about_bills",
    "receive_wages": "wage_payment_channel",
    "receive_agriculture": "agri_payment_channel",
    "pay_utilities": "utility_payment_channel",
    "fin26": "sent_domestic_remittance",
    "fin28": "received_domestic_remittance",
    "fin37": "received_govt_transfer",
    "merchantpay_dig": "made_digital_merchant_payment",
}

# --- code mappings -------------------------------------------------------

YES_NO = {1: 1, 2: 0}  # 3 = don't know, 4 = refused -> missing
ZERO_ONE = {1: 1, 0: 0}  # constructed indicators are already 0/1
FEMALE = {1: 1, 2: 0}
WORKFORCE = {1: 1, 2: 0}
EDUCATION = {1: "primary_or_less", 2: "secondary", 3: "tertiary"}
INCOME_QUINTILE = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}  # 1 = poorest 20%, 5 = richest 20%
EMERGENCY_SOURCE = {
    1: "savings",
    2: "family_friends",
    3: "work",
    4: "borrowing",
    5: "selling_assets",
    6: "other",
    7: "could_not",
}
WORRY = {1: "very", 2: "somewhat", 3: "not_at_all"}
PAYMENT_CHANNEL = {1: "account", 2: "cash_only", 3: "other", 4: "none"}

# Readable column name -> mapping to apply
CODE_MAPS = {
    "is_female": FEMALE,
    "education_level": EDUCATION,
    "income_quintile": INCOME_QUINTILE,
    "in_workforce": WORKFORCE,
    "has_bank_account": ZERO_ONE,
    "has_mobile_money": ZERO_ONE,
    "owns_mobile_phone": YES_NO,
    "has_internet_access": YES_NO,
    "saved_past_year": ZERO_ONE,
    "saved_in_savings_club": YES_NO,
    "borrowed_past_year": ZERO_ONE,
    "borrowed_from_fin_institution": YES_NO,
    "borrowed_from_family_friends": YES_NO,
    "borrowed_for_medical": YES_NO,
    "emergency_fund_source": EMERGENCY_SOURCE,
    "worried_about_bills": WORRY,
    "wage_payment_channel": PAYMENT_CHANNEL,
    "agri_payment_channel": PAYMENT_CHANNEL,
    "utility_payment_channel": PAYMENT_CHANNEL,
    "sent_domestic_remittance": YES_NO,
    "received_domestic_remittance": YES_NO,
    "received_govt_transfer": YES_NO,
    "made_digital_merchant_payment": ZERO_ONE,
}


def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns in COLUMN_MAP and give them readable names."""
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise KeyError(f"Input is missing expected Findex columns: {sorted(missing)}")
    return df[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)


def decode(df: pd.DataFrame) -> pd.DataFrame:
    """Replace numeric answer codes with values; unlisted codes become NaN."""
    out = df.copy()
    for col, mapping in CODE_MAPS.items():
        out[col] = out[col].map(mapping)
        if all(isinstance(v, int) for v in mapping.values()):
            out[col] = out[col].astype("Int8")  # nullable integer
        else:
            out[col] = out[col].astype("category")
    return out


def keep_adults(df: pd.DataFrame, min_age: int = config.MIN_AGE) -> pd.DataFrame:
    """Drop respondents below min_age and those with no recorded age."""
    return df[df["age"] >= min_age].reset_index(drop=True)


def clean_findex(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning step: select + rename, decode codes, keep adults."""
    out = select_and_rename(df)
    out = decode(out)
    out = keep_adults(out)
    out["age"] = out["age"].astype("int16")
    return out
