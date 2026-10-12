"""Step 3.7: the model's input columns, grouped the way an applicant experiences them.

Each group is something we either ask the applicant (a block of questions on the form),
take from their uploaded M-Pesa statement, or take from the loan they are applying for.
Ablation removes whole groups, because dropping one question from a block saves little;
dropping a whole block shortens the application form.
"""

from mkopoguard.features.transactions import FEATURES as STATEMENT_FEATURES

FEATURE_GROUPS = {
    # The loan being applied for
    "loan": ["loan_amount_kes", "term_months"],
    # Survey questions: who the applicant is
    "profile": ["age", "income_quintile", "in_workforce", "education_level"],
    # Survey questions: which financial services they have and use
    "financial_access": [
        "has_bank_account",
        "has_mobile_money",
        "owns_mobile_phone",
        "has_internet_access",
        "made_digital_merchant_payment",
    ],
    # Survey questions: how they are paid and pay bills
    "income_channels": ["wage_payment_channel", "agri_payment_channel", "utility_payment_channel"],
    # Survey questions: saving habits and financial worry
    "savings_resilience": [
        "saved_past_year",
        "saved_in_savings_club",
        "emergency_fund_source",
        "worried_about_bills",
    ],
    # Survey questions: past borrowing
    "borrowing_history": [
        "borrowed_past_year",
        "borrowed_from_fin_institution",
        "borrowed_from_family_friends",
        "borrowed_for_medical",
    ],
    # Survey questions: money sent and received
    "transfers": [
        "sent_domestic_remittance",
        "received_domestic_remittance",
        "received_govt_transfer",
    ],
    # The uploaded statement: whether there is one, and the 17 features built from it
    "statement": ["has_statement", *STATEMENT_FEATURES],
}

# Groups that come from the application form (the statement and loan come from elsewhere)
SURVEY_GROUPS = [
    "profile",
    "financial_access",
    "income_channels",
    "savings_resilience",
    "borrowing_history",
    "transfers",
]


def columns_in(groups) -> list[str]:
    """Every column in the named groups, in group order."""
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise KeyError(f"Unknown feature groups: {sorted(unknown)}")
    return [col for group in groups for col in FEATURE_GROUPS[group]]
