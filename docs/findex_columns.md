# Findex columns kept for MkopoGuard

**Source:** Global Findex 2021 microdata codebook, World Bank Microdata Library (WLD_2021_FINDEX_v03_M):
https://microdata.worldbank.org/catalog/4607/data-dictionary/F1?file_name=micro_world_139countries.dta

**Coding conventions**
- Yes/no questions (`fin...`): 1 = yes, 2 = no, 3 = don't know, 4 = refused. 3 and 4 become missing.
- Derived indicators (`account_*`, `saved`, `borrowed`, `merchantpay_dig`): 1 = yes, 0 = no.
- "Don't know", "refused" and "does not apply" always become missing.

**Verified** = codes checked on the codebook page. ☐ = still to check before step 1.3.

## Identifiers and metadata (not model features)

| Original | New name | Meaning |
| --- | --- | --- |
| economycode | country_code | ISO3 country code |
| economy | country | Country name |
| wpid_random | respondent_id | Unique respondent ID |
| wgt | survey_weight | Survey weight, valid within one country only |
| year | survey_year | 2021, or 2022 for Eswatini, Lesotho, Botswana |

## Fairness audit only (never a model feature)

| Original | New name | Codes | Verified |
| --- | --- | --- | --- |
| female | is_female | 1 = female, 2 = male | ✔ |

## Model features (23)

| Original | New name | Question | Codes | Verified |
| --- | --- | --- | --- | --- |
| age | age | Respondent age | Years (15+) | ✔ |
| educ | education_level | Education level | 1 primary or less, 2 secondary, 3 tertiary; 4 dk, 5 ref | ✔ |
| inc_q | income_quintile | Household income quintile within own country | 1 poorest 20% … 5 richest 20% | ✔ |
| emp_in | in_workforce | Respondent is in the workforce | 1 in workforce, 2 out of workforce | ✔ |
| account_fin | has_bank_account | Has an account at a financial institution | 1/0 | ✔ |
| account_mob | has_mobile_money | Has a mobile money account | 1/0 | ✔ |
| mobileowner | owns_mobile_phone | Owns a mobile phone | 1 yes, 2 no, 3 dk, 4 ref | ✔ |
| internetaccess | has_internet_access | Has internet access | 1 yes, 2 no, 3 dk, 4 ref | ✔ |
| saved | saved_past_year | Saved in the past year | 1/0 | ✔ |
| fin17b | saved_in_savings_club | Saved using an informal savings club (e.g. chama) | yes/no | ✔ |
| borrowed | borrowed_past_year | Borrowed in the past year | 1/0 | ✔ |
| fin22a | borrowed_from_fin_institution | Borrowed from a financial institution | yes/no | ✔ |
| fin22b | borrowed_from_family_friends | Borrowed from family or friends | yes/no | ✔ |
| fin20 | borrowed_for_medical | Borrowed for medical purposes | yes/no | ✔ |
| fin24 | emergency_fund_source | Main source of emergency funds within 30 days | 1 savings, 2 family/friends, 3 work, 4 borrowing, 5 selling assets, 6 other, 7 could not come up with the money; 8 dk, 9 ref | ✔ |
| fin44c | worried_about_bills | Financially worried: paying bills | 1 very, 2 somewhat, 3 not at all; 4 does not apply, 5 dk, 6 ref | ✔ |
| receive_wages | wage_payment_channel | Received a wage payment, and how | 1 into account, 2 cash only, 3 other methods, 4 did not receive; 5 dk/ref | ✔ |
| receive_agriculture | agri_payment_channel | Received payment for selling agricultural goods | Expected to match receive_wages | ✔ |
| pay_utilities | utility_payment_channel | Paid a utility bill, and how | Check on codebook page | ✔ |
| fin26 | sent_domestic_remittance | Sent money to someone in the country | yes/no | ✔ |
| fin28 | received_domestic_remittance | Received money from someone in the country | yes/no | ✔ |
| fin37 | received_govt_transfer | Received a government transfer | yes/no | ✔ |
| merchantpay_dig | made_digital_merchant_payment | Made a digital payment to a merchant | 1/0 | ✔ |

## Considered and left out

- **urbanicity_f2f** (rural/urban): empty in some countries.
- **fin24a, fin24b** (how hard it would be to find emergency money): missing for up to 48% of respondents in some countries. fin24 covers the same idea with no gaps.
- **fin13c** (used mobile money to borrow) and similar follow-ups: only asked of people who already said yes to an earlier question.
- **fin44a, fin44b, fin44d** (worries about old age, medical costs, school fees): fin44c (bills) is the one most tied to repayment.
- **fin14_2, fin14c_2, fin31b1** (first digital payment after COVID-19): not related to loan risk.
- **remittances** (combined indicator): covered more clearly by fin26 and fin28.
