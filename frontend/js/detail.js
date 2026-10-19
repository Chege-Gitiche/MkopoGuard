// The saved-assessment page (step 6.6): detail.html?id=123.
//
// Everything the officer's assessment was based on, read back from the server: the score and
// decision, the reasons, every answer, and how much of a statement there was. Someone else's
// assessment, or a missing one, gets the same "not in your history" answer.

import { api } from "./api.js";
import { day, renderReasons, renderVerdict, shillings, termText, when } from "./render.js";
import { session } from "./session.js";
import { lastHistory } from "./store.js";

const YES_NO = (value) => (value ? "Yes" : "No");
const CHOICES = {
  education_level: { primary_or_less: "Primary or less", secondary: "Secondary", tertiary: "Tertiary" },
  emergency_fund_source: {
    savings: "Savings",
    work: "Work",
    family_friends: "Family or friends",
    borrowing: "Borrowing",
    selling_assets: "Selling assets",
    other: "Somewhere else",
    could_not: "Could not raise it",
  },
  worried_about_bills: { very: "Very worried", somewhat: "Somewhat worried", not_at_all: "Not worried" },
};
const INCOME = { 1: "1, poorest fifth", 5: "5, richest fifth" };

// Every answer on the form, in the form's order (a test checks none is missing).
export const ANSWERS = [
  ["application_date", "Application date", day],
  ["loan_amount_kes", "Loan amount", shillings],
  ["term_months", "Loan term", termText],
  ["age", "Age", String],
  ["income_quintile", "Income group", (v) => INCOME[v] || String(v)],
  ["in_workforce", "In the workforce", YES_NO],
  ["education_level", "Education", (v) => CHOICES.education_level[v]],
  ["saved_past_year", "Saved any money in the past year", YES_NO],
  ["saved_in_savings_club", "Saved in a chama or savings group", YES_NO],
  ["borrowed_past_year", "Borrowed any money in the past year", YES_NO],
  ["borrowed_from_fin_institution", "From a bank or other financial institution", YES_NO],
  ["borrowed_from_family_friends", "From family or friends", YES_NO],
  ["borrowed_for_medical", "For medical bills", YES_NO],
  ["emergency_fund_source", "In an emergency, money would come from", (v) => CHOICES.emergency_fund_source[v]],
  ["worried_about_bills", "Worried about paying bills", (v) => CHOICES.worried_about_bills[v]],
];

const status = document.getElementById("detail-status");
const content = document.getElementById("detail");

function showMessage(text) {
  status.textContent = text;
  status.hidden = false;
  content.hidden = true;
}

function answerItem([name, label, format], profile) {
  const value = profile[name];
  const wrapper = document.createElement("div");
  const term = document.createElement("dt");
  const detail = document.createElement("dd");
  term.textContent = label;
  detail.textContent = value === null || value === undefined ? "Not asked" : format(value);
  wrapper.append(term, detail);
  return wrapper;
}

function render(record, user) {
  document.title = `Assessment #${record.id} | MkopoGuard`;
  document.getElementById("detail-title").textContent = `Assessment #${record.id}`;
  const loan = record.profile.loan_amount_kes ? `, ${shillings(record.profile.loan_amount_kes)}` : "";
  const term = record.term_months ? ` over ${termText(record.term_months)}` : "";
  document.getElementById("detail-lede").textContent =
    `Applicant ${record.applicant_ref}${loan}${term}. Assessed ${when(record.created_at)} by ${user.full_name}.`;

  renderVerdict(record);
  renderReasons(record.reasons);
  document.getElementById("answers").replaceChildren(...ANSWERS.map((a) => answerItem(a, record.profile)));

  const n = record.statement_transactions;
  document.getElementById("statement-summary").textContent = record.has_statement
    ? `Included: ${n.toLocaleString("en-KE")} transactions were read from it. The transactions themselves are not kept.`
    : "None: this applicant was scored on their answers only.";
  document.getElementById("detail-model").textContent =
    `Scored by model version ${record.model_version}. Later versions may score the same answers differently; this record keeps the score as it was given.`;

  status.hidden = true;
  content.hidden = false;
}

// "Back to history" returns to the filters and page the officer was last looking at.
document.querySelector(".back-link").href = lastHistory();

const user = await session;
if (user) {
  const id = new URLSearchParams(window.location.search).get("id");
  if (!id || !/^\d+$/.test(id)) {
    showMessage("No assessment was chosen. Pick one from your history.");
  } else {
    try {
      render(await api.prediction(id), user);
    } catch (error) {
      showMessage(error.status === 404 ? "This assessment is not in your history." : error.message);
    }
  }
}
