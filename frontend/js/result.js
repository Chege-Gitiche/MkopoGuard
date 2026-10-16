// The Result page (step 6.5).
//
// Shows the assessment just made. result.html?id=123 is a saved assessment; with no id it is
// the demo account's unsaved one. The copy the assessment page kept is used when it matches
// (it also has the input warnings); otherwise the saved record is fetched from the server.

import { api } from "./api.js";
import { renderReasons, renderVerdict, renderWarnings, shillings, termText, when } from "./render.js";
import { session } from "./session.js";
import { readResult } from "./store.js";

const status = document.getElementById("result-status");
const content = document.getElementById("result");

function showMessage(text) {
  status.textContent = text;
  status.hidden = false;
  content.hidden = true;
}

async function load(user) {
  const id = new URLSearchParams(window.location.search).get("id");
  const kept = readResult(user.id);
  if (kept && String(kept.prediction.id) === String(id ?? null)) return kept;
  if (!id) return null;
  const record = await api.prediction(id);
  return { prediction: { ...record, warnings: [] }, profile: record.profile };
}

function render({ prediction, profile }) {
  const loan = profile.loan_amount_kes ? `, ${shillings(profile.loan_amount_kes)}` : "";
  const term = prediction.term_months ? ` over ${termText(prediction.term_months)}` : "";
  document.getElementById("result-lede").textContent =
    `Applicant ${prediction.applicant_ref}${loan}${term}`;

  renderVerdict(prediction);
  renderWarnings(prediction.warnings);
  renderReasons(prediction.reasons);

  const saved = prediction.id !== null && prediction.id !== undefined;
  document.getElementById("saved-as").textContent = saved
    ? `Assessment #${prediction.id}`
    : "Not saved: the demo account keeps nothing";
  document.getElementById("assessed-at").textContent = when(prediction.created_at);
  document.getElementById("has-statement").textContent = prediction.has_statement
    ? "Included"
    : "None: scored on the answers only";
  document.getElementById("model-version").textContent = `Version ${prediction.model_version}`;

  const link = document.getElementById("view-record");
  link.hidden = !saved;
  if (saved) link.href = `/detail.html?id=${encodeURIComponent(prediction.id)}`;

  status.hidden = true;
  content.hidden = false;
}

const user = await session;
if (user) {
  try {
    const result = await load(user);
    if (result) render(result);
    else showMessage("There is no result to show yet. Start a new assessment to see one here.");
  } catch (error) {
    showMessage(
      error.status === 404 ? "This assessment is not in your history." : error.message,
    );
  }
}
