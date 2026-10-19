// The Model info page (step 6.6): the numbers come from the model actually being served, so the
// page can never disagree with it after the model is updated.

import { api } from "./api.js";
import { BAND_LABELS, bandClass, termText } from "./render.js";
import { session } from "./session.js";

const pct = (fraction) => `${Math.round(fraction * 1000) / 10}%`;
const packagedDay = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Africa/Nairobi",
  day: "numeric",
  month: "short",
  year: "numeric",
});

function fact(term, text) {
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = term;
  dd.textContent = text;
  return [dt, dd];
}

function tableRow(...cells) {
  const tr = document.createElement("tr");
  for (const content of cells) {
    const td = document.createElement("td");
    if (content instanceof Node) td.append(content);
    else td.textContent = content;
    tr.append(td);
  }
  return tr;
}

function bandRange([low, high], last) {
  if (low === 0) return `Under ${pct(high)}`;
  if (last) return `${pct(low)} and over`;
  return `${pct(low)} to under ${pct(high)}`;
}

function render(info) {
  document.getElementById("model-facts").replaceChildren(
    ...fact("Name", info.name),
    ...fact("Version", info.version),
    ...fact("Packaged", packagedDay.format(new Date(info.packaged_at))),
    ...fact("Trained on", `${info.trained_on_applicants.toLocaleString("en-KE")} synthetic applicants`),
    ...fact(
      "Test PR-AUC",
      `${info.test_pr_auc.toFixed(3)} on a locked test set it never trained on. PR-AUC measures how well the model ranks defaulters above repayers; 1.0 is perfect.`,
    ),
  );

  const terms = Object.entries(info.thresholds_by_term_months).sort((a, b) => a[0] - b[0]);
  document.getElementById("threshold-rows").replaceChildren(
    ...terms.map(([months, threshold]) => {
      const tr = tableRow(termText(Number(months)), pct(threshold));
      tr.lastChild.className = "number";
      return tr;
    }),
  );

  const bands = Object.entries(info.bands);
  document.getElementById("band-rows").replaceChildren(
    ...bands.map(([name, range], i) => {
      const tag = document.createElement("span");
      tag.className = `band band-${bandClass(name)}`;
      tag.textContent = BAND_LABELS[name];
      return tableRow(tag, bandRange(range, i === bands.length - 1));
    }),
  );

  const c = info.costs;
  document.getElementById("model-costs").textContent =
    `Assumes a fee of ${pct(c.monthly_fee_rate)} a month, a funding cost of ${pct(c.monthly_funding_cost)} a month, and that ${pct(c.loss_given_default)} of a defaulted loan is lost.`;
  document.getElementById("model-notice").textContent = info.notice;
}

const user = await session;
if (user) {
  try {
    render(await api.modelInfo());
  } catch (error) {
    document.getElementById("model-notice").textContent =
      `Couldn't load the latest model details, so the figures below may be out of date. ${error.message}`;
  }
}
