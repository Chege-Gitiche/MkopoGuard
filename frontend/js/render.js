// Showing a prediction on the page (step 6.5): the Result page now, the saved-record page in 6.6.
//
// Everything goes in as text (textContent), never as HTML, so nothing an officer typed can
// turn into markup. The band is always written out in words as well as shown in colour.

import { placeMarkers } from "./ui.js";

export const BAND_LABELS = {
  low: "Low",
  medium: "Medium",
  high: "High",
  very_high: "Very high",
};
const BAND_RANGES = {
  low: "under 10%",
  medium: "10% to under 20%",
  high: "20% to under 35%",
  very_high: "35% and over",
};
const DECISION_LABELS = { approve: "Approve", decline: "Decline" };
const STRENGTH_LABELS = { major: "Major", moderate: "Moderate", minor: "Minor" };
const MINUS = "\u2212";

const kenyanTime = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Africa/Nairobi",
  day: "numeric",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});
const kenyanDay = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Africa/Nairobi",
  day: "numeric",
  month: "short",
  year: "numeric",
});

// --- formatting -------------------------------------------------------------------------------

export function percent(probability) {
  return `${(probability * 100).toFixed(1)}%`;
}

export function bandClass(band) {
  return band.replace("_", "-"); // very_high -> very-high, the CSS class name
}

export function shillings(amount) {
  return `KSh ${Number(amount).toLocaleString("en-KE")}`;
}

export function termText(months) {
  return months === 1 ? "1 month" : `${months} months`;
}

/** "16 Oct 2026, 10:05" in Kenyan time. */
export function when(iso) {
  return kenyanTime.format(new Date(iso)).replace(" at ", ", ");
}

/** "5 May 2025" for a plain date such as the application date. */
export function day(isoDate) {
  return kenyanDay.format(new Date(`${isoDate}T12:00:00+03:00`));
}

function points(value, direction) {
  const size = Math.abs(value).toFixed(1);
  return direction === "lowers" ? `${MINUS}${size} points` : `+${size} points`;
}

function setText(id, text) {
  const element = document.getElementById(id);
  if (element) element.textContent = text;
  return element;
}

function make(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

// --- the verdict: probability, band, decision, scale ------------------------------------------

export function renderVerdict(prediction) {
  const { probability_of_default: p, band, decision, threshold, term_months: term } = prediction;
  const value = setText("probability", percent(p));
  value.className = `probability-value tone-${bandClass(band)}`;

  const tag = setText("band", BAND_LABELS[band]);
  tag.className = `band band-${bandClass(band)}`;
  setText("decision", DECISION_LABELS[decision]);

  let rule;
  if (threshold === null || threshold === undefined) {
    rule = "The decision threshold was not recorded for this assessment.";
  } else if (term) {
    rule = `Declines at ${percent(threshold)} or more for a ${term}-month loan`;
  } else {
    rule = `Declines at ${percent(threshold)} or more`;
  }
  setText("rule", rule);

  const marker = document.getElementById("scale-marker");
  marker.dataset.at = percent(Math.min(p, 1));
  marker.querySelector(".scale-marker-label").textContent = percent(p);
  const line = document.getElementById("scale-threshold");
  line.hidden = threshold === null || threshold === undefined;
  if (!line.hidden) line.dataset.at = percent(threshold);
  placeMarkers();

  let caption = `${percent(p)} sits in the ${BAND_LABELS[band].toLowerCase()} band (${BAND_RANGES[band]}).`;
  if (!line.hidden) caption += ` The dashed line is the decline threshold, ${percent(threshold)}.`;
  setText("scale-caption", caption);
}

// --- reasons ------------------------------------------------------------------------------------

function reasonItem(reason) {
  const item = make("li");
  item.append(
    make("span", "", reason.text),
    make("span", "reason-impact", points(reason.impact_points, reason.direction)),
    make("span", "reason-strength", STRENGTH_LABELS[reason.strength] || reason.strength),
  );
  return item;
}

export function renderReasons(reasons) {
  document.getElementById("reasons-raising").replaceChildren(...reasons.raising.map(reasonItem));
  document.getElementById("reasons-lowering").replaceChildren(...reasons.lowering.map(reasonItem));
  const other = reasons.other_factors;
  const size = Math.abs(other.impact_points).toFixed(1);
  let text = `Everything else together ${other.impact_points < 0 ? "took away" : "added"} ${size} points`;
  text += other.material ? ", as much as one of the reasons listed." : ".";
  setText("other-factors", text);
}

// --- warnings ---------------------------------------------------------------------------------

export function renderWarnings(warnings = []) {
  const box = document.getElementById("warnings");
  if (!box) return;
  document.getElementById("warning-list").replaceChildren(...warnings.map((w) => make("li", "", w)));
  box.hidden = warnings.length === 0;
}
