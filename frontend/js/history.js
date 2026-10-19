// The History page (step 6.6).
//
// The filters are a plain GET form, so every filtered view and every page has its own address
// (it can be bookmarked, and Back works). This script reads the address, asks the API for that
// page of the officer's own assessments, and draws the table.

import { api } from "./api.js";
import { BAND_LABELS, bandClass, percent, when } from "./render.js";
import { session } from "./session.js";
import { rememberHistory } from "./store.js";

const PAGE_SIZE = 20;
const FILTERS = ["band", "decision", "ref", "date_from", "date_to"];
const DECISION_LABELS = { approve: "Approve", decline: "Decline" };

const form = document.getElementById("history-filters");
const table = document.getElementById("history-table");
const rows = document.getElementById("history-rows");
const empty = document.getElementById("history-empty");
const count = document.getElementById("history-count");
const filterError = document.getElementById("filter-error");

/** The filters and page in the address bar, ignoring anything else. */
function readAddress() {
  const params = new URLSearchParams(window.location.search);
  const filters = {};
  for (const name of FILTERS) {
    const value = (params.get(name) || "").trim();
    if (value) filters[name] = value;
  }
  const page = Number.parseInt(params.get("page") || "1", 10);
  return { filters, page: Number.isInteger(page) && page > 0 ? page : 1 };
}

function pageAddress(filters, page) {
  const params = new URLSearchParams({ ...filters, page: String(page) });
  return `/history.html?${params}`;
}

function cell(label, ...content) {
  const td = document.createElement("td");
  td.setAttribute("role", "cell");
  td.dataset.label = label;
  td.append(...content);
  return td;
}

function row(item) {
  const tr = document.createElement("tr");
  tr.setAttribute("role", "row");

  const band = document.createElement("span");
  band.className = `band band-${bandClass(item.band)}`;
  band.textContent = BAND_LABELS[item.band];

  const link = document.createElement("a");
  link.href = `/detail.html?id=${encodeURIComponent(item.id)}`;
  const hidden = document.createElement("span");
  hidden.className = "visually-hidden";
  hidden.textContent = ` ${item.applicant_ref}`;
  link.append("View", hidden);

  const probability = cell("Probability", percent(item.probability_of_default));
  probability.className = "number";
  tr.append(
    cell("Date", when(item.created_at)),
    cell("Reference", item.applicant_ref),
    probability,
    cell("Band", band),
    cell("Decision", DECISION_LABELS[item.decision]),
    cell("Record", link),
  );
  return tr;
}

function setPageLink(id, filters, page, usable) {
  const link = document.getElementById(id);
  if (usable) {
    link.href = pageAddress(filters, page);
    link.removeAttribute("aria-disabled");
  } else {
    link.removeAttribute("href"); // not a link any more: can't be focused or followed
    link.setAttribute("aria-disabled", "true");
  }
}

function emptyMessage(user, filtered) {
  if (user.role === "demo") return "The demo account doesn't save assessments, so its history is empty.";
  if (filtered) return "No assessments match these filters. Change or clear a filter.";
  return "You haven't made any assessments yet. Start one from New assessment.";
}

function showFilterError(message) {
  filterError.textContent = message;
  filterError.hidden = false;
}

// Catch the one mistake the browser can see before asking the server.
form.addEventListener("submit", (event) => {
  const from = form.elements.date_from.value;
  const to = form.elements.date_to.value;
  if (from && to && from > to) {
    event.preventDefault();
    showFilterError("The From date must be on or before the To date.");
    form.elements.date_from.focus();
  }
});

const user = await session;
if (user) {
  const { filters, page } = readAddress();
  rememberHistory(window.location.pathname + window.location.search);
  for (const [name, value] of Object.entries(filters)) form.elements[name].value = value;
  try {
    const result = await api.predictions({ ...filters, page, size: PAGE_SIZE });
    rows.replaceChildren(...result.items.map(row));
    const filtered = Object.keys(filters).length > 0;
    table.hidden = result.items.length === 0;
    empty.textContent = emptyMessage(user, filtered);
    empty.hidden = result.items.length > 0;
    const noun = result.total === 1 ? "assessment" : "assessments";
    count.textContent =
      result.total === 0
        ? "No assessments"
        : `Page ${result.page} of ${result.pages}, ${result.total.toLocaleString("en-KE")} ${noun}${filtered ? " matching" : ""}`;
    setPageLink("page-prev", filters, page - 1, page > 1);
    setPageLink("page-next", filters, page + 1, page < result.pages);
    if (page > result.pages && result.pages > 0) {
      window.location.replace(pageAddress(filters, result.pages)); // e.g. an old bookmark
    }
  } catch (error) {
    table.hidden = true;
    count.textContent = "";
    if (error.status === 422) {
      showFilterError(
        error.fields.some((f) => f.field === "date_from")
          ? "The From date must be on or before the To date."
          : "One of the filters isn't valid. Check them and try again.",
      );
    } else {
      showFilterError(error.message);
    }
  }
}
