// The New assessment page (step 6.5).
//
// - "Load sample" fills the form with one of the ten synthetic applicants and attaches their
//   M-Pesa statement, so the result can be compared with the score the API recorded.
// - The form is checked here first, so an officer sees every mistake at once, next to the
//   field. The server checks everything again and is the real judge: anything it rejects is
//   shown the same way.
// - While scoring, the button says "Assessing..." and the page can't be submitted twice.
// - On success the result goes to the Result page.

import { api } from "./api.js";
import { session } from "./session.js";
import { saveResult } from "./store.js";

// The same rules as the API (src/mkopoguard/api/schemas.py); a test keeps them equal.
export const REF_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_/-]{2,63}$/;
export const LIMITS = {
  age: [18, 100],
  loan_amount_kes: [1000, 100000],
  income_quintile: [1, 5],
  statement_bytes: 1048576,
};

const REQUIRED_YES_NO = {
  in_workforce: "In the workforce",
  saved_past_year: "Saved any money in the past year",
  borrowed_past_year: "Borrowed any money in the past year",
};
const OPTIONAL_YES_NO = [
  "saved_in_savings_club",
  "borrowed_from_fin_institution",
  "borrowed_from_family_friends",
  "borrowed_for_medical",
];
const OPTIONAL_CHOICES = ["education_level", "emergency_fund_source", "worried_about_bills"];
// A follow-up can't be "Yes" when the question it follows was answered "No"
const FOLLOW_UPS = {
  saved_past_year: ["saved_in_savings_club"],
  borrowed_past_year: [
    "borrowed_from_fin_institution",
    "borrowed_from_family_friends",
    "borrowed_for_medical",
  ],
};

const form = document.getElementById("assess-form");
const sampleForm = document.getElementById("sample-form");
const sampleSelect = document.getElementById("sample");
const assessButton = document.getElementById("assess-button");
const statusLine = document.getElementById("assess-status");
const summary = document.getElementById("form-errors");
const summaryList = document.getElementById("form-error-list");
const statementInput = document.getElementById("statement");

// --- reading and filling the form ---------------------------------------------------------

function checked(name) {
  const input = form.querySelector(`input[name="${name}"]:checked`);
  return input ? input.value : null;
}

function setRadio(name, value) {
  const wanted = value === null || value === undefined ? "" : String(value);
  for (const input of form.querySelectorAll(`input[name="${name}"]`)) {
    input.checked = input.value === wanted;
  }
}

/** The profile exactly as the API expects it: real numbers, true/false, null for "Not asked". */
function readProfile() {
  const value = (name) => form.elements[name].value.trim();
  const yesNo = (raw) => (raw === "true" ? true : raw === "false" ? false : null);
  const profile = {
    applicant_ref: value("applicant_ref"),
    application_date: value("application_date"),
    age: Number(value("age")),
    income_quintile: Number(value("income_quintile")),
    in_workforce: yesNo(checked("in_workforce")),
    education_level: value("education_level") || null,
    loan_amount_kes: Number(value("loan_amount_kes")),
    term_months: Number(checked("term_months")),
    saved_past_year: yesNo(checked("saved_past_year")),
    borrowed_past_year: yesNo(checked("borrowed_past_year")),
    emergency_fund_source: value("emergency_fund_source") || null,
    worried_about_bills: value("worried_about_bills") || null,
  };
  for (const name of OPTIONAL_YES_NO) profile[name] = yesNo(checked(name));
  return profile;
}

function fillForm(profile) {
  form.elements.applicant_ref.value = profile.applicant_ref;
  form.elements.application_date.value = profile.application_date;
  form.elements.age.value = profile.age;
  form.elements.income_quintile.value = String(profile.income_quintile);
  form.elements.loan_amount_kes.value = profile.loan_amount_kes;
  setRadio("term_months", profile.term_months);
  for (const name of [...Object.keys(REQUIRED_YES_NO), ...OPTIONAL_YES_NO]) {
    setRadio(name, profile[name]);
  }
  for (const name of OPTIONAL_CHOICES) form.elements[name].value = profile[name] ?? "";
  syncFollowUps();
}

function attachStatement(file) {
  const files = new DataTransfer();
  if (file) files.items.add(file);
  statementInput.files = files.files;
}

function syncFollowUps() {
  for (const [parent, children] of Object.entries(FOLLOW_UPS)) {
    const answeredNo = checked(parent) === "false";
    for (const child of children) {
      const yes = form.querySelector(`input[name="${child}"][value="true"]`);
      yes.disabled = answeredNo;
      if (answeredNo && yes.checked) setRadio(child, false);
    }
  }
}

// --- checking the form ------------------------------------------------------------------------

function todayHere() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
}

function wholeNumberIn(raw, [low, high]) {
  const n = Number(raw);
  return raw !== "" && Number.isInteger(n) && n >= low && n <= high;
}

/** Every problem the browser can see: [{ field, message }], in the order of the form. */
function check(profile) {
  const problems = [];
  const add = (field, message) => problems.push({ field, message });
  const raw = (name) => form.elements[name].value.trim();

  if (!profile.applicant_ref) add("applicant_ref", "Enter the applicant reference.");
  else if (!REF_PATTERN.test(profile.applicant_ref) || !/[A-Za-z]/.test(profile.applicant_ref)) {
    add("applicant_ref", "Use 3 to 64 letters, digits, - _ or /, with at least one letter.");
  }
  if (!profile.application_date) add("application_date", "Enter the date the applicant applied.");
  else if (profile.application_date > todayHere()) {
    add("application_date", "The application date can't be in the future.");
  }
  if (!wholeNumberIn(raw("loan_amount_kes"), LIMITS.loan_amount_kes)) {
    add("loan_amount_kes", "Enter a whole amount from KSh 1,000 to KSh 100,000.");
  }
  if (!checked("term_months")) add("term_months", "Choose a loan term.");
  if (!wholeNumberIn(raw("age"), LIMITS.age)) add("age", "Enter an age from 18 to 100.");
  if (!raw("income_quintile")) add("income_quintile", "Choose an income group.");
  for (const [name, label] of Object.entries(REQUIRED_YES_NO)) {
    if (profile[name] === null) add(name, `Choose Yes or No for "${label}".`);
  }
  const file = statementInput.files[0];
  if (file && !/\.csv$/i.test(file.name)) add("statement", "Choose a CSV file.");
  else if (file && file.size > LIMITS.statement_bytes) {
    add("statement", "The statement is larger than 1 MB.");
  }
  return problems;
}

// --- showing problems -----------------------------------------------------------------------

/** Where a field's error goes, and what gets focus when the officer jumps to it. */
function fieldParts(field) {
  const control = form.elements[field];
  if (!control) return null;
  if (control instanceof RadioNodeList || control.type === "radio") {
    const first = form.querySelector(`input[name="${field}"]`);
    return { box: first.closest("fieldset"), focus: first, described: first.closest("fieldset") };
  }
  return { box: control.closest(".field"), focus: control, described: control };
}

function clearProblem(field) {
  const parts = fieldParts(field);
  const error = document.getElementById(`${field}-error`);
  if (error) error.remove();
  if (!parts) return;
  parts.focus.removeAttribute("aria-invalid");
  const ids = (parts.described.getAttribute("aria-describedby") || "")
    .split(" ")
    .filter((id) => id && id !== `${field}-error`);
  if (ids.length) parts.described.setAttribute("aria-describedby", ids.join(" "));
  else parts.described.removeAttribute("aria-describedby");
}

function clearProblems() {
  for (const error of form.querySelectorAll(".field-error")) {
    clearProblem(error.id.replace(/-error$/, ""));
  }
  summaryList.replaceChildren();
  summary.hidden = true;
}

function showProblems(problems, heading = "Fix these before assessing") {
  clearProblems();
  summary.querySelector("h2").textContent = heading;
  const seen = new Set();
  for (const { field, message } of problems) {
    const parts = fieldParts(field);
    const item = document.createElement("li");
    if (parts) {
      const link = document.createElement("a");
      link.href = `#${parts.focus.id || field}`;
      link.textContent = message;
      link.addEventListener("click", (event) => {
        event.preventDefault();
        parts.focus.focus();
      });
      item.append(link);
      if (!seen.has(field)) {
        seen.add(field);
        const error = document.createElement("p");
        error.className = "field-error";
        error.id = `${field}-error`;
        error.textContent = message;
        parts.box.append(error);
        if (parts.focus.type !== "radio") parts.focus.setAttribute("aria-invalid", "true");
        const ids = (parts.described.getAttribute("aria-describedby") || "").split(" ");
        parts.described.setAttribute("aria-describedby", [...ids, error.id].join(" ").trim());
      }
    } else {
      item.textContent = message;
    }
    summaryList.append(item);
  }
  summary.hidden = false;
  summary.focus();
}

/** Server problems use the API's field names; statement problems say where in the file. */
function fromServer(fields) {
  return fields.map(({ field, message }) => {
    const text = message.replace(/^Value error, /, "");
    if (field.startsWith("statement")) {
      const where = field.slice("statement.".length);
      const place = where && where !== "file" ? `Statement ${where}: ` : "";
      return { field: "statement", message: `${place}${text}` };
    }
    return { field: form.elements[field] ? field : "", message: text };
  });
}

// --- busy states ----------------------------------------------------------------------------

function setBusy(busy, message = "") {
  assessButton.disabled = busy;
  sampleForm.querySelector("button").disabled = busy;
  assessButton.textContent = busy ? "Assessing\u2026" : "Assess risk";
  form.setAttribute("aria-busy", String(busy));
  statusLine.textContent = message;
}

// --- events -----------------------------------------------------------------------------------

let manifest = null;

sampleForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = sampleForm.querySelector("button");
  button.disabled = true;
  statusLine.textContent = "Loading the sample\u2026";
  try {
    manifest = manifest || (await api.samples());
    const entry = manifest.find((sample) => sample.name === sampleSelect.value);
    const [profile, statement] = await Promise.all([
      api.sampleProfile(entry),
      api.sampleStatement(entry),
    ]);
    clearProblems();
    fillForm(profile);
    attachStatement(statement);
    const label = sampleSelect.selectedOptions[0].textContent;
    statusLine.textContent = statement
      ? `Loaded ${label} with its statement. Check the answers, then select Assess risk.`
      : `Loaded ${label}, who has no statement. Check the answers, then select Assess risk.`;
  } catch (error) {
    statusLine.textContent = `Couldn't load the sample. ${error.message}`;
  } finally {
    button.disabled = false;
  }
});

form.addEventListener("change", (event) => {
  if (event.target.name) clearProblem(event.target.name);
  if (event.target.name in FOLLOW_UPS) syncFollowUps();
});

form.addEventListener("reset", () => {
  setTimeout(() => {
    clearProblems();
    syncFollowUps();
    statusLine.textContent = "Form cleared.";
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const profile = readProfile();
  const problems = check(profile);
  if (problems.length) {
    showProblems(problems);
    return;
  }
  clearProblems();
  const user = await session;
  if (!user) return; // on the way to the login page
  setBusy(true, "Scoring the applicant\u2026");
  try {
    const prediction = await api.predict(profile, statementInput.files[0] || null);
    saveResult(user.id, prediction, profile);
    const saved = prediction.id !== null && prediction.id !== undefined;
    window.location.assign(saved ? `/result.html?id=${prediction.id}` : "/result.html");
  } catch (error) {
    setBusy(false);
    if (error.status === 422 && error.fields.length) {
      showProblems(fromServer(error.fields));
    } else {
      const reference = error.reference ? ` Reference: ${error.reference}.` : "";
      showProblems([{ field: "", message: `${error.message}${reference}` }], "Couldn't assess");
    }
  }
});

syncFollowUps();
