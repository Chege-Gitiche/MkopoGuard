// The login page (step 6.4).
//
// Signs in through api.login(): the server sets the HttpOnly session cookie, and the officer
// goes on to the page they were sent from (?next=), or New assessment. Errors are shown on the
// page itself: wrong details, too many attempts, server unreachable.

import { api, safeNext } from "./api.js";

const REASONS = {
  expired: "Sign in to continue. Sessions end 30 minutes after signing in.",
  "signed-out": "You have logged out.",
};

const params = new URLSearchParams(window.location.search);
const next = safeNext(params.get("next"));
const form = document.getElementById("login-form");
const email = document.getElementById("email");
const password = document.getElementById("password");
const button = form.querySelector('button[type="submit"]');
const errorBox = document.getElementById("login-error");
const errorText = document.getElementById("login-error-text");
const statusBox = document.getElementById("login-status");

function showError(message) {
  errorText.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
  errorText.textContent = "";
}

function setBusy(busy) {
  button.disabled = busy;
  button.textContent = busy ? "Signing in\u2026" : "Sign in";
  form.setAttribute("aria-busy", String(busy));
}

function messageFor(error) {
  if (error.status === 422) return "Enter your email and password.";
  return error.message; // 401, 429 and network errors already read well
}

const reason = REASONS[params.get("reason")];
if (reason) {
  statusBox.textContent = reason;
  statusBox.hidden = false;
}

// Already signed in (a second tab, or the back button)? Go straight on.
api
  .currentUser()
  .then((user) => {
    if (user) window.location.replace(next);
  })
  .catch(() => {}); // server unreachable: the form will say so if they try

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideError();
  if (!email.value.trim() || !password.value) {
    showError("Enter your email and password.");
    (email.value.trim() ? password : email).focus();
    return;
  }
  setBusy(true);
  try {
    await api.login(email.value.trim(), password.value);
    window.location.replace(next);
  } catch (error) {
    password.value = "";
    setBusy(false);
    showError(messageFor(error));
    password.focus();
  }
});
