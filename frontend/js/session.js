// Every page behind the login (step 6.4).
//
// Checks the session as the page opens: no valid session means api.js sends the officer to the
// login page and brings them back here afterwards. Shows who is signed in, shows the demo
// notice for the demo account, and makes "Log out" clear the session cookie.
//
// Pages that need the user (steps 6.5, 6.6) can wait for it:  const user = await session;

import { api } from "./api.js";
import { forgetResults } from "./store.js";

const ROLE_LABELS = { officer: "Officer", admin: "Admin", demo: "Demo" };

function showUser(user) {
  const name = document.getElementById("account-name");
  const role = document.getElementById("account-role");
  if (name) name.textContent = user.full_name;
  if (role) {
    role.textContent = ROLE_LABELS[user.role] || user.role;
    role.hidden = false;
  }
  const demo = document.getElementById("demo-notice");
  if (demo) demo.hidden = user.role !== "demo";
}

/** The signed-in user, or null (not signed in: the page is already on its way to login). */
export const session = api
  .me()
  .then((user) => {
    showUser(user);
    return user;
  })
  .catch(() => null);

const logoutForm = document.querySelector('form[action="/auth/logout"]');
if (logoutForm) {
  logoutForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = logoutForm.querySelector("button");
    button.disabled = true;
    button.textContent = "Logging out\u2026";
    try {
      await api.logout();
    } catch (error) {
      if (error.status === 0) {
        // The server can't be reached, so the cookie is still set: stay, and let them retry.
        button.disabled = false;
        button.textContent = "Log out (try again)";
        return;
      }
    }
    forgetResults(); // the last result shown in this tab goes with the session
    window.location.replace("/?reason=signed-out");
  });
}
