// The last result, kept for this browser tab only (step 6.5).
//
// The assessment page hands its result to the Result page through sessionStorage: needed for
// the demo account, whose results are never saved on the server, and it keeps the input
// warnings, which the server does not store. It is tied to the signed-in user, lives only as
// long as the tab, and is wiped on log out.

const KEY = "mkopoguard.lastResult";
const HISTORY_KEY = "mkopoguard.historyAddress";

export function saveResult(userId, prediction, profile) {
  try {
    sessionStorage.setItem(KEY, JSON.stringify({ userId, prediction, profile }));
  } catch {
    // storage blocked or full: the Result page falls back to the server copy
  }
}

/** The kept result, but only if it belongs to this user. */
export function readResult(userId) {
  try {
    const kept = JSON.parse(sessionStorage.getItem(KEY));
    return kept && kept.userId === userId ? kept : null;
  } catch {
    return null;
  }
}

export function forgetResults() {
  try {
    sessionStorage.removeItem(KEY);
    sessionStorage.removeItem(HISTORY_KEY);
  } catch {
    // nothing to forget
  }
}

// The History page the officer last looked at (step 6.6), so "Back to history" on a saved
// assessment returns to the same filters and page. (The browser sends no Referer header here:
// the server's Referrer-Policy is no-referrer, so the page can't ask where it came from.)

export function rememberHistory(address) {
  try {
    sessionStorage.setItem(HISTORY_KEY, address);
  } catch {
    // the plain History link still works
  }
}

/** The last History address, only ever a History page on this site. */
export function lastHistory() {
  try {
    const address = sessionStorage.getItem(HISTORY_KEY) || "";
    return /^\/history\.html(\?[^#]*)?$/.test(address) ? address : "/history.html";
  } catch {
    return "/history.html";
  }
}
