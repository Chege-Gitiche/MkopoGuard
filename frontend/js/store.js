// The last result, kept for this browser tab only (step 6.5).
//
// The assessment page hands its result to the Result page through sessionStorage: needed for
// the demo account, whose results are never saved on the server, and it keeps the input
// warnings, which the server does not store. It is tied to the signed-in user, lives only as
// long as the tab, and is wiped on log out.

const KEY = "mkopoguard.lastResult";

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
  } catch {
    // nothing to forget
  }
}
