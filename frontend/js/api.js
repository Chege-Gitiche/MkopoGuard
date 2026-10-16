// MkopoGuard API client (step 6.3): every call the pages make to the server goes through here.
//
// - Same origin only. The login lives in an HttpOnly cookie that this code never sees; the
//   browser attaches it because of credentials: "same-origin".
// - Every request carries X-Requested-With: fetch. The server only sets the cookie, and only
//   accepts a cookie-logged-in change (POST ...), when that header is present. Another site
//   can't add it, which blocks cross-site request forgery.
// - 401 (not logged in, or the 30-minute session ran out) sends the officer to the login page,
//   remembering the page they were on so they come back to it.
// - Every other failure becomes an ApiError with a message an officer can act on, so the
//   pages show it as it is.

const TIMEOUT_MS = 30_000;
const LOGIN_PAGE = "/";
const HOME_PAGE = "/assess.html";

export const NETWORK_ERROR = "Can't reach the server. Check your connection and try again.";
export const TIMEOUT_ERROR = "The server took too long to answer. Try again.";
export const SERVER_ERROR = "Something went wrong on our side. Please try again.";

/** A failed request. status 0 means the request never got an answer. */
export class ApiError extends Error {
  constructor(status, message, { fields = [], retryAfter = null, reference = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.fields = fields; // [{ field: "age", message: "..." }] for 422
    this.retryAfter = retryAfter; // seconds, for 429
    this.reference = reference; // request id, for 500
  }
}

// --- where to go after login ---------------------------------------------------------------

/** A page on this site to return to, or the home page. Never another site (open redirect). */
export function safeNext(value) {
  if (typeof value !== "string" || !value.startsWith("/")) return HOME_PAGE;
  if (value.startsWith("//") || value.includes("\\")) return HOME_PAGE;
  if (value === LOGIN_PAGE || value.startsWith("/?") || value.startsWith("/index.html")) {
    return HOME_PAGE;
  }
  return value;
}

/** The login page address that brings the officer back here afterwards. */
export function loginUrl(reason = "expired", here = window.location) {
  const params = new URLSearchParams({ reason, next: here.pathname + here.search });
  return `${LOGIN_PAGE}?${params}`;
}

function goToLogin(reason) {
  window.location.assign(loginUrl(reason));
}

// --- turning a response into data or an ApiError --------------------------------------------

/** "age" for 422 locations like ["profile", "age"] or ["query", "page"]: the form field name. */
function fieldName(loc) {
  const parts = (Array.isArray(loc) ? loc : [loc]).map(String);
  const start = ["body", "query", "path", "form", "profile"].includes(parts[0]) ? 1 : 0;
  return parts.slice(start).join(".") || "request";
}

async function readBody(response) {
  const type = response.headers.get("content-type") || "";
  if (!type.includes("application/json")) return null;
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function errorFrom(response, body) {
  const detail = body && body.detail;
  if (response.status === 422 && Array.isArray(detail)) {
    const fields = detail.map((item) => ({ field: fieldName(item.loc), message: item.msg }));
    return new ApiError(422, "Some answers need fixing.", { fields });
  }
  if (response.status === 429) {
    const seconds = Number.parseInt(response.headers.get("Retry-After") || "", 10);
    return new ApiError(429, typeof detail === "string" ? detail : "Too many attempts.", {
      retryAfter: Number.isFinite(seconds) ? seconds : null,
    });
  }
  if (response.status >= 500) {
    const reference = (body && body.reference) || response.headers.get("X-Request-ID");
    const message = typeof detail === "string" ? detail : SERVER_ERROR;
    return new ApiError(response.status, message, { reference });
  }
  const message = typeof detail === "string" ? detail : `Request failed (${response.status}).`;
  return new ApiError(response.status, message);
}

// --- the one place fetch is called -------------------------------------------------------

/**
 * Send one request and return the parsed JSON body.
 * loginOn401: false for the calls where 401 is an answer, not an expired session (login).
 */
export async function request(path, { method = "GET", body, headers = {}, loginOn401 = true } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  let response;
  try {
    response = await fetch(path, {
      method,
      body,
      credentials: "same-origin",
      cache: "no-store",
      redirect: "error",
      signal: controller.signal,
      headers: { Accept: "application/json", "X-Requested-With": "fetch", ...headers },
    });
  } catch (error) {
    const timedOut = error && error.name === "AbortError";
    throw new ApiError(0, timedOut ? TIMEOUT_ERROR : NETWORK_ERROR);
  } finally {
    clearTimeout(timer);
  }
  const data = await readBody(response);
  if (response.status === 401 && loginOn401) {
    goToLogin("expired");
    throw new ApiError(401, "Your session has ended. Please sign in again.");
  }
  if (!response.ok) throw errorFrom(response, data);
  return data;
}

function query(params) {
  const kept = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "");
  const text = new URLSearchParams(kept.map(([k, v]) => [k, String(v)])).toString();
  return text ? `?${text}` : "";
}

// --- the API, one function per endpoint ----------------------------------------------------

export const api = {
  /** Log in. On success the server sets the session cookie; nothing is returned to keep. */
  login(email, password) {
    const body = new URLSearchParams({ username: email, password });
    return request("/auth/login", { method: "POST", body, loginOn401: false });
  },

  /** Log out: the server clears the session cookie. */
  logout() {
    return request("/auth/logout", { method: "POST", loginOn401: false });
  },

  /** The logged-in officer: { id, email, full_name, role, created_at }. 401 -> login page. */
  me() {
    return request("/auth/me");
  },

  /** The logged-in officer, or null if nobody is signed in (no redirect: for the login page). */
  async currentUser() {
    try {
      return await request("/auth/me", { loginOn401: false });
    } catch (error) {
      if (error.status === 401) return null;
      throw error;
    }
  },

  /** Score an applicant. profile is a plain object; statement is a File or null. */
  predict(profile, statement = null) {
    const form = new FormData();
    form.append("profile", JSON.stringify(profile));
    if (statement) form.append("statement", statement, statement.name);
    return request("/predict", { method: "POST", body: form });
  },

  /** One page of the officer's history. Filters: page, size, band, decision, ref, dates. */
  predictions(filters = {}) {
    return request(`/predictions${query(filters)}`);
  },

  /** One saved assessment, in full. */
  prediction(id) {
    return request(`/predictions/${encodeURIComponent(id)}`);
  },

  /** The model's version, test score, thresholds and bands. */
  modelInfo() {
    return request("/model/info");
  },

  /** Is the server up? Public: works without logging in. */
  health() {
    return request("/health", { loginOn401: false });
  },
};
