# Accessibility audit (step 6.7)

Target: a Lighthouse accessibility score of **90 or more** on every screen, full keyboard use,
visible focus, checked colour contrast, and the risk band never shown by colour alone.

**Result: 100 on every screen, phone and desktop.** Audited on 2026-10-19 in headless
Chromium against the app running locally, signed in as an officer with 27 saved assessments,
then confirmed in Chrome on Windows (last column of the scores table).

## How it was checked

| Check | Tool | What it covers |
|---|---|---|
| Lighthouse accessibility | Lighthouse 12.8 (Chromium), mobile and desktop presets | Names, labels, contrast, landmarks, headings, ARIA, document language |
| Automated rules | axe-core 4.10 (WCAG 2.0 and 2.1 A and AA, best practice) | The same engine Lighthouse uses, run on states Lighthouse can't reach: the form with errors showing, the empty History |
| Keyboard tour | Scripted: Tab through every page, record each stop and its outline | Every control reachable, focus visible on each stop, no traps |
| Reflow | Pages loaded 320 px wide (a 1280 px screen at 400% zoom) | Nothing wider than the screen, no sideways scrolling |
| High contrast | Windows high-contrast mode emulated (forced colours) | Band tags, the score scale and focus outlines still visible |
| Contrast | Unit test in `tests/unit/test_frontend.py` | Every text colour at least 4.5 : 1 on its background (WCAG AA) |
| Manual | Keyboard only, in Chrome on Windows | A full assessment from sample to result without a mouse |

## Lighthouse scores

| Screen | Address | Phone | Desktop | Chrome on Windows |
|---|---|---|---|---|
| Login | `/` (signed out) | 100 | 100 | |
| New assessment | `/assess.html` | 100 | 100 | |
| Result | `/result.html?id=N` | 100 | 100 | |
| History | `/history.html` | 100 | 100 | |
| Saved assessment | `/detail.html?id=N` | 100 | 100 | |
| Model info | `/about.html` | 100 | 100 | |

Best practices also scored 100 on every screen after the fix below. Lighthouse won't score the
404 page (it refuses pages that answer 404); axe found no problems on it.

## Keyboard tour

Every page starts with "Skip to main content", which moves focus past the header. Every stop
showed the 3 px blue outline.

| Page | Tab stops | Notes |
|---|---|---|
| Login | 5 | Skip link, the MkopoGuard name (a link), email, password, Sign in |
| New assessment | 30 | Each yes/no group is one stop (arrow keys move inside it, as for any radio group); the date box has three stops (day, month, year) |
| Result | 8 | |
| History | 40 | Includes 20 "View" links and the two date boxes |
| Saved assessment | 7 | |
| Model info | 6 | |

A whole assessment was done keyboard-only: choose a sample, Load sample, Assess risk, read the
result. Error messages are announced (the error summary takes focus and every item links to its
field) and the "Assessing..." state is announced through a status line.

## Problems found and fixed in this step

1. **No focus outline on date boxes.** Chrome moves focus into the day, month and year parts
   inside a date box, so the box itself never matched `:focus-visible`. Fixed with
   `input[type="date"]:focus-within`. A test now fails if the rule disappears or if any
   `outline: none` is added.
2. **The saved-assessment answers were 1 px too wide at 320 px.** Their grid asked for columns of
   at least 18rem (288 px), more than a 320 px screen has left after padding. Every grid now asks
   for `min(18rem, 100%)`, and a test checks all of them.
3. **High-contrast mode hid the score marker's label** (dark text on a black background). Band
   tags, the scale and its marker now keep visible edges and text in forced-colour mode.
4. **The login page logged a console error for every signed-out visitor.** It checked "am I
   signed in?" by asking `/auth/me`, whose 401 is the right answer but shows as an error.
   Added `GET /auth/session`, which answers `{"user": null}` instead. (Best practices went from
   96 to 100 on the login page.)

## Already in place from earlier steps

- The band is always written in words inside its tag; colour is a second signal (6.1, 6.2).
- Band colours chosen for contrast (5.7 : 1 or better), and steadily darker from Low to Very high
  so they still read in order for colour-blind users (6.1).
- One heading 1 per page, no skipped heading levels, unique ids, every field labelled (tests, 6.2).
- Errors shown next to each field and linked from a summary; fields marked `aria-invalid` (6.5).
- Messages that change without a page load use `role="status"` or `role="alert"` (6.4, 6.5).
- The History table keeps its table roles when shown as cards on phones (6.2).
- `prefers-reduced-motion` respected; the app has no animation anyway (6.2).

## Re-checking

Lighthouse in Chrome: open a page, press F12, choose **Lighthouse**, tick only
**Accessibility**, pick **Mobile** or **Desktop**, and click **Analyze page load**. Sign in first;
the login page must be checked signed out.
