// Shared page helpers (step 6.3).
//
// The Content-Security-Policy allows no inline styles, so a position on the risk scale is
// written as data-at="49.1%" in the HTML and turned into the --at CSS variable here.
// (Setting a style from a script file is allowed; a style="" attribute in the HTML is not.)

export function placeMarkers(root = document) {
  for (const element of root.querySelectorAll("[data-at]")) {
    element.style.setProperty("--at", element.dataset.at);
  }
}

placeMarkers();
