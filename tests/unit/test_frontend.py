"""Step 6.2: the static pages. Structure checks that keep them accessible as they change.

These read the HTML with the standard library's parser, so no browser is needed. The full
accessibility audit (axe / Lighthouse) is step 6.7; these catch the common slips early:
a missing label, a page without a heading, a form field the API doesn't know.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mkopoguard import config
from mkopoguard.api.main import create_app
from mkopoguard.api.schemas import MAX_LOAN_KES, MIN_LOAN_KES, REF_PATTERN, ApplicantProfile
from mkopoguard.api.settings import Settings
from mkopoguard.api.statement import MAX_BYTES

FRONTEND = config.PROJECT_ROOT / "frontend"
PAGES = ["index.html", "assess.html", "result.html", "history.html", "detail.html", "about.html"]
ALL_PAGES = PAGES + ["404.html"]
APP_PAGES = PAGES[1:]  # behind the login: header with navigation
DISCLAIMER = "Demo trained on synthetic data. Not for real lending decisions."
BANDS = ("low", "medium", "high", "very-high")
FORM_ONLY = {"statement"}  # sent as its own multipart part, not inside the profile JSON
FONT = "fonts/atkinson-hyperlegible-next-latin-wght-normal.woff2"


class Page(HTMLParser):
    """Collects what the checks need from one page."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []  # (tag, attrs) in order
        self.text = []
        self.label_for = set()
        self.wrapped = 0  # inputs inside a <label>
        self.inside_label = 0
        self.headings = []
        self._heading = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if tag == "label":
            self.inside_label += 1
            if "for" in attrs:
                self.label_for.add(attrs["for"])
        if tag == "input" and self.inside_label:
            self.wrapped += 1
            attrs["_wrapped"] = True
        if re.fullmatch(r"h[1-6]", tag):
            self._heading = [int(tag[1]), ""]

    def handle_endtag(self, tag):
        if tag == "label":
            self.inside_label -= 1
        if self._heading and tag == f"h{self._heading[0]}":
            self.headings.append(tuple(self._heading))
            self._heading = None

    def handle_data(self, data):
        self.text.append(data)
        if self._heading:
            self._heading[1] += data

    def all(self, tag):
        return [a for t, a in self.tags if t == tag]

    def ids(self):
        return [a["id"] for _, a in self.tags if "id" in a]

    @property
    def plain_text(self):
        return " ".join(" ".join(self.text).split())


def parse(name: str) -> Page:
    page = Page()
    page.feed((FRONTEND / name).read_text(encoding="utf-8"))
    return page


# --- every page --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_PAGES)
def test_page_basics(name):
    page = parse(name)
    assert page.all("html")[0].get("lang") == "en"
    assert any(a.get("name") == "viewport" for a in page.all("meta"))
    titles = [t for t, _ in page.tags if t == "title"]
    assert len(titles) == 1 and "| MkopoGuard" in page.plain_text
    assert len(page.all("main")) == 1 and page.all("main")[0].get("id") == "main"
    assert any(a.get("href") == "#main" for a in page.all("a")), "skip link"
    assert [a["href"] for a in page.all("link") if a.get("rel") == "stylesheet"] == [
        "/css/style.css"
    ]


@pytest.mark.parametrize("name", ALL_PAGES)
def test_every_page_carries_the_disclaimer_in_its_footer(name):
    html = (FRONTEND / name).read_text(encoding="utf-8")
    footer = re.search(r"<footer[^>]*>(.*?)</footer>", html, re.S)
    assert footer and DISCLAIMER in " ".join(footer.group(1).split())


@pytest.mark.parametrize("name", ALL_PAGES)
def test_one_h1_and_no_skipped_heading_levels(name):
    levels = [level for level, _ in parse(name).headings]
    assert levels.count(1) == 1 and levels[0] == 1
    for before, after in zip(levels, levels[1:], strict=False):
        assert after <= before + 1, f"h{before} followed by h{after}"


@pytest.mark.parametrize("name", ALL_PAGES)
def test_ids_are_unique(name):
    ids = parse(name).ids()
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("name", ALL_PAGES)
def test_every_form_control_has_a_label(name):
    page = parse(name)
    for tag in ("input", "select", "textarea"):
        for attrs in page.all(tag):
            if attrs.get("type") in ("hidden", "submit", "reset", "button"):
                continue
            labelled = (
                attrs.get("_wrapped")
                or attrs.get("id") in page.label_for
                or attrs.get("aria-label")
            )
            assert labelled, f"{tag} {attrs.get('name')} has no label"


@pytest.mark.parametrize("name", ALL_PAGES)
def test_described_by_points_at_real_elements(name):
    page = parse(name)
    ids = set(page.ids())
    for _, attrs in page.tags:
        for ref in attrs.get("aria-describedby", "").split():
            assert ref in ids, ref


@pytest.mark.parametrize("name", ALL_PAGES)
def test_no_inline_scripts_or_event_handlers(name):
    """Ready for a strict Content-Security-Policy: script only from files (step 6.3)."""
    page = parse(name)
    for attrs in page.all("script"):
        assert "src" in attrs, "inline <script>"
    for _, attrs in page.tags:
        assert not any(key.startswith("on") for key in attrs), "inline event handler"


@pytest.mark.parametrize("name", ALL_PAGES)
def test_links_to_our_own_pages_exist(name):
    for attrs in parse(name).all("a"):
        href = attrs.get("href", "")
        if href.startswith("/") and not href.startswith("/docs"):
            path = href.split("?")[0].split("#")[0].lstrip("/") or "index.html"
            assert (FRONTEND / path).exists(), href


@pytest.mark.parametrize("name", APP_PAGES)
def test_app_pages_have_the_navigation_and_mark_the_current_page(name):
    page = parse(name)
    navs = [a for a in page.all("nav") if a.get("aria-label") == "Main"]
    assert len(navs) == 1
    current = [a["href"] for a in page.all("a") if a.get("aria-current") == "page"]
    assert len(current) == 1
    assert any(a.get("action") == "/auth/logout" for a in page.all("form"))


# --- colour is never the only signal ------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_PAGES)
def test_every_band_tag_names_its_band_in_text(name):
    html = (FRONTEND / name).read_text(encoding="utf-8")
    names = {"low": "Low", "medium": "Medium", "high": "High", "very-high": "Very high"}
    for band, text in re.findall(r'class="band band-([a-z-]+)"[^>]*>([^<]*)<', html):
        assert text.strip() == names[band]


def test_the_stylesheet_defines_every_band_colour_once():
    css = (FRONTEND / "css/style.css").read_text(encoding="utf-8")
    for band in BANDS:
        assert css.count(f"--band-{band}-bg:") == 1
        assert css.count(f"--band-{band}-fg:") == 1
        assert f".band-{band}" in css
    root = re.search(r":root\s*{(.*?)}", css, re.S).group(1)
    outside_root = css.replace(root, "")
    hexes = re.findall(r"#[0-9a-fA-F]{3,6}\b", outside_root)
    assert set(h.lower() for h in hexes) <= {"#ffffff"}, "colours belong in :root variables"


def test_band_colours_pass_contrast():
    """WCAG AA: text needs 4.5 : 1 on its background."""
    css = (FRONTEND / "css/style.css").read_text(encoding="utf-8")

    def colour(name):
        return re.search(rf"--{name}:\s*(#[0-9a-fA-F]{{6}})", css).group(1)

    def luminance(hex_colour):
        rgb = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    def ratio(a, b):
        hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    for band in BANDS:
        fg = colour(f"band-{band}-fg")
        assert ratio(fg, colour(f"band-{band}-bg")) >= 4.5, band
        assert ratio(fg, "#ffffff") >= 4.5, band
    for name in ("ink", "muted", "primary", "error"):
        assert ratio(colour(name), colour("page")) >= 4.5, name
        assert ratio(colour(name), colour("surface")) >= 4.5, name


# --- the forms match the API -------------------------------------------------------------------


def test_assessment_form_asks_exactly_what_the_api_needs():
    page = parse("assess.html")
    form_names = {a["name"] for t, a in page.tags if t in ("input", "select") and "name" in a} - {
        "sample"
    }
    assert form_names - FORM_ONLY == set(ApplicantProfile.model_fields)
    assert "statement" in form_names


def test_assessment_choices_use_the_values_the_api_accepts():
    html = (FRONTEND / "assess.html").read_text(encoding="utf-8")
    page = parse("assess.html")
    schema = ApplicantProfile.model_json_schema()

    def allowed(field):
        spec = schema["properties"][field]
        options = spec.get("anyOf", [spec])
        values = set()
        for option in options:
            ref = option.get("$ref")
            target = schema["$defs"][ref.split("/")[-1]] if ref else option
            values |= {str(v) for v in target.get("enum", [])}
            if "const" in target:
                values.add(str(target["const"]))
            if target.get("type") == "null":
                values.add("")
        return values

    for field in ("education_level", "emergency_fund_source", "worried_about_bills"):
        block = re.search(rf'<select id="{field}".*?</select>', html, re.S).group(0)
        offered = set(re.findall(r'value="([^"]*)"', block))
        assert offered == allowed(field), field
    terms = {a["value"] for a in page.all("input") if a.get("name") == "term_months"}
    assert terms == {"1", "3", "6"}


def test_login_form_uses_the_oauth2_field_names():
    page = parse("index.html")
    names = {a.get("name") for a in page.all("input")}
    assert names == {"username", "password"}
    password = next(a for a in page.all("input") if a.get("name") == "password")
    assert password.get("type") == "password"
    assert password.get("autocomplete") == "current-password"


def test_history_filters_match_the_api_query_parameters():
    page = parse("history.html")
    names = {a.get("name") for t, a in page.tags if t in ("input", "select") and a.get("name")}
    assert names == {"band", "decision", "ref", "date_from", "date_to"}


def test_samples_offered_are_the_committed_samples():
    page = parse("assess.html")
    offered = {
        a["value"]
        for t, a in page.tags
        if t == "option" and a.get("value", "").startswith("sample_")
    }
    committed = {p.stem for p in (config.PROJECT_ROOT / "samples/profiles").glob("*.json")}
    assert offered == committed


# --- served by the app --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    if not (config.BUNDLE_DIR / "manifest.json").exists():
        pytest.skip("bundle not packaged yet")
    settings = Settings(
        _env_file=None,
        secret_key="test-secret-key-that-is-long-enough-1234567890",
        environment="test",
    )
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.mark.parametrize("name", PAGES)
def test_pages_are_served(client, name):
    response = client.get("/" if name == "index.html" else f"/{name}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["x-frame-options"] == "DENY"


@pytest.mark.parametrize(
    "path, kind",
    [
        ("/css/style.css", "text/css"),
        (f"/{FONT}", "font/woff2"),
        ("/img/favicon.svg", "image/svg"),
        ("/js/api.js", "text/javascript"),
        ("/js/ui.js", "text/javascript"),
    ],
)
def test_assets_are_served(client, path, kind):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(kind)


def test_the_font_licence_ships_with_the_font():
    assert (FRONTEND / FONT).stat().st_size > 10_000
    assert "SIL Open Font License" in (FRONTEND / "fonts/OFL.txt").read_text(encoding="utf-8")


def test_unknown_pages_get_the_friendly_404(client):
    response = client.get("/no-such-page")
    assert response.status_code == 404
    assert "Page not found" in response.text


def test_page_names_do_not_fall_under_the_api_no_store_rule(client):
    """Pages may be cached; only API paths (/auth, /predict, ...) are no-store."""
    for name in PAGES[1:]:
        assert client.get(f"/{name}").headers.get("cache-control") != "no-store", name


def test_frontend_folder_has_only_what_the_site_needs():
    allowed = {".html", ".css", ".js", ".woff2", ".svg", ".txt"}
    for path in Path(FRONTEND).rglob("*"):
        if path.is_file() and path.name != ".gitkeep":  # placeholder that keeps an empty folder
            assert path.suffix in allowed, path


# --- scripts and the Content-Security-Policy (step 6.3) --------------------------------------


@pytest.mark.parametrize("name", ALL_PAGES)
def test_no_inline_styles(name):
    """The policy blocks style="" attributes and <style> blocks; positions use data-at."""
    page = parse(name)
    assert not page.all("style")
    for tag, attrs in page.tags:
        assert "style" not in attrs, tag


@pytest.mark.parametrize("name", ALL_PAGES)
def test_scripts_are_modules_from_our_js_folder(name):
    for attrs in parse(name).all("script"):
        src = attrs.get("src", "")
        assert src.startswith("/js/") and attrs.get("type") == "module", src
        assert (FRONTEND / src.lstrip("/")).exists(), src


@pytest.mark.parametrize("name", ["result.html", "detail.html"])
def test_scale_positions_are_data_attributes_set_by_ui_js(name):
    page = parse(name)
    positions = [attrs["data-at"] for _, attrs in page.tags if "data-at" in attrs]
    assert positions and all(re.fullmatch(r"\d+(\.\d+)?%", p) for p in positions)
    assert any(a.get("src") == "/js/ui.js" for a in page.all("script"))


def test_only_the_api_client_calls_fetch():
    for path in (FRONTEND / "js").glob("*.js"):
        text = path.read_text(encoding="utf-8")
        expected = 1 if path.name == "api.js" else 0
        assert text.count("fetch(") == expected, path.name


def test_the_api_client_sends_same_origin_credentials_and_the_app_header():
    text = (FRONTEND / "js/api.js").read_text(encoding="utf-8")
    assert 'credentials: "same-origin"' in text
    assert '"X-Requested-With": "fetch"' in text
    for path in ("/auth/login", "/auth/logout", "/auth/me", "/predict", "/predictions",
                 "/model/info", "/health"):  # fmt: skip
        assert f'"{path}' in text or f"`{path}" in text, path


# --- login and logout (step 6.4) ------------------------------------------------------------


def test_login_page_runs_login_js_and_has_places_for_messages():
    page = parse("index.html")
    assert [a["src"] for a in page.all("script")] == ["/js/login.js"]
    ids = set(page.ids())
    assert {"login-form", "login-status", "login-error", "login-error-text"} <= ids
    roles = {a.get("id"): a.get("role") for _, a in page.tags if a.get("id")}
    assert roles["login-status"] == "status" and roles["login-error"] == "alert"


@pytest.mark.parametrize("name", APP_PAGES + ["404.html"])
def test_app_pages_check_the_session_first(name):
    sources = [a["src"] for a in parse(name).all("script")]
    assert sources[0] == "/js/session.js"


@pytest.mark.parametrize("name", APP_PAGES + ["404.html"])
def test_no_officer_name_is_hard_coded_in_the_header(name):
    html = (FRONTEND / name).read_text(encoding="utf-8")
    assert re.search(r'<span id="account-name"></span>', html)
    assert re.search(r'<span class="role" id="account-role" hidden></span>', html)


def test_scripts_never_write_html():
    """Text goes in with textContent, so a name or reference can never become markup."""
    for path in (FRONTEND / "js").glob("*.js"):
        text = path.read_text(encoding="utf-8")
        for risky in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
            assert risky not in text, f"{path.name}: {risky}"


# --- the assessment flow (step 6.5) -----------------------------------------------------------


def js(name: str) -> str:
    return (FRONTEND / "js" / name).read_text(encoding="utf-8")


def ids_used(*scripts: str) -> set[str]:
    """Every element id the scripts look up with getElementById("...")."""
    text = "".join(js(name) for name in scripts)
    return set(re.findall(r'getElementById\("([^"]+)"\)', text))


def test_each_page_loads_its_script_after_the_session_check():
    for page_name, script in (("assess.html", "/js/assess.js"), ("result.html", "/js/result.js")):
        sources = [a["src"] for a in parse(page_name).all("script")]
        assert sources[0] == "/js/session.js" and script in sources, page_name


@pytest.mark.parametrize(
    "page_name, scripts",
    [("assess.html", ["assess.js"]), ("result.html", ["result.js", "render.js"])],
)
def test_every_element_a_script_needs_is_on_its_page(page_name, scripts):
    missing = ids_used(*scripts) - set(parse(page_name).ids())
    assert not missing, missing


def test_browser_checks_match_the_api_rules():
    """The browser checks first, the server decides: the two must agree on the rules."""
    text = js("assess.js")
    pattern = re.search(r"REF_PATTERN = /(.+)/;", text).group(1)
    assert pattern == REF_PATTERN
    limits = dict(re.findall(r"(\w+): \[?(\d+(?:, \d+)?)\]?,", text))
    fields = ApplicantProfile.model_fields

    def bounds(name):
        metadata = {type(m).__name__: m for m in fields[name].metadata}
        return f"{metadata['Ge'].ge}, {metadata['Le'].le}"

    assert limits["age"] == bounds("age")
    assert limits["income_quintile"] == bounds("income_quintile")
    assert limits["loan_amount_kes"] == f"{MIN_LOAN_KES}, {MAX_LOAN_KES}"
    assert limits["statement_bytes"] == str(MAX_BYTES)


def test_the_reference_box_allows_as_many_characters_as_the_api():
    ref = next(a for a in parse("assess.html").all("input") if a.get("name") == "applicant_ref")
    assert ref["maxlength"] == "64" and "{2,63}" in REF_PATTERN


def test_the_result_page_waits_for_its_data():
    page = parse("result.html")
    attrs = {a["id"]: a for _, a in page.tags if "id" in a}
    assert "hidden" in attrs["result"]
    assert attrs["result-status"].get("role") == "status"
    html = (FRONTEND / "result.html").read_text(encoding="utf-8")
    assert '<p class="lede" id="result-lede"></p>' in html


def test_samples_are_served_for_load_sample(client):
    manifest = client.get("/samples/manifest.json")
    assert manifest.status_code == 200
    assert manifest.headers["content-type"].startswith("application/json")
    for entry in manifest.json():
        profile = client.get(f"/samples/{entry['profile']}")
        assert profile.status_code == 200, entry["name"]
        ApplicantProfile.model_validate_json(profile.text)  # a valid profile, as the form sends
        if entry["statement"]:
            statement = client.get(f"/samples/{entry['statement']}")
            assert statement.status_code == 200, entry["name"]
            assert statement.headers["content-type"].startswith("text/csv")


def test_the_samples_folder_serves_nothing_else(client):
    assert client.get("/samples/../src/mkopoguard/config.py").status_code == 404
    assert client.get("/samples/").status_code == 404  # no folder listing


# --- history, saved assessments and model info (step 6.6) --------------------------------------


@pytest.mark.parametrize(
    "page_name, script",
    [("history.html", "/js/history.js"), ("detail.html", "/js/detail.js"),
     ("about.html", "/js/about.js")],
)  # fmt: skip
def test_live_pages_load_their_script_after_the_session_check(page_name, script):
    sources = [a["src"] for a in parse(page_name).all("script")]
    assert sources[0] == "/js/session.js" and script in sources


@pytest.mark.parametrize(
    "page_name, scripts",
    [("history.html", ["history.js"]), ("detail.html", ["detail.js", "render.js"]),
     ("about.html", ["about.js"])],
)  # fmt: skip
def test_live_pages_have_every_element_their_scripts_need(page_name, scripts):
    optional = {"warnings", "warning-list"}  # render.js skips warnings on pages without them
    missing = ids_used(*scripts) - set(parse(page_name).ids()) - optional
    assert not missing, missing


def test_history_has_no_made_up_rows_and_starts_hidden():
    html = (FRONTEND / "history.html").read_text(encoding="utf-8")
    body = re.search(r'<tbody role="rowgroup" id="history-rows">(.*?)</tbody>', html, re.S)
    assert body and not body.group(1).strip()
    attrs = {a["id"]: a for _, a in parse("history.html").tags if "id" in a}
    assert "hidden" in attrs["history-table"]
    assert attrs["filter-error"].get("role") == "alert"


def test_detail_page_shows_every_answer():
    """Every field the form sends (except the reference, shown in the heading) has a label."""
    block = re.search(r"export const ANSWERS = \[(.*?)\n\];", js("detail.js"), re.S).group(1)
    shown = re.findall(r'^\s*\["(\w+)",', block, re.M)
    assert len(shown) == len(set(shown))
    assert set(shown) == set(ApplicantProfile.model_fields) - {"applicant_ref"}


def test_detail_page_has_words_for_every_choice_the_api_accepts():
    schema = ApplicantProfile.model_json_schema()
    text = js("detail.js")
    for field in ("education_level", "emergency_fund_source", "worried_about_bills"):
        values = set()
        for option in schema["properties"][field].get("anyOf", []):
            target = schema["$defs"][option["$ref"].split("/")[-1]] if "$ref" in option else option
            values |= set(target.get("enum", []))
        block = re.search(rf"{field}: \{{(.*?)\}}", text, re.S).group(1)
        assert set(re.findall(r"(\w+):", block)) == set(values), field


def test_the_back_link_can_only_return_to_a_history_page():
    pattern = re.search(r"return (/\^.*?\$/)\.test\(address\)", js("store.js")).group(1)
    assert pattern.startswith(r"/^\/history\.html")


# --- accessibility pass (step 6.7) ---------------------------------------------------------------


def css() -> str:
    return (FRONTEND / "css/style.css").read_text(encoding="utf-8")


def test_focus_is_always_visible():
    text = css()
    rule = re.search(r":focus-visible,\s*input\[type=\"date\"\]:focus-within\s*\{([^}]*)\}", text)
    assert rule, "date boxes need :focus-within (Chrome focuses the parts inside them)"
    width = re.search(r"outline:\s*(\d+)px solid", rule.group(1))
    assert width and int(width.group(1)) >= 2
    assert not re.search(r"outline:\s*(none|0)\b", text), "never remove the focus outline"


def test_grids_never_need_more_room_than_the_screen():
    """At 400% zoom a laptop screen is 320 px wide: no column may demand more."""
    for minimum in re.findall(r"minmax\(([^,]+),", css()):
        assert minimum.startswith("min(") or minimum.strip() in ("0", "max-content"), minimum


def test_high_contrast_and_reduced_motion_are_handled():
    text = css()
    assert "@media (forced-colors: active)" in text
    assert "@media (prefers-reduced-motion: reduce)" in text


def test_the_login_page_checks_the_session_without_a_401():
    text = js("api.js")
    current = re.search(r"async currentUser\(\) \{(.*?)\n  \},", text, re.S).group(1)
    assert '"/auth/session"' in current and "/auth/me" not in current
