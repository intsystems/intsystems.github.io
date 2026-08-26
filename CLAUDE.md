# Intelligent Systems Department Website — Claude Instructions

Jekyll 4.3 static site for the Intelligent Systems Department (MIPT), deployed to GitHub Pages via GitHub Actions.
Bilingual (EN default at `/`, RU at `/ru/`) using `jekyll-multiple-languages-plugin` 1.8.

## Build & deploy

- Local: `bundle install && bundle exec jekyll serve` (Ruby 3.1 in CI).
  With macOS system Ruby use `BUNDLE_PATH=vendor/bundle bundle install && BUNDLE_PATH=vendor/bundle bundle exec jekyll build`.
- Sass is compiled by `jekyll-sass-converter` 2.x (libsass) on purpose: no dart-sass binary and no `@import` deprecation noise.
  libsass evaluates CSS `min()`/`max()` as Sass math — wrap them in `unquote("...")` (see `_sass/_toc.scss`).
- Deploy: push to `main` triggers `.github/workflows/jekyll.yml` (optimize images → build → deploy to Pages).
  Pull requests to `main` run the same build without deploying.
  Optimized images are cached by the hash of the originals (`actions/cache`), so the slow step runs only when images change.
- `.github/workflows/links.yml` checks all external links weekly (lychee); a red run means a dead link in content.
- There is no other deploy path; `_site/`, `vendor/` and `Gemfile.lock` are gitignored.

## Architecture: how i18n works (critical)

Every content page exists in TWO places:

1. A **stub** at the repo root (e.g. `nir.md`) or in a collection (`_people/*.md`, `_course/*.md`).
   The stub holds ONLY front matter (title key, permalink, layout data) and a `{% tf <file>.md %}` tag.
2. The **actual content** in `_i18n/en/<file>.md` and `_i18n/ru/<file>.md`.
   `{% tf %}` pulls the file for the current language.

Translation strings (nav names, page titles, people/course display names) live in `_i18n/en.yml` and `_i18n/ru.yml`.
Templates use `{% t key.path %}` to resolve them.
Missing RU content files fall back to EN (currently missing in `_i18n/ru/`: `conferences.md`, `nir.md`, `seminars.md`, `templates.md`, `thesis.md` — intentional, they render EN content).

**Rule: never put visible text in a root stub — put it in `_i18n/{en,ru}/`.
Never add a person/course without registering its display name in BOTH `en.yml` and `ru.yml`.**

## Directory map

| Path | Purpose |
|---|---|
| `_config.yml` | Site config, nav menu (`nav.pages`), collections, defaults, roles/types enums (`global.people.roles`, `global.course.types`), `logo` (large, for Open Graph) / `navbar_logo` (small) |
| `*.md` (root) | Page stubs: front matter + `{% tf %}` only. Permalinks like `/materials/nir/` are set here |
| `404.html` | Not-found page (bilingual, served by GitHub Pages for any missing URL) |
| `_i18n/en.yml`, `_i18n/ru.yml` | ALL translation strings: nav, titles, `people:` names, `courses:` names |
| `_i18n/{en,ru}/*.md` | Real page content per language |
| `_i18n/{en,ru}/_people/*.md` | Person biographies (rendered under profile header) |
| `_i18n/{en,ru}/_course/*.md` | Course descriptions |
| `_i18n/{en,ru}/_posts/` | News posts (`YYYY-MM-DD-slug.md`, layout `news`, shown on home page carousel) |
| `_people/*.md` | Person stubs: `position` (role), `avatar`, contact/scholar links. Filename = `lastname_ii.md` |
| `_course/*.md` | Course stubs: `type` (bachelor/master/deprecated/draft), `lecturers` (comma-separated people IDs), `site` |
| `_blogs/*.md` | Blog posts — full content directly here (NOT via `_i18n`), EN only. Filename = URL slug (`/materials/blog/<slug>/`) |
| `redirects/*.md` | Front-matter-only stubs with `layout: redirect` that keep old URLs alive (`permalink:` old, `redirect:` new) |
| `_data/schedule.yml` | Timetable data, rendered by `_includes/schedule_table.html` from `_i18n/*/education.md` |
| `_data/conferences.yml` | Conference deadlines, rendered by Liquid in `_i18n/en/conferences.md` |
| `_layouts/` | `default` (base HTML, all `<head>` assets), `page`, `profile`, `course`, `blog`, `news`, `redirect` |
| `_includes/` | `navbar`, `footer`, `seo` (meta + JSON-LD), `edit` (floating edit button), `toc`, `blog-grid`, `schedule_table`, `person-card` (avatar+name card, pass `id=` or `profile=`), `social-links` (the department's social links, pass `link_class=`) |
| `_data/profile_links.yml` | Ordered list of contact/academic link types on profile pages; add an entry here to support a new link field in `_people/*.md` |
| `_sass/` + `style.scss` | Styles; `style.scss` imports all partials; variables in `_sass/_variables.scss`; `_fonts.scss` (self-hosted Open Sans `@font-face`), `_icons.scss` (SVG-mask icons replacing Font Awesome/academicons), `_bootstrap_shim.scss` (structural navbar rules copied from Bootstrap 3) |
| `fonts/` | Self-hosted Open Sans woff2 (variable, latin+cyrillic; declared in `_sass/_fonts.scss`) |
| `javascript/` | Vanilla JS, no jQuery/Bootstrap: `navbar.js` (collapse+dropdowns), `carousel.js` (scroll-snap carousel), `toc.js`, `fade_in.js`, `copy_email.js`, `table_scroll.js`, `mermaid_init.js`, `prevent_flash.js` |
| `images/people/` | Avatars (filename referenced in `_people/*.md` `avatar:` field; `default.jpg` fallback) |
| `images/blog/<post-slug>/` | Blog images, folder named after the `_blogs/*.md` filename |

## Recipes

### Add a person
1. Create `_people/lastname_ii.md` (copy an existing one): set `title: people.lastname_ii`, `position:` (one of `hotd|dos|phd|pgs|gs`), `avatar:`, links.
2. Create `_i18n/en/_people/lastname_ii.md` and `_i18n/ru/_people/lastname_ii.md` with the bio.
3. Add `lastname_ii: Name Surname` under `people:` in `_i18n/en.yml` and the Russian name in `_i18n/ru.yml`.
4. Add the photo to `images/people/` (any jpg/png as is — CI compresses it at deploy time).
URL becomes `/people/lastname_ii/`.

### Add a course
1. Create `_course/course_name.md`: `title: courses.course_name`, `type:`, `lecturers:` (people IDs, comma-separated, no spaces), `site:`.
2. Create `_i18n/{en,ru}/_course/course_name.md` with the description.
3. Register `course_name:` under `courses:` in both `en.yml` and `ru.yml`.
The course auto-appears on `/course/`, home page, lecturer profiles ("Teaches" section), and the schedule if referenced in `_data/schedule.yml`.

### Add a blog post
1. Copy an existing post (e.g. `_blogs/hyperband.md`) to `_blogs/<slug>.md` — a short descriptive slug, it becomes the URL; fill `title`, `date`, `authors`, `summary`, `tags`, `read_time`, `cover`.
2. Put images in `images/blog/<slug>/` (as is — CI compresses at deploy time).
3. Start body headings at `##` — the layout already renders the title as `<h1>`.
URL becomes `/materials/blog/<slug>/`. Never rename a published post without adding a stub in `redirects/`.

### Add news
Create `_i18n/en/_posts/YYYY-MM-DD-slug.md` AND `_i18n/ru/_posts/YYYY-MM-DD-slug.md` with `title`, `date`, `important: true|false`.
Home page shows the 10 latest.

### Update schedule / NIR / thesis tables
- Schedule: edit `_data/schedule.yml` (course IDs must match `_course/` filenames); keep EN and RU `education.md` in sync (same semesters, same order).
- NIR reports: append rows to tables in `_i18n/en/nir.md` (see README for the row format).
- Theses: `_i18n/en/thesis.md`.
- Semester headings are always **Season Year**: `### Spring 2026` / `### Fall 2026` (RU: `### Весна 2026` / `### Осень 2026`); data keys follow the same order (`spring_2026`). Thesis sections are by year only (`### 2026`).

## Gotchas

- The i18n plugin builds the site once per language; EN pages live at `/...`, RU at `/ru/...`.
  `site.baseurl` is language-prefixed; `site.baseurl_root` is not — use `baseurl_root` for assets (images, JS).
  In `seo.html`, image URLs are built from `site.url` + root path for the same reason.
- SEO lives in `_includes/seo.html` (description, canonical, hreflang, Open Graph, JSON-LD incl. breadcrumbs) and the `<title>` template in `default.html`
  (`<page> — <brand>`, strings under `site.seo` in `en.yml`/`ru.yml`).
  A root stub with `untranslated: true` (nir, thesis, conferences, seminars, templates) and every blog post canonicalize their RU copy to the EN URL — remove the flag when RU content is added.
- Every page has exactly one `<h1>`: layouts render the title as `h1.page-heading` (styled like h2), content pages start with a single `#` heading, list pages (`people.md`, `course.md`) carry an `h1.sr-only`.
  Keep this when adding pages — Lighthouse/SEO and screen readers depend on it.
- `_plugins/i18n_filter_cache.rb` clears Jekyll 4's `absolute_url` cache between the per-language builds; without it RU canonicals/sitemap point at EN URLs.
- `{% t key %}` with a MISSING key fails the build (plugin 1.8 returns nil).
  `page.title` may be a translation key (`titles.nir`, `people.x` — has a dot, no spaces) or a literal string (blog/news posts);
  `_layouts/default.html` resolves it into `page_title` once, and `seo.html` reuses that — never call `{% t page.title %}` elsewhere.
- People/course lookups are done by `page.url contains lecturer_id` string matching — keep IDs unique and never rename a person file without checking `lecturers:` fields in `_course/`.
- `edit: true` in front matter shows the floating "edit on GitHub" button (`_includes/edit.html` maps the path into `_i18n/<lang>/`).
- `toc: true` + optional `toc_headings` enables the floating table of contents (`javascript/toc.js`).
- Front matter defaults in `_config.yml` assign layouts by collection type — new files usually need no `layout:` key.
- Tables are auto-wrapped for horizontal scroll by `table_scroll.js`; don't wrap them manually.
- Images are NOT compressed manually.
  The deploy workflow (`.github/workflows/jekyll.yml`, "Optimize images" step) resizes and compresses all jpg/png at build time; the repo keeps originals.
- Do NOT delete unused images from `images/` — they are kept on purpose as an archive and may be reused later.
- There is NO Bootstrap, jQuery, Font Awesome, academicons, or bulma-carousel — do not add CDN links back.
  Navbar behavior is `javascript/navbar.js`; navbar structural CSS is `_sass/_bootstrap_shim.scss`.
  Icons: `<i class="fa fa-NAME"></i>` / `<i class="ai ai-NAME"></i>` markup still works, but only for the classes defined in `_sass/_icons.scss` — to add a new icon, add its SVG there as a mask.
- MathJax loads only on `layout: blog` pages or pages with `math: true`; mermaid only on `blog`/`news` layouts or `mermaid: true` (see conditions in `_layouts/default.html`).
- No inline `style="..."` in markup — every visual rule lives in `_sass/`.
  Home page sections use dedicated classes in `_sass/_home.scss` (`.hero*`, `.stats*`, `.research-*`, `.tag-list`, `.fullwidth-figure`); person cards and social links are includes, never copy-pasted markup.
- Responsive container widths come from the `$container-widths` map in `_sass/_variables.scss` — change breakpoint behavior there, not in per-file media queries.
- EN and RU `index.md` must keep identical markup (only text differs, plus the `hero__title--sm` modifier on RU for the longer title).
