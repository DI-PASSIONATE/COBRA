# COBRA GUI Design Rules

Rules for any agent or person editing `src/cobra/gui/`. They define the visual
language, the theming architecture, and the checks to run before a GUI change is
finished. `.claude/CLAUDE.md` points here; read this file before touching GUI code.

## 1. Design intent

- **Tone:** professional, executive, calm. The tool is used for long optimization
  runs in engineering labs; it should feel like a well-made instrument, not a game.
- **Style:** Material-inspired — tiered surfaces, consistent corner radii, a clear
  button hierarchy, restrained colour, and typography that carries structure.
  Qt Style Sheets cannot render shadows or ripples; elevation is expressed with
  surface tiers and 1 px borders, never with fake shadows.
- **Palette:** muted and laid-back. COBRA's and ORCA's logos share a navy plus
  circuit-green identity; the GUI uses desaturated relatives of those colours
  (a steel-navy "tide", a sage "moss"). No neon, no saturated primaries, no pure
  black or pure white surfaces.
- **Naming:** nature names live in code — tokens, theme names, identifiers.
  User-facing copy stays plain ("Start Optimization", not "Strike").

## 2. Themes and colour tokens

Two themes, both shipped, both first-class:

| Theme | Mode | Idea |
| --- | --- | --- |
| **Sandbank** | light | sunlit shallows, warm sand, pale stone |
| **Deepwater** | dark | deep slate water, low-glare surfaces |

Every colour in the GUI comes from a token. Hex values appear **only** in
`src/cobra/gui/theme.py`, in the token definitions below. No hex literals, named
Qt colours (`Qt.GlobalColor.lightGray`), or single-letter pyqtgraph colours
(`"k"`, `"g"`, `"r"`) anywhere else in `gui/`.

### Surfaces and text

| Token | Role | Sandbank | Deepwater |
| --- | --- | --- | --- |
| `canvas` | window background | `#F3F4F1` | `#1B1F24` |
| `surface` | cards, inputs, tables, plots | `#FBFBF9` | `#23282E` |
| `surface_alt` | headers, secondary buttons, striping | `#E8EAE5` | `#2C3239` |
| `border` | default 1 px borders | `#D3D7D0` | `#3A4149` |
| `border_strong` | hover/active borders, dividers | `#B6BCB3` | `#4C5560` |
| `text` | body text | `#2B3138` | `#E3E6E3` |
| `text_muted` | labels, captions, axis text | `#5F6870` | `#A3ABB2` |
| `text_disabled` | disabled text | `#9AA2A8` | `#6C757D` |
| `selection` | selected rows, text selection | `#D6E2EC` | `#34475A` |

Elevation order is `canvas` < `surface` < `surface_alt`. A widget on `surface`
gets a `border`; nothing is separated by colour alone.

### Semantic colours

| Token | Role | Sandbank | Deepwater |
| --- | --- | --- | --- |
| `tide` | primary / interactive / focus / info | `#3C5A78` | `#7FA3C4` |
| `tide_hover` | | `#34506B` | `#91B1CE` |
| `tide_pressed` | | `#2C4359` | `#6E92B3` |
| `tide_subtle` | tinted backgrounds for primary state | `#E4EBF2` | `#2A3A4A` |
| `on_tide` | text/icons on a `tide` fill | `#FFFFFF` | `#101820` |
| `moss` | success, running, resume, "meets goal" | `#6B8F5E` | `#8FAF82` |
| `moss_subtle` | | `#E6EEE1` | `#2E3A2C` |
| `ochre` | warning, paused, attention | `#B08A3E` | `#CBA45B` |
| `ochre_subtle` | | `#F3ECDC` | `#3A3222` |
| `ember` | danger, stop, error, "violates goal" | `#A85A4E` | `#C77A6E` |
| `ember_subtle` | | `#F2E3E0` | `#3E2A27` |
| `slate` | neutral/inactive state (e.g. stopping) | `#6B7580` | `#9AA5B0` |

Strong variants for the status colours — `moss_strong`, `ochre_strong`,
`ember_strong` — are used wherever the colour carries text: status text on a
surface, and filled buttons with `on_tide` text (pause, resume, danger). The
base tints are for plot series, borders, badges and subtle fills only.

| Token | Sandbank | Deepwater |
| --- | --- | --- |
| `moss_strong` | `#4E6E44` | `#8FAF82` (= base) |
| `ochre_strong` | `#7F6226` | `#CBA45B` (= base) |
| `ember_strong` | `#8A4438` | `#C77A6E` (= base) |

In Sandbank the base `moss`/`ochre` only reach ~3.5:1 under white text, which
is why the strong variants exist; in Deepwater the base tokens already pass.

Contrast targets (WCAG, against the background the token sits on): body text
≥ 7:1, UI text and icons ≥ 4.5:1, focus rings and status indicators ≥ 3:1.
Decorative 1 px borders are exempt but must stay visibly distinct (≥ 1.3:1).
All tokens above were checked against these; re-check when adding one.

### Plot series (pyqtgraph)

`series` is an ordered list; take colours by index, cycling. Always pair a
series colour with a legend entry or label.

| # | Name | Sandbank | Deepwater |
| --- | --- | --- | --- |
| 0 | tide | `#3C5A78` | `#7FA3C4` |
| 1 | moss | `#6B8F5E` | `#8FAF82` |
| 2 | ochre | `#B08A3E` | `#CBA45B` |
| 3 | ember | `#A85A4E` | `#C77A6E` |
| 4 | heather | `#7A6A93` | `#A394BE` |
| 5 | lagoon | `#4A8A8A` | `#78B3B0` |
| 6 | slate | `#6B7580` | `#9AA5B0` |
| 7 | bark | `#8A6B4F` | `#B49072` |

Plot chrome: background `surface`, axis lines and tick text `text_muted`, grid
at 15 % alpha of `border_strong`, plot title `text`. Goal bounds: minimum bound
in `moss`, maximum bound in `ember`, width 3, dashed; the current trial curve is
width 3, history curves width 1.5 at 60 % alpha.

## 3. Shape, spacing, typography

| Token | Value | Use |
| --- | --- | --- |
| `radius_sm` | 6 px | chips, badges, small inputs |
| `radius_md` | 8 px | line edits, combo/spin boxes, tables, progress bars |
| `radius_lg` | 10 px | buttons, group boxes, cards |
| `space_1..5` | 4 / 8 / 12 / 16 / 24 px | the only spacing values; layouts use these |
| `control_h` | 36 px | minimum height of buttons and inputs |
| `control_h_lg` | 48 px | the top-level panel switch and primary action buttons |

Typography: family `"Segoe UI", "Noto Sans", "Helvetica Neue", Arial, sans-serif`
(unchanged). Sizes: `body 13`, `caption 11`, `heading 15 / 600`, `title 18 / 600`.
Weights 400 and 600 only. Group-box titles and table headers are `600`,
`text_muted`. Sentence case everywhere; Title Case only for window titles.

## 4. Component rules

**Buttons** — four roles, expressed with the existing dynamic properties:

| Role | Property | Look |
| --- | --- | --- |
| Primary | `primaryAction="true"` | filled `tide`, `on_tide` text, no border |
| Secondary (default) | none | `surface_alt` fill, `border`, `text` |
| Tertiary / quiet | `flat="true"` | transparent, `tide` text, `tide_subtle` on hover |
| Danger | `dangerAction="true"` | filled `ember_strong`, `on_tide` text |

`actionState` on the primary button maps `start → tide`, `pause → ochre_strong`,
`resume → moss_strong`, `stopping → slate`. Disabled: `surface_alt` fill,
`text_disabled` text, `border`. Never lighten a fill by hand for disabled; use
the tokens. Panel switch buttons (`tabButton`) render as a segmented control:
inactive `surface_alt`, active `surface` with a 2 px `tide` border and `tide`
text.

**Inputs** — `surface` fill, `border`, `radius_md`, 6 px 8 px padding. Focus:
the border becomes `tide` (QSS has no outline ring, so the border colour is the
focus signal). Invalid: `ember` border and an `ember_strong` message
under the field — never a message box for validation.

**Group boxes / cards** — `surface` fill on `canvas`, `border`, `radius_lg`,
title in `text_muted 600`. Use them to group; do not nest more than two deep.

**Tables** — `surface` fill, header `surface_alt`, alternating rows off,
`selection` for selected rows, row height ≥ 28 px. Section-header rows inside
a table use `surface_alt`, not a Qt global colour.

**Progress and status** — the progress bar chunk is `tide`; when paused it is
`ochre_strong`, when finished `moss_strong`. Status text is always paired with colour (colour
is never the only signal). Elapsed-time and iteration labels are `text_muted`.

**Dialogs** — inherit the window theme automatically (stylesheet is applied to
the `QApplication`, see §5). Primary action bottom-right, cancel to its left.

**Plots** — styled from tokens through one helper (`style_plot(plot, tokens)`);
never call `setBackground`/`setPen` with literals in `main_window.py`.

## 5. Theming architecture

`src/cobra/gui/theme.py` is the single owner of appearance. Target structure:

```python
@dataclass(frozen=True)
class ThemeTokens:
    name: str            # "Sandbank" | "Deepwater"
    dark: bool
    canvas: str
    surface: str
    ...                  # every token in §2 and §3
    series: tuple[str, ...]

SANDBANK = ThemeTokens(...)
DEEPWATER = ThemeTokens(...)

def build_stylesheet(tokens: ThemeTokens) -> str: ...   # one QSS template, tokens substituted
def style_plot(plot: pg.PlotWidget, tokens: ThemeTokens) -> None: ...
def icon(name: str, tokens: ThemeTokens, *, on_fill: bool = False) -> QIcon: ...
```

Mode resolution:

- Setting `appearance/mode` in `QSettings("COBRA", "COBRA")` holds
  `"system" | "light" | "dark"`, default `"system"`.
- `"system"` reads `QGuiApplication.styleHints().colorScheme()` and follows
  `colorSchemeChanged` live. Manual modes ignore the OS.
- The main window offers a toggle (system → light → dark, or a three-way menu)
  in the global controls row; the change is persisted immediately.
- Applying a theme: set the stylesheet on the `QApplication` (not on individual
  windows), then emit a `theme_changed(tokens)` signal that plots and icon
  holders listen to. Widgets must not cache a `ThemeTokens` beyond the current
  theme; re-derive on the signal.
- Dynamic-property changes (`setProperty("actionState", ...)`) are followed by
  `style().unpolish(w); style().polish(w)` as today.

Rules:

- No `setStyleSheet` on individual widgets. Add a selector or property to the
  QSS template instead.
- No new hex literals outside `theme.py`. Add a token if one is missing, in both
  themes, with its contrast checked.
- `theme.py` must remain importable without a display (no widget creation at
  import time) so the token tables and `build_stylesheet` can be unit-tested.
- Keep the two token sets structurally identical; a test compares the field
  sets and asserts every token is defined in both.

## 6. Icons

- Source: `qtawesome`, Material Design Icons set (`"mdi6.*"`). It is an approved
  dependency; add it to `pyproject.toml` and run `uv sync` in the implementation
  step. No other icon fonts, no bundled PNGs.
- Always obtain icons through `theme.icon(name, tokens, on_fill=...)` so colour
  follows the theme: `text_muted` on surfaces, `on_tide` on filled buttons,
  `tide` for tertiary buttons. Re-request icons on `theme_changed`.
- Sizes: 16 px inline in tables/labels, 20 px on standard buttons, 24 px on the
  48 px primary and panel-switch buttons.
- Primary and danger actions keep a text label next to the icon. Icon-only
  buttons are allowed only for secondary/tertiary actions and must have a
  tooltip and `setAccessibleName`.
- Icon vocabulary (extend here, keep it consistent):
  `play` start, `pause` pause, `stop` stop, `folder-open-outline` browse,
  `content-save-outline` save, `plus` add, `delete-outline` remove,
  `pencil-outline` edit, `help-circle-outline` help/tutorial,
  `theme-light-dark` theme toggle, `download-outline` fetch/HF download,
  `chart-line` visualization, `tune-variant` configuration.

## 7. Copy and interaction

- Labels are nouns or short verb phrases in sentence case with units:
  "Center frequency (GHz)", "Max iterations". No trailing colons in form labels
  when the layout already aligns label and field.
- Tooltips come from `help_texts.py`; every new control gets one.
- Destructive actions (stop a run, remove a goal) confirm only when work would
  be lost; otherwise act immediately and make it reversible.
- Long operations run in `worker.py` threads; the UI shows state via the
  primary button's `actionState`, the progress bar, and a status label — never
  via a modal.
- Errors from configuration validation surface inline and in the log panel;
  message boxes are reserved for failures that block the action entirely.
- Keep GUI configuration and headless JSON in lockstep: a field that exists in
  the GUI exists in `RunConfiguration`, and vice versa.

## 8. Process for GUI changes

1. Read `theme.py` and this file. Reuse an existing token, property, or
   component before adding a new one.
2. Make the change with tokens only. If a new visual state is needed, add it to
   the QSS template and to both token sets.
3. Run `.venv/bin/ruff check` and `.venv/bin/ty check` on changed files.
4. Run the token/stylesheet tests (`tests/test_gui_theme.py` once it exists);
   `theme.py` needs no display.
5. Launch the GUI and check the change in **both** themes (toggle via the
   window control or set `appearance/mode`). Confirm contrast, hover, pressed,
   focus, disabled, and the plots. Do not run Xyce or Optuna in the foreground
   for this check.
6. Update `docs/user-guide/gui.md` if anything user-visible changed (new
   control, new theme behaviour, renamed label).

Reviewer checklist: no hex outside `theme.py` · no per-widget `setStyleSheet` ·
both themes updated · icons via `theme.icon` · plots via `style_plot` · colour
never the only signal · sentence case, units, tooltip · docs touched.

## 9. Known deviations in the current GUI

To be resolved in the GUI update step, not incrementally by unrelated changes:

- `theme.py` is light-only with Material-blue literals and duplicates ORCA's file.
- `main_window.py` uses `Qt.GlobalColor.lightGray` for table section rows and
  `"k"`/`"g"`/`"r"` pyqtgraph pens; `_style_plot_for_light_background` should
  become `theme.style_plot`.
- The stylesheet is applied to the main window rather than the application, so
  dialogs created before `apply_theme` may miss it.
- No icons, no theme toggle, no persisted appearance setting.
