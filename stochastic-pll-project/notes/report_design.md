# Report design

Reading this as a standalone scientific teaching and research report for a university reader, in an academic editorial visual language. Dials: **ENERGY 1 / RHYTHM 2 / MOTION 1**.

- Ivory page, ink type, one dark teal accent; dark mode uses warm charcoal and pale text. Serif reading text keeps long derivations comfortable; sans serif controls and headings give navigational clarity. Monospace is confined to code and filenames.
- A narrow fixed contents rail accompanies a generous reading column. Sections alternate prose/derivation, full-width figure, and precise tabular evidence according to content. No generic promotional cards, shadows, gradient ornament or invented claims.
- Section letters and numbered mathematical explanations are the identity motif. Rules and spacing provide hierarchy. The title and current section heading carry the focal point.
- Responsive tables and equations scroll within their own frames. At narrow widths the contents rail becomes an explicit toggle. Controls use visible labels, sufficient touch targets and a visible focus ring. Native details preserve keyboard access.
- Motion is limited to state changes, respects reduced-motion preference and is unnecessary for comprehension. Theme changes are immediate. No animated charts or scroll effects.
- SVG diagrams illustrate mechanisms and are explicitly conceptual. Measured plots use actual saved output only. Captions separate observation, interpretation and limits.
- Search hides whole nonmatching sections, reports the match count, exposes a no-results/reset state and keeps the matching navigation synchronized. The initial no-JavaScript document remains readable.
- MathJax 3.2.2 tex-svg is saved locally under report/vendor and copied inline into the HTML. SVG mathematical glyphs avoid external font requests. Tables, sources, source-page images, JavaScript, CSS and scientific figures are embedded.

Verification responsibilities: report generator validates required executed artifacts, records embedded asset hashes and checks that image/script/style references are internal. Root agent performs browser verification of offline network behavior, typography, both themes, search/empty state, diagrams, actual-data plot controls, collapse controls, mobile layout and absence of horizontal page overflow. Never label this design gate passed before those checks are recorded.
