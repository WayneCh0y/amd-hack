"""AMD Developer Hackathon (ACT II) — Track 1 presentation deck.

Rendered with manim-slides (built on Manim CE 0.19). Each scene is one slide,
closed by `self.next_slide()` so the deck can also be driven interactively.

    Title  ->  Objective  ->  Architecture  ->  Iteration  ->  Results  ->  Closing

See presentation/README.md for the render / stitch / PDF commands.

All text is typeset with LaTeX (`Tex`). To keep the modern tech look from
assets/cover.png (rather than Computer Modern serif), the LaTeX template uses
Helvetica sans (`helvet` + `sansmath`) for headings/body and typewriter
(`\\texttt`) for the mono labels. Theme: near-black canvas, deep red radial
glow, crimson primary accent, cyan + green secondary accents.

Layout grammar, applied consistently across the deck:
  * two-column slides (Objective, Results) stack the headline on two lines and
    left-align a non-flanked eyebrow;
  * full-width slides (Architecture, Iteration) run the headline on one line and
    center a flanked eyebrow, to buy vertical room for the diagram below.
"""

from manim import *
from manim_slides import Slide

# ── Palette (from assets/cover.png) ────────────────────────────────────────
BG        = "#0A0A0F"   # canvas
GLOW      = "#F5333F"   # red glow tint
RED       = "#F5333F"   # primary accent
RED_SOFT  = "#FF5A63"
CYAN      = "#4FC6EC"   # eyebrow / secondary
GREEN     = "#42C767"   # success / zero-cost path
WHITE     = "#F5F7FA"
MUTED     = "#7C828E"   # secondary text
MUTED_HI  = "#9AA0AC"
HAIR      = "#242730"   # hairlines / borders

# ── Pacing ─────────────────────────────────────────────────────────────────
# Animations were originally timed for an interactive talk, where the speaker
# holds each slide; as a flat video they ran too fast to read. PACE stretches
# every animation, BEAT separates the groups within a slide, and each scene
# closes on a `self.wait(...)` sized to its reading load — the dense slides
# (Iteration, Results) hold longest. Raise PACE to slow the whole deck.
PACE = 1.4     # multiplier applied to every run_time
BEAT = 0.35    # short pause between beats inside a slide

# The footer hairline spans `frame_width - 2`, and it is the deck's margin.
# Full-width content (the pipeline, the iteration ledger) is held to the same
# measure so nothing overhangs the rule beneath it.
CONTENT_W = 12.2


def rt(seconds):
    """Scale an animation duration by the deck-wide pacing dial."""
    return seconds * PACE


# ── LaTeX template: Helvetica sans (keeps the grotesque/tech feel) ──────────
TEX = TexTemplate()
TEX.add_to_preamble(
    r"\usepackage{helvet}"
    r"\renewcommand{\familydefault}{\sfdefault}"
    r"\usepackage{sansmath}\sansmath"
)

# Separator dot as sans-math so it also renders inside \texttt runs
# (Computer Modern typewriter has no ·/— glyphs, so use math-mode \cdot).
DOT = r"$\cdot$"           # ·
ARROW = r"$\rightarrow$"   # →


def tex(body, size, color=WHITE, bold=False):
    """Sans LaTeX text mobject."""
    s = r"\textbf{%s}" % body if bold else body
    return Tex(s, tex_template=TEX, font_size=size, color=color)


def mono(body, size=26, color=MUTED):
    """Monospace (typewriter) LaTeX label."""
    return Tex(r"\texttt{%s}" % body, tex_template=TEX, font_size=size, color=color)


def tracked(body):
    """Letterspaced caps: space every glyph, and open the word breaks wider.

    LaTeX has no tracking primitive outside microtype, so insert the spaces by
    hand. The word gap must stay clearly wider than the letter gap or the words
    run together. Caps set this way read as a label rather than as prose, which
    is what the eyebrow and the names want.
    """
    return r"\quad ".join(r"\ \ ".join(w) for w in body.upper().split(" "))


def rule(length, color=HAIR, width=2, opacity=1.0):
    """Horizontal hairline."""
    return Line(
        LEFT * length / 2, RIGHT * length / 2,
        stroke_color=color, stroke_width=width, stroke_opacity=opacity,
    )


def diamond(size=0.13, color=RED):
    """Small rotated square — the separator between the two names."""
    return Square(side_length=size, fill_color=color, fill_opacity=1, stroke_width=0).rotate(PI / 4)


def status(label, color):
    """A colored dot + muted caption, as along the bottom of the cover art."""
    return VGroup(
        Dot(radius=0.045, color=color),
        mono(label, size=20, color=MUTED),
    ).arrange(RIGHT, buff=0.13)


def panel(content, width, accent=HAIR, pad=0.6):
    """The deck's one container: hairline border over a near-transparent fill.

    Every boxed thing on every slide is this shape, so cards, pipeline stages
    and the escalation callout all read as members of the same family. `accent`
    carries the emphasis — HAIR is neutral, a colour means "this one matters".
    """
    return RoundedRectangle(
        width=max(width, content.width + 0.9), height=content.height + pad,
        corner_radius=0.16,
        stroke_color=accent, stroke_width=1.5,
        stroke_opacity=0.55 if accent != HAIR else 1.0,
        fill_color=WHITE, fill_opacity=0.02,
    )


# ── Shared slide chrome ────────────────────────────────────────────────────
# Every slide draws the same backdrop, eyebrow and footer so the deck reads as
# one artifact. Built here once rather than per-scene.

def backdrop(glow_at=ORIGIN):
    """Faint dot grid + red radial glow.

    The grid gives the flat canvas texture; the glow is a stack of low-opacity
    concentric discs, which fakes a radial falloff with no visible hard edge.
    Point the glow at whatever the slide wants looked at.
    """
    fw, fh = config.frame_width, config.frame_height
    grid = VGroup(*[
        Dot(point=[x, y, 0], radius=0.013, color=HAIR, fill_opacity=0.9)
        for x in np.arange(-fw / 2, fw / 2, 0.5)
        for y in np.arange(-fh / 2, fh / 2, 0.5)
    ]).set_z_index(-20)
    glow = VGroup(*[
        Circle(radius=0.75 * i, color=GLOW, fill_opacity=0.012, stroke_width=0)
        for i in range(11, 0, -1)
    ]).move_to(glow_at).set_z_index(-10)
    return grid, glow


def eyebrow(*labels, flanked=False):
    """Cyan tracked caps behind a short rule (both sides when `flanked`).

    Segments are passed separately and joined with a dot here — `tracked()`
    upper-cases what it's given, which would mangle the LaTeX in `DOT`.
    """
    body = (r"\ \ " + DOT + r"\ \ ").join(tracked(l) for l in labels)
    parts = [rule(0.6, CYAN, width=2), tex(body, 20, color=CYAN)]
    if flanked:
        parts.append(rule(0.6, CYAN, width=2))
    return VGroup(*parts).arrange(RIGHT, buff=0.3)


def headline(first, second, size=58, stacked=True):
    """The two-tone headline: a white clause resolving into a crimson one.

    Stacked on two-column slides, inline on full-width ones. The gradient runs
    over glyphs in reading order either way, so the colour lands on `second`.
    """
    a, b = tex(first, size, WHITE, bold=True), tex(second, size, RED, bold=True)
    group = (
        VGroup(a, b).arrange(DOWN, buff=0.16, aligned_edge=LEFT) if stacked
        else VGroup(a, b).arrange(RIGHT, buff=0.36)
    )
    return group.set_color_by_gradient(WHITE, RED_SOFT, RED)


def card(kicker, kicker_color, display, display_color, caption, note, accent=HAIR, width=5.3):
    """A bordered panel: kicker, big display figure, caption, footnote.

    Echoes the pipeline cards on the cover art. An accent border marks the card
    that carries the stakes.
    """
    content = VGroup(
        tex(tracked(kicker), 15, color=kicker_color),
        tex(display, 38, color=display_color),
        mono(caption, size=18, color=MUTED_HI),
        mono(note, size=17, color=MUTED),
    ).arrange(DOWN, buff=0.16, aligned_edge=LEFT)

    box = panel(content, width, accent)
    # Pin to a fixed left inset rather than centering, so the kicker/figure/
    # caption share one margin regardless of which line is widest.
    content.move_to(box).align_to(box, LEFT).shift(RIGHT * 0.45)
    return VGroup(box, content)


def footer():
    """Full-bleed hairline, credit left, tech stack right."""
    foot_rule = rule(config.frame_width - 2.0, HAIR, width=1.5).to_edge(DOWN, buff=0.85)
    credit = mono("AMD Developer Hackathon " + DOT + " ACT II " + DOT + " Track 1", 20, color=MUTED)
    credit.next_to(foot_rule, DOWN, buff=0.28).align_to(foot_rule, LEFT)
    stack = VGroup(
        status("Docker " + DOT + " linux/amd64", RED),
        status("Python", CYAN),
        status("Fireworks AI", GREEN),
    ).arrange(RIGHT, buff=0.5)
    stack.next_to(foot_rule, DOWN, buff=0.24).align_to(foot_rule, RIGHT)
    return foot_rule, credit, stack


def masthead(brow, head, sub, top_buff=0.55, gap=0.34):
    """Centered eyebrow / headline / one-line standfirst, pinned to the top."""
    block = VGroup(brow, head, sub).arrange(DOWN, buff=gap)
    block.to_edge(UP, buff=top_buff)
    return block


class Title(Slide):
    def construct(self):
        self.camera.background_color = ManimColor(BG)

        grid, glow = backdrop(glow_at=[0, 0.4, 0])
        self.add(grid, glow)

        brow = eyebrow("ACT II", "TRACK 1", flanked=True)

        # ── Hero title: one continuous gradient across both lines, so
        #    "General-Purpose" fades warm and "AI Agent." resolves to crimson.
        title = headline("General-Purpose", "AI Agent.", size=108)
        title1, title2 = title

        # ── Divider: hairline with a lit red core at the center ─────────────
        divider = VGroup(
            rule(3.2, HAIR, width=1.5),
            rule(0.8, RED, width=2),
        )

        # ── Names: tracked caps, red diamond between them ───────────────────
        names = VGroup(
            tex(tracked("Wayne"), 34, color=WHITE, bold=True),
            diamond(),
            tex(tracked("Jermaine"), 34, color=WHITE, bold=True),
        ).arrange(RIGHT, buff=0.55)
        subtitle = mono("batch inference agent", 20, color=MUTED)

        # Stack top-down with per-gap spacing, then center the block as a whole.
        title.next_to(brow, DOWN, buff=0.45)
        divider.next_to(title, DOWN, buff=0.6)
        names.next_to(divider, DOWN, buff=0.5)
        subtitle.next_to(names, DOWN, buff=0.3)
        hero = VGroup(brow, title, divider, names, subtitle)
        hero.move_to(UP * 0.35)

        foot_rule, credit, stack = footer()

        # ── Animate ────────────────────────────────────────────────────────
        self.play(FadeIn(grid), run_time=rt(0.5))
        self.play(
            LaggedStart(
                *[GrowFromCenter(r) for r in (brow[0], brow[2])],
                FadeIn(brow[1], shift=DOWN * 0.12),
                lag_ratio=0.2, run_time=rt(0.6),
            )
        )
        self.play(
            LaggedStart(
                FadeIn(title1, shift=UP * 0.3),
                FadeIn(title2, shift=UP * 0.3),
                lag_ratio=0.35, run_time=rt(0.9),
            )
        )
        self.wait(BEAT)
        self.play(GrowFromCenter(divider), run_time=rt(0.5))
        self.play(
            LaggedStart(
                *[FadeIn(n, shift=UP * 0.2) for n in names],
                FadeIn(subtitle, shift=UP * 0.15),
                lag_ratio=0.25, run_time=rt(0.9),
            )
        )
        self.play(
            LaggedStart(
                Create(foot_rule),
                FadeIn(credit, shift=UP * 0.1),
                FadeIn(stack, shift=UP * 0.1),
                lag_ratio=0.3, run_time=rt(0.8),
            )
        )
        self.wait(2.6)
        self.next_slide()


class Objective(Slide):
    """Track 1's objective: the two-stage scoring, and what it implies.

    Content is drawn from documentation/track-1.md — the 80% accuracy gate
    (16 of 19 fixed tasks), then ascending rank by total Fireworks tokens.
    """

    def construct(self):
        self.camera.background_color = ManimColor(BG)

        # Glow pushed right, behind the scoring cards, so it lights the side of
        # the slide that carries the numbers.
        grid, glow = backdrop(glow_at=[3.4, 0.4, 0])
        self.add(grid, glow)

        # ── Left column: the objective, stated ─────────────────────────────
        brow = eyebrow("THE OBJECTIVE")
        head = headline("Enough accuracy.", "Minimum tokens.")
        head1, head2 = head

        body = VGroup(
            tex(r"Not a chatbot --- a \textbf{batch agent}. Read every prompt", 26, color=MUTED_HI),
            tex(r"from \texttt{/input/tasks.json}, solve it, write", 26, color=MUTED_HI),
            tex(r"\texttt{/output/results.json}, exit 0. No frontend, no user.", 26, color=MUTED_HI),
        ).arrange(DOWN, buff=0.16, aligned_edge=LEFT)

        left = VGroup(brow, head, body).arrange(DOWN, buff=0.45, aligned_edge=LEFT)

        # ── Right column: the two scoring stages, gate then rank ───────────
        stage1 = card(
            "STAGE 1", CYAN, r"$\ge$ \textbf{80\%}", RED,
            "accuracy gate " + DOT + " 16 of 19 tasks",
            "below the gate " + ARROW + " off the leaderboard",
            accent=RED,
        )
        stage2 = card(
            "STAGE 2", CYAN, r"\textbf{Fewest tokens}", WHITE,
            "token efficiency " + DOT + " passers only",
            "ranked ascending " + ARROW + " fewest wins",
            accent=HAIR,
        )
        flow = Arrow(
            UP * 0.24, DOWN * 0.24, buff=0,
            stroke_width=2, max_tip_length_to_length_ratio=0.4, color=MUTED,
        )
        right = VGroup(stage1, flow, stage2).arrange(DOWN, buff=0.2)

        columns = VGroup(left, right).arrange(RIGHT, buff=0.9, aligned_edge=UP)
        if columns.width > CONTENT_W:      # hold the deck's margin
            columns.scale_to_fit_width(CONTENT_W)
        columns.move_to(UP * 0.7)

        # ── Bottom band: the eight capability areas the agent must cover ───
        areas = ["factual", "math", "sentiment", "summarisation",
                 "NER", "debugging", "logic", "code gen"]
        items = []
        for i, a in enumerate(areas):
            if i:
                items.append(Line(UP * 0.1, DOWN * 0.1, stroke_color=HAIR, stroke_width=1.5))
            items.append(mono(a, size=19, color=MUTED))
        area_row = VGroup(*items).arrange(RIGHT, buff=0.28)
        area_kicker = tex(tracked("8 capability areas"), 15, color=CYAN)
        areas_block = VGroup(area_kicker, area_row).arrange(DOWN, buff=0.26)
        areas_block.move_to(DOWN * 2.32)

        foot_rule, credit, stack = footer()

        # ── Animate ────────────────────────────────────────────────────────
        self.add(foot_rule, credit, stack)   # chrome carries over from Title
        self.play(
            LaggedStart(
                GrowFromCenter(brow[0]),
                FadeIn(brow[1], shift=RIGHT * 0.15),
                lag_ratio=0.3, run_time=rt(0.5),
            )
        )
        self.play(
            LaggedStart(
                FadeIn(head1, shift=UP * 0.25),
                FadeIn(head2, shift=UP * 0.25),
                lag_ratio=0.35, run_time=rt(0.8),
            )
        )
        self.play(FadeIn(body, shift=UP * 0.15), run_time=rt(0.6))
        self.wait(BEAT)
        self.play(
            LaggedStart(
                FadeIn(stage1, shift=LEFT * 0.25),
                GrowArrow(flow),
                FadeIn(stage2, shift=LEFT * 0.25),
                lag_ratio=0.4, run_time=rt(1.1),
            )
        )
        self.play(
            LaggedStart(
                FadeIn(area_kicker, shift=UP * 0.1),
                *[FadeIn(m, shift=UP * 0.1) for m in area_row],
                lag_ratio=0.06, run_time=rt(0.9),
            )
        )
        self.wait(3.4)
        self.next_slide()


def chip(label):
    """A bare mono path in a hairline pill — the pipeline's input/output ends."""
    text = mono(label, 18, color=MUTED_HI)
    box = SurroundingRectangle(
        text, color=HAIR, corner_radius=0.12, buff=0.22,
        stroke_width=1.5, fill_opacity=0,
    )
    return VGroup(box, text)


def stage(index, name, detail, cost="0 tokens", cost_color=GREEN, width=2.5):
    """One pipeline stage: numbered, named, costed."""
    head = VGroup(
        mono(index, size=19, color=RED),
        tex(name, 25, color=WHITE, bold=True),
    ).arrange(RIGHT, buff=0.18)
    content = VGroup(
        head,
        mono(detail, size=16, color=MUTED),
        mono(cost, size=16, color=cost_color),
    ).arrange(DOWN, buff=0.16)
    box = panel(content, width, HAIR, pad=0.55)
    content.move_to(box)
    return VGroup(box, content)


class Architecture(Slide):
    """The local-first hybrid: classify, answer locally, verify, escalate.

    Straight from documentation/architecture-1.md and agent/src/agent/pipeline.py.
    The point of the slide is the cost annotation — three stages at zero
    Fireworks tokens, and one branch that spends them.
    """

    def construct(self):
        self.camera.background_color = ManimColor(BG)

        # Glow low and central, sitting under the pipeline itself.
        grid, glow = backdrop(glow_at=[0.6, -1.3, 0])
        self.add(grid, glow)

        brow = eyebrow("THE AGENT", flanked=True)
        head = headline("Answer locally.", "Escalate rarely.", size=50, stacked=False)
        # Split by hand: LaTeX wraps a long `Tex` at its own measure, which
        # breaks the line in an ugly place.
        sub = VGroup(
            tex(r"Only tokens through \texttt{FIREWORKS\_BASE\_URL} are scored.", 23, color=MUTED_HI),
            tex("Every verified local answer is free.", 23, color=MUTED_HI),
        ).arrange(DOWN, buff=0.14)
        masthead(brow, head, sub)

        # ── The spine: input, three zero-cost stages, output ───────────────
        src = chip("/input/tasks.json")
        dst = chip("/output/results.json")

        s1 = stage("01", "Classify", "keyword heuristic")
        s2 = stage("02", "Answer", "Qwen2.5-3B, 4-bit")
        s3 = stage("03", "Verify", "format \\& sanity")

        spine = VGroup(src, s1, s2, s3, dst).arrange(RIGHT, buff=0.45)
        # Open the last gap so the green "verified" label has room to sit over
        # the arrow rather than over the boxes on either side of it.
        dst.shift(RIGHT * 0.6)

        arrows = VGroup(*[
            Arrow(
                a.get_right(), b.get_left(), buff=0.1, color=MUTED,
                stroke_width=2, max_tip_length_to_length_ratio=0.32,
            )
            for a, b in zip(spine[:-1], spine[1:])
        ])
        keep = mono("verified", 16, color=GREEN).next_to(arrows[3], UP, buff=0.18)

        # ── The branch: the only place tokens are spent ────────────────────
        esc_head = VGroup(
            tex("Escalate", 25, color=RED_SOFT, bold=True),
            tex(ARROW, 22, color=MUTED),
            tex("Fireworks API", 25, color=WHITE, bold=True),
        ).arrange(RIGHT, buff=0.22)
        esc_content = VGroup(
            esc_head,
            mono("only on failed verification " + DOT + " tokens spent here", 16, color=MUTED),
        ).arrange(DOWN, buff=0.18)
        esc = VGroup(panel(esc_content, 4.6, RED, pad=0.6), esc_content)
        esc_content.move_to(esc[0])
        esc.move_to([(s3.get_center()[0] + dst.get_center()[0]) / 2, -2.05, 0])

        down = Arrow(
            s3.get_bottom(), esc.get_top() + LEFT * 1.2, buff=0.14, color=RED,
            stroke_width=2, max_tip_length_to_length_ratio=0.32,
        )
        rejoin = Arrow(
            esc.get_top() + RIGHT * 1.2, dst.get_bottom(), buff=0.14, color=RED_SOFT,
            stroke_width=2, max_tip_length_to_length_ratio=0.32,
        )
        fails = mono("fails", 16, color=RED).next_to(down, LEFT, buff=0.16)

        diagram = VGroup(spine, arrows, keep, esc, down, rejoin, fails)
        if diagram.width > CONTENT_W:
            diagram.scale_to_fit_width(CONTENT_W)
        diagram.move_to([0, -0.6, 0])

        foot_rule, credit, stack = footer()

        # ── Animate: walk the happy path first, then reveal what it costs ──
        self.add(foot_rule, credit, stack)   # chrome carries over
        self.play(
            LaggedStart(
                *[GrowFromCenter(r) for r in (brow[0], brow[2])],
                FadeIn(brow[1], shift=DOWN * 0.12),
                lag_ratio=0.2, run_time=rt(0.5),
            )
        )
        self.play(FadeIn(head, shift=UP * 0.22), run_time=rt(0.7))
        self.play(FadeIn(sub, shift=UP * 0.12), run_time=rt(0.5))
        self.wait(BEAT)
        self.play(FadeIn(src, shift=RIGHT * 0.2), run_time=rt(0.4))
        self.play(
            LaggedStart(
                *[
                    anim
                    for a, s in zip(arrows[:3], (s1, s2, s3))
                    for anim in (GrowArrow(a), FadeIn(s, shift=RIGHT * 0.2))
                ],
                lag_ratio=0.45, run_time=rt(1.8),
            )
        )
        self.play(
            LaggedStart(
                GrowArrow(arrows[3]),
                FadeIn(keep, shift=DOWN * 0.1),
                FadeIn(dst, shift=RIGHT * 0.2),
                lag_ratio=0.35, run_time=rt(0.9),
            )
        )
        self.wait(BEAT)
        self.play(
            LaggedStart(
                GrowArrow(down),
                FadeIn(fails, shift=RIGHT * 0.1),
                FadeIn(esc, shift=UP * 0.2),
                GrowArrow(rejoin),
                lag_ratio=0.35, run_time=rt(1.4),
            )
        )
        self.wait(3.4)
        self.next_slide()


def ledger_row(index, head, detail, figure, fig_note, width=CONTENT_W):
    """One line of the iteration ledger: what broke, and the number that fixed it.

    Columns are pinned to absolute x positions rather than arranged, so the
    index / body / figure line up across rows regardless of text width.
    """
    idx = mono(index, size=24, color=RED)
    body = VGroup(
        tex(head, 29, color=WHITE, bold=True),
        mono(detail, size=19, color=MUTED),
    ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
    right = VGroup(
        tex(figure, 40, color=RED, bold=True),
        mono(fig_note, size=17, color=MUTED_HI),
    ).arrange(DOWN, buff=0.12, aligned_edge=RIGHT)

    idx.move_to([-width / 2, 0, 0], aligned_edge=LEFT)
    body.move_to([-width / 2 + 0.9, 0, 0], aligned_edge=LEFT)
    right.move_to([width / 2, 0, 0], aligned_edge=RIGHT)
    return VGroup(idx, body, right)


class Iteration(Slide):
    """What the benchmark harness changed our minds about.

    Every figure here is a real measurement recorded in .claude/memory:
    reasoning_effort (-27% completion tokens), the ~12% ceiling on per-category
    model routing, and the 3B model scoring 100% on math / logic / code — the
    three categories we had predicted would need the API.
    """

    def construct(self):
        self.camera.background_color = ManimColor(BG)

        # Glow left, under the numbers column's counterweight.
        grid, glow = backdrop(glow_at=[-3.0, -0.9, 0])
        self.add(grid, glow)

        brow = eyebrow("IMPLEMENTATION", "ITERATION", flanked=True)
        head = headline("We guessed.", "The benchmark disagreed.", size=50, stacked=False)
        sub = tex("Every number below is measured, not assumed.", 23, color=MUTED_HI)
        masthead(brow, head, sub)

        rows = [
            ledger_row(
                "01", "Hidden reasoning ate the token cap",
                "thinking filled the cap, answers came back empty",
                r"$-$27\%", r"with reasoning\_effort = low",
            ),
            ledger_row(
                "02", "Routing between models wasn't the lever",
                "optimal per-category routing vs one frontier model",
                r"12\%", "total token saving",
            ),
            ledger_row(
                "03", "Our hypothesis was backwards",
                "we predicted math, logic and code would need the API",
                r"100\%", "local score on those three",
            ),
        ]
        ys = [0.55, -0.72, -1.99]
        for row, y in zip(rows, ys):
            row.shift(UP * y)
        seps = VGroup(*[
            rule(CONTENT_W, HAIR, width=1.2).move_to([0, y, 0])
            for y in ((ys[0] + ys[1]) / 2, (ys[1] + ys[2]) / 2)
        ])

        foot_rule, credit, stack = footer()

        # ── Animate: each row lands as a unit, separated by a hairline ─────
        self.add(foot_rule, credit, stack)   # chrome carries over
        self.play(
            LaggedStart(
                *[GrowFromCenter(r) for r in (brow[0], brow[2])],
                FadeIn(brow[1], shift=DOWN * 0.12),
                lag_ratio=0.2, run_time=rt(0.5),
            )
        )
        self.play(FadeIn(head, shift=UP * 0.22), run_time=rt(0.7))
        self.play(FadeIn(sub, shift=UP * 0.12), run_time=rt(0.5))
        self.wait(BEAT)
        for i, row in enumerate(rows):
            idx, body, right = row
            self.play(
                LaggedStart(
                    FadeIn(idx, shift=RIGHT * 0.15),
                    FadeIn(body, shift=RIGHT * 0.15),
                    FadeIn(right, shift=LEFT * 0.15),
                    lag_ratio=0.3, run_time=rt(0.85),
                )
            )
            if i < len(seps):
                self.play(Create(seps[i]), run_time=rt(0.35))
        self.wait(4.0)
        self.next_slide()


# Per-category results from the 40-task local benchmark (see
# .claude/memory/track1-local-model-results.md). Five tasks per category.
LOCAL_SCORES = [
    ("math", 5), ("logic", 5), ("code debug", 5), ("code gen", 5),
    ("factual", 4), ("sentiment", 4), ("summarisation", 4), ("NER", 4),
]
GATE_HITS = 4          # 80% of 5 tasks — the pass mark for a category
DOT_GAP = 0.72
DOT_R = 0.13
ROW_GAP = 0.58


def score_row(label, hits, total=5):
    """A label and `total` dots — filled where the local model was correct."""
    lab = mono(label, size=21, color=MUTED_HI)
    lab.move_to([-0.62, 0, 0], aligned_edge=RIGHT)
    marks = VGroup(*[
        (
            Dot(radius=DOT_R, color=GREEN) if i < hits
            else Circle(radius=DOT_R, stroke_color=HAIR, stroke_width=3, fill_opacity=0)
        ).move_to([i * DOT_GAP, 0, 0])
        for i in range(total)
    ])
    return VGroup(lab, marks)


class Results(Slide):
    """The outcome: 36 of 40 locally, every category over the gate.

    The dot matrix is the argument — the block left of the gate line is solid,
    which is precisely what "all eight categories pass" looks like.
    """

    def construct(self):
        self.camera.background_color = ManimColor(BG)

        grid, glow = backdrop(glow_at=[3.6, 0.2, 0])
        self.add(grid, glow)

        # ── Left column: the claim ────────────────────────────────────────
        brow = eyebrow("THE RESULT")
        head = headline(r"90\% accuracy.", "Zero tokens.")
        head1, head2 = head

        body = VGroup(
            tex("The bundled 3B model clears all eight", 25, color=MUTED_HI),
            tex(r"categories alone --- 36 of 40 correct.", 25, color=MUTED_HI),
            tex("Fireworks is the fallback, not the default.", 25, color=MUTED_HI),
        ).arrange(DOWN, buff=0.16, aligned_edge=LEFT)

        caveat = VGroup(
            mono("measured on our own 40-task harness", 17, color=MUTED),
            mono("the live gate is an LLM judge over 19 unseen tasks", 17, color=MUTED),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)

        left = VGroup(brow, head, body, caveat).arrange(DOWN, buff=0.42, aligned_edge=LEFT)

        # ── Right column: per-category dot matrix with the gate marked ─────
        # Keep this short: a long tracked run overflows LaTeX's measure and wraps.
        kicker = tex(tracked("5 tasks per category"), 15, color=CYAN)
        matrix = VGroup(*[score_row(l, h) for l, h in LOCAL_SCORES])
        for i, row in enumerate(matrix):
            row.shift(DOWN * i * ROW_GAP)

        gate_x = (GATE_HITS - 0.5) * DOT_GAP
        gate_line = DashedLine(
            [gate_x, matrix.get_top()[1] + 0.22, 0],
            [gate_x, matrix.get_bottom()[1] - 0.22, 0],
            dash_length=0.11, stroke_width=1.6, color=RED, stroke_opacity=0.7,
        )
        # `tracked()` letterspaces per glyph, which would split the backslash
        # off "\%" — so the gate label stays a plain mono run.
        gate_label = mono(r"80\% gate", 17, color=RED)
        gate_label.next_to(gate_line, DOWN, buff=0.16)

        right = VGroup(kicker, VGroup(matrix, gate_line, gate_label).center())
        right.arrange(DOWN, buff=0.34)

        # Flush both columns to the deck's measure rather than centring the
        # pair, so this slide's outer margins match Objective's and the footer's.
        edge = (config.frame_width - CONTENT_W) / 2
        left.to_edge(LEFT, buff=edge)
        right.to_edge(RIGHT, buff=edge)
        right.align_to(left, UP)
        columns = VGroup(left, right)
        columns.move_to(UP * 0.35, coor_mask=np.array([0, 1, 0]))

        foot_rule, credit, stack = footer()

        # ── Animate: state the claim, then let the matrix prove it ────────
        self.add(foot_rule, credit, stack)   # chrome carries over
        self.play(
            LaggedStart(
                GrowFromCenter(brow[0]),
                FadeIn(brow[1], shift=RIGHT * 0.15),
                lag_ratio=0.3, run_time=rt(0.5),
            )
        )
        self.play(
            LaggedStart(
                FadeIn(head1, shift=UP * 0.25),
                FadeIn(head2, shift=UP * 0.25),
                lag_ratio=0.35, run_time=rt(0.8),
            )
        )
        self.play(FadeIn(body, shift=UP * 0.15), run_time=rt(0.6))
        self.wait(BEAT)
        self.play(FadeIn(kicker, shift=UP * 0.1), run_time=rt(0.4))
        self.play(
            LaggedStart(
                *[FadeIn(row, shift=RIGHT * 0.18) for row in matrix],
                lag_ratio=0.28, run_time=rt(1.7),
            )
        )
        self.play(
            LaggedStart(
                Create(gate_line),
                FadeIn(gate_label, shift=UP * 0.08),
                lag_ratio=0.4, run_time=rt(0.8),
            )
        )
        self.play(FadeIn(caveat, shift=UP * 0.1), run_time=rt(0.5))
        self.wait(3.8)
        self.next_slide()


class Closing(Slide):
    """Bookend of Title: same hero geometry, same divider, same names."""

    def construct(self):
        self.camera.background_color = ManimColor(BG)

        grid, glow = backdrop(glow_at=[0, 0.4, 0])
        self.add(grid, glow)

        brow = eyebrow("TRACK 1", "SUBMISSION", flanked=True)

        thanks = tex("Thank you.", 100, color=WHITE, bold=True)
        thanks.set_color_by_gradient(WHITE, RED_SOFT, RED)
        sub = tex("A 3B model, a verifier, and an API we barely call.", 26, color=MUTED_HI)

        divider = VGroup(rule(3.2, HAIR, width=1.5), rule(0.8, RED, width=2))
        names = VGroup(
            tex(tracked("Wayne"), 34, color=WHITE, bold=True),
            diamond(),
            tex(tracked("Jermaine"), 34, color=WHITE, bold=True),
        ).arrange(RIGHT, buff=0.55)
        repo = mono("github.com/WayneCh0y/amd-hack", 20, color=MUTED)

        thanks.next_to(brow, DOWN, buff=0.45)
        sub.next_to(thanks, DOWN, buff=0.4)
        divider.next_to(sub, DOWN, buff=0.55)
        names.next_to(divider, DOWN, buff=0.5)
        repo.next_to(names, DOWN, buff=0.3)
        hero = VGroup(brow, thanks, sub, divider, names, repo)
        hero.move_to(UP * 0.35)

        foot_rule, credit, stack = footer()

        self.add(foot_rule, credit, stack)   # chrome carries over
        self.play(
            LaggedStart(
                *[GrowFromCenter(r) for r in (brow[0], brow[2])],
                FadeIn(brow[1], shift=DOWN * 0.12),
                lag_ratio=0.2, run_time=rt(0.6),
            )
        )
        self.play(FadeIn(thanks, shift=UP * 0.3), run_time=rt(0.9))
        self.play(FadeIn(sub, shift=UP * 0.15), run_time=rt(0.6))
        self.wait(BEAT)
        self.play(GrowFromCenter(divider), run_time=rt(0.5))
        self.play(
            LaggedStart(
                *[FadeIn(n, shift=UP * 0.2) for n in names],
                FadeIn(repo, shift=UP * 0.15),
                lag_ratio=0.25, run_time=rt(0.9),
            )
        )
        self.wait(3.0)
        self.next_slide()
