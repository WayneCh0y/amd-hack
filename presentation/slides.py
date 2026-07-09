"""AMD Developer Hackathon (ACT II) — Track 1 presentation deck.

Rendered with manim-slides (built on Manim CE 0.19). Each slide is a
`self.next_slide()` boundary so the deck can be driven interactively:

    manim-slides render presentation/slides.py Title
    manim-slides Title

Static preview (last frame -> PNG):

    manim -s -r 1920,1080 presentation/slides.py Title

All text is typeset with LaTeX (`Tex`). To keep the modern tech look from
assets/cover.png (rather than Computer Modern serif), the LaTeX template uses
Helvetica sans (`helvet` + `sansmath`) for headings/body and typewriter
(`\\texttt`) for the mono labels. Theme: near-black canvas, deep red radial
glow, crimson primary accent, cyan + green secondary accents.
"""

from manim import *
from manim_slides import Slide

# ── Palette (from assets/cover.png) ────────────────────────────────────────
BG        = "#0A0A0F"   # canvas
GLOW      = "#F5333F"   # red glow tint
RED       = "#F5333F"   # primary accent
RED_SOFT  = "#FF5A63"
CYAN      = "#4FC6EC"   # eyebrow / secondary
GREEN     = "#42C767"   # success / Fireworks
WHITE     = "#F5F7FA"
MUTED     = "#7C828E"   # secondary text
MUTED_HI  = "#9AA0AC"
HAIR      = "#242730"   # hairlines / borders

# ── LaTeX template: Helvetica sans (keeps the grotesque/tech feel) ──────────
TEX = TexTemplate()
TEX.add_to_preamble(
    r"\usepackage{helvet}"
    r"\renewcommand{\familydefault}{\sfdefault}"
    r"\usepackage{sansmath}\sansmath"
)

# Separator dot as sans-math so it also renders inside \texttt runs
# (Computer Modern typewriter has no ·/— glyphs, so use math-mode \cdot).
DOT = r"$\cdot$"   # ·


def tex(body, size, color=WHITE, bold=False):
    """Sans LaTeX text mobject."""
    s = r"\textbf{%s}" % body if bold else body
    return Tex(s, tex_template=TEX, font_size=size, color=color)


def mono(body, size=26, color=MUTED):
    """Monospace (typewriter) LaTeX label."""
    return Tex(r"\texttt{%s}" % body, tex_template=TEX, font_size=size, color=color)


class Title(Slide):
    def construct(self):
        self.camera.background_color = ManimColor(BG)

        # ── Ambient red glow: stacked low-opacity discs pushed off-canvas
        #    to the right → soft radial falloff, no visible hard edge.
        glow = VGroup(*[
            Circle(radius=0.75 * i, color=GLOW, fill_opacity=0.012, stroke_width=0)
            for i in range(11, 0, -1)
        ])
        glow.move_to([6.2, 0.3, 0]).set_z_index(-10)
        self.add(glow)

        # ── Top bar: logo mark + brand (left) · track pill (right) ─────────
        r = 0.19  # isometric cube glyph: pointy-top hexagon + inner "Y"
        hexagon = RegularPolygon(n=6, radius=r, start_angle=PI / 2,
                                 color=WHITE, stroke_width=2.5)
        spokes = VGroup(*[
            Line(ORIGIN, v, color=WHITE, stroke_width=2.5)
            for v in (UP * r,
                      RIGHT * (0.866 * r) + DOWN * (0.5 * r),
                      LEFT * (0.866 * r) + DOWN * (0.5 * r))
        ])
        cube = VGroup(hexagon, spokes)
        logo_box = RoundedRectangle(
            width=0.62, height=0.62, corner_radius=0.16,
            color=RED, fill_color=RED, fill_opacity=1, stroke_width=0,
        )
        cube.move_to(logo_box.get_center())
        logo = VGroup(logo_box, cube)
        brand = VGroup(
            tex("Group", 30, color=WHITE, bold=True),
            tex("AMD Hackathon", 30, color=MUTED_HI),
        ).arrange(RIGHT, buff=0.18)
        header = VGroup(logo, brand).arrange(RIGHT, buff=0.28)
        header.to_corner(UL, buff=0.9).set_y(3.55)

        pill_txt = mono(rf"ACT II \ {DOT}\ TRACK 1 \ {DOT}\ General-Purpose AI Agent",
                        size=24, color=MUTED_HI)
        pill = SurroundingRectangle(
            pill_txt, buff=0.22, corner_radius=0.16,
            color=HAIR, stroke_width=1.5, fill_opacity=0,
        )
        pill_grp = VGroup(pill, pill_txt).to_corner(UR, buff=0.9).set_y(3.55)

        # ── Hero block (left aligned) ──────────────────────────────────────
        eyebrow_line = Line(ORIGIN, RIGHT * 0.5, color=CYAN, stroke_width=3)
        eyebrow_txt = mono("BATCH INFERENCE AGENT", size=28, color=CYAN)
        eyebrow = VGroup(eyebrow_line, eyebrow_txt).arrange(RIGHT, buff=0.3)

        title1 = tex("General-Purpose", 132, color=WHITE, bold=True)
        title2 = tex("AI Agent.", 132, color=RED, bold=True)

        divider = Line(ORIGIN, RIGHT * 4.2, color=HAIR, stroke_width=2)

        presented = mono("PRESENTED BY", size=24, color=MUTED)
        names = VGroup(
            tex("Wayne", 58, color=WHITE, bold=True),
            Dot(radius=0.07, color=RED),
            tex("Jermaine", 58, color=WHITE, bold=True),
        ).arrange(RIGHT, buff=0.4)

        # Stack with per-gap spacing, left edges aligned.
        eyebrow.to_edge(LEFT, buff=0.9)
        title1.next_to(eyebrow, DOWN, buff=0.5, aligned_edge=LEFT)
        title2.next_to(title1, DOWN, buff=0.28, aligned_edge=LEFT)
        divider.next_to(title2, DOWN, buff=0.6, aligned_edge=LEFT)
        presented.next_to(divider, DOWN, buff=0.42, aligned_edge=LEFT)
        names.next_to(presented, DOWN, buff=0.3, aligned_edge=LEFT)

        hero = VGroup(eyebrow, title1, title2, divider, presented, names)
        hero.set_y(-0.15)

        # ── Footer: submission (left) · stack chips (right) ────────────────
        rule = Line(LEFT * 6.4, RIGHT * 6.4, color=HAIR, stroke_width=1).set_y(-3.35)
        # em dash rendered in the sans family (typewriter lacks the glyph)
        foot_left = tex(r"\texttt{ACT II }\,\textemdash\,\texttt{ Track 1 Submission}",
                        22, color=MUTED)
        foot_left.to_corner(DL, buff=0.9).set_y(-3.72)

        def chip(label, col):
            dot = Dot(radius=0.05, color=col)
            t = mono(label, size=22, color=MUTED_HI)
            return VGroup(dot, t).arrange(RIGHT, buff=0.16)

        chips = VGroup(
            chip(rf"Docker \ {DOT}\ linux/amd64", RED),
            chip("Python", CYAN),
            chip("Fireworks AI", GREEN),
        ).arrange(RIGHT, buff=0.45)
        chips.to_corner(DR, buff=0.9).set_y(-3.72)

        # ── Animate ────────────────────────────────────────────────────────
        self.play(
            LaggedStart(
                FadeIn(logo, scale=0.8), FadeIn(brand, shift=RIGHT * 0.2),
                FadeIn(pill_grp, shift=LEFT * 0.2),
                lag_ratio=0.3, run_time=1.0,
            )
        )
        self.play(
            Create(eyebrow_line), FadeIn(eyebrow_txt, shift=RIGHT * 0.2),
            run_time=0.7,
        )
        self.play(
            LaggedStart(
                FadeIn(title1, shift=UP * 0.35),
                FadeIn(title2, shift=UP * 0.35),
                lag_ratio=0.35, run_time=1.0,
            )
        )
        self.play(Create(divider), run_time=0.5)
        self.play(
            FadeIn(presented, shift=UP * 0.15),
            LaggedStart(
                *[FadeIn(n, shift=UP * 0.2) for n in names],
                lag_ratio=0.25,
            ),
            run_time=0.8,
        )
        self.play(
            Create(rule),
            FadeIn(foot_left), FadeIn(chips),
            run_time=0.7,
        )
        self.wait(0.3)
        self.next_slide()
