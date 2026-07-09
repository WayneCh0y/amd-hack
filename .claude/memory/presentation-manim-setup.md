---
name: presentation-manim-setup
description: How the Track-1 video presentation is built and rendered (manim/manim-slides) — env, commands, and gotchas
metadata:
  type: project
---

The hackathon **video presentation** (text-only, no voiceover) lives in `presentation/slides.py`,
built with **Manim CE 0.19.1 + manim-slides 5.5.2**. Theme is derived from `assets/cover.png`
(near-black canvas, red `#F5333F` primary, cyan `#4FC6EC`, green `#42C767`, Segoe UI Black headlines
over Cascadia Mono labels). First slide implemented: scene `Title` (names Wayne · Jermaine, task
"General-Purpose AI Agent"). Slides are `manim_slides.Slide` subclasses using `self.next_slide()`.

**Rendering (use the GLOBAL Python 3.13, NOT the agent `.venv` — manim isn't in the agent venv):**
`C:\Users\Wayne\AppData\Local\Programs\Python\Python313\` has manim + manim-slides.
- Static PNG preview (fast): `manim -s -r 1920,1080 presentation/slides.py Title`
- Video (1080p60 mp4): `manim -qh presentation/slides.py Title` → `media/videos/slides/1080p60/Title.mp4`

**Key env facts / gotchas:**
- Manim 0.19 encodes video via **PyAV (bundled libav), NOT an external ffmpeg** — no ffmpeg on PATH
  needed to render. The only standalone ffmpeg present (`anaconda3\pkgs\ffmpeg-8.1.1...\Library\bin`)
  is **broken (exit 53, missing DLLs)** — don't rely on it for probe/convert; validate mp4s with a
  PyAV script instead (`import av; av.open(path)`).
- Under `-s` (image) mode, `manim_slides.Slide` teardown raises a harmless
  `Failed to merge basenames` — the PNG is already written; ignore it.
- Set `self.camera.background_color = ManimColor(BG)` (NOT a raw hex string) or manim-slides teardown
  throws `'str' object has no attribute 'to_hex'` after a video render.
- `LaggedStartMap(FadeIn, group, shift=...)` fails — `shift` isn't forwarded per-anim; build
  `LaggedStart(*[FadeIn(m, shift=...) for m in group])` instead.
- For the final non-interactive video, `next_slide()` boundaries just play continuously under plain
  `manim`; stitch multiple scenes with `manim-slides convert` (PyAV-based) or one big Scene.
