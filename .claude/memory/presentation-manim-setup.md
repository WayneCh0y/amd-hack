---
name: presentation-manim-setup
description: How the Track-1 video presentation is built and rendered (manim/manim-slides) — env, commands, and gotchas
metadata:
  type: project
---

The hackathon **video presentation** (text-only, no voiceover) lives in `presentation/slides.py`,
built with **Manim CE 0.19.1 + manim-slides 5.5.2**. Theme derived from `assets/cover.png`
(near-black canvas, red `#F5333F` primary, cyan `#4FC6EC`, green `#42C767`). Six `manim_slides.Slide`
scenes, each closed by `self.next_slide()`: **Title → Objective → Architecture → Iteration → Results
→ Closing**. Deliverables: `presentation/final.mp4` (1080p60, ~62 s) and `presentation/deck.pdf`
(6 pages). **Render / stitch / PDF commands live in `presentation/README.md` — read it, don't
re-derive.**

**Use the GLOBAL Python 3.13 as a module, never the bare console scripts:**
`C:\Users\Wayne\AppData\Local\Programs\Python\Python313\python.exe` has manim + manim-slides + img2pdf
(manim is NOT in the agent `.venv`). Both `manim.exe` and `manim-slides.exe` first on PATH are
**Anaconda's and broken** — bare `manim --version` tracebacks, `manim-slides` raises
`numpy.dtype size changed`. Always `& $py -m manim ...` / `& $py -m manim_slides ...`.

**Key env facts / gotchas:**
- Manim 0.19 encodes via **PyAV (bundled libav), NOT an external ffmpeg**. The only standalone ffmpeg
  present (`anaconda3\pkgs\ffmpeg-8.1.1...`) is **broken (exit 53, missing DLLs)** — validate mp4s
  with PyAV instead (`import av; av.open(path)`).
- **`manim-slides convert` has no MP4 writer** (HTML/PDF/PPTX only), and its `concatenate_video_files`
  hands ffmpeg's concat demuxer a backslash path → spurious `FileNotFoundError` on Windows. Hence
  `presentation/build_video.py`, which remuxes the per-scene MP4s through PyAV; with no args it
  stitches the whole deck in order.
- **PDF export needs `img2pdf`** (installed) and must run *after* a video render — `convert` reads
  `slides/<Scene>.json` + the clips under `slides/files/`, else `Cannot merge an empty list of files`.
  Each page is that slide's last frame.
- Under `-s` (image) mode, teardown raises `Failed to merge basenames` — harmless (the PNG is already
  written) but it **aborts the process**, so only the first scene renders. Preview one scene at a time.
- Set `self.camera.background_color = ManimColor(BG)` (NOT a raw hex string) or teardown throws
  `'str' object has no attribute 'to_hex'` after a video render.
- `LaggedStartMap(FadeIn, group, shift=...)` silently drops `shift`; build
  `LaggedStart(*[FadeIn(m, shift=...) for m in group])` instead.
- **LaTeX (all text is `Tex`):** the `tracked()` letterspacing helper inserts `\ \ ` between glyphs and
  therefore **mangles escaped characters** — never pass it `\%` or `$\cdot$` (track each plain segment
  and join, as `eyebrow()` does). A long `Tex` string wraps at LaTeX's own measure in an ugly place —
  split body copy into one `Tex` per line.
- Pacing is one dial at the top of `slides.py`: `PACE` (run_time multiplier, currently 1.4) + `BEAT` +
  a per-scene trailing `self.wait()` sized to reading load. Layout measure is `CONTENT_W = 12.2`, equal
  to the footer hairline, so no full-width element overhangs it.
