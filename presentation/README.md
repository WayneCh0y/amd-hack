# Presentation — Render & Play

The AMD Hackathon (Track 1) video presentation, built with
[Manim CE](https://www.manim.community/) + [manim-slides](https://manim-slides.eertmans.be/).
Slides live in [`slides.py`](slides.py); each scene is one slide.

All commands below are **PowerShell**, run from the repo root
(`C:\Me\Projects\amd-hack`).

---

## Prerequisites (already set up on this machine)

- **Python 3.13** at `C:\Users\Wayne\AppData\Local\Programs\Python\Python313\`
  with `manim` (0.19.1) and `manim-slides` (5.5.2).
  > ⚠️ Manim is **not** in the project's `agent/.venv` — that venv is only for the
  > agent. Use the global Python above to render.
- **LaTeX** (MiKTeX) on PATH — the slides typeset all text with `Tex`.
- **ffmpeg is not required.** Manim 0.19 encodes video through PyAV (bundled),
  so rendering works with nothing extra installed.

If `manim` isn't found directly, use the full path:
`& "C:\Users\Wayne\AppData\Local\Programs\Python\Python313\Scripts\manim.exe"`

---

## 1. Quick preview (static image, fastest)

Renders just the last frame to a PNG — no video encoding. Great for checking layout.

```powershell
manim -s -r 1920,1080 --media_dir presentation\media presentation\slides.py Title
start presentation\media\images\slides\Title_ManimCE_v0.19.1.png
```

> A harmless `Failed to merge basenames` error prints at the very end in `-s` mode
> (it's the manim-slides teardown expecting video files). The PNG is already written
> before it fires — ignore it.

---

## 2. Render the video (MP4)

```powershell
manim -qh --media_dir presentation\media presentation\slides.py Title
```

- `-qh` = high quality (1920×1080, 60 fps). Other quality flags:
  `-ql` (480p, fast draft) · `-qm` (720p) · `-qk` (4K).
- Output: `presentation\media\videos\slides\1080p60\Title.mp4`

**Render + auto-play in one line:**

```powershell
manim -qh -p --media_dir presentation\media presentation\slides.py Title
```

The `-p` flag tells manim to open the finished video in your default player.

---

## 3. Play an already-rendered video

```powershell
start presentation\media\videos\slides\1080p60\Title.mp4
```

`start` hands the file to your default video player (e.g. Windows Media Player).

---

## 4. Interactive slideshow (present live)

Instead of a flat video, drive the deck with your keyboard (arrow keys / space to
advance, `q` to quit):

```powershell
manim-slides render presentation\slides.py Title
manim-slides present Title
```

---

## 5. Multiple slides → one video

As more scenes are added to `slides.py` (e.g. `Title`, `Problem`, `Scoring`, …):

**Render each, then stitch into a single MP4** with manim-slides `convert`
(PyAV-based, so it works without a standalone ffmpeg):

```powershell
manim-slides render presentation\slides.py Title Problem Scoring
manim-slides convert Title Problem Scoring presentation\final.mp4
```

`convert` can also target `.html` (self-contained slideshow) or `.pptx`:

```powershell
manim-slides convert Title Problem Scoring presentation\deck.html
```

---

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `Failed to merge basenames` after `-s` | Harmless manim-slides teardown in image mode; PNG is still written. |
| `'str' object has no attribute 'to_hex'` after a video render | The scene's `background_color` must be a `ManimColor(...)`, not a raw hex string. (Already fixed in `slides.py`.) |
| `manim` not found | Use the full path to `Scripts\manim.exe` under Python 3.13 (see Prerequisites). |
| `latex`/`dvisvgm` errors | Ensure MiKTeX is on PATH; first run may prompt to install missing packages. |
| Standalone `ffmpeg`/`ffprobe` fails (exit 53) | The conda-cache copy is broken — not needed. Manim/manim-slides use PyAV. |
| Stale frames after editing | Delete `presentation\media\` and re-render (or add `--disable_caching`). |
