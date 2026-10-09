#!/usr/bin/env python3
"""Generate every SVG in assets/ for the GitHub profile README.

Run from the repository root:
    pip install pillow numpy
    python scripts/generate_assets.py                        # default cloud + lambda
    python scripts/generate_assets.py --image foto.png       # your own image
    python scripts/generate_assets.py --image foto.png --invert   # light subject on dark bg

VISUAL.MAP loops: image -> each logo in LOGOS (assets/source/logos/<name>.png,
black silhouette on transparent) -> image. Missing logo files are skipped.

Writes (dark + light each): banner-*.v<N>.svg, whoami-*.svg, radar-*.svg, radar-langs-*.svg
Texts live in PROFILE, SKILLS, LANGS and whoami(); radar values are self-rated 0-100.
"""

from __future__ import annotations

import argparse
import html
from collections import deque
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BANNER_VERSION = 10  # bump after changing the banner so caches fetch the new file
ASSETS = ROOT / "assets"
SEED = 1611
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
SANS = "ui-sans-serif,-apple-system,Segoe UI,Helvetica,Arial,sans-serif"
LOGO_DIR = ASSETS / "source" / "logos"
LOGOS = ("astro", "react", "python", "aws")
LOOP_BEGIN, PORTRAIT_HOLD, MOVE, LOGO_HOLD = 2.4, 3.0, 1.3, 3.0  # seconds
TRAVELLERS = 600

# (indent, key, value) rows shown in the banner's vim profile.yml panel.
PROFILE = [
    (0, "profile", ""),
    (1, "subject", "Johan Zuluaga"),
    (1, "role", "Backend & Data Developer (AWS)"),
    (1, "origin", "San Luis, Antioquia, Colombia"),
    (1, "focus", "Pipelines de datos · Serverless · Scraping"),
    (1, "status", "Full Stack MERN certificado"),
    (1, "learning", "AWS Certified Data Engineer"),
    (0, "stack", ""),
    (1, "aws", "Lambda · S3 · SQS · EventBridge · ECR"),
    (1, "backend", "Python · Node.js · Express · FastAPI"),
    (1, "data", "MySQL (RDS) · PostgreSQL · MongoDB"),
    (1, "frontend", "React · React Native · Astro"),
    (1, "automation", "Puppeteer · ETL · Gemini · OCR"),
    (0, "contact", ""),
    (1, "linkedin", "/in/johan-zuluaga-870343257"),
    (1, "github", "JohanZ1611"),
    (1, "email", "johan16zulu@gmail.com"),
]

SKILLS = ("Backend & Data Skill Radar", [
    ("AWS Serverless", 85),
    ("Pipelines & ETL", 85),
    ("Web Scraping", 90),
    ("Backend & APIs", 85),
    ("Frontend & Mobile", 78),
    ("IA (Gemini / LLMs)", 72),
])

LANGS = ("Language Stack", [
    ("Python", 90),
    ("JavaScript", 88),
    ("TypeScript", 82),
    ("SQL", 80),
    ("Kotlin", 45),
])

# Blue / sky palette only.
THEMES = {
    "dark": {
        "bg": "#0A0F1E", "panel": "#0D1628", "panel2": "#101B30", "line": "#25344C",
        "muted": "#8291A8", "text": "#E6EDF3", "dots": "#58A6FF", "chrome": "#7DD3FC",
        "key": "#58A6FF", "accent": "#38BDF8", "warm": "#BAE6FD", "shadow": "#02050B",
    },
    "light": {
        "bg": "#EEF4FF", "panel": "#FFFFFF", "panel2": "#F3F8FF", "line": "#C8D8F0",
        "muted": "#64748B", "text": "#0F1B2D", "dots": "#1F6FEB", "chrome": "#0369A1",
        "key": "#1F6FEB", "accent": "#0284C7", "warm": "#1E40AF", "shadow": "#A8C0E0",
    },
}

# --------------------------------------------------------------------------- #
# VISUAL.MAP dither
# --------------------------------------------------------------------------- #

GRID_W, GRID_H = 300, 340


def default_ink() -> np.ndarray:
    """Cloud silhouette with a lambda cut out, shaded so the dither has texture."""
    s = 4  # supersample, then shrink for smooth edges
    img = Image.new("L", (GRID_W * s, GRID_H * s), 0)
    d = ImageDraw.Draw(img)
    for cx, cy, r in ((105, 180, 52), (160, 150, 70), (215, 178, 50), (75, 205, 32), (240, 207, 32)):
        d.ellipse([(cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s], fill=255)
    d.rounded_rectangle([75 * s, 180 * s, 240 * s, 238 * s], radius=29 * s, fill=255)
    w = 15 * s  # lambda strokes
    d.line([(130 * s, 112 * s), (190 * s, 222 * s)], fill=0, width=w)
    d.line([(160 * s, 167 * s), (128 * s, 222 * s)], fill=0, width=w)
    d.line([(122 * s, 112 * s), (138 * s, 112 * s)], fill=0, width=w)
    mask = np.asarray(img.resize((GRID_W, GRID_H), Image.Resampling.LANCZOS), np.float32) / 255
    yy, xx = np.mgrid[0:GRID_H, 0:GRID_W]
    light = 1 - np.hypot(xx - 120, yy - 120) / 260  # light source top-left
    return mask * np.clip(0.35 + 0.65 * light, 0, 1)


def edge_connected(mask: np.ndarray) -> np.ndarray:
    """True cells of ``mask`` reachable from the image border (the background)."""
    out = np.zeros(mask.shape, bool)
    h, w = mask.shape
    queue = deque([(y, x) for y in range(h) for x in (0, w - 1)] + [(y, x) for x in range(w) for y in (0, h - 1)])
    while queue:
        y, x = queue.popleft()
        if 0 <= y < h and 0 <= x < w and mask[y, x] and not out[y, x]:
            out[y, x] = True
            queue.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
    return out


def image_inks(path: Path, invert: bool) -> dict[str, np.ndarray]:
    """Per-theme ink so the photo reads like a photo in both modes: shadows are
    dots on the light banner, highlights are dots on the dark one. The flat
    background (white, or black with --invert) never gets dots."""
    src = Image.open(path).convert("RGBA")
    w, h = src.size
    crop_w = min(w, int(h * GRID_W / GRID_H))
    crop_h = int(crop_w * GRID_H / GRID_W)
    left, top = (w - crop_w) // 2, max(0, (h - crop_h) // 3)
    src = src.crop((left, top, left + crop_w, top + crop_h)).resize((GRID_W, GRID_H), Image.Resampling.LANCZOS)
    bg = Image.new("RGBA", src.size, "black" if invert else "white")
    bg.alpha_composite(src)
    gray = ImageOps.autocontrast(ImageOps.grayscale(bg.convert("RGB")), cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.3).filter(ImageFilter.UnsharpMask(2, 150, 1))
    lum = np.asarray(gray, np.float32) / 255
    subject = ~edge_connected(lum < 0.08 if invert else lum > 0.92)
    # Cap at .85 so solid areas (a black shirt) keep a dithered texture.
    return {
        "light": np.clip(1 - lum, 0, 0.85) * subject,
        "dark": np.clip(lum, 0, 0.85) * subject,
    }


def floyd_steinberg(ink: np.ndarray) -> np.ndarray:
    """Serpentine 1-bit error diffusion; True = dot."""
    work = ink.astype(np.float32).copy()
    out = np.zeros(work.shape, bool)
    h, w = work.shape
    for y in range(h):
        step = 1 if y % 2 == 0 else -1
        for x in (range(w) if step == 1 else range(w - 1, -1, -1)):
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = new > 0
            err = old - new
            if 0 <= x + step < w:
                work[y, x + step] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - step < w:
                    work[y + 1, x - step] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= x + step < w:
                    work[y + 1, x + step] += err * 1 / 16
    return out


def logo_mask(path: Path) -> np.ndarray:
    """Silhouette fitted into a 230px box, centred on the 300x340 grid."""
    icon = Image.open(path).convert("RGBA")
    icon.thumbnail((230, 230), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (GRID_W, GRID_H), (0, 0, 0, 0))
    canvas.alpha_composite(icon, ((GRID_W - icon.width) // 2, (GRID_H - icon.height) // 2))
    return np.asarray(canvas.getchannel("A")) > 127


def ordered(points: np.ndarray) -> np.ndarray:
    """Order points in y-bands, then x within each band. Travellers pair up by
    index across frames, so this keeps each dot moving to a nearby spot."""
    # ponytail: band sort instead of optimal transport (scipy); swap in
    # linear_sum_assignment if the morph ever looks tangled.
    idx = np.argsort(points[:, 1], kind="stable")
    bands = [b[np.argsort(points[b, 0], kind="stable")] for b in np.array_split(idx, 28)]
    return points[np.concatenate(bands)]


def runs_path(points: list[tuple[int, int]]) -> str:
    """Merge horizontally adjacent 1px dots into short path runs."""
    pts = sorted(points, key=lambda p: (p[1], p[0]))
    out, i = [], 0
    while i < len(pts):
        x0, y = pts[i]
        x1 = x0
        i += 1
        while i < len(pts) and pts[i][1] == y and pts[i][0] == x1 + 1:
            x1 = pts[i][0]
            i += 1
        out.append(f"M{x0} {y}.5h{x1 - x0 + 1}")  # .5 = stroke centred on the pixel row
    return "".join(out)


# --------------------------------------------------------------------------- #
# banner (vim profile.yml)
# --------------------------------------------------------------------------- #


def txt(x, y, s, fill, size=13, extra=""):
    return f'<text x="{x}" y="{y}" fill="{fill}" font-family="{MONO}" font-size="{size}" {extra}>{s}</text>'


def banner(theme: str, dots: np.ndarray, logos: list[np.ndarray]) -> str:
    t = THEMES[theme]
    ox, oy = 94, 161  # centre the 300x340 grid in the 390x414 frame
    ys, xs = np.nonzero(dots)
    rng = np.random.default_rng(SEED)
    groups = rng.integers(0, 48, len(xs))
    starts = rng.permutation(np.linspace(0.05, 1.6, 48))

    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="610" viewBox="0 0 1180 610" '
        'role="img" aria-labelledby="title desc">',
        '<title id="title">Johan Zuluaga — Backend &amp; Data Developer</title>',
        '<desc id="desc">Animated vim-style terminal with a dithered visual map and a YAML profile.</desc>',
        '<defs><filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
        f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" flood-opacity=".28"/></filter>'
        '<clipPath id="vclip"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>'
        f'<linearGradient id="scan" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{t["chrome"]}" stop-opacity="0"/>'
        f'<stop offset="1" stop-color="{t["chrome"]}" stop-opacity=".12"/></linearGradient></defs>',
        f'<rect width="1180" height="610" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/><circle cx="59" cy="38" r="6" fill="#FEBC2E"/>'
        '<circle cx="80" cy="38" r="6" fill="#28C840"/>',
        txt(590, 43, "vim profile.yml", t["muted"], 13, 'text-anchor="middle" letter-spacing=".4"'),
        # left frame
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        txt(49, 111, "VISUAL.MAP", t["chrome"], 13, 'font-weight="700" letter-spacing="1.2"'),
        txt(438, 111, "300×340 / 1-BIT", t["muted"], 11, 'text-anchor="end"'),
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12M439 539h-12M439 539v-12" '
        f'fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        '<g clip-path="url(#vclip)" shape-rendering="crispEdges">',
    ]
    # Portrait layer: fades out while the logos play (loop starts after the intro).
    if logos:
        times = [0.0, PORTRAIT_HOLD]
        for _ in logos:
            times += [times[-1] + MOVE, times[-1] + MOVE + LOGO_HOLD]
        times.append(times[-1] + MOVE)
        loop = f'begin="{LOOP_BEGIN}s" dur="{times[-1]:g}s" repeatCount="indefinite" '                f'keyTimes="{";".join(f"{v / times[-1]:.4f}" for v in times)}"'
        fade = ["1", "1"] + ["0"] * (len(times) - 3) + ["1"]
        p.append(f'<g><animate attributeName="opacity" {loop} values="{";".join(fade)}"/>')
    # Scattered intro: interleaved dot groups fade in once, then stay.
    for g in range(48):
        sel = groups == g
        pts = [(int(x) + ox, int(y) + oy) for x, y in zip(xs[sel], ys[sel])]
        if pts:
            p.append(f'<path d="{runs_path(pts)}" fill="none" stroke="{t["dots"]}" stroke-width="1" opacity="0">'
                     f'<animate attributeName="opacity" begin="{starts[g]:.2f}s" dur=".8s" values="0;.92" fill="freeze"/></path>')
    if logos:
        p.append("</g>")
        portrait = np.column_stack((xs, ys)).astype(float)
        frames = [ordered(portrait[rng.choice(len(portrait), TRAVELLERS, replace=False)])] * 2
        for k, mask in enumerate(logos):
            ly, lx = np.nonzero(mask)
            pts = np.column_stack((lx, ly)).astype(float)
            frames += [ordered(pts[rng.choice(len(pts), TRAVELLERS)])] * 2
            # Dense dithered logo takes over while the travellers hold still.
            hy, hx = np.nonzero(floyd_steinberg(mask * 0.6))
            show = ["0"] * len(times)
            show[2 + 2 * k] = show[3 + 2 * k] = ".9"
            p.append(f'<path d="{runs_path([(int(x) + ox, int(y) + oy) for x, y in zip(hx, hy)])}" fill="none" '
                     f'stroke="{t["dots"]}" stroke-width="1" opacity="0">'
                     f'<animate attributeName="opacity" {loop} values="{";".join(show)}"/></path>')
        frames.append(frames[0])
        fly = ";".join(["0", "0"] + ["1"] * (len(times) - 3) + ["0"])
        for i in range(TRAVELLERS):
            pos = ";".join(f"{f[i, 0] + ox:g} {f[i, 1] + oy:g}" for f in frames)
            p.append(f'<path d="M0 0h1.4v1.4h-1.4z" fill="{t["dots"]}" opacity="0">'
                     f'<animateTransform attributeName="transform" type="translate" {loop} values="{pos}"/>'
                     f'<animate attributeName="opacity" {loop} values="{fly}"/></path>')
    # Looping scanline over the map.
    p.append(f'<rect x="49" y="0" width="390" height="30" fill="url(#scan)">'
             '<animate attributeName="y" values="78;540" dur="5s" begin="2.4s" repeatCount="indefinite"/></rect></g>')
    p.append(txt(58, 551, f"PTS {len(xs):05d} · FS/SERPENTINE", t["muted"], 10))

    # right panel: YAML in vim
    p += [
        f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
        txt(490, 111, "profile.yml", t["chrome"], 13, 'font-weight="700" letter-spacing=".5"'),
        txt(580, 111, "[YAML]", t["muted"], 11),
        f'<rect x="996" y="94" width="132" height="24" rx="12" fill="{t["chrome"]}" fill-opacity=".14" stroke="{t["chrome"]}"/>',
        txt(1062, 111, "@JohanZ1611", t["chrome"], 13, 'text-anchor="middle" font-weight="700"'),
    ]
    y = 148.0
    for i, (indent, key, value) in enumerate(PROFILE, 1):
        if indent == 0:
            body, x = f'<tspan fill="{t["chrome"]}" font-weight="700">{html.escape(key)}:</tspan>', 525
        else:
            body = (f'<tspan fill="{t["key"]}">{html.escape(key)}: </tspan>'
                    f'<tspan fill="{t["text"]}">{html.escape(value)}</tspan>')
            x = 542
        p.append(txt(506, f"{y:g}", f"{i:2d}", t["muted"], 13, 'text-anchor="end" opacity=".5"'))
        p.append(txt(x, f"{y:g}", body, t["text"]))
        y += 21.5
    nbytes = len("\n".join(("  " * i) + k + ": " + v for i, k, v in PROFILE).encode())
    p += [
        # blinking block cursor on the line after the YAML
        f'<rect x="525" y="{y - 13:g}" width="8" height="16" fill="{t["accent"]}">'
        '<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;.5;.5;1" dur="1.1s" repeatCount="indefinite"/></rect>',
        f'<path d="M474 526H1146" stroke="{t["line"]}"/>',
        f'<rect x="475" y="527" width="670" height="32" fill="{t["panel"]}"/>',
        f'<rect x="485" y="533" width="72" height="20" rx="3" fill="{t["dots"]}"/>',
        txt(521, 547, "NORMAL", t["bg"], 11, 'text-anchor="middle" font-weight="700"'),
        txt(569, 547, "profile.yml", t["text"], 12, 'font-weight="600"'),
        txt(740, 547, "[utf-8]", t["muted"], 11),
        txt(1134, 547, f"{len(PROFILE)}L, {nbytes}B  100%  {len(PROFILE)}:1", t["muted"], 11, 'text-anchor="end"'),
        "</svg>",
    ]
    return "".join(p)


# --------------------------------------------------------------------------- #
# whoami terminal card
# --------------------------------------------------------------------------- #


def whoami(theme: str) -> str:
    t = THEMES[theme]
    card = "#111827" if theme == "dark" else "#F8FBFF"
    inner = "#162033" if theme == "dark" else "#FFFFFF"
    soft = "#B6C2D4" if theme == "dark" else "#334155"

    def T(x, y, s, fill, size=17, bold=False):
        return txt(x, y, html.escape(s), fill, size, 'font-weight="700"' if bold else "")

    return "".join([
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 390" width="960" height="390" role="img" '
        'aria-labelledby="title desc"><title id="title">Johan Zuluaga — Backend &amp; Data Developer</title>'
        '<desc id="desc">Terminal-style profile card: mission and areas of expertise.</desc>',
        f'<defs><linearGradient id="sky" x1="0" x2="1" y1="0" y2="1"><stop stop-color="{t["dots"]}"/>'
        f'<stop offset="1" stop-color="{t["chrome"]}"/></linearGradient>'
        '<filter id="glow" x="-20%" y="-30%" width="140%" height="160%"><feGaussianBlur stdDeviation="2.2" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>',
        f'<rect width="960" height="390" rx="18" fill="{card}"/>',
        '<rect x="9" y="9" width="942" height="372" rx="13" fill="none" stroke="url(#sky)" stroke-width="2"/>',
        f'<circle cx="35" cy="35" r="5" fill="{t["dots"]}"/><circle cx="54" cy="35" r="5" fill="{t["accent"]}"/>'
        f'<circle cx="73" cy="35" r="5" fill="{t["chrome"]}"/>',
        T(98, 41, "JohanZ1611  ·  profile shell  ·  online", t["muted"], 15),
        f'<path d="M29 58H931" stroke="{t["line"]}"/>',
        f'<rect x="29" y="78" width="594" height="271" rx="9" fill="{inner}" stroke="{t["dots"]}" stroke-width="1.5"/>',
        T(48, 111, "❯ whoami", t["accent"], 20, True),
        T(48, 143, "johan_zuluaga", soft, 18), T(196, 143, "—", t["muted"], 18),
        T(221, 143, "Backend & Data Developer", t["warm"], 18),
        f'<path d="M48 166H604" stroke="{t["line"]}"/>',
        T(48, 196, "context:", t["muted"]), T(166, 196, "San Luis, Antioquia · Colombia", t["dots"]),
        T(48, 225, "mission:", t["muted"]), T(166, 225, "datos confiables, de la fuente al dashboard", t["chrome"]),
        T(48, 267, "❯ ls expertise/", t["accent"], 20, True),
        T(48, 299, "data+cloud/", t["dots"], 16), T(190, 299, "AWS Serverless · ETL · Scraping · Python", soft, 16),
        T(48, 326, "fullstack/", t["dots"], 16), T(190, 326, "MERN · React Native · Astro · Gemini", soft, 16),
        # right panel: night-sky runtime (blue moon over waves)
        f'<rect x="651" y="78" width="280" height="271" rx="9" fill="{inner}" stroke="{t["dots"]}" stroke-width="1.5"/>',
        T(674, 111, "serverless runtime", t["warm"]),
        f'<circle cx="791" cy="203" r="56" fill="{t["dots"]}" opacity=".95"/>',
        f'<path d="M674 202H908M674 216H908M674 230H908M674 244H908M674 258H908" stroke="{inner}" stroke-width="5" opacity=".9"/>',
        f'<path d="M674 277C711 247 746 303 785 272S861 249 908 281" fill="none" stroke="{t["chrome"]}" stroke-width="3" filter="url(#glow)"/>',
        f'<path d="M674 294C716 264 746 320 789 289S862 270 908 298" fill="none" stroke="{t["accent"]}" stroke-width="2"/>',
        T(674, 326, "status: aprendiendo siempre", t["muted"], 14),
        "</svg>",
    ])


# --------------------------------------------------------------------------- #
# radar charts (ported from macu-dev/scripts/radar.py, blue palette)
# --------------------------------------------------------------------------- #

RADAR = {
    "dark": {"grid": "#30363d", "spoke": "#21262d", "label": "#c9d1d9", "value": "#8b949e",
             "title": "#e6edf3", "fill": "#2F81F7", "stroke": "#58A6FF"},
    "light": {"grid": "#d0d7de", "spoke": "#e6eaef", "label": "#1f2328", "value": "#57606a",
              "title": "#1f2328", "fill": "#1F6FEB", "stroke": "#0969DA"},
}


def ring(r, n):
    return [(r * math.cos(-math.pi / 2 + i * 2 * math.pi / n), r * math.sin(-math.pi / 2 + i * 2 * math.pi / n))
            for i in range(n)]


def radar(title: str, axes: list[tuple[str, float]], theme: str, size=360, rings=4, values=False) -> str:
    c, n = RADAR[theme], len(axes)
    r, gap, LBL, VAL, TTL, pad = size / 2 - 8, 20, 13, 11, 15, 10
    vals = [max(0.0, min(100.0, v)) for _, v in axes]
    outer = ring(r, n)
    labels = []
    for i, (label, v) in enumerate(axes):
        cs, sn = outer[i][0] / r, outer[i][1] / r
        anchor = "middle" if abs(cs) < 0.25 else ("start" if cs > 0 else "end")
        dy = 4 if abs(sn) < 0.25 else (14 if sn > 0 else -5)
        labels.append(((r + gap) * cs, (r + gap) * sn + dy, anchor, label, vals[i]))
    minx, maxx, miny, maxy = -r, r, -r, r
    for lx, ly, anchor, label, v in labels:
        w = len(label) * LBL * 0.62
        x0 = lx if anchor == "start" else lx - w if anchor == "end" else lx - w / 2
        minx, maxx = min(minx, x0), max(maxx, x0 + w)
        miny, maxy = min(miny, ly - LBL), max(maxy, ly + 4 + (VAL + 4 if values else 0))
    th = TTL + 14
    W, H = round(maxx - minx + 2 * pad), round(maxy - miny + 2 * pad + th)
    ox, oy = -minx + pad, -miny + pad + th
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="{html.escape(title)}" font-family="{SANS}">',
         f'<text x="{W / 2:.1f}" y="{pad + TTL}" text-anchor="middle" font-size="{TTL}" font-weight="700" '
         f'fill="{c["title"]}">{html.escape(title)}</text>',
         f'<g transform="translate({ox:.1f},{oy:.1f})">']
    for k in range(rings, 0, -1):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in ring(r * k / rings, n))
        p.append(f'<polygon points="{pts}" fill="none" stroke="{c["grid"]}" opacity="{0.35 + 0.5 * k / rings:.2f}"/>')
    p += [f'<line x1="0" y1="0" x2="{x:.1f}" y2="{y:.1f}" stroke="{c["spoke"]}"/>' for x, y in outer]
    shape = [(x * v / 100, y * v / 100) for (x, y), v in zip(outer, vals)]
    # No grow-in animation: viewers that freeze SMIL would show it stuck at 4%.
    p.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in shape)}" fill="{c["fill"]}" '
             f'fill-opacity="0.22" stroke="{c["stroke"]}" stroke-width="2.5" stroke-linejoin="round"/>')
    p += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{c["stroke"]}"/>' for x, y in shape]
    for lx, ly, anchor, label, v in labels:
        p.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="{LBL}" font-weight="600" '
                 f'fill="{c["label"]}">{html.escape(label)}</text>')
        if values:
            p.append(f'<text x="{lx:.1f}" y="{ly + VAL + 4:.1f}" text-anchor="{anchor}" font-size="{VAL}" '
                     f'fill="{c["value"]}">{v:g}</text>')
    p.append("</g></svg>")
    return "".join(p)


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, help="logo/photo for the VISUAL.MAP panel (default: cloud + lambda)")
    ap.add_argument("--invert", action="store_true", help="use when the subject is light on a dark background")
    args = ap.parse_args()

    inks = image_inks(args.image, args.invert) if args.image else dict.fromkeys(THEMES, default_ink())
    dots = {theme: floyd_steinberg(ink) for theme, ink in inks.items()}
    assert all(d.any() for d in dots.values()), "the image produced no dots; try --invert"
    logos = [logo_mask(LOGO_DIR / f"{n}.png") for n in LOGOS if (LOGO_DIR / f"{n}.png").exists()]
    print(f"VISUAL.MAP loop: image -> {len(logos)} logos")

    ASSETS.mkdir(exist_ok=True)
    for theme in THEMES:
        out = {
            f"banner-{theme}.v{BANNER_VERSION}.svg": banner(theme, dots[theme], logos),
            f"whoami-{theme}.svg": whoami(theme),
            f"radar-{theme}.svg": radar(SKILLS[0], SKILLS[1], theme),
            f"radar-langs-{theme}.svg": radar(LANGS[0], LANGS[1], theme, size=340, values=True),
        }
        for name, svg in out.items():
            (ASSETS / name).write_text(svg, encoding="utf-8")
            print(f"assets/{name}: {len(svg.encode()) / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
