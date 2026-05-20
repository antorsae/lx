#!/usr/bin/env python3
"""
Generate docs landing pages from synced docs/ output.

This keeps docs/index.html and docs/lx521-system.html in sync with
the current measurement sets and generated plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import config

DOCS_DIR = Path("docs")
SYSTEM_SET = "lx521-system"

SET_TITLES = {
    "andres": "Andres Measurements",
    "juan-baffleless": "Juan Baffleless Drivers",
    SYSTEM_SET: "LX521.4 Complete System",
}

SET_DESCRIPTIONS = {
    "andres": "4 Drivers in the LX521.4 baffle. No EQ/cross-overs applied.",
    "juan-baffleless": (
        "Baffleless drivers measured in free air. Dipole behavior expected. "
        "Includes rear measurements for complete polar pattern visualization."
    ),
}

DRIVER_ROLES = {
    "10F8424": "Upper Mid",
    "L22MG": "Lower Mid",
    "MU10": "Upper Mid",
    "MU10RB-SL": "Upper Mid",
    "SEAS27T": "Tweeter",
    "SS10F8424G00": "Upper Mid",
}


def detect_drivers(set_docs_dir: Path) -> List[str]:
    interactive_dir = set_docs_dir / "interactive"
    drivers = []
    for path in interactive_dir.glob("*_freq_response_angles.html"):
        stem = path.stem
        drivers.append(stem.replace("_freq_response_angles", ""))
    return sorted(drivers)


def link_list(items: Sequence[Tuple[str, str]], indent: str = " " * 8) -> str:
    if not items:
        return ""
    lines = [f'{indent}<ul class="link-list">']
    for href, label in items:
        lines.append(f'{indent}    <li><a href="{href}">{label}</a></li>')
    lines.append(f"{indent}</ul>")
    return "\n".join(lines)


def image_grid(items: Sequence[Tuple[str, str, str, str]], indent: str = " " * 8) -> str:
    if not items:
        return ""
    lines = [f'{indent}<div class="image-grid">']
    for href, src, alt, caption in items:
        lines.append(f'{indent}    <div class="image-card">')
        lines.append(f'{indent}        <a href="{href}" target="_blank">')
        lines.append(f'{indent}            <img src="{src}" alt="{alt}">')
        lines.append(f'{indent}            <div class="caption">{caption}</div>')
        lines.append(f'{indent}        </a>')
        lines.append(f'{indent}    </div>')
    lines.append(f"{indent}</div>")
    return "\n".join(lines)


def render_interactive_card(
    set_name: str,
    set_cfg: Dict,
    *,
    wrap_card: bool = True,
    include_header: bool = True,
    include_driver_tags: bool = True,
) -> str:
    set_docs_dir = DOCS_DIR / set_name
    interactive_dir = set_docs_dir / "interactive"
    has_rear = bool(set_cfg.get("has_rear"))

    drivers = detect_drivers(set_docs_dir)
    title = SET_TITLES.get(set_name, set_name.replace("-", " ").title())
    subtitle = f"{len(drivers)} Drivers - " + (
        "Full 360 deg coverage" if has_rear else "Front hemisphere (0-90 deg)"
    )
    description = SET_DESCRIPTIONS.get(set_name)

    header_lines: List[str] = []
    body_lines: List[str] = []
    if include_header:
        if wrap_card:
            header_lines = [
                '                <div class="card-header">',
                f"                    <h2>{title}</h2>",
                f'                    <div class="subtitle">{subtitle}</div>',
                "                </div>",
            ]
        else:
            body_lines.append(f"                <h2>{title}</h2>")
            body_lines.append(f'                <div class="subtitle">{subtitle}</div>')

        if description:
            body_lines.append(
                f'                    <p style="margin-bottom: 1rem; color: var(--text-muted);">{description}</p>'
            )

    # Polar & Directivity
    polar_items: List[Tuple[str, str]] = []
    if (interactive_dir / "polar/polar_explorer.html").exists():
        polar_items.append(
            (f"{set_name}/interactive/polar/polar_explorer.html",
             "Polar Explorer (360 deg)" if has_rear else "Polar Explorer")
        )
    if (interactive_dir / "di_comparison.html").exists():
        polar_items.append((f"{set_name}/interactive/di_comparison.html", "Directivity Index"))
    if (interactive_dir / "beamwidth_comparison.html").exists():
        polar_items.append((f"{set_name}/interactive/beamwidth_comparison.html", "Beamwidth"))

    if polar_items:
        body_lines.append("                    <h3>Polar & Directivity</h3>")
        body_lines.append(link_list(polar_items))

    # Frequency response explorer + per-driver angles
    freq_items: List[Tuple[str, str]] = []
    if (interactive_dir / "freq_response_explorer.html").exists():
        freq_items.append(
            (f"{set_name}/interactive/freq_response_explorer.html",
             "<strong>Multi-Driver Explorer</strong>")
        )
    for driver in drivers:
        fname = f"{driver}_freq_response_angles.html"
        if (interactive_dir / fname).exists():
            freq_items.append((f"{set_name}/interactive/{fname}", driver))

    if freq_items:
        body_lines.append("                    <h3>Frequency Response by Angle</h3>")
        body_lines.append(link_list(freq_items))

    # Crossover analysis (if present)
    crossover_files = sorted(interactive_dir.glob("crossover_*.html"))
    if crossover_files:
        cross_items = []
        for f in crossover_files:
            label = f.stem.replace("_", " ")
            cross_items.append((f"{set_name}/interactive/{f.name}", label.title()))
        body_lines.append("                    <h3>Crossover Analysis</h3>")
        body_lines.append(link_list(cross_items))

    # Contour plots
    norm_items = []
    abs_items = []
    for driver in drivers:
        norm_fname = f"{driver}_contour_normalized.html"
        abs_fname = f"{driver}_contour_absolute.html"
        if (interactive_dir / norm_fname).exists():
            caption = driver
            if driver in DRIVER_ROLES:
                caption += f" ({DRIVER_ROLES[driver]})"
            norm_items.append((f"{set_name}/interactive/{norm_fname}", caption))
        if (interactive_dir / abs_fname).exists():
            caption = driver
            if driver in DRIVER_ROLES:
                caption += f" ({DRIVER_ROLES[driver]})"
            abs_items.append((f"{set_name}/interactive/{abs_fname}", caption))

    if norm_items:
        body_lines.append("                    <h3>Contour Plots (Normalized)</h3>")
        body_lines.append(link_list(norm_items))
    if abs_items:
        body_lines.append("                    <h3>Contour Plots (Absolute)</h3>")
        body_lines.append(link_list(abs_items))

    # Measurement summary
    if (interactive_dir / "measurement_summary.html").exists():
        body_lines.append("                    <h3>Measurement Details</h3>")
        body_lines.append(
            link_list([(f"{set_name}/interactive/measurement_summary.html", "Measurement Summary")])
        )

    if drivers and include_driver_tags:
        tags = " ".join(f'<span class="driver-tag">{d}</span>' for d in drivers)
        body_lines.append(f'                    <div class="drivers">{tags}</div>')

    if not wrap_card:
        return "\n".join(line for line in body_lines if line)

    card_lines = [
        '            <div class="card">',
        *header_lines,
        '                <div class="card-body">',
        *body_lines,
        "                </div>",
        "            </div>",
    ]
    return "\n".join(line for line in card_lines if line)


def render_static_section(set_name: str, set_cfg: Dict) -> str:
    set_docs_dir = DOCS_DIR / set_name
    static_dir = set_docs_dir / "static_plots"
    has_rear = bool(set_cfg.get("has_rear"))
    drivers = detect_drivers(set_docs_dir)
    title = SET_TITLES.get(set_name, set_name.replace("-", " ").title())

    lines = []
    header = f"Static Plots - {title} (PNG)"
    if has_rear:
        header += " - 360 deg"
    lines.append(f'        <h2 class="section-title">{header}</h2>')

    # Core analysis images
    core_items = []
    for fname, caption in [
        ("di_comparison.png", "Directivity Index" if set_name != "andres" else "Directivity Index Comparison"),
        ("beamwidth_comparison.png", "Beamwidth Comparison" if set_name == "andres" else "Beamwidth"),
        ("dipole_null_analysis.png", "Dipole Null Analysis"),
    ]:
        path = static_dir / "core" / fname
        if path.exists():
            rel = f"{set_name}/static_plots/core/{fname}"
            core_items.append((rel, rel, caption, caption))

    if core_items:
        lines.append("        <h3>Core Analysis</h3>")
        lines.append(image_grid(core_items, indent=" " * 8))

    # Polar diagrams
    polar_items = []
    gallery = static_dir / "polar" / "polar_gallery_overlaid.png"
    if gallery.exists():
        rel = f"{set_name}/static_plots/polar/polar_gallery_overlaid.png"
        caption = "All Drivers Overlaid" if set_name == "andres" else "Polar Gallery"
        if has_rear:
            caption += " (360 deg)"
        polar_items.append((rel, rel, "Polar Gallery", caption))
    for driver in drivers:
        fname = f"{driver}_polar_circular.png"
        path = static_dir / "polar" / fname
        if path.exists():
            rel = f"{set_name}/static_plots/polar/{fname}"
            caption = driver
            role = DRIVER_ROLES.get(driver)
            if role:
                caption += f" ({role})"
            if has_rear:
                caption += " (360 deg)"
            polar_items.append((rel, rel, f"{driver} Polar", caption))

    if polar_items:
        heading = "Polar Diagrams"
        if has_rear:
            heading += " (360 deg)"
        lines.append(f"        <h3>{heading}</h3>")
        lines.append(image_grid(polar_items, indent=" " * 8))

    # Contour normalized / absolute
    for suffix, heading in [
        ("contour_normalized", "Contour Plots (Normalized)"),
        ("contour_absolute", "Contour Plots (Absolute)"),
    ]:
        items = []
        for driver in drivers:
            fname = f"{driver}_{suffix}.png"
            path = static_dir / "core" / fname
            if path.exists():
                rel = f"{set_name}/static_plots/core/{fname}"
                caption = driver
                role = DRIVER_ROLES.get(driver)
                if role:
                    caption += f" ({role})"
                items.append((rel, rel, fname, caption))
        if items:
            lines.append(f"        <h3>{heading}</h3>")
            lines.append(image_grid(items, indent=" " * 8))

    # Frequency response by angle / normalized
    for suffix, heading in [
        ("freq_response_angles", "Frequency Response by Angle"),
        ("freq_response_normalized", "Frequency Response Normalized (0° = ref)"),
    ]:
        items = []
        for driver in drivers:
            fname = f"{driver}_{suffix}.png"
            path = static_dir / "core" / fname
            if path.exists():
                rel = f"{set_name}/static_plots/core/{fname}"
                items.append((rel, rel, fname, driver))
        if items:
            lines.append(f"        <h3>{heading}</h3>")
            lines.append(image_grid(items, indent=" " * 8))

    # Crossover analysis (if present)
    crossover_dir = static_dir / "crossover"
    if crossover_dir.exists():
        items = []
        for f in sorted(crossover_dir.glob("crossover_*.png")):
            rel = f"{set_name}/static_plots/crossover/{f.name}"
            caption = f.stem.replace("_", " ").title()
            items.append((rel, rel, f.name, caption))
        if items:
            lines.append("        <h3>Crossover Analysis</h3>")
            lines.append(image_grid(items, indent=" " * 8))

    return "\n".join(lines)


def write_index():
    non_system_sets = [k for k in config.MEASUREMENT_SETS.keys() if k != SYSTEM_SET]

    interactive_cards = "\n".join(render_interactive_card(s, config.MEASUREMENT_SETS[s]) for s in non_system_sets)
    static_sections = "\n".join(render_static_section(s, config.MEASUREMENT_SETS[s]) for s in non_system_sets)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LX521 Polar Analysis</title>
    <link rel="stylesheet" href="assets/css/styles.css">
</head>
<body>
    <header>
        <h1>LX521 Polar Analysis</h1>
        <p>Comprehensive polar response measurements and directivity analysis</p>
    </header>

    <main>
        <div class="card" style="margin-bottom: 2rem; border-left: 4px solid #7c3aed;">
            <div class="card-body">
                <h3>Complete System Measurements</h3>
                <p style="margin-bottom: 1rem; color: var(--text-muted);">
                    View polar response analysis of the complete LX521.4 system with all drivers summed.
                </p>
                <ul class="link-list">
                    <li><a href="lx521-system.html">LX521.4 Complete System (Juan's measurements, 360 deg)</a></li>
                </ul>
            </div>
        </div>

        <div class="card" style="margin-bottom: 2rem; border-left: 4px solid var(--primary);">
            <div class="card-body">
                <h3>Download Raw Measurements (REW .mdat files)</h3>
                <p style="margin-bottom: 1rem; color: var(--text-muted);">
                    Download the original REW measurement files to run your own analysis.
                    Available from the <a href="https://github.com/antorsae/lx/releases/latest">latest release</a>.
                </p>
                <ul class="link-list">
                    <li><a href="https://github.com/antorsae/lx/releases/latest/download/andres-measurements.zip">Andres Measurements (4 drivers, 0-90 deg)</a></li>
                </ul>
            </div>
        </div>

        <h2 class="section-title">Interactive Plots (HTML)</h2>
        <div class="measurement-sets">
{interactive_cards}
        </div>

{static_sections}
    </main>

    <footer>
        <p><a href="https://github.com/antorsae/lx">Source Code</a></p>
    </footer>
</body>
</html>
"""

    (DOCS_DIR / "index.html").write_text(html)


def write_system_page():
    set_cfg = config.MEASUREMENT_SETS[SYSTEM_SET]
    set_docs_dir = DOCS_DIR / SYSTEM_SET
    drivers = detect_drivers(set_docs_dir)
    system_name = drivers[0] if drivers else "LX521 System"
    angles = ", ".join(str(a) for a in set_cfg.get("angles", []))

    interactive_body = render_interactive_card(
        SYSTEM_SET,
        set_cfg,
        wrap_card=False,
        include_header=False,
        include_driver_tags=False,
    )
    static_section = render_static_section(SYSTEM_SET, set_cfg)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LX521.4 Complete System - Polar Analysis</title>
    <link rel="stylesheet" href="assets/css/styles.css">
    <style>
        :root {{
            --primary: #7c3aed;
            --primary-dark: #6d28d9;
            --highlight-bg: #f3e8ff;
            --highlight-border: #7c3aed;
            --highlight-text: #6d28d9;
        }}
    </style>
</head>
<body>
    <header>
        <h1>LX521.4 Complete System</h1>
        <p>Full system polar response measurements (all drivers summed)</p>
        <a href="index.html" class="back-link">Back to Main Page</a>
    </header>

    <main>
        <div class="card">
            <div class="card-header">
                <h2>Measurement Details</h2>
                <div class="subtitle">Juan's measurements - Full 360 deg coverage</div>
            </div>
            <div class="card-body">
                <div class="highlight">
                    <strong>Complete System:</strong> These measurements capture the LX521.4 with all drivers active and crossover applied.
                </div>
                <p><strong>Configuration:</strong> {system_name}</p>
                <p><strong>Angles:</strong> {angles} degrees</p>
                <p><strong>Coverage:</strong> Front + Rear measurements (full 360 deg)</p>
            </div>
        </div>

        <h2 class="section-title">Interactive Plots (HTML)</h2>
        <div class="card">
            <div class="card-body">
{interactive_body}
            </div>
        </div>

{static_section}
    </main>

    <footer>
        <p><a href="https://github.com/antorsae/lx">Source Code</a> | <a href="index.html">Back to Main Page</a></p>
    </footer>
</body>
</html>
"""

    (DOCS_DIR / "lx521-system.html").write_text(html)


def main():
    DOCS_DIR.mkdir(exist_ok=True)
    write_index()
    if (DOCS_DIR / SYSTEM_SET).exists():
        write_system_page()


if __name__ == "__main__":
    main()
