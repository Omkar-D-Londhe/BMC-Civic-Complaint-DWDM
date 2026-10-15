---
name: Obsidian Precision Engine
colors:
  surface: '#131313'
  surface-dim: '#131313'
  surface-bright: '#393939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1b1b1b'
  surface-container: '#1f1f1f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353535'
  on-surface: '#e2e2e2'
  on-surface-variant: '#c1c6d7'
  inverse-surface: '#e2e2e2'
  inverse-on-surface: '#303030'
  outline: '#8b90a0'
  outline-variant: '#414754'
  surface-tint: '#aec6ff'
  primary: '#aec6ff'
  on-primary: '#002e6b'
  primary-container: '#0070f3'
  on-primary-container: '#ffffff'
  inverse-primary: '#0059c5'
  secondary: '#a7c8ff'
  on-secondary: '#003061'
  secondary-container: '#3291ff'
  on-secondary-container: '#002a55'
  tertiary: '#ffb596'
  on-tertiary: '#581e00'
  tertiary-container: '#ca4e00'
  on-tertiary-container: '#fffeff'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#d8e2ff'
  primary-fixed-dim: '#aec6ff'
  on-primary-fixed: '#001a43'
  on-primary-fixed-variant: '#004397'
  secondary-fixed: '#d5e3ff'
  secondary-fixed-dim: '#a7c8ff'
  on-secondary-fixed: '#001b3c'
  on-secondary-fixed-variant: '#004788'
  tertiary-fixed: '#ffdbcd'
  tertiary-fixed-dim: '#ffb596'
  on-tertiary-fixed: '#360f00'
  on-tertiary-fixed-variant: '#7d2d00'
  background: '#131313'
  on-background: '#e2e2e2'
  surface-variant: '#353535'
  surface-pitch: '#000000'
  surface-charcoal: '#0a0a0a'
  surface-subtle: '#111111'
  surface-elevated: '#171717'
  border-subtle: '#1f1f1f'
  border-contrast: '#2e2e2e'
  text-primary: '#ffffff'
  text-secondary: '#ededed'
  text-muted: '#888888'
  text-faint: '#444444'
  accent-cyan: '#3291ff'
  status-success: '#10b981'
  status-warning: '#f5a623'
  status-critical: '#e00000'
typography:
  display-hero:
    fontFamily: Geist
    fontSize: 56px
    fontWeight: '600'
    lineHeight: 64px
    letterSpacing: -0.04em
  display-hero-mobile:
    fontFamily: Geist
    fontSize: 36px
    fontWeight: '600'
    lineHeight: 44px
    letterSpacing: -0.03em
  headline-lg:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.03em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Geist
    fontSize: 20px
    fontWeight: '500'
    lineHeight: 28px
    letterSpacing: -0.02em
  headline-sm:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '500'
    lineHeight: 24px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: -0.01em
  body-md:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: -0.005em
  body-sm:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0em
  code-md:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0em
  annotation-mono:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.04em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1.5rem
  gutter-mobile: 1rem
  margin: 2rem
  margin-mobile: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.5rem
---

## Brand & Style

This design system reflects hyper-engineered modern software craftsmanship: uncompromising, stark, and architectural. Designed for engineers, researchers, and technical leaders, the aesthetic rejects ornamental distraction in favor of mathematical rigor, rapid visual processing, and high-density information display.

The visual style is characterized by deep pitch-black canvases, razor-thin structural delineation, and intense typographic contrast. Surfaces rely on subtle charcoal tiers, micro-borders, and tactile glass accents that suggest cutting-edge laboratory precision. The emotional tone is authoritative, quiet, fast, and intensely competent.

## Colors

The palette is anchored in absolute pitch black (`#000000`) and tiered dark charcoal foundations. Chromatic usage is disciplined and surgical: color is never applied purely for decoration; it strictly communicates telemetry, operational status, and decisive calls to action.

### Palette Hierarchy & Intent
- **Canvas & Backgrounds:** The base viewport is pure `#000000`. Structural tiers transition to `#0a0a0a` and `#111111` for inset panels or cards.
- **Micro-Borders:** Boundaries between components rely on precise 1px strokes of `#1f1f1f` for resting states, shifting to `#2e2e2e` for elevated containers and interactive hover bounds.
- **Monochrome Typography:** Primary copy commands maximum clarity with pure `#ffffff` and `#ededed`. Supporting information, metadata, and structural framing drop back to `#888888` and `#444444` to enforce strict scannability.
- **Precision Accents:** `#0070F3` acts as the primary interactive and brand beacon, energized by `#3291FF` for focused states and high-priority highlights.
- **Telemetry Indicators:** System states are communicated with pinpoint emerald (`#10B981`), caution amber (`#F5A623`), and surgical red (`#E00000`).

## Typography

Typographic execution centers on razor-sharp legibility and mathematical spacing. Geist provides high geometric consistency across prose and headings, while a specialized monospaced tier handles technical parameters, metric keys, telemetry readouts, and runtime logs.

### Application Rules
- **Negative Tracking:** Headings use tight negative letter-spacing (`-0.02em` to `-0.04em`) to create a cohesive, dense editorial presence characteristic of high-performance developer documentation.
- **Monospace Lab Annotations:** Metadata tags, hash representations, commit references, and status metrics must strictly use the label tier (`annotation-mono`). Monospace typography should be paired with uppercase styling when conveying structural tags or keys.
- **Vertical Alignment:** Text frames adhere strictly to regular line-height baselines to avoid sub-pixel jitter in dense tabular views.

## Layout & Spacing

The layout model adheres to an explicit 12-column structural grid system built upon 8px base increments. Structural scaffolding emphasizes linear orientation, hairline column delimiters, and strict alignment across complex multi-panel dashboards.

### Responsive Behavior
- **Desktop (1024px+):** 12-column layout with 24px (`1.5rem`) gutters and outer canvas margins of 32px (`2rem`). Panels align along persistent 1px grid guides.
- **Tablet (768px – 1023px):** 8-column layout with 20px gutters. Secondary sidebars collapse into toggleable vertical slide-outs or pinned iconic rails.
- **Mobile (< 768px):** 4-column layout with 16px (`1rem`) gutters and 16px margins. Complex tabular grids reflow into vertically stacked cards separated by fine horizontal dividing rules.

## Elevation & Depth

Visual hierarchy is constructed through luminosity tiers and micro-line boundaries rather than heavy, diffuse drop shadows. Surfaces feel engineered, flat, and crystalline.

### Structural Layering
1. **Base (0dp):** Pitch black canvas (`#000000`). Used for global viewport framing and root containers.
2. **Sunken Surface (-1dp):** `#050505` with inset 1px inner borders (`#1a1a1a`), reserved for code editors, logs, and technical telemetry outputs.
3. **Elevated Surface (1dp):** `#0a0a0a` framed by a 1px solid `#1f1f1f` border. Used for standard cards, table containers, and navigational headers.
4. **Interactive Hover (2dp):** Background shifts slightly to `#111111` with the border brightening to `#2e2e2e`. Shadows are restrained to an ultra-subtle directional wash: `0 4px 20px rgba(0, 0, 0, 0.7)`.
5. **Overlays & Dialogs (3dp):** Translucent backdrop blur (`backdrop-filter: blur(12px)`) combined with an opaque `#0a0a0a/90` fill, framed by a high-contrast micro-border `#2e2e2e` and a hard perimeter shadow: `0 16px 40px rgba(0, 0, 0, 0.8)`.

## Shapes

The design system enforces a precise, soft geometry defined by `roundedness: 1`. Corners are subtly softened to prevent visual harshness while preserving the architectural, technical feel of an analytical workspace.

### Curvature Tokens
- **Micro Elements (Inputs, Buttons, Badges):** `4px` (`0.25rem`). Provides crisp, punchy interactors without rounding into toy-like or pill shapes.
- **Standard Surfaces (Cards, Modals, Panels):** `8px` (`0.5rem`). Applied to modular sections, data tables, and major structural boundaries.
- **Interactive Focus Rings:** Conform exactly to the target shape's outline with a 2px offset, rendered in `#0070F3` with zero corner distortion.

## Components

### Buttons
- **Primary:** Solid `#FFFFFF` fill with `#000000` text, bold weight (`600`), 4px corner radius. On hover, background shifts to `#EAEAEA`. Focus state renders an outer 2px `#0070F3` ring offset by 2px pitch black.
- **Secondary:** Pitch black surface with a solid 1px `#1F1F1F` border and `#EDEDED` text. On hover, border shifts to `#2E2E2E` and background to `#111111`.
- **Destructive:** Pitch black surface with a 1px border of `rgba(224, 0, 0, 0.3)` and `#E00000` text. Hover shifts background to `rgba(224, 0, 0, 0.1)`.

### Inputs & Form Fields
- Height is fixed at 36px for dense data entry. Background is `#0A0A0A` enclosed in a 1px `#1F1F1F` border.
- Text is `#EDEDED` with `#444444` placeholders.
- Focus transitions border instantly to `#3291FF` without layout shift.
- Monospace variant available for API keys, terminal inputs, and numerical parameters.

### Cards & Analytical Panels
- Composed of `#0A0A0A` background with a continuous 1px `#1F1F1F` perimeter rule.
- Optional top header band highlighted with a 1px horizontal rule separating metadata or actions from the body content.
- Interactive cards exhibit a subtle border illumination to `#2E2E2E` and a faint glow on hover.

### Badges & Monospace Data Tags
- Compact height (20px) with `JetBrains Mono` at 11px.
- Subtle background tint (`rgba(255, 255, 255, 0.05)`) with an integrated 1px border matching the accent color.
- Status variants leverage an active 6px radial dot indicator (green for `Ready`, amber for `Building`, red for `Error`).

### Lists & Data Tables
- Flush, borderless inner rows bounded by 1px horizontal dividers (`#1A1A1A`).
- Column headers set in `JetBrains Mono` 11px uppercase with `#888888` text.
- Row hover triggers a subtle background transition to `#0E0E0E` across the full row width.

### Checkboxes & Radios
- Size: 16x16px square (checkbox) or circle (radio), 1px solid `#2E2E2E` border, background `#0A0A0A`.
- Checked state: `#FFFFFF` fill with an inset `#000000` mark or `#0070F3` fill for active system states.