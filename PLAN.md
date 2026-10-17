# Filter — Parametric EQ Plugin

A surgical parametric EQ plugin for electronic music production, built in Rust.

## Identity
- **Name:** Filter
- **Vibe:** Dark, minimal, spectrum-forward — FabFilter Pro-Q aesthetic
- **Target:** Techno/house mixing & mastering
- **Formats:** CLAP + VST3 (no AU)

## Tech Stack
| Layer | Choice |
|-------|--------|
| Language | Rust |
| Framework | nih_plug |
| DSP | Custom biquad filters |
| GUI | iced (via nih_plug) |
| Spectrum | Real-time FFT (RustFFT) |

## Phases

### Phase 1 — Core (MVP)
- [ ] Biquad filter DSP (Bell, Low Shelf, High Shelf, Low Cut, High Cut)
- [ ] Up to 24 bands with per-band: freq, gain, Q, type, enable
- [ ] Stereo L/R processing
- [ ] CLAP + VST3 build output
- [ ] Dark UI with per-band controls (knobs/sliders)
- [ ] Basic parameter automation support

### Phase 2 — Spectrum & Spatial UX
- [ ] Real-time FFT spectrum analyzer (input + output overlay)
- [ ] Click on spectrum to create a band at that frequency
- [ ] Drag bands on spectrum (freq + gain)
- [ ] Q handles (drag to widen/narrow)
- [ ] Colored band nodes (unique color per band)
- [ ] Hover parameter readout

### Phase 3 — Curve Display
- [ ] Summed frequency response curve
- [ ] Individual band curve overlays
- [ ] Scale zoom (frequency + amplitude)

### Phase 4 — Advanced
- [ ] Dynamic EQ (envelope follower → gain modulation on bell bands)
- [ ] Mid/Side processing
- [ ] Oversampling (2x / 4x)
- [ ] Linear phase mode
- [ ] Match EQ
- [ ] Preset system (factory + user)

## UI Color Palette (Pro-Q inspired)
- Background: `#1a1a1e`
- Grid lines: `#2a2a30`
- Spectrum fill: green → yellow → red gradient
- Band colors: cycle through 12 distinct hues
- Text: `#cccccc`
- Active band: brighter, inactive: dimmed

## Project Structure
```
filter/
├── Cargo.toml
├── src/
│   ├── lib.rs          # Plugin entry + params
│   ├── dsp/
│   │   ├── mod.rs
│   │   ├── biquad.rs   # Filter implementations
│   │   └── spectrum.rs # FFT analyzer
│   └── editor.rs       # GUI
├── xtask/              # Build tasks (nih_plug convention)
└── assets/             # UI assets
```

## Band Management (FabFilter-style)
- Start with 0 bands
- Click spectrum → create band (defaults to Bell at click frequency, 0dB gain)
- Right-click → band with shelf/cut default
- Drag up/down → adjust gain
- Drag left/right → adjust frequency
- Scroll on node → adjust Q
- Max 24 bands, bands can be toggled on/off
- Double-click band node → reset to flat

## Decisions (resolved)
- Cut slopes: 6, 12, 24, 48 dB/oct (all)
- Band limit: 24 from the start
- Branding: Generic (no Verksmiðjan)
- Name: Filter
