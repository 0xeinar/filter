# Filter

A surgical parametric EQ plugin built in Rust. Inspired by FabFilter Pro-Q — dark UI, up to 24 bands, spectrum-forward design.

## Features

- **24-band parametric EQ** — Bell, Low Shelf, High Shelf, Low Cut, High Cut, Notch, Band Pass, All Pass, Tilt Shelf
- **Per-band controls** — frequency (20Hz–20kHz), gain (±24dB), Q (0.1–20), type, enable
- **Cut slopes** — 6 to 96 dB/oct in nine steps, as cascaded Butterworth sections
- **Biquad DSP** — Direct Form II Transposed, Audio EQ Cookbook coefficients
- **Dark themed GUI** — FabFilter-style aesthetic, colored band labels
- **CLAP + VST3** — works in Ableton, Bitwig, Reaper, FL Studio
- **Sample-accurate automation**
- **Dynamic EQ** — any gain type can follow the level inside its own band
- **Output stage** — ±24 dB trim, pink-weighted auto-gain, click-free bypass
- **Spectrum** — input and post-EQ traces overlaid, so the curve is visible in the signal

### Auto-gain

Auto-gain estimates the level change of the current curve by integrating its
power response over 20 Hz–20 kHz, weighted uniformly in log frequency (pink
noise — a reasonable stand-in for music), and applies the inverse as makeup. It
is a model, not a measurement, so it is accurate to roughly ±1.5 dB on typical
curves rather than exact; `tests/auto_gain.rs` measures real pink noise through
the real biquads to keep it honest.


## Filters and slopes

| Type | Controls |
|---|---|
| Bell | freq, gain, Q |
| Low/High Shelf | freq, gain, Q |
| Tilt Shelf | freq, gain, Q — ±g/2 across a pivot at 0 dB |
| Low/High Cut | freq, **slope** (gain and Q are ignored) |
| Notch, Band Pass, All Pass | freq, Q |

A cut is a **Butterworth cascade**: 12 dB/oct is one biquad, 24 is two, 96 is
eight. Odd slopes (6, 18, 30) are one second-order section short of the next even
slope and finish with a first-order section, so 6 dB/oct really is 6 and not a
mislabelled 12. The section Qs come from `1 / (2 cos((2k+1)π/4n))`, and
`tests/null_reference.rs` measures the achieved steepness for all nine settings:
every one lands within 3% of its label and nulls at about -145 dB against an
independently-derived reference.

Gain and Q are ignored on cuts rather than silently doing nothing: the panel
shows a slope picker instead of those sliders, the node sits on the 0 dB line
where the curve starts, and scrolling it steps the slope.

## Dynamics

Any band with a gain can be made dynamic. The gain knob becomes the **maximum**
change, and the band is inert until the level inside it exceeds the threshold:

| Control | Meaning |
|---|---|
| Threshold | level inside the band, dBFS, where the gain starts to apply |
| Range | how many dB above the threshold it takes to reach the full gain |
| Attack / Release | how fast the detector follows the signal (63% in the stated time) |

The band listens through a band-pass at **its own frequency and Q**, so a narrow
bell is triggered by a narrow slice of spectrum and a loud tone elsewhere does
not open it. Detection runs on the linked signal (the louder channel) and one
gain feeds both channels, so a reduction cannot move the phantom centre.

Because the gain depends on the signal, the drawn curve shows the *static*
maximum — but the band's node readout and the panel's `DYN` row both show the
gain it is applying **right now**, published from the audio thread through a
lock-free atomic per band. Without that you can hear a dynamic band working and
not see it.

`tests/dynamic_eq.rs` drives the detector, the gain law and a biquad in the same
16-sample shape the audio loop uses, and measures real levels: a −10 dBFS tone
through a −12 dB band at −30 dB threshold comes out at −22 dBFS, the same band
passes a −45 dBFS tone untouched, and an 8 kHz tone does not open a 1 kHz band
at all.

## Layout

The editor is laid out Pro-Q style: graph on top, a strip of numbered band
chips under it, then a panel for **one** band at a time.

| Region | Height |
|---|---|
| Header (Bands, Out, Auto gain, Bypass) | 28 |
| Graph | 300 |
| Band chips | 24 |
| Selected-band panel | 66 |
| Peak readout | 14 |

Every region has an explicit height, so the layout is a constant — 24 bands
occupy exactly as much space as one. (The previous design had one row per band
at ~24 px; past five bands it ran off the bottom of the window.)

## Graph interaction

| Action | Result |
|---|---|
| Click empty graph | Create a band at that frequency (Bell, 0 dB, enabled) |
| Click-drag | Set the new band's gain in the same gesture |
| Click a node | Select it — the panel below edits that band |
| Click a chip | Select that band from the strip |
| Drag a node | Frequency (x) and gain (y) |
| Scroll over a node | Q (hold Alt for fine steps); on a cut, steps the slope |
| Double-click a node | Gain back to 0 dB |
| Right-click a node | Delete the band; later bands shift down, selection follows |
| ← / → | Nudge the selected band's frequency by a semitone (Shift: an octave) |
| ↑ / ↓ | Nudge gain by 0.1 dB (Shift: 1 dB) |
| Tab / Shift+Tab | Select the next / previous band |
| Backspace / Delete | Delete the selected band |
| Esc | Clear the selection |

## Build

```bash
# Clone with submodules (nih_plug is referenced via git)
git clone https://github.com/0xeinar/filter.git
cd filter

# Build the plugin (CLAP + VST3)
cargo run -p xtask -- bundle filter --release

# Output:
#   target/bundled/filter.clap
#   target/bundled/filter.vst3
```

## Install

### macOS

```bash
# VST3 (Ableton, Reaper, FL Studio)
cp -r target/bundled/filter.vst3 ~/Library/Audio/Plug-Ins/VST3/

# CLAP (Bitwig, Reaper)
cp -r target/bundled/filter.clap ~/Library/Audio/Plug-Ins/CLAP/
```

Rescan plugins in your DAW after copying.

## Test

```bash
# Run all tests
cargo test

# 183 tests:
#  148 unit tests — biquad coefficients, frequency response, stability,
#                   dynamic-EQ detector (ballistics, selectivity, threshold
#                   law, stability),
#                   response-curve math (log mapping, combined EQ response),
#                   per-channel state independence, band selection and
#                   delete-compaction bookkeeping, the editor's fixed-height
#                   layout budget, the canvas draw path (layers, curve
#                   position in pixel space, cache invalidation, selection
#                   ring, click-to-select), keyboard nudging, all nine filter
#                   types, cut slopes and cascades, and 5 end-to-end tests
#                   (a click-drag, a key press and a cut drag →
#                   normalized value → plain value → coefficients → audio)
#    3 loudness tests — auto-gain vs measured pink-noise level change
#   10 FFT integration tests — sine sweep → spectrum verification
#    7 property-based fuzz tests — 1000 random parameter combos each
#    7 null tests — plugin output vs an independent f64 DF-I reference
#                   (every filter type, tilt as two sections, sloped cuts at
#                   12–96 dB/oct, an 8-band cascade; ~-145 dB), plus measured
#                   slope accuracy and monotonicity
#    8 dynamic-EQ tests — detector + gain law + biquad driven in the audio
#                   loop's 16-sample shape; measured levels for loud, quiet,
#                   off-band, boost and attack/release cases

# Optional: dump the GUI's canvas geometry for offline rasterization
FILTER_CANVAS_DUMP=/tmp/layers.json cargo test --lib dump_layers_for_visual_inspection
```

## Development Setup

After cloning, enable the pre-push hook (runs formatting check + tests before every push):

```bash
git config core.hooksPath .githooks
```

This is required for local development — the hook won't activate without it.

## Project Structure

```
filter/
├── src/
│   ├── lib.rs           # Plugin entry point, params, DSP routing
│   ├── editor.rs        # iced GUI (dark theme, band controls)
│   └── dsp/
│       ├── mod.rs
│       └── biquad.rs    # Biquad filter + 22 unit tests
├── tests/
│   ├── frequency_response.rs  # FFT-based integration tests
│   ├── fuzz_property.rs       # Property-based fuzzing (proptest)
│   └── null_reference.rs      # Null test vs independent f64 reference
├── benches/
│   └── hotpath_bench.rs       # Per-sample cost of each process-loop shape
├── xtask/               # Build task runner (nih_plug convention)
├── .githooks/
│   └── pre-push         # fmt + test gate
└── .github/workflows/
    └── ci.yml           # CI: Ubuntu + macOS
```

## Tech Stack

| Layer | Choice |
|-------|--------|
| Language | Rust |
| Framework | [nih_plug](https://github.com/robbert-vdh/nih-plug) |
| DSP | Custom biquad (Audio EQ Cookbook) |
| GUI | iced (via nih_plug_iced) |
| Testing | approx, proptest, rustfft |

## Roadmap

- [x] Phase 1 — Core EQ (bands, filters, DSP, basic UI)
- [ ] Phase 2 — Spectrum analyzer, click-to-create bands, drag controls
- [ ] Phase 3 — Frequency response curve overlay
- [ ] Phase 4 — Dynamic EQ, Mid/Side, oversampling, linear phase

## License

GPL-3.0-or-later
