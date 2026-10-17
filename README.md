# Filter

A surgical parametric EQ plugin built in Rust. Inspired by FabFilter Pro-Q — dark UI, up to 24 bands, spectrum-forward design.

## Features

- **24-band parametric EQ** — Bell, Low Shelf, High Shelf, Low Cut, High Cut
- **Per-band controls** — frequency (20Hz–20kHz), gain (±24dB), Q (0.1–20), type, enable
- **Biquad DSP** — Direct Form II Transposed, Audio EQ Cookbook coefficients
- **Dark themed GUI** — FabFilter-style aesthetic, colored band labels
- **CLAP + VST3** — works in Ableton, Bitwig, Reaper, FL Studio
- **Sample-accurate automation**

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

# 69 tests:
#   49 unit tests — biquad coefficients, frequency response, stability,
#                   response-curve math (log mapping, combined EQ response),
#                   per-channel state independence
#   10 FFT integration tests — sine sweep → spectrum verification
#    7 property-based fuzz tests — 1000 random parameter combos each
#    3 null tests — f32 plugin output vs an independent f64 DF-I reference
#                   (all five filter types + an 8-band cascade, ~-148 dB)
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
