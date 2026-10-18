//! Selected-band bookkeeping and the readout formatting the Pro-Q style
//! band panel uses.
//!
//! Selection lives in a single `AtomicUsize` shared by the graph (which
//! selects on click) and the editor (which selects from the band strip and
//! renders the panel). `NONE` is `usize::MAX` so the atomic needs no extra
//! flag.

/// Sentinel meaning "no band is selected".
pub const NONE: usize = usize::MAX;

/// True when `selection` is the sentinel.
pub fn is_none(selection: usize) -> bool {
    selection == NONE
}

/// Clamp a selection to a band count: `NONE` stays `NONE`, an index past the
/// end becomes the last band, and an empty bank has nothing to select.
///
/// This is what the view applies before using a selection, so lowering the
/// band count can never make the panel point at a band that no longer exists.
pub fn clamp_to(selection: usize, count: usize) -> usize {
    if count == 0 || is_none(selection) {
        return NONE;
    }
    selection.min(count - 1)
}

/// Where the selection should point after deleting band `deleted`, when
/// `remaining` bands are left.
///
/// Deletion compacts the tail down one slot, so every band above the deleted
/// one shifts down: a selection above it must follow, and a selection on the
/// deleted band falls to the band that took its place (or the last band, or
/// nothing).
pub fn after_delete(selection: usize, deleted: usize, remaining: usize) -> usize {
    if remaining == 0 {
        return NONE;
    }
    if is_none(selection) {
        return NONE;
    }
    if selection < deleted {
        selection
    } else if selection > deleted {
        selection - 1
    } else {
        deleted.min(remaining - 1)
    }
}

/// Frequency readout: `440 Hz`, `1.20 kHz`, `20.0 kHz`.
pub fn format_freq(hz: f32) -> String {
    if hz < 1000.0 {
        format!("{:.0} Hz", hz)
    } else if hz < 10_000.0 {
        format!("{:.2} kHz", hz / 1000.0)
    } else {
        format!("{:.1} kHz", hz / 1000.0)
    }
}

/// Gain readout: `+3.5 dB`, `0.0 dB`, `-12.0 dB`.
pub fn format_gain(db: f32) -> String {
    // `+ 0.0` and `-0.0` both print as "0.0 dB" — a flat band must not read
    // as "-0.0".
    let rounded = (db * 10.0).round() / 10.0;
    if rounded == 0.0 {
        "0.0 dB".to_string()
    } else {
        format!("{rounded:+.1} dB")
    }
}

/// Q readout: `Q 0.71`.
pub fn format_q(q: f32) -> String {
    format!("Q {q:.2}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_keeps_a_live_selection_and_drops_a_dead_one() {
        assert_eq!(clamp_to(2, 5), 2);
        assert_eq!(clamp_to(9, 5), 4, "past the end clamps to the last band");
        assert_eq!(clamp_to(NONE, 5), NONE);
        assert_eq!(clamp_to(0, 0), NONE, "an empty bank has nothing to select");
    }

    #[test]
    fn deleting_a_band_below_the_selection_moves_the_selection_down() {
        // Bands 0,1,2,3; delete 0; old band 2 becomes band 1.
        assert_eq!(after_delete(2, 0, 3), 1);
    }

    #[test]
    fn deleting_a_band_above_the_selection_leaves_it_alone() {
        assert_eq!(after_delete(0, 2, 3), 0);
    }

    #[test]
    fn deleting_the_selected_band_selects_the_band_that_took_its_place() {
        // Bands 0,1,2,3; delete 1; band 2 slides into slot 1 and is selected.
        assert_eq!(after_delete(1, 1, 3), 1);
        // Deleting the last band falls back to the new last band.
        assert_eq!(after_delete(3, 3, 3), 2);
        // Deleting the only band leaves nothing selected.
        assert_eq!(after_delete(0, 0, 0), NONE);
    }

    #[test]
    fn deleting_with_nothing_selected_keeps_nothing_selected() {
        assert_eq!(after_delete(NONE, 1, 4), NONE);
    }

    #[test]
    fn frequency_readout_switches_units_like_pro_q() {
        assert_eq!(format_freq(440.0), "440 Hz");
        assert_eq!(format_freq(20.0), "20 Hz");
        assert_eq!(format_freq(1_200.0), "1.20 kHz");
        assert_eq!(format_freq(20_000.0), "20.0 kHz");
    }

    #[test]
    fn gain_readout_always_carries_a_sign_except_at_zero() {
        assert_eq!(format_gain(3.5), "+3.5 dB");
        assert_eq!(format_gain(-12.0), "-12.0 dB");
        assert_eq!(format_gain(0.0), "0.0 dB");
        assert_eq!(format_gain(-0.0), "0.0 dB", "no negative zero");
        assert_eq!(format_gain(0.04), "0.0 dB", "sub-0.1 dB is flat");
    }

    #[test]
    fn q_readout_has_two_decimals() {
        assert_eq!(format_q(0.707), "Q 0.71");
        assert_eq!(format_q(10.0), "Q 10.00");
    }
}
