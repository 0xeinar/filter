//! Patch snapshots, undo/redo history, and the A/B pair.
//!
//! Everything here is plain data plus the transitions between states, so the
//! awkward parts — coalescing a drag into one undo step, what undo does to
//! redo, what a slot holds when you switch away from it — are testable without
//! a host, a window or a plugin instance.
//!
//! Applying a patch *does* need a `GuiContext` (the host owns the parameter
//! store), so `apply` lives here but is driven by the editor.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use nih_plug::prelude::{GuiContext, Param, ParamSetter};

use crate::{FilterParams, MAX_BANDS};

/// How long the parameters must sit unchanged before a change counts as one
/// undo step. A drag writes continuously; without this every frame of it would
/// be its own entry, and undo would walk back through the drag one pixel at a
/// time.
pub const SETTLE: Duration = Duration::from_millis(400);

/// How many steps each direction keeps.
pub const HISTORY_LIMIT: usize = 50;

/// Every parameter of one band.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandPatch {
    pub freq: f32,
    pub gain: f32,
    pub q: f32,
    pub filter_type: i32,
    pub slope: i32,
    pub enable: bool,
    pub dyn_enable: bool,
    pub dyn_threshold: f32,
    pub dyn_range: f32,
    pub dyn_attack: f32,
    pub dyn_release: f32,
    pub dyn_knee: f32,
    pub dyn_sc_enable: bool,
    pub dyn_sc_freq: f32,
}

/// A complete plugin state: the global parameters plus every band.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Patch {
    pub num_bands: i32,
    pub output_gain: f32,
    pub auto_gain: bool,
    pub bands: [BandPatch; MAX_BANDS],
}

impl BandPatch {
    fn capture(bp: &crate::BandParams) -> Self {
        Self {
            freq: bp.freq.value(),
            gain: bp.gain.value(),
            q: bp.q.value(),
            filter_type: bp.filter_type.value(),
            slope: bp.slope.value(),
            enable: bp.enable.value(),
            dyn_enable: bp.dyn_enable.value(),
            dyn_threshold: bp.dyn_threshold.value(),
            dyn_range: bp.dyn_range.value(),
            dyn_attack: bp.dyn_attack.value(),
            dyn_release: bp.dyn_release.value(),
            dyn_knee: bp.dyn_knee.value(),
            dyn_sc_enable: bp.dyn_sc_enable.value(),
            dyn_sc_freq: bp.dyn_sc_freq.value(),
        }
    }
}

impl Patch {
    /// Read the current state out of the parameters.
    pub fn capture(params: &FilterParams) -> Self {
        Self {
            num_bands: params.num_bands.value(),
            output_gain: params.output_gain.value(),
            auto_gain: params.auto_gain.value(),
            bands: core::array::from_fn(|i| BandPatch::capture(&params.band_params[i])),
        }
    }

    /// Write this patch into the parameters, touching only what differs.
    ///
    /// Writing everything on every undo would emit 24 bands × 15 parameters of
    /// host traffic for a one-knob change, and hosts treat a parameter write as
    /// automation input. Skipping equal values keeps an undo to the size of the
    /// edit it reverses.
    ///
    /// Bypass is deliberately **not** part of a patch: it is a monitoring
    /// switch, not part of the sound, and undoing into a bypassed state is
    /// confusing. The A/B pair does carry it separately at the call site if it
    /// ever needs to.
    pub fn apply(&self, params: &FilterParams, ctx: &dyn GuiContext) -> usize {
        let setter = ParamSetter::new(ctx);
        let mut written = 0;

        written += set(&setter, &params.num_bands, self.num_bands);
        written += set(&setter, &params.output_gain, self.output_gain);
        written += set(&setter, &params.auto_gain, self.auto_gain);

        for (i, band) in self.bands.iter().enumerate() {
            let bp = &params.band_params[i];
            written += set(&setter, &bp.freq, band.freq);
            written += set(&setter, &bp.gain, band.gain);
            written += set(&setter, &bp.q, band.q);
            written += set(&setter, &bp.filter_type, band.filter_type);
            written += set(&setter, &bp.slope, band.slope);
            written += set(&setter, &bp.enable, band.enable);
            written += set(&setter, &bp.dyn_enable, band.dyn_enable);
            written += set(&setter, &bp.dyn_threshold, band.dyn_threshold);
            written += set(&setter, &bp.dyn_range, band.dyn_range);
            written += set(&setter, &bp.dyn_attack, band.dyn_attack);
            written += set(&setter, &bp.dyn_release, band.dyn_release);
            written += set(&setter, &bp.dyn_knee, band.dyn_knee);
            written += set(&setter, &bp.dyn_sc_enable, band.dyn_sc_enable);
            written += set(&setter, &bp.dyn_sc_freq, band.dyn_sc_freq);
        }
        written
    }
}

/// Write one parameter if it differs, as a single automation gesture.
///
/// Compares against the *unmodulated* value: under host modulation the
/// modulated value moves on its own, and treating that as a user edit would
/// fill the history with steps nobody took.
fn set<P>(setter: &ParamSetter, param: &P, value: P::Plain) -> usize
where
    P: Param,
{
    if param.unmodulated_plain_value() == value {
        return 0;
    }
    setter.begin_set_parameter(param);
    setter.set_parameter(param, value);
    setter.end_set_parameter(param);
    1
}

/// Undo/redo over settled patches.
#[derive(Debug, Default)]
pub struct History {
    undo: Vec<Patch>,
    redo: Vec<Patch>,
    /// The state the parameters were last known to be at rest in.
    settled: Option<Patch>,
    /// When a change was first noticed, so a drag can settle into one step.
    pending_since: Option<Instant>,
}

impl History {
    /// Start tracking from `initial` with nothing to undo.
    pub fn new(initial: Patch) -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            settled: Some(initial),
            pending_since: None,
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    /// Watch the live parameters, and record a step once a change has stopped
    /// moving. Returns true if a step was added.
    ///
    /// This is a polling design rather than a hook in every edit path: the
    /// editor redraws every frame anyway, and polling catches edits that never
    /// pass through our own handlers (host automation, another editor instance,
    /// a preset load) without a single one of them needing to know about undo.
    pub fn observe(&mut self, current: &Patch, now: Instant) -> bool {
        let Some(settled) = &self.settled else {
            self.settled = Some(current.clone());
            return false;
        };
        if current == settled {
            self.pending_since = None;
            return false;
        }
        match self.pending_since {
            None => {
                self.pending_since = Some(now);
                false
            }
            Some(since) if now.duration_since(since) < SETTLE => false,
            Some(_) => {
                let previous = settled.clone();
                self.settled = Some(current.clone());
                self.pending_since = None;
                self.redo.clear();
                self.undo.push(previous);
                if self.undo.len() > HISTORY_LIMIT {
                    self.undo.remove(0);
                }
                true
            }
        }
    }

    /// Step back. `current` is the live state, which becomes the redo entry.
    pub fn undo(&mut self, current: &Patch) -> Option<Patch> {
        let previous = self.undo.pop()?;
        self.redo.push(current.clone());
        self.settled = Some(previous.clone());
        self.pending_since = None;
        Some(previous)
    }

    /// Step forward again.
    pub fn redo(&mut self, current: &Patch) -> Option<Patch> {
        let next = self.redo.pop()?;
        self.undo.push(current.clone());
        self.settled = Some(next.clone());
        self.pending_since = None;
        Some(next)
    }

    /// Forget the history of a state that no longer applies, e.g. after loading
    /// a different patch from the other A/B slot.
    pub fn rebase(&mut self, current: Patch) {
        self.settled = Some(current);
        self.pending_since = None;
    }
}

/// The A and B states.
///
/// A switch saves what is live into the slot being left and returns the other
/// slot to apply. Both slots start identical, so switching does nothing until
/// one of them is edited — which is what makes an A/B comparison honest.
#[derive(Debug)]
pub struct AbSlots {
    slots: [Patch; 2],
    active: usize,
}

impl AbSlots {
    pub fn new(initial: Patch) -> Self {
        Self {
            slots: [initial.clone(), initial],
            active: 0,
        }
    }

    /// Which slot is live: 0 = A, 1 = B.
    pub fn active(&self) -> usize {
        self.active
    }

    pub fn active_name(&self) -> &'static str {
        if self.active == 0 {
            "A"
        } else {
            "B"
        }
    }

    /// The stored patch for a slot.
    pub fn slot(&self, index: usize) -> &Patch {
        &self.slots[index.min(1)]
    }

    /// Swap slots: save `current` into the live one, return the other.
    pub fn toggle(&mut self, current: &Patch) -> Patch {
        self.slots[self.active] = current.clone();
        self.active = 1 - self.active;
        self.slots[self.active].clone()
    }

    /// Copy the live state into the other slot without switching to it.
    pub fn copy_to_other(&mut self, current: &Patch) {
        let other = 1 - self.active;
        self.slots[self.active] = current.clone();
        self.slots[other] = current.clone();
    }

    /// Write the live state into the active slot. Returns whether it changed
    /// anything, so the caller can avoid republishing an identical snapshot
    /// every frame.
    pub fn store_active(&mut self, current: &Patch) -> bool {
        if self.slots[self.active] == *current {
            return false;
        }
        self.slots[self.active] = current.clone();
        true
    }

    /// Overwrite both slots (used when restoring a session).
    fn restore(&mut self, slots: [Patch; 2], active: usize) {
        self.slots = slots;
        self.active = active.min(1);
    }
}

/// Schema version of the persisted A/B pair.
///
/// Bump this whenever `Patch` gains, loses or reorders a field. A snapshot from
/// an older version is ignored rather than deserialized into a plausible but
/// wrong state: `serde` would happily accept a blob missing the fields added
/// since, and the band parameters would then silently land on their defaults.
pub const AB_SCHEMA: u32 = 3;

/// The A/B pair as it is stored in the session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbSnapshot {
    pub version: u32,
    pub active: u8,
    pub slots: [Patch; 2],
}

impl AbSnapshot {
    pub fn is_current(&self) -> bool {
        self.version == AB_SCHEMA
    }
}

/// History and the A/B pair behind one lock, shared by the editor (which draws
/// the slot state and drives undo) and the canvas (which owns the keyboard).
///
/// GUI-thread only — this is `Mutex` rather than `RefCell` purely so the editor
/// struct can stay `Send + Sync` without another unsafe impl.
pub struct PatchState {
    pub history: History,
    pub ab: AbSlots,
}

impl PatchState {
    pub fn new(params: &FilterParams) -> Self {
        let patch = Patch::capture(params);
        Self {
            history: History::new(patch.clone()),
            ab: AbSlots::new(patch),
        }
    }

    /// Rebuild from a persisted snapshot, if there is a usable one.
    ///
    /// The history always starts from the *live* parameters, not from a slot:
    /// after a reload there is nothing to undo back into, and seeding the
    /// history with a slot would let undo jump to a state the user never heard
    /// in this session.
    pub fn restored(params: &FilterParams, snapshot: Option<&AbSnapshot>) -> Self {
        let live = Patch::capture(params);
        let mut state = Self {
            history: History::new(live),
            ab: AbSlots::new(Patch::capture(params)),
        };
        if let Some(snapshot) = snapshot.filter(|s| s.is_current()) {
            state
                .ab
                .restore(snapshot.slots.clone(), snapshot.active as usize);
        }
        state
    }

    /// A snapshot of both slots for persistence.
    pub fn snapshot(&self) -> AbSnapshot {
        AbSnapshot {
            version: AB_SCHEMA,
            active: self.ab.active() as u8,
            slots: [self.ab.slot(0).clone(), self.ab.slot(1).clone()],
        }
    }

    /// Keep the live slot in step with the parameters.
    ///
    /// Without this the active slot only catches up when you switch away from
    /// it, so an edit made and left in place would be lost on reload while the
    /// *other* slot survived — the worst of both.
    pub fn sync(&mut self, live: &Patch) -> bool {
        self.ab.store_active(live)
    }

    /// Record a settled change, if there is one. Called once per GUI frame.
    pub fn observe(&mut self, params: &FilterParams, now: Instant) -> bool {
        let current = Patch::capture(params);
        self.observe_patch(&current, now)
    }

    /// As `observe`, for a patch the caller already captured — the editor needs
    /// the same patch for `sync`, and capturing it twice per frame is wasted
    /// work on a 24-band state.
    pub fn observe_patch(&mut self, current: &Patch, now: Instant) -> bool {
        self.history.observe(current, now)
    }

    /// Step back one edit and write it to the host. Returns whether anything
    /// was written, so a caller can avoid a repaint when the stack is empty.
    pub fn undo(&mut self, params: &FilterParams, ctx: &dyn GuiContext) -> bool {
        let current = Patch::capture(params);
        match self.history.undo(&current) {
            Some(patch) => {
                patch.apply(params, ctx);
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self, params: &FilterParams, ctx: &dyn GuiContext) -> bool {
        let current = Patch::capture(params);
        match self.history.redo(&current) {
            Some(patch) => {
                patch.apply(params, ctx);
                true
            }
            None => false,
        }
    }

    /// Switch to the other A/B slot.
    ///
    /// The switch is *not* an undo step: the history is rebased onto the patch
    /// that is now live, so undo after a switch walks back the edits made in the
    /// new slot rather than flipping slots again. Two independent states, one
    /// history, no ping-pong.
    pub fn toggle_slot(&mut self, params: &FilterParams, ctx: &dyn GuiContext) {
        let current = Patch::capture(params);
        let target = self.ab.toggle(&current);
        target.apply(params, ctx);
        self.history.rebase(target);
    }

    /// Copy the live state into the other slot without switching.
    pub fn copy_slot(&mut self, params: &FilterParams) {
        let current = Patch::capture(params);
        self.ab.copy_to_other(&current);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{test_params, RecordingGuiContext};

    fn patch(params: &FilterParams) -> Patch {
        Patch::capture(params)
    }

    #[test]
    fn applying_a_patch_writes_the_patchs_values_to_the_host() {
        // A test cannot move a parameter: the store belongs to nih-plug and its
        // setters are crate-private. What it *can* see is the exact write
        // stream the host would receive, which is what `apply` exists to
        // produce. (Every layer above this point is already covered by the
        // e2e tests, which go through the real gesture path.)
        let source = test_params(&[(100.0, 6.0, 2.5, crate::FilterType::LowShelf)]);
        let snapshot = patch(&source);
        let target = test_params(&[(8_000.0, -9.0, 0.7, crate::FilterType::HighCut)]);
        let ctx = RecordingGuiContext::default();

        let written = snapshot.apply(&target, &ctx);
        assert_eq!(written, ctx.writes().len(), "one write per parameter");

        // Band 0 differs in every field, so the stream starts with its values
        // in the order `apply` uses.
        let tb = &target.band_params[0];
        let expected_prefix = [
            tb.freq.preview_normalized(100.0),
            tb.gain.preview_normalized(6.0),
            tb.q.preview_normalized(2.5),
            tb.filter_type.preview_normalized(1),
        ];
        assert_eq!(
            &ctx.writes()[..4],
            &expected_prefix,
            "the first four writes must be band 0's frequency, gain, Q and type"
        );
    }

    #[test]
    fn applying_an_identical_patch_writes_nothing() {
        // The thing that keeps undo from spraying the host with 264 writes.
        let params = test_params(&[(1_000.0, 3.0, 1.0, crate::FilterType::Bell)]);
        let snapshot = patch(&params);
        let ctx = RecordingGuiContext::default();
        assert_eq!(snapshot.apply(&params, &ctx), 0);
        assert!(ctx.writes().is_empty());
    }

    #[test]
    fn a_one_band_change_costs_three_writes_to_undo() {
        // The live state is one band's freq/gain/Q away from `before`. Undoing
        // must write exactly those three, with the old values — not all 264.
        let before = patch(&test_params(&[(
            1_000.0,
            3.0,
            1.0,
            crate::FilterType::Bell,
        )]));
        let live = test_params(&[(2_000.0, -4.0, 3.0, crate::FilterType::Bell)]);
        let ctx = RecordingGuiContext::default();

        assert_eq!(before.apply(&live, &ctx), 3);
        let bp = &live.band_params[0];
        assert_eq!(
            ctx.writes(),
            vec![
                bp.freq.preview_normalized(1_000.0),
                bp.gain.preview_normalized(3.0),
                bp.q.preview_normalized(1.0),
            ]
        );
    }

    #[test]
    fn a_change_settles_into_one_undo_step() {
        let start = test_params(&[(1_000.0, 0.0, 1.0, crate::FilterType::Bell)]);
        let mut history = History::new(patch(&start));
        let t0 = Instant::now();

        // A drag: the value wanders, then stops.
        let mut dragged = patch(&start);
        for i in 1..10 {
            dragged.bands[0].gain = i as f32;
            assert!(
                !history.observe(&dragged, t0 + Duration::from_millis(i * 16)),
                "a moving value must not record a step"
            );
        }
        // Still nothing until it has been quiet for SETTLE.
        assert!(!history.observe(&dragged, t0 + Duration::from_millis(200)));
        assert!(history.observe(&dragged, t0 + Duration::from_millis(200) + SETTLE));
        assert_eq!(history.undo_len(), 1, "one drag is one undo step");
    }

    #[test]
    fn undo_restores_the_patch_before_the_change_and_redo_returns() {
        let start = test_params(&[(1_000.0, 0.0, 1.0, crate::FilterType::Bell)]);
        let mut history = History::new(patch(&start));
        let t0 = Instant::now();

        let mut changed = patch(&start);
        changed.bands[0].gain = 9.0;
        assert!(!history.observe(&changed, t0));
        assert!(history.observe(&changed, t0 + SETTLE));

        assert!(history.can_undo());
        let restored = history.undo(&changed).expect("there is a step to undo");
        assert_eq!(restored, patch(&start));
        assert!(!history.can_undo());
        assert!(history.can_redo());

        let forward = history.redo(&restored).expect("there is a step to redo");
        assert_eq!(forward, changed);
        assert!(history.can_undo());
        assert!(!history.can_redo());
    }

    #[test]
    fn undo_does_not_record_itself_as_a_new_change() {
        // After an undo the parameters are back at `settled`, so the observer
        // must see no difference — otherwise every undo would add a step and
        // the stack would never empty.
        let start = test_params(&[(1_000.0, 0.0, 1.0, crate::FilterType::Bell)]);
        let mut history = History::new(patch(&start));
        let t0 = Instant::now();
        let mut changed = patch(&start);
        changed.bands[0].gain = 9.0;
        history.observe(&changed, t0);
        history.observe(&changed, t0 + SETTLE);

        let restored = history.undo(&changed).unwrap();
        assert!(!history.observe(&restored, t0 + SETTLE * 3));
        assert_eq!(history.undo_len(), 0);
    }

    #[test]
    fn a_new_edit_clears_the_redo_branch() {
        // Standard undo semantics: once you edit after undoing, the future you
        // abandoned is gone.
        let start = test_params(&[(1_000.0, 0.0, 1.0, crate::FilterType::Bell)]);
        let mut history = History::new(patch(&start));
        let t0 = Instant::now();
        let mut first = patch(&start);
        first.bands[0].gain = 6.0;
        history.observe(&first, t0);
        history.observe(&first, t0 + SETTLE);

        let restored = history.undo(&first).unwrap();
        assert!(history.can_redo());

        let mut second = restored.clone();
        second.bands[1].gain = -3.0;
        history.observe(&second, t0 + SETTLE * 2);
        history.observe(&second, t0 + SETTLE * 2 + SETTLE);
        assert!(!history.can_redo(), "a new edit must drop the redo branch");
    }

    #[test]
    fn history_is_bounded() {
        let start = test_params(&[]);
        let mut history = History::new(patch(&start));
        let t0 = Instant::now();
        for i in 0..HISTORY_LIMIT + 10 {
            let mut step = patch(&start);
            step.output_gain = i as f32;
            history.observe(&step, t0 + Duration::from_secs(i as u64 * 2));
            history.observe(&step, t0 + Duration::from_secs(i as u64 * 2) + SETTLE);
        }
        assert!(
            history.undo_len() <= HISTORY_LIMIT,
            "history grew to {} steps",
            history.undo_len()
        );
    }

    #[test]
    fn nothing_to_undo_or_redo_at_the_start() {
        let params = test_params(&[]);
        let mut history = History::new(patch(&params));
        let current = patch(&params);
        assert!(history.undo(&current).is_none());
        assert!(history.redo(&current).is_none());
    }

    #[test]
    fn ab_slots_start_identical_and_then_diverge() {
        let params = test_params(&[(1_000.0, 0.0, 1.0, crate::FilterType::Bell)]);
        let mut ab = AbSlots::new(patch(&params));
        assert_eq!(ab.active(), 0);
        assert_eq!(ab.active_name(), "A");

        // Switching with no edits is a no-op.
        let same = ab.toggle(&patch(&params));
        assert_eq!(same, patch(&params));
        assert_eq!(ab.active_name(), "B");

        // Edit in B, then go back: A must be untouched.
        let mut edited_b = patch(&params);
        edited_b.bands[0].gain = 7.0;
        let back_to_a = ab.toggle(&edited_b);
        assert_eq!(back_to_a, patch(&params), "A must be unchanged");
        assert_eq!(ab.active_name(), "A");

        // ... and returning to B restores the edit.
        let back_to_b = ab.toggle(&back_to_a);
        assert_eq!(back_to_b, edited_b, "B must remember its own edit");
    }

    #[test]
    fn copying_the_active_slot_overwrites_the_other() {
        let params = test_params(&[(1_000.0, 0.0, 1.0, crate::FilterType::Bell)]);
        let mut ab = AbSlots::new(patch(&params));
        let mut edited = patch(&params);
        edited.bands[0].gain = -5.0;

        ab.copy_to_other(&edited);
        assert_eq!(ab.slot(0), &edited);
        assert_eq!(ab.slot(1), &edited);
        // Copying must not switch: you stay where you were editing.
        assert_eq!(ab.active(), 0);
    }

    // ---- persistence ----------------------------------------------------

    #[test]
    fn a_snapshot_round_trips_through_the_session_blob() {
        // Both slots, including the inactive one that the user cannot see —
        // that is the whole point: B is the reference version.
        let params = test_params(&[(1_000.0, 0.0, 1.0, crate::FilterType::Bell)]);
        let mut state = PatchState::new(&params);

        let mut edited = patch(&params);
        edited.bands[0].gain = 6.0;
        edited.num_bands = 3;
        state.ab.toggle(&edited); // B is now live, A holds `edited`
        let in_b = patch(&params);
        state.ab.toggle(&in_b); // back to A

        let blob = serde_json::to_string(&state.snapshot()).expect("snapshot must serialize");
        let restored: AbSnapshot = serde_json::from_str(&blob).expect("and deserialize");

        assert!(restored.is_current());
        assert_eq!(restored.active, 0);
        assert_eq!(restored.slots[0].bands[0].gain, 6.0, "A keeps its edit");
        assert_eq!(
            restored.slots[1],
            patch(&params),
            "B keeps the untouched copy"
        );
    }

    #[test]
    fn restoring_rebuilds_both_slots_and_the_active_one() {
        let params = test_params(&[(1_000.0, 0.0, 1.0, crate::FilterType::Bell)]);
        let mut a = patch(&params);
        a.bands[0].gain = -3.0;
        let mut b = patch(&params);
        b.bands[0].gain = 4.5;

        let snapshot = AbSnapshot {
            version: AB_SCHEMA,
            active: 1,
            slots: [a.clone(), b.clone()],
        };
        let state = PatchState::restored(&params, Some(&snapshot));

        assert_eq!(
            state.ab.active(),
            1,
            "the live slot is restored, not just the data"
        );
        assert_eq!(state.ab.slot(0), &a);
        assert_eq!(state.ab.slot(1), &b);
        // Nothing to undo: a fresh session has no history to walk back into.
        assert!(!state.history.can_undo());
    }

    #[test]
    fn an_edit_left_in_the_active_slot_is_persisted() {
        // The regression this is for: the active slot only caught up when you
        // switched away from it, so an edit made and left in place vanished on
        // reload while the other slot survived.
        let params = test_params(&[(1_000.0, 0.0, 1.0, crate::FilterType::Bell)]);
        let mut state = PatchState::new(&params);

        let mut live = patch(&params);
        live.bands[0].gain = -8.0;
        assert!(state.sync(&live), "a changed live state must be stored");
        assert_eq!(state.snapshot().slots[0].bands[0].gain, -8.0);

        // Unchanged state is not republished every frame.
        assert!(
            !state.sync(&live),
            "an identical state must not report a change"
        );
    }

    #[test]
    fn a_snapshot_from_another_schema_is_ignored() {
        // serde would happily fill missing fields with defaults and hand back a
        // plausible-looking wrong state; the version is what refuses it.
        let params = test_params(&[(1_000.0, 0.0, 1.0, crate::FilterType::Bell)]);
        let mut stale = patch(&params);
        stale.bands[0].gain = 12.0;
        let snapshot = AbSnapshot {
            version: AB_SCHEMA - 1,
            active: 1,
            slots: [stale.clone(), stale],
        };

        let state = PatchState::restored(&params, Some(&snapshot));
        assert_eq!(state.ab.active(), 0);
        assert_eq!(
            state.ab.slot(0),
            &patch(&params),
            "an old blob must be ignored, not half-applied"
        );
    }

    #[test]
    fn no_snapshot_means_a_fresh_pair() {
        let params = test_params(&[(1_000.0, 0.0, 1.0, crate::FilterType::Bell)]);
        let state = PatchState::restored(&params, None);
        assert_eq!(state.ab.active(), 0);
        assert_eq!(state.ab.slot(0), state.ab.slot(1));
        assert_eq!(state.ab.slot(0), &patch(&params));
    }
}
