//! Validation probes: neurodynamic coupling and inference feedback.
//!
//! These tests probe *claims about neurodynamics* that the Python
//! substrate tier cannot reach — they live at the daemon layer:
//!
//! - **Effective-level coupling**: the coupling matrix reads
//!   `effective_level` (level × receptor_sensitivity × desensitization
//!   × internalization), not raw level. Downregulating a driver's
//!   receptors must attenuate its influence on a coupled chemical —
//!   the systemic mechanism of tolerance.
//! - **Sleep-phase gating**: `apply_inference_feedback` skips adenosine
//!   impulses during NREM/REM — a "rest" policy while asleep would pump
//!   the same sleep pressure glymphatic clearance is removing. Gating
//!   must be selective: other impulses in the same result still apply.
//! - **Quiet control**: an inference result carrying no impulses changes
//!   no chemical level. The feedback path itself must not inject energy.
//!
//! Existing coverage lives in `tests/active_inference.rs` (feedback
//! impulses, dyadic coupling, layout) and `tests/state_schema.rs`
//! (coupling direction, metaplasticity) — this file probes the claims
//! they do not.

use genesis::daemon::active_inference::{
    ActiveInferenceEngine, apply_inference_feedback,
};
use genesis::state::MentalPhase;
use genesis::state::neurochemical::{
    NEUROCHEMICAL_COUNT, NeuroTickParams, NeurochemicalId, NeurochemicalVector,
};

/// Build a real inference result and append the requested impulses.
fn result_with_impulses(extra: &[(u8, f32)]) -> genesis::daemon::active_inference::InferenceResult {
    let mut engine = ActiveInferenceEngine::new();
    let state = [0.5f32; NEUROCHEMICAL_COUNT];
    let mut result = engine.cycle(&state, &state, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
    result.impulses.extend_from_slice(extra);
    result
}

/// Snapshot of every chemical's raw level — the field impulses write.
fn levels(vec: &NeurochemicalVector) -> [f32; NEUROCHEMICAL_COUNT] {
    let mut out = [0.0f32; NEUROCHEMICAL_COUNT];
    for chem in &vec.chemicals {
        out[chem.id as usize] = chem.level;
    }
    out
}

// ─── Quiet control ──────────────────────────────────────────────

#[test]
fn probe_feedback_without_impulses_is_silent() {
    // The feedback path must not inject energy on its own: a result
    // carrying no impulses and no baseline adjustment leaves every
    // level bit-identical.
    let mut neuro = NeurochemicalVector::new(0);
    let before = levels(&neuro);

    let result = result_with_impulses(&[]);
    assert!(result.impulses.is_empty());
    apply_inference_feedback(&mut neuro, &result, 0);

    let after = levels(&neuro);
    assert_eq!(
        before, after,
        "feedback with no impulses changed levels: {before:?} → {after:?}"
    );
}

// ─── Sleep-phase gating ─────────────────────────────────────────

#[test]
fn probe_sleep_gates_adenosine_but_not_dopamine() {
    // During NREM the adenosine impulse is dropped (self-defeating:
    // pumping sleep pressure the glymphatic mechanism is clearing).
    // Gating must be *selective* — the dopamine impulse in the same
    // result still applies.
    let result = result_with_impulses(&[
        (NeurochemicalId::Adenosine as u8, 0.5),
        (NeurochemicalId::Dopamine as u8, 0.5),
    ]);

    let mut sleeping = NeurochemicalVector::new(0);
    sleeping.emergent_phase = MentalPhase::NREM as u8;
    let mut awake = NeurochemicalVector::new(0);
    awake.emergent_phase = MentalPhase::Active as u8;

    let ad_sleep_before = sleeping.get(NeurochemicalId::Adenosine).unwrap().level;
    let da_sleep_before = sleeping.get(NeurochemicalId::Dopamine).unwrap().level;
    let ad_awake_before = awake.get(NeurochemicalId::Adenosine).unwrap().level;
    let da_awake_before = awake.get(NeurochemicalId::Dopamine).unwrap().level;

    apply_inference_feedback(&mut sleeping, &result, 0);
    apply_inference_feedback(&mut awake, &result, 0);

    let ad_sleep_after = sleeping.get(NeurochemicalId::Adenosine).unwrap().level;
    let da_sleep_after = sleeping.get(NeurochemicalId::Dopamine).unwrap().level;
    let ad_awake_after = awake.get(NeurochemicalId::Adenosine).unwrap().level;
    let da_awake_after = awake.get(NeurochemicalId::Dopamine).unwrap().level;

    assert_eq!(
        ad_sleep_before, ad_sleep_after,
        "adenosine impulse should be gated during NREM: {ad_sleep_before} → {ad_sleep_after}"
    );
    assert!(
        ad_awake_after > ad_awake_before,
        "the same adenosine impulse should apply while awake: {ad_awake_before} → {ad_awake_after}"
    );
    assert!(
        da_sleep_after > da_sleep_before,
        "gating must be selective — dopamine impulse still applies during NREM: {da_sleep_before} → {da_sleep_after}"
    );
}

#[test]
fn probe_rem_also_gates_adenosine() {
    let result = result_with_impulses(&[(NeurochemicalId::Adenosine as u8, 0.5)]);
    let mut rem = NeurochemicalVector::new(0);
    rem.emergent_phase = MentalPhase::REM as u8;

    let before = rem.get(NeurochemicalId::Adenosine).unwrap().level;
    apply_inference_feedback(&mut rem, &result, 0);
    let after = rem.get(NeurochemicalId::Adenosine).unwrap().level;

    assert_eq!(
        before, after,
        "adenosine impulse should be gated during REM: {before} → {after}"
    );
}

// ─── Effective-level coupling (systemic tolerance) ─────────────

#[test]
fn probe_receptor_downregulation_attenuates_coupling() {
    // Claim: coupling_force[i] = Σ_j matrix[i][j]·(effective[j] − base[j])
    // — the *effective* level, not the raw level. Holding cortisol's raw
    // level fixed while halving its receptor sensitivity must reduce its
    // suppression of BDNF (matrix[BDNF][CORT] = −0.20). This is the
    // mechanism by which tolerance propagates through the whole system.
    //
    // One tick only: the directed effect is isolated before cascades
    // (HPA feedback, adaptation) and homeostatic pull confound it.
    let suppress = |sensitivity: f32| -> f32 {
        let mut vec = NeurochemicalVector::new(0);
        let cort = vec.get_mut(NeurochemicalId::Cortisol).unwrap();
        cort.level = 0.80;
        cort.receptor_sensitivity = sensitivity;
        vec.recompute_derived();

        let bdnf_before = vec.get(NeurochemicalId::BDNF).unwrap().level;
        // Hold the driver fixed across the tick so the measurement is
        // the coupling term, not homeostatic decay of the impulse.
        vec.get_mut(NeurochemicalId::Cortisol).unwrap().level = 0.80;
        vec.tick_default();
        bdnf_before - vec.get(NeurochemicalId::BDNF).unwrap().level
    };

    let suppression_full = suppress(1.0);
    let suppression_halved = suppress(0.5);

    assert!(
        suppression_full > 0.0,
        "elevated effective cortisol should suppress BDNF: {suppression_full}"
    );
    assert!(
        suppression_halved < suppression_full,
        "halved receptor sensitivity should reduce suppression: \
         full={suppression_full}, halved={suppression_halved}"
    );
}

#[test]
fn probe_coupling_silent_at_effective_baseline() {
    // Quiet control for the coupling channel, isolated by subtraction.
    //
    // The claim under test is precise: coupling_force[i] =
    // Σ_j matrix[i][j]·(effective[j] − baseline[j]). At construction
    // every effective level sits at baseline, so every coupling term is
    // exactly zero — zeroing the channel must change nothing.
    //
    // An earlier version of this probe asserted all 18 levels stay flat
    // across a tick. That assertion is wrong on the mechanism's own
    // terms: documented always-on forces (activity-dependent eCB
    // synthesis, adenosine accumulation, stochastic noise) legitimately
    // move levels at chemical equilibrium. Instead, tick two identical
    // vectors — one with coupling_scale on, one off. Every other force
    // runs in both arms and cancels; the arms differ iff coupling is
    // nonzero at equilibrium.
    let mut coupled = NeurochemicalVector::new(0);
    let mut decoupled = NeurochemicalVector::new(0);

    coupled.tick_with_params(&NeuroTickParams::DEFAULT);
    decoupled.tick_with_params(&NeuroTickParams {
        coupling_scale: 0.0,
        ..NeuroTickParams::DEFAULT
    });

    let max_diff = levels(&coupled)
        .iter()
        .zip(levels(&decoupled).iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "coupling contributed {max_diff} of drift at effective-level equilibrium — \
         expected ~0 (only within-tick ordering effects may differ)"
    );
}
