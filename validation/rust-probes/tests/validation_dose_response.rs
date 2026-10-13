//! Validation probes: dose-response sweep across all 18 chemicals.
//!
//! For every chemical: hold it at depleted / low / baseline / high,
//! run the real coupled dynamics, and record the functional readouts
//! the mind consumes — valence, arousal, global tone, plasticity gate,
//! encoding/consolidation/retrieval weights, emergent phase.
//!
//! This is the "what does each chemical do" map. Where a chemical is
//! supposed to matter and doesn't, that is a finding. Where removing
//! it changes nothing downstream, it is decorative. Where the response
//! curve is non-monotone or saturates, the map records it.
//!
//! Run with `--nocapture` to see the full dose-response tables.

use genesis::state::MentalPhase;
use genesis::state::memory::MemoryPointers;
use genesis::state::neurochemical::{
    NEUROCHEMICAL_COUNT, NeuroTickParams, NeurochemicalId, NeurochemicalVector,
};

const QUIET: NeuroTickParams = NeuroTickParams {
    noise_amplitude: 0.0,
    ..NeuroTickParams::DEFAULT
};

/// What the mind can actually feel — the functional readouts.
#[derive(Debug, Clone, Copy)]
struct Readout {
    valence: f32,
    arousal: f32,
    global_tone: f32,
    plasticity_gate: f32,
    encoding_weight: f32,
    consolidation_weight: f32,
    retrieval_weight: f32,
    phase: u8,
}

/// Hold `id` at `level` for `ticks`, then report the functional state.
/// `level < 0` means "its own baseline" (sustained clamped level).
fn dose_readout(id: NeurochemicalId, level: f32, ticks: usize) -> Readout {
    let mut vec = NeurochemicalVector::new(0);
    let level = if level < 0.0 {
        vec.get(id).unwrap().baseline
    } else {
        level
    };
    for _ in 0..ticks {
        // Hold the level each tick — sustained manipulation, not a
        // pulse. This is the "the chemical is missing/high" condition.
        vec.get_mut(id).unwrap().level = level;
        vec.tick_with_params(&QUIET);
    }
    let eff = |i: NeurochemicalId| vec.effective_levels[i as usize];
    let mut mem = MemoryPointers::new();
    mem.update_gating(
        eff(NeurochemicalId::Dopamine),
        eff(NeurochemicalId::Serotonin),
        eff(NeurochemicalId::Norepinephrine),
        eff(NeurochemicalId::Acetylcholine),
        eff(NeurochemicalId::Cortisol),
        eff(NeurochemicalId::BDNF),
    );
    Readout {
        valence: vec.valence,
        arousal: vec.arousal,
        global_tone: vec.global_tone,
        plasticity_gate: vec.plasticity_gate,
        encoding_weight: mem.encoding_weight,
        consolidation_weight: mem.consolidation_weight,
        retrieval_weight: mem.retrieval_weight,
        phase: vec.emergent_phase,
    }
}

fn fmt_phase(p: u8) -> &'static str {
    MentalPhase::from_u8(p).label()
}

/// The sweep: every chemical, four doses, all readouts printed.
#[test]
fn probe_dose_response_sweep() {
    let levels = [("depleted", 0.0), ("low", 0.15), ("baseline", -1.0), ("high", 0.95)];
    println!("\n=== DOSE-RESPONSE SWEEP (level held 200 ticks) ===");
    println!(
        "{:<16} {:<9} | {:>7} {:>7} {:>6} {:>6} | {:>6} {:>6} {:>6} | {}",
        "chemical", "dose", "valence", "arousal", "tone", "plas",
        "enc", "cons", "retr", "phase"
    );
    for j in 0..NEUROCHEMICAL_COUNT {
        let id = NeurochemicalId::from_u8(j as u8);
        for (name, lvl) in levels {
            let r = dose_readout(id, lvl, 200);
            println!(
                "{:<16} {:<9} | {:>7.3} {:>7.3} {:>6.3} {:>6.3} | {:>6.3} {:>6.3} {:>6.3} | {}",
                id.name(),
                name,
                r.valence,
                r.arousal,
                r.global_tone,
                r.plasticity_gate,
                r.encoding_weight,
                r.consolidation_weight,
                r.retrieval_weight,
                fmt_phase(r.phase),
            );
        }
        println!();
    }
}

// ─── Targeted dose-response assertions ──────────────────────────
//
// Directed effects the architecture documents. Each has a quiet arm
// (depleted/baseline) and a manipulated arm (high).

#[test]
fn probe_dopamine_dose_response() {
    // DA feeds valence (+0.25 + synergy) and encoding weight (+0.4).
    let depleted = dose_readout(NeurochemicalId::Dopamine, 0.0, 200);
    let baseline = dose_readout(NeurochemicalId::Dopamine, -1.0, 200);
    let high = dose_readout(NeurochemicalId::Dopamine, 0.95, 200);

    assert!(
        high.encoding_weight > baseline.encoding_weight,
        "high DA must raise encoding weight: {:.3} vs {:.3}",
        high.encoding_weight,
        baseline.encoding_weight
    );
    assert!(
        depleted.encoding_weight < baseline.encoding_weight,
        "depleted DA must lower encoding weight"
    );
    assert!(
        high.valence > baseline.valence && baseline.valence > depleted.valence,
        "valence must be monotone in DA: {:.3} < {:.3} < {:.3}",
        depleted.valence,
        baseline.valence,
        high.valence
    );
}

#[test]
fn probe_cortisol_stress_signature() {
    // Cortisol is the stress axis endpoint: it should raise
    // consolidation (stressful memories consolidate), LOWER retrieval
    // (stress impairs recall), close plasticity, and lower valence.
    let none = dose_readout(NeurochemicalId::Cortisol, 0.0, 200);
    let high = dose_readout(NeurochemicalId::Cortisol, 0.95, 200);

    assert!(
        high.consolidation_weight > none.consolidation_weight,
        "stress must strengthen consolidation"
    );
    assert!(
        high.retrieval_weight < none.retrieval_weight,
        "stress must impair retrieval: {:.3} vs {:.3}",
        high.retrieval_weight,
        none.retrieval_weight
    );
    assert!(
        high.plasticity_gate < none.plasticity_gate,
        "stress must close the plasticity gate"
    );
    assert!(
        high.valence < none.valence,
        "stress must lower valence"
    );
}

#[test]
fn probe_bdnf_is_the_plasticity_chemical() {
    // BDNF is the only input to plasticity_gate besides cortisol.
    // Depleted BDNF → gate → 0 → can_form_new_memories() false.
    let depleted = dose_readout(NeurochemicalId::BDNF, 0.0, 200);
    let high = dose_readout(NeurochemicalId::BDNF, 0.95, 200);

    assert!(
        depleted.plasticity_gate < 0.1,
        "no BDNF → plasticity gate must close: {}",
        depleted.plasticity_gate
    );
    assert!(
        high.plasticity_gate > depleted.plasticity_gate + 0.3,
        "BDNF must open the gate substantially"
    );
}

#[test]
fn probe_adenosine_drives_sleep_pressure() {
    // High sustained adenosine must move the system toward sleep:
    // either the phase flips or arousal collapses. Depleted adenosine
    // must do neither.
    let depleted = dose_readout(NeurochemicalId::Adenosine, 0.0, 400);
    let high = dose_readout(NeurochemicalId::Adenosine, 0.95, 400);

    let high_phase = MentalPhase::from_u8(high.phase);
    let slept = matches!(high_phase, MentalPhase::NREM | MentalPhase::REM)
        || high.arousal < depleted.arousal;
    assert!(
        slept,
        "sustained high adenosine must drive sleep pressure: \
         phase={high_phase:?} arousal={:.3} (depleted={:.3})",
        high.arousal,
        depleted.arousal
    );
}

#[test]
fn probe_acetylcholine_gates_memory_channels() {
    // ACh is in all three memory weights — it is the "attention to
    // memory" chemical. Depleting it must hit encoding, consolidation,
    // and retrieval simultaneously.
    let depleted = dose_readout(NeurochemicalId::Acetylcholine, 0.0, 200);
    let baseline = dose_readout(NeurochemicalId::Acetylcholine, -1.0, 200);

    assert!(depleted.encoding_weight < baseline.encoding_weight);
    assert!(depleted.consolidation_weight < baseline.consolidation_weight);
    assert!(depleted.retrieval_weight < baseline.retrieval_weight);
}

#[test]
fn probe_serotonin_stabilizes() {
    // Serotonin contributes to retrieval weight and valence, and
    // suppresses stress (via coupling → cortisol). Depleting it must
    // lower retrieval and valence.
    let depleted = dose_readout(NeurochemicalId::Serotonin, 0.0, 200);
    let baseline = dose_readout(NeurochemicalId::Serotonin, -1.0, 200);
    assert!(
        depleted.retrieval_weight < baseline.retrieval_weight,
        "SRT depletion must impair retrieval"
    );
    assert!(
        depleted.valence < baseline.valence,
        "SRT depletion must lower valence"
    );
}
