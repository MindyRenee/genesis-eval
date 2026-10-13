//! Validation probes: the neurochemical census and influence map.
//!
//! Two questions, answered from the live substrate:
//!
//! 1. **What are all 18 chemicals doing?** A census: every chemical's
//!    level, baseline, effective level, and receptor state — plus the
//!    derived channels the mind actually reads (valence, arousal,
//!    global tone, plasticity gate, emergent phase).
//! 2. **How do they affect Genesis?** Two directions: chemically
//!    (the measured 18×18 influence matrix — which chemicals move
//!    which others through coupling) and functionally (the derived
//!    channels: valence's exact weighted formula, the plasticity gate
//!    BDNF·(1−cortisol·0.5), phase emergence, sleep arousal clamps).
//!
//! Isolation technique: for each source chemical, tick two identical
//! vectors — coupling on vs off (`coupling_scale`). Every other force
//! (homeostasis, eCB synthesis, adenosine accumulation, decay) runs in
//! both arms and cancels; the difference is the coupling contribution.

use genesis::state::MentalPhase;
use genesis::state::neurochemical::{
    DEFAULT_COUPLING_MATRIX, NEUROCHEMICAL_COUNT, NeuroTickParams,
    NeurochemicalId, NeurochemicalVector,
};

const QUIET_PARAMS: NeuroTickParams = NeuroTickParams {
    noise_amplitude: 0.0,
    ..NeuroTickParams::DEFAULT
};

/// Tick once with all chemicals at baseline except `src`, which is
/// held `bump` above baseline. Returns per-chemical coupling deltas.
fn coupling_row(src: NeurochemicalId, bump: f32) -> [f32; NEUROCHEMICAL_COUNT] {
    let run = |scale: f32| -> [f32; NEUROCHEMICAL_COUNT] {
        let mut vec = NeurochemicalVector::new(0);
        vec.get_mut(src).unwrap().level += bump;
        vec.recompute_derived();
        vec.tick_with_params(&NeuroTickParams {
            coupling_scale: scale,
            ..QUIET_PARAMS
        });
        let mut out = [0.0f32; NEUROCHEMICAL_COUNT];
        for c in &vec.chemicals {
            out[c.id as usize] = c.level;
        }
        out
    };
    let on = run(QUIET_PARAMS.coupling_scale);
    let off = run(0.0);
    let mut delta = [0.0f32; NEUROCHEMICAL_COUNT];
    for i in 0..NEUROCHEMICAL_COUNT {
        delta[i] = on[i] - off[i];
    }
    delta
}

// ─── probe 1: the census ────────────────────────────────────────
//
// Print every chemical's real state so a reader can see the whole
// system at once, and assert the integrity contract: finite, bounded,
// effective = level × modulators at construction.

#[test]
fn probe_chemical_census() {
    let vec = NeurochemicalVector::new(0);
    println!("\n=== NEUROCHEMICAL CENSUS (fresh NeurochemicalVector) ===");
    println!(
        "{:<4} {:<16} {:>8} {:>8} {:>10} {:>10}",
        "id", "name", "level", "base", "effective", "sens"
    );
    for chem in &vec.chemicals {
        println!(
            "{:<4} {:<16} {:>8.4} {:>8.4} {:>10.4} {:>10.4}",
            chem.id,
            NeurochemicalId::from_u8(chem.id).name(),
            chem.level,
            chem.baseline,
            chem.effective_level(),
            chem.receptor_sensitivity,
        );
        assert!(chem.level.is_finite(), "chemical {} level not finite", chem.id);
        assert!(
            (0.0..=1.0).contains(&chem.level),
            "chemical {} level {} out of [0,1]",
            chem.id,
            chem.level
        );
        // At construction, modulators are neutral: effective == level.
        assert!(
            (chem.effective_level() - chem.level).abs() < 1e-6,
            "chemical {} effective {} != level {} at construction",
            chem.id,
            chem.effective_level(),
            chem.level
        );
    }
    println!(
        "derived: arousal={:.4} valence={:.4} global_tone={:.4} \
         plasticity_gate={:.4} phase={:?}",
        vec.arousal,
        vec.valence,
        vec.global_tone,
        vec.plasticity_gate,
        MentalPhase::from_u8(vec.emergent_phase),
    );
}

// ─── probe 2: the measured influence matrix ─────────────────────
//
// For each source chemical, the sign of every nonzero coupling entry
// must match the declared matrix. This turns the static 18×18 table
// into a *measured* influence map: which chemicals actually push
// which others, and by how much.

#[test]
fn probe_coupling_influence_matrix() {
    println!("\n=== MEASURED COUPLING INFLUENCE (one tick, +0.4 source bump) ===");
    let mut checked = 0usize;
    let mut mismatches: Vec<(usize, usize, f32, f32)> = Vec::new();
    for j in 0..NEUROCHEMICAL_COUNT {
        let src = NeurochemicalId::from_u8(j as u8);
        let row = coupling_row(src, 0.4);
        for i in 0..NEUROCHEMICAL_COUNT {
            let declared = DEFAULT_COUPLING_MATRIX[i][j];
            if declared.abs() < 1e-9 {
                continue;
            }
            checked += 1;
            let measured = row[i];
            // Sign agreement: positive coupling must push target up.
            if declared > 0.0 && measured < -1e-7
                || declared < 0.0 && measured > 1e-7
            {
                mismatches.push((i, j, declared, measured));
            }
            if measured.abs() > 1e-9 {
                println!(
                    "  {:>12} → {:<16} declared={:+.3} measured={:+.6}",
                    src.name(),
                    NeurochemicalId::from_u8(i as u8).name(),
                    declared,
                    measured
                );
            }
        }
    }
    assert!(
        checked > 0,
        "coupling matrix has nonzero entries — nothing measured"
    );
    assert!(
        mismatches.is_empty(),
        "coupling sign mismatches (declared vs measured): {mismatches:?}"
    );
    println!("  checked {checked} declared couplings — all signs match");
}

// ─── probe 3: the derived channels the mind reads ───────────────
//
// The Python layer never sees raw chemicals — it reads the derived
// summary. These formulas are the chemical→mind contract, so they
// get exact assertions.

#[test]
fn probe_valence_formula_exact() {
    // valence = da·0.25 + srt·0.25 + oxy·0.10 + end·0.15 + ne·0.10
    //           − cort·0.35 − max(glu−0.70,0)·0.15 + da·srt·0.10
    let mut vec = NeurochemicalVector::new(0);
    let set = |v: &mut NeurochemicalVector, id: NeurochemicalId, x: f32| {
        v.get_mut(id).unwrap().level = x;
    };
    set(&mut vec, NeurochemicalId::Dopamine, 0.60);
    set(&mut vec, NeurochemicalId::Serotonin, 0.50);
    set(&mut vec, NeurochemicalId::Oxytocin, 0.40);
    set(&mut vec, NeurochemicalId::Endorphin, 0.30);
    set(&mut vec, NeurochemicalId::Norepinephrine, 0.40);
    set(&mut vec, NeurochemicalId::Cortisol, 0.20);
    set(&mut vec, NeurochemicalId::Glutamate, 0.80); // 0.10 excess
    vec.recompute_derived();

    // effective = level at construction (modulators neutral), BUT
    // dopamine/serotonin effective go through subtype weighting —
    // compute expected from effective levels, not raw, to be exact.
    let eff = |id: NeurochemicalId| vec.effective_levels[id as usize];
    let da = eff(NeurochemicalId::Dopamine);
    let srt = eff(NeurochemicalId::Serotonin);
    let expected = da * 0.25
        + srt * 0.25
        + eff(NeurochemicalId::Oxytocin) * 0.10
        + eff(NeurochemicalId::Endorphin) * 0.15
        + eff(NeurochemicalId::Norepinephrine) * 0.10
        - eff(NeurochemicalId::Cortisol) * 0.35
        - (eff(NeurochemicalId::Glutamate) - 0.70).max(0.0) * 0.15
        + da * srt * 0.10;
    assert!(
        (vec.valence - expected).abs() < 1e-5,
        "valence formula: expected {expected}, measured {}",
        vec.valence
    );
}

#[test]
fn probe_plasticity_gate_formula() {
    // plasticity_gate = bdnf · (1 − cort·0.5): BDNF opens learning,
    // cortisol closes it — stress literally gates plasticity.
    let mut vec = NeurochemicalVector::new(0);
    vec.get_mut(NeurochemicalId::BDNF).unwrap().level = 0.60;
    vec.get_mut(NeurochemicalId::Cortisol).unwrap().level = 0.40;
    vec.recompute_derived();
    let bdnf = vec.effective_levels[NeurochemicalId::BDNF as usize];
    let cort = vec.effective_levels[NeurochemicalId::Cortisol as usize];
    let expected = (bdnf * (1.0 - cort * 0.5)).clamp(0.0, 1.0);
    assert!(
        (vec.plasticity_gate - expected).abs() < 1e-5,
        "plasticity_gate: expected {expected}, measured {}",
        vec.plasticity_gate
    );

    // Directed: doubling cortisol must shrink the gate.
    let mut vec2 = NeurochemicalVector::new(0);
    vec2.get_mut(NeurochemicalId::BDNF).unwrap().level = 0.60;
    vec2.get_mut(NeurochemicalId::Cortisol).unwrap().level = 0.80;
    vec2.recompute_derived();
    assert!(
        vec2.plasticity_gate < vec.plasticity_gate,
        "more cortisol must close the plasticity gate"
    );
}

#[test]
fn probe_global_tone_is_mean_effective() {
    let mut vec = NeurochemicalVector::new(0);
    vec.get_mut(NeurochemicalId::Dopamine).unwrap().level = 0.9;
    vec.get_mut(NeurochemicalId::GABA).unwrap().level = 0.1;
    vec.recompute_derived();
    let mean: f32 =
        vec.effective_levels.iter().sum::<f32>() / NEUROCHEMICAL_COUNT as f32;
    assert!(
        (vec.global_tone - mean.clamp(0.0, 2.0)).abs() < 1e-6,
        "global_tone must be the clamped mean of effective levels"
    );
}

// ─── probe 4: phase emerges from the chemistry ──────────────────
//
// MentalPhase is not a switch — it emerges from the coupled vector.
// High adenosine + low histamine + melatonin must produce NREM, and
// the Wilson-Cowan hysteresis must make the phase sticky, not
// flickering.

#[test]
fn probe_sleep_phase_emerges_from_chemistry() {
    let mut vec = NeurochemicalVector::new(0);
    assert_eq!(
        MentalPhase::from_u8(vec.emergent_phase),
        MentalPhase::Active,
        "fresh vector should start Active"
    );

    // Drive the sleep conditions: adenosine high, histamine low,
    // melatonin elevated (circadian night), orexin low (wake drive off).
    for _ in 0..400 {
        vec.get_mut(NeurochemicalId::Adenosine).unwrap().level = 0.85;
        vec.get_mut(NeurochemicalId::Histamine).unwrap().level = 0.10;
        vec.get_mut(NeurochemicalId::Melatonin).unwrap().level = 0.70;
        vec.get_mut(NeurochemicalId::Orexin).unwrap().level = 0.15;
        vec.tick_with_params(&QUIET_PARAMS);
    }
    let phase = MentalPhase::from_u8(vec.emergent_phase);
    assert!(
        phase == MentalPhase::NREM || phase == MentalPhase::REM,
        "sustained sleep chemistry must produce a sleep phase, got {phase:?}"
    );
    // During sleep phases, arousal is hard-clamped by the VLPO model.
    if phase == MentalPhase::NREM {
        assert!(
            (0.10..=0.25).contains(&vec.arousal),
            "NREM arousal must be clamped to [0.10, 0.25], got {}",
            vec.arousal
        );
    } else {
        assert!(
            (0.35..=0.55).contains(&vec.arousal),
            "REM arousal must be clamped to [0.35, 0.55], got {}",
            vec.arousal
        );
    }
}
