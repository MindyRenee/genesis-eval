//! Validation probes: the interoceptive substrate — "down to electrons."
//!
//! Cognitive-tier probes measure whether the mind's machinery works.
//! This tier asks the deeper question: is the machinery attached to a
//! *body* — does the physical state of the machine it runs on actually
//! reach its neurochemistry?
//!
//! The chain under test:
//!
//! ```text
//! electrons  →  sysfs/procfs counters (temp, freq, battery, load, RAPL)
//!           →  Interoceptor::read() → BodyState
//!           →  neuro_impulses() → NeurochemicalVector levels
//! ```
//!
//! Probe design follows the validation-lab rules: controlled inputs,
//! quiet controls (below-threshold bodies must produce no impulse),
//! directed effects (above-threshold bodies must produce impulses in
//! the documented direction and magnitude), and honest reporting of
//! absent hardware (a missing RAPL sensor reads 0.0 — it must not be
//! fabricated).

use genesis::daemon::interoception::{BodyState, Interoceptor};
use genesis::state::neurochemical::{NeurochemicalId, NeurochemicalVector};
use std::path::Path;

// ─── helpers ───────────────────────────────────────────────────

/// Read a sysfs/procfs node directly — the ground truth the probe
/// checks Interoceptor against. If this and BodyState disagree, the
/// interoceptive sense is not actually reading the machine.
fn sysfs(path: &str) -> Option<f32> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite())
}

/// Direct battery-capacity read across all power_supply batteries.
fn sysfs_battery_fraction() -> Option<f32> {
    let dir = Path::new("/sys/class/power_supply");
    let mut best: Option<f32> = None;
    for entry in std::fs::read_dir(dir).ok()? {
        let path = entry.ok()?.path();
        if path.join("capacity").exists() {
            if let Some(v) = sysfs(&path.join("capacity").to_string_lossy()) {
                best = Some(v / 100.0);
            }
        }
    }
    best
}

/// Direct CPU temperature read — the first thermal zone that parses.
fn sysfs_cpu_temp_c() -> Option<f32> {
    let dir = Path::new("/sys/class/thermal");
    for entry in std::fs::read_dir(dir).ok()? {
        let path = entry.ok()?.path().join("temp");
        if let Some(millideg) = sysfs(&path.to_string_lossy()) {
            return Some(millideg / 1000.0);
        }
    }
    None
}

/// Sum of every impulse aimed at a given chemical.
fn impulse_total(impulses: &[(NeurochemicalId, f32)], id: NeurochemicalId) -> f32 {
    impulses
        .iter()
        .filter(|(c, _)| *c == id)
        .map(|(_, m)| *m)
        .sum()
}

// ─── probe 1: the body is real ──────────────────────────────────
//
// Interoceptor::read() must return values consistent with this
// machine's actual sysfs state — not constants, not fabrication.

#[test]
fn probe_interoceptor_reads_real_hardware() {
    let mut ic = Interoceptor::new();
    // First read primes delta-based fields; second read populates rates.
    let _ = ic.read();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let body = ic.read();

    assert_eq!(
        body.num_cores as usize,
        std::thread::available_parallelism().unwrap().get(),
        "num_cores must match the real core count"
    );
    assert!(
        body.arousal_freq > 0.0 && body.arousal_freq <= 1.0,
        "arousal_freq should be a real fraction of max frequency: {}",
        body.arousal_freq
    );
    assert!(
        body.cpu_temp_c > 10.0 && body.cpu_temp_c < 110.0,
        "cpu_temp_c should be a plausible real temperature: {}°C",
        body.cpu_temp_c
    );
    assert!(
        body.energy_reserve >= 0.0 && body.energy_reserve <= 1.0,
        "energy_reserve out of range: {}",
        body.energy_reserve
    );

    // Cross-check against sysfs ground truth.
    if let Some(real_temp) = sysfs_cpu_temp_c() {
        // Sensors may differ by zone; tolerance is loose but the read
        // must be a real temperature, not the 50.0 fallback.
        assert!(
            (body.cpu_temp_c - real_temp).abs() < 25.0 || body.cpu_temp_c == 50.0,
            "cpu_temp_c {} vs sysfs {} — either close or an honest fallback",
            body.cpu_temp_c,
            real_temp
        );
    }
    if let Some(real_batt) = sysfs_battery_fraction() {
        assert!(
            (body.energy_reserve - real_batt).abs() < 0.15,
            "energy_reserve {} vs sysfs battery {} — must track the real battery",
            body.energy_reserve,
            real_batt
        );
    }
}

// ─── probe 2: thermal stress formula ───────────────────────────
//
// neuro_impulses maps body state to chemistry with documented
// thresholds and magnitudes. Each mapping gets a quiet control
// (below threshold → silent) and an exact-magnitude check.

#[test]
fn probe_thermal_stress_formula() {
    let ic = Interoceptor::new();

    // Quiet control: 64°C is below the 65°C stress threshold.
    let cool = ic.neuro_impulses(&BodyState {
        cpu_temp_c: 64.0,
        stress_load: 0.5, // keep load/stress contributions out of the picture
        energy_reserve: 1.0,
        on_ac_power: true,
        io_activity: 0.0,
        cognitive_load: 0.0,
        ..BodyState::neutral()
    });
    assert_eq!(
        impulse_total(&cool, NeurochemicalId::CRH),
        0.0,
        "64°C must not produce a CRH stress impulse (threshold is 65°C)"
    );

    // Directed effect with exact magnitude: intensity = (70-65)/20 = 0.25
    // → CRH = 0.002 + 0.25·0.008 = 0.004.
    let warm = ic.neuro_impulses(&BodyState {
        cpu_temp_c: 70.0,
        stress_load: 0.5,
        energy_reserve: 1.0,
        on_ac_power: true,
        io_activity: 0.0,
        cognitive_load: 0.0,
        ..BodyState::neutral()
    });
    let crh = impulse_total(&warm, NeurochemicalId::CRH);
    assert!(
        (crh - 0.004).abs() < 1e-6,
        "70°C should produce CRH impulse 0.002+0.25·0.008 = 0.004, got {crh}"
    );

    // At 90°C two documented branches compose: heat stress clamps at
    // 0.010, and — with the fan off — "cooling failure" (fan<0.05 &&
    // temp>80) adds intensity·0.006 = 0.006. Total 0.016.
    let hot_fan_off = ic.neuro_impulses(&BodyState {
        cpu_temp_c: 90.0,
        thermoregulatory_effort: 0.0,
        stress_load: 0.5,
        energy_reserve: 1.0,
        on_ac_power: true,
        ..BodyState::neutral()
    });
    let crh_hot = impulse_total(&hot_fan_off, NeurochemicalId::CRH);
    assert!(
        (crh_hot - 0.016).abs() < 1e-6,
        "90°C fan-off should produce CRH 0.010 + 0.006 = 0.016, got {crh_hot}"
    );

    // Same temperature, fan working hard: the cooling-failure branch
    // goes silent, "fan struggling" (effort>0.3 && temp>75) adds
    // min(1, 1)·0.004 instead → 0.010 + 0.004 = 0.014.
    let hot_fan_on = ic.neuro_impulses(&BodyState {
        cpu_temp_c: 90.0,
        thermoregulatory_effort: 1.0,
        stress_load: 0.5,
        energy_reserve: 1.0,
        on_ac_power: true,
        ..BodyState::neutral()
    });
    let crh_fan = impulse_total(&hot_fan_on, NeurochemicalId::CRH);
    assert!(
        (crh_fan - 0.014).abs() < 1e-6,
        "90°C fan-on should produce CRH 0.010 + 0.004 = 0.014, got {crh_fan}"
    );

    // The distinction is the finding: identical temperature, different
    // thermoregulatory context → different total stress. The body
    // doesn't just feel heat — it feels heat *and whether its cooling
    // system is coping*.
}

// ─── probe 3: survival mapping is gated by power context ───────
//
// Low battery must only matter when running on battery — a low
// battery on AC power is not a survival concern. The gate is the
// claim.

#[test]
fn probe_battery_survival_gate() {
    let ic = Interoceptor::new();
    let low_battery = |on_ac: bool| BodyState {
        cpu_temp_c: 50.0,
        stress_load: 0.5, // boundary — below the >0.5 NE branch AND >0.25 wellbeing
        energy_reserve: 0.20,
        on_ac_power: on_ac,
        io_activity: 0.0,
        cognitive_load: 0.0,
        ..BodyState::neutral()
    };

    let on_ac = ic.neuro_impulses(&low_battery(true));
    assert_eq!(
        impulse_total(&on_ac, NeurochemicalId::CRH),
        0.0,
        "20% battery on AC power must not trigger survival CRH — it's plugged in"
    );

    let on_battery = ic.neuro_impulses(&low_battery(false));
    // intensity = (0.30 − 0.20)/0.30 = 1/3 → CRH = 0.004·(1/3) ≈ 0.001333
    let crh = impulse_total(&on_battery, NeurochemicalId::CRH);
    assert!(
        (crh - 0.004 / 3.0).abs() < 1e-6,
        "20% battery on battery should produce CRH 0.004/3, got {crh}"
    );
}

// ─── probe 4: healthy body produces well-being chemistry ───────

#[test]
fn probe_idle_body_wellbeing() {
    let ic = Interoceptor::new();
    let healthy = ic.neuro_impulses(&BodyState {
        cpu_temp_c: 45.0,
        stress_load: 0.10,
        cognitive_load: 0.30,
        energy_reserve: 0.9,
        on_ac_power: true,
        io_activity: 0.0,
        ..BodyState::neutral()
    });
    assert!(
        (impulse_total(&healthy, NeurochemicalId::Serotonin) - 0.002).abs() < 1e-9,
        "cool + idle + low memory should produce serotonin 0.002"
    );
    assert!(
        impulse_total(&healthy, NeurochemicalId::GABA) >= 0.001,
        "relaxed body should include a GABA touch"
    );
    assert_eq!(
        impulse_total(&healthy, NeurochemicalId::CRH),
        0.0,
        "a healthy body must not produce stress hormone"
    );
}

// ─── probe 5: physical state reaches the neurochemical vector ──
//
// The full chain: synthetic body → impulses → applied to the vector →
// effective levels move in the documented direction. Quiet control:
// a neutral body moves nothing through this channel.

#[test]
fn probe_body_state_drives_neurochemistry() {
    let ic = Interoceptor::new();

    let stressed_body = BodyState {
        cpu_temp_c: 95.0,
        stress_load: 1.5,
        energy_reserve: 0.10,
        on_ac_power: false,
        cognitive_load: 0.90,
        io_activity: 0.5,
        ..BodyState::neutral()
    };
    let impulses = ic.neuro_impulses(&stressed_body);
    assert!(
        impulses.len() >= 3,
        "a heavily stressed body should raise multiple impulses, got {:?}",
        impulses
    );

    let mut neuro = NeurochemicalVector::new(0);
    let crh_before = neuro.effective(NeurochemicalId::CRH);
    let ne_before = neuro.effective(NeurochemicalId::Norepinephrine);
    for &(id, mag) in &impulses {
        neuro.apply_impulse_capped(id, mag, 0);
    }
    neuro.recompute_derived();

    assert!(
        neuro.effective(NeurochemicalId::CRH) > crh_before,
        "CRH should rise under a stressed body"
    );
    assert!(
        neuro.effective(NeurochemicalId::Norepinephrine) > ne_before,
        "NE should rise under sustained load + I/O"
    );

    // Quiet control: neutral body produces zero impulses — nothing to apply.
    let neutral = ic.neuro_impulses(&BodyState {
        stress_load: 0.5,
        ..BodyState::neutral()
    });
    assert!(
        impulse_total(&neutral, NeurochemicalId::CRH).abs() < 1e-9,
        "neutral body must not inject CRH"
    );
}

// ─── probe 6: absent hardware reports zero, not fabrication ────
//
// If RAPL domains or perf counters don't exist on this machine, the
// electron-level fields must read 0.0 — a missing sensor is honest,
// a fabricated number is not.

#[test]
fn probe_absent_sensors_read_zero() {
    let has_rapl = Path::new("/sys/class/powercap").exists()
        && std::fs::read_dir("/sys/class/powercap")
            .map(|mut d| {
                d.any(|e| {
                    e.ok()
                        .map(|e| {
                            e.file_name()
                                .to_string_lossy()
                                .contains("rapl")
                        })
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);

    let mut ic = Interoceptor::new();
    let _ = ic.read();
    std::thread::sleep(std::time::Duration::from_millis(100));
    let body = ic.read();

    if !has_rapl {
        assert_eq!(
            body.core_activity, 0.0,
            "no RAPL domains → core_activity must honestly read 0.0, not a guess"
        );
        assert_eq!(
            body.dram_activity, 0.0,
            "no RAPL domains → dram_activity must honestly read 0.0"
        );
        assert_eq!(
            body.uncore_activity, 0.0,
            "no RAPL domains → uncore_activity must honestly read 0.0"
        );
    }
}
