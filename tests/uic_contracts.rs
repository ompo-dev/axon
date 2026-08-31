use std::process::Command;

use axon_uic::{
    Authority, Capability, CapabilityGate, Effect, ExecutionMode, Feasibility, OptimizationFailure,
    run_checked,
};

#[test]
fn public_capability_and_fallback_contracts_hold_together() {
    let gate = CapabilityGate::new(Feasibility::ready(), Authority::from([Effect::ReadData]));
    let read = Capability::new("read local state", [Effect::ReadData]);
    let network = Capability::new("send remote report", [Effect::Network]);

    assert!(gate.evaluate(&read).is_ok());
    assert!(gate.evaluate(&network).is_err());
    assert_eq!(
        run_checked::<u64>(Err(OptimizationFailure::InvalidCertificate), || 7).mode(),
        ExecutionMode::ExactFallback
    );
}

#[test]
fn lab_binary_exposes_ambiguous_decision_and_denied_effect() {
    let output = Command::new(env!("CARGO_BIN_EXE_axon-uic"))
        .output()
        .expect("lab binary starts");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("lab output is UTF-8");
    assert!(stdout.contains("decision certified: None"));
    assert!(stdout.contains("Unauthorized(Network)"));
}
