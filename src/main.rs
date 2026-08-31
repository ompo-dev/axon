use axon_uic::{
    Authority, Capability, CapabilityGate, CostPrices, DecisionCertificate, Effect, Feasibility,
    Interval, Morphology, PhysicalCost, Refinement, Region, select_refinement,
};

fn interval(lower: i64, upper: i64) -> Interval {
    Interval::try_new(lower, upper).expect("constant bounds are valid")
}

fn main() {
    let decision = DecisionCertificate::try_from_utilities([
        ("observe", interval(48, 84)),
        ("act", interval(51, 79)),
    ])
    .expect("static demo action names are unique");
    let refinements = [
        Refinement::new("observe", 24, PhysicalCost::new(8, 4, 1)),
        Refinement::new("prove", 30, PhysicalCost::new(30, 8, 3)),
    ];
    let next = select_refinement(&refinements, CostPrices::unit()).unwrap();
    let plan = Morphology::allocate(
        100,
        [
            Region::new("trusted", 40, [(0, 0)]),
            Region::new("memory", 0, [(20, 50), (40, 70)]),
            Region::new("programs", 0, [(20, 60), (40, 65)]),
        ],
    )
    .expect("static demo budget is sufficient");
    let gate = CapabilityGate::new(Feasibility::ready(), Authority::from([Effect::ReadData]));
    let network = Capability::new("send report", [Effect::Network]);

    println!("AXON-UIC lab");
    println!(
        "decision certified: {:?}; ambiguity: {}",
        decision.certified_action(),
        decision.ambiguity()
    );
    println!("next refinement: {}", next.name());
    println!(
        "morphology: trusted={} memory={} programs={}",
        plan.bytes_for("trusted").unwrap(),
        plan.bytes_for("memory").unwrap(),
        plan.bytes_for("programs").unwrap()
    );
    println!("capability gate: {:?}", gate.evaluate(&network));
}
