fn main() {
    let report = axon::experiments::run_scientific_suite().v5_omega;
    println!("{}", report.to_markdown());
}
