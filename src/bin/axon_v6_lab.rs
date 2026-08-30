fn main() {
    println!(
        "{}",
        axon::experiments::run_scientific_suite()
            .v6_omega
            .to_markdown()
    );
}
