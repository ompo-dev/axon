fn main() {
    println!(
        "{}",
        axon::experiments::run_scientific_suite()
            .v7_morphogenic
            .to_markdown()
    );
}
