fn main() {
    println!(
        "{}",
        axon::experiments::run_scientific_suite()
            .lambda
            .to_markdown()
    );
}
