fn main() {
    println!(
        "{}",
        axon::experiments::run_scientific_suite().to_markdown()
    );
}
