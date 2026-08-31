fn main() {
    println!(
        "{}",
        axon::experiments::run_scientific_suite()
            .lambda_squared
            .to_markdown()
    );
}
