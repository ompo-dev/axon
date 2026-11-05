fn main() {
    if let Err(error) = axon_uic::curriculum::command() {
        eprintln!("curriculum: {error}");
        std::process::exit(1);
    }
}
