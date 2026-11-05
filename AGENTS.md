# AXON Engineering

The single canonical project document is `docs/AXON.md`. Read its current-state,
architecture and known-errors sections before planning changes. Historical
appendices are evidence, not current instructions. Update this document rather
than creating competing vision/result guides; keep immutable experimental data.

For every coding task in this repository, read and apply the user's skill:
`C:/Users/maico/Downloads/zero-write-engineering/zero-write-engineering/SKILL.md`.
Use balanced mode by default. Reuse existing contracts before creating code;
preserve unrelated working-tree changes; use scoped patches and verify results.
If that local skill is unavailable, report the limitation rather than claiming use.

The UI lives in `ui/`: React, TypeScript, Bun, Tailwind and Motion. The local
server reuses the Rust observer and CLI. Never simulate telemetry or present
stored experiments as live activity. Unknown measurements remain unavailable.
Run `bun run check`, `bun run build`, and `bun run test` from `ui/` after UI edits.
The real-core integration suite is opt-in: `AXON_UI_REAL_CORE=1`.
