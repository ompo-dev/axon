use super::*;

fn reheader(bytes: &[u8], magic: &[u8; 8], change: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
    let (b, end) = body(bytes, magic).unwrap();
    let mut header: serde_json::Value = serde_json::from_slice(&b[12..end]).unwrap();
    change(&mut header);
    let mut altered = framed(magic, &header).unwrap();
    altered.extend_from_slice(&b[end..]);
    finish(altered).unwrap()
}

#[test]
fn valid_checksums_do_not_bypass_typed_state_and_genome_boundaries() {
    let model = Model::new(genes(Rule::Hebbian, Position::Rotary)).unwrap();
    let bytes = model.to_bytes().unwrap();
    for (name, value) in [
        ("revision", serde_json::json!(2)),
        ("extra", serde_json::json!(true)),
    ] {
        let altered = reheader(&bytes, MAGIC, |h| h[name] = value);
        assert!(Model::from_bytes(&altered).is_err());
    }
    let state = model.begin("owner", Mode::Sequence).unwrap();
    let checkpoint = model.snapshot(&state).unwrap();
    for (name, value) in [
        ("revision", serde_json::json!(2)),
        ("extra", serde_json::json!(true)),
        ("slots", serde_json::json!(u64::MAX)),
        ("position", serde_json::json!(MAX_POSITION + 1)),
        ("scope", serde_json::json!("other")),
        ("mode", serde_json::json!("association")),
        ("model", serde_json::json!(model.identity.wrapping_add(1))),
    ] {
        let altered = reheader(&checkpoint, STATE_MAGIC, |h| h[name] = value);
        assert!(model.restore("owner", Mode::Sequence, &altered).is_err());
    }
    let (body, end) = body(&checkpoint, STATE_MAGIC).unwrap();
    for value in [f32::NAN, f32::INFINITY, 1e10] {
        let mut altered = body.to_vec();
        altered[end..end + 4].copy_from_slice(&value.to_le_bytes());
        assert!(
            model
                .restore("owner", Mode::Sequence, &finish(altered).unwrap())
                .is_err()
        );
    }
    let mut g = genes(Rule::Hebbian, Position::Rotary);
    g.dimensions = 32;
    g.neurons = 256;
    g.layers = 4;
    let large = Model::new(g).unwrap();
    let mut state = large.begin("bounded", Mode::Sequence).unwrap();
    let before = large.snapshot(&state).unwrap();
    assert!(
        large
            .step(
                &mut state,
                &vectors(32, 1)[0],
                2_000_000,
                262144 - large.retained_bytes()
            )
            .is_err()
    );
    assert_eq!(before, large.snapshot(&state).unwrap());
}

fn genes(rule: Rule, position: Position) -> Genes {
    Genes {
        seed: 47,
        dimensions: 8,
        neurons: 32,
        layers: 2,
        position,
        rule,
        decay: 1.0,
        learning_rate: 1.0,
    }
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= 1e-4 + 1e-4 * y.abs(), "{x} vs {y}");
    }
}
fn vectors(d: usize, count: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            (0..d)
                .map(|j| (((i * 17 + j * 11 + 3) % 31) as f32 - 15.0) / 8.0)
                .collect()
        })
        .collect()
}

#[test]
fn plasticity_changes_state_not_genes_and_delta_corrects_repeated_associations() {
    let key = vectors(8, 1).remove(0);
    let mut old = vec![0.0; 8];
    old[2] = 1.0;
    let mut new = vec![0.0; 8];
    new[5] = 1.0;
    for rule in [Rule::Frozen, Rule::Hebbian, Rule::Delta] {
        let model = Model::new(genes(rule, Position::None)).unwrap();
        let original = model.to_bytes().unwrap();
        let mut state = model.begin("private-owner", Mode::Association).unwrap();
        let zero = model.query(&state, &key, 2_000_000, 262144).unwrap();
        assert_eq!(zero.output, vec![0.0; 8]);
        for _ in 0..4 {
            model
                .associate(&mut state, &key, &old, 2_000_000, 262144)
                .unwrap();
        }
        model
            .associate(&mut state, &key, &new, 2_000_000, 262144)
            .unwrap();
        let before = model.snapshot(&state).unwrap();
        let answer = model.query(&state, &key, 2_000_000, 262144).unwrap();
        assert_eq!(before, model.snapshot(&state).unwrap());
        assert_eq!(answer.synapses_written, 0);
        match rule {
            Rule::Frozen => assert_eq!(answer.output, vec![0.0; 8]),
            Rule::Hebbian => assert!(answer.output[2] > answer.output[5]),
            Rule::Delta => close(&answer.output, &new),
        }
        assert_eq!(model.to_bytes().unwrap(), original);
        let fresh = model.begin("private-owner", Mode::Association).unwrap();
        assert_eq!(
            model.query(&fresh, &key, 2_000_000, 262144).unwrap().output,
            vec![0.0; 8]
        );
        let restored = model
            .restore("private-owner", Mode::Association, &before)
            .unwrap();
        assert_eq!(model.snapshot(&restored).unwrap(), before);
        assert!(
            model
                .restore("another-owner", Mode::Association, &before)
                .is_err()
        );
        assert!(
            model
                .restore("private-owner", Mode::Sequence, &before)
                .is_err()
        );
        let mut other = genes(rule, Position::None);
        other.seed = 71;
        assert!(
            Model::new(other)
                .unwrap()
                .restore("private-owner", Mode::Association, &before)
                .is_err()
        );
        assert!(Model::from_bytes(&before).is_err());
        assert!(
            model
                .restore("private-owner", Mode::Association, &original)
                .is_err()
        );
    }
}

// Independent dense, strictly-lower-triangular attention oracle, not streaming S.
fn dense(model: &Model, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let g = model.genes();
    let (d, n) = (g.dimensions, g.neurons);
    let [enc, enc_v, dec] = model.reference_parameters();
    let norm = |v: &[f32]| {
        let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
        let variance = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / v.len() as f64;
        v.iter()
            .map(|x| ((*x as f64 - mean) / (variance + 1e-5).sqrt()) as f32)
            .collect::<Vec<_>>()
    };
    let mul = |v: &[f32], m: &[f32], width: usize| {
        (0..width)
            .map(|j| (0..v.len()).map(|i| v[i] * m[i * width + j]).sum::<f32>())
            .collect::<Vec<_>>()
    };
    let mut x = inputs.iter().map(|v| norm(v)).collect::<Vec<_>>();
    for _ in 0..g.layers {
        let values = x.clone();
        let positive = x
            .iter()
            .map(|v| {
                mul(v, &enc, n)
                    .into_iter()
                    .map(|v| v.max(0.0))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut keys = positive.clone();
        if g.position == Position::Rotary {
            for (t, k) in keys.iter_mut().enumerate() {
                for j in (0..n).step_by(2) {
                    let a = t as f64 / 65536f64.powf(j as f64 / n as f64);
                    let (c, s) = (a.cos() as f32, a.sin() as f32);
                    let (v, w) = (k[j], k[j + 1]);
                    k[j] = v * c - w * s;
                    k[j + 1] = w * c + v * s;
                }
            }
        }
        for t in 0..x.len() {
            let mut attention = vec![0.0; d];
            for past in 0..t {
                let weight = keys[t]
                    .iter()
                    .zip(&keys[past])
                    .map(|(q, k)| q * k)
                    .sum::<f32>();
                for (j, a) in attention.iter_mut().enumerate() {
                    *a += weight * values[past][j];
                }
            }
            let mapped = mul(&norm(&attention), &enc_v, n);
            let gates = mapped
                .iter()
                .zip(&positive[t])
                .map(|(a, b)| a.max(0.0) * b)
                .collect::<Vec<_>>();
            let y = norm(&mul(&gates, &dec, d));
            x[t] = norm(
                &values[t]
                    .iter()
                    .zip(y)
                    .map(|(a, b)| a + b)
                    .collect::<Vec<_>>(),
            );
        }
    }
    x
}

#[test]
fn recurrent_bdh_cell_matches_dense_past_attention_and_checkpoint_chunks() {
    for position in [Position::None, Position::Rotary] {
        let model = Model::new(genes(Rule::Hebbian, position)).unwrap();
        let inputs = vectors(8, 17);
        let expected = dense(&model, &inputs);
        let mut state = model.begin("sequence-a", Mode::Sequence).unwrap();
        let mut outputs = Vec::new();
        let size = state.retained_bytes();
        for (i, input) in inputs.iter().enumerate() {
            let result = model.step(&mut state, input, 2_000_000, 262144).unwrap();
            close(&result.output, &expected[i]);
            assert_eq!(result.position, i);
            assert_eq!(result.neurons_active.len(), 2);
            assert!(result.live_bytes < 262144);
            assert_eq!(state.retained_bytes(), size);
            outputs.push(result.output);
            if i == 7 {
                state = model
                    .restore(
                        "sequence-a",
                        Mode::Sequence,
                        &model.snapshot(&state).unwrap(),
                    )
                    .unwrap();
            }
        }
        let prefix = dense(&model, &inputs[..8]);
        for i in 0..8 {
            close(&prefix[i], &outputs[i]);
        }
        let mut uninterrupted = model.begin("sequence-a", Mode::Sequence).unwrap();
        for input in &inputs {
            model
                .step(&mut uninterrupted, input, 2_000_000, 262144)
                .unwrap();
        }
        assert_eq!(
            model.snapshot(&state).unwrap(),
            model.snapshot(&uninterrupted).unwrap()
        );
    }
}

#[test]
fn failures_are_atomic_and_codecs_refuse_cross_identity_and_corruption() {
    let model = Model::new(genes(Rule::Hebbian, Position::Rotary)).unwrap();
    let bytes = model.to_bytes().unwrap();
    let restored = Model::from_bytes(&bytes).unwrap();
    assert_eq!(restored.to_bytes().unwrap(), bytes);
    assert_eq!(restored.retained_bytes(), model.retained_bytes());
    let mut state = model.begin("scope", Mode::Sequence).unwrap();
    let before = model.snapshot(&state).unwrap();
    for input in [vec![0.0; 7], vec![f32::NAN; 8], vec![1e7; 8]] {
        assert!(model.step(&mut state, &input, 2_000_000, 262144).is_err());
        assert_eq!(before, model.snapshot(&state).unwrap());
    }
    for (work, mem) in [
        (0, 262144),
        (1, 262144),
        (2_000_001, 262144),
        (2_000_000, 0),
    ] {
        assert!(model.step(&mut state, &[1.0; 8], work, mem).is_err());
        assert_eq!(before, model.snapshot(&state).unwrap());
    }
    let mut broken = bytes.clone();
    let end = broken.len() - 9;
    broken[end] ^= 1;
    assert!(Model::from_bytes(&broken).is_err());
    assert!(Model::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    let mut corrupt = before.clone();
    let end = corrupt.len() - 9;
    corrupt[end] ^= 1;
    assert!(model.restore("scope", Mode::Sequence, &corrupt).is_err());
    state.position = MAX_POSITION;
    let last = model.snapshot(&state).unwrap();
    assert!(
        model
            .step(&mut state, &[1.0; 8], 2_000_000, 262144)
            .is_err()
    );
    assert_eq!(last, model.snapshot(&state).unwrap());
    assert!(model.begin("", Mode::Sequence).is_err());
    for (d, n, l) in [(3, 32, 2), (8, 15, 2), (8, 33, 2), (8, 32, 0), (8, 32, 5)] {
        let mut g = genes(Rule::Hebbian, Position::None);
        g.dimensions = d;
        g.neurons = n;
        g.layers = l;
        assert!(Model::new(g).is_err());
    }
}

#[test]
fn neuron_and_latent_dimensions_are_independent_and_bounded() {
    for d in [4, 8, 16, 32] {
        for n in [16, 64, 128] {
            let mut g = genes(Rule::Hebbian, Position::Rotary);
            g.dimensions = d;
            g.neurons = n;
            let model = Model::new(g).unwrap();
            let mut state = model.begin("dimensions", Mode::Sequence).unwrap();
            for v in vectors(d, 5) {
                let out = model.step(&mut state, &v, 2_000_000, 262144).unwrap();
                assert_eq!(out.output.len(), d);
                assert!(out.neurons_active.iter().all(|a| *a <= n));
            }
            assert!(model.snapshot(&state).unwrap().len() < MAX_BYTES);
        }
    }
}
