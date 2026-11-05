//! Optional lab checkpoints. Disabled telemetry does not serialize or write.
use std::{
    fmt::Write as _,
    io::{self, Write},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy)]
pub enum Value<'a> {
    Text(&'a str),
    Count(u64),
    Number(f64),
    Flag(bool),
    Null,
    Counts(&'a [u64]),
    Texts(&'a [&'a str]),
}
fn quoted(out: &mut String, value: &str) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                write!(out, "\\u{:04x}", c as u32).expect("string");
            }
            c => out.push(c),
        }
    }
    out.push('"');
}
pub struct LabTelemetry<W: Write> {
    output: W,
    enabled: bool,
    sequence: u64,
    started: Instant,
    experiment: &'static str,
}
impl LabTelemetry<io::Stdout> {
    pub fn from_environment(experiment: &'static str) -> Self {
        Self::new(
            io::stdout(),
            std::env::var_os("AXON_LAB_TELEMETRY").is_some_and(|v| v == "1"),
            experiment,
        )
    }
}
impl<W: Write> LabTelemetry<W> {
    pub fn new(output: W, enabled: bool, experiment: &'static str) -> Self {
        Self {
            output,
            enabled,
            sequence: 0,
            started: Instant::now(),
            experiment,
        }
    }
    pub fn enabled(&self) -> bool {
        self.enabled
    }
    pub fn emit(&mut self, event: &str, fields: &[(&str, Value<'_>)]) -> io::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        for (key, value) in fields {
            if !key.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
                || key.is_empty()
                || ["event", "experiment", "coreSeq", "coreNs", "unixMs"].contains(key)
                || matches!(value, Value::Number(v) if !v.is_finite())
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "invalid telemetry field",
                ));
            }
        }
        self.sequence += 1;
        let mut row = String::from("{\"event\":");
        quoted(&mut row, event);
        row.push_str(",\"experiment\":");
        quoted(&mut row, self.experiment);
        write!(
            row,
            ",\"coreSeq\":{},\"coreNs\":{},\"unixMs\":{}",
            self.sequence,
            self.started.elapsed().as_nanos(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        )
        .expect("string");
        for (key, value) in fields {
            row.push(',');
            quoted(&mut row, key);
            row.push(':');
            match value {
                Value::Text(v) => quoted(&mut row, v),
                Value::Count(v) => write!(row, "{v}").expect("string"),
                Value::Number(v) => write!(row, "{v}").expect("string"),
                Value::Flag(v) => write!(row, "{v}").expect("string"),
                Value::Null => row.push_str("null"),
                Value::Counts(values) => {
                    row.push('[');
                    for (i, v) in values.iter().enumerate() {
                        if i > 0 {
                            row.push(',');
                        }
                        write!(row, "{v}").expect("string");
                    }
                    row.push(']');
                }
                Value::Texts(values) => {
                    row.push('[');
                    for (i, value) in values.iter().enumerate() {
                        if i > 0 {
                            row.push(',');
                        }
                        quoted(&mut row, value);
                    }
                    row.push(']');
                }
            }
        }
        row.push_str("}\n");
        self.output.write_all(row.as_bytes())?;
        self.output.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_never_writes_or_serializes() {
        struct Forbidden;
        impl Write for Forbidden {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                panic!("disabled write")
            }
            fn flush(&mut self) -> io::Result<()> {
                panic!("disabled flush")
            }
        }
        LabTelemetry::new(Forbidden, false, "test")
            .emit("x", &[("invalid-key", Value::Number(f64::NAN))])
            .unwrap();
    }
    #[test]
    fn typed_records_escape_text_and_reject_nonfinite_reserved_keys() {
        let mut writer = LabTelemetry::new(Vec::new(), true, "test");
        writer
            .emit(
                "progress",
                &[
                    ("text", Value::Text("a\n\"\\\t\u{1}")),
                    ("count", Value::Count(2)),
                    ("steps", Value::Texts(&["read\"source", "derive\\value"])),
                ],
            )
            .unwrap();
        let line = String::from_utf8(writer.output.clone()).unwrap();
        assert!(line.contains("\"coreSeq\":1"));
        assert!(line.contains("a\\n\\\"\\\\\\t\\u0001"));
        assert!(line.contains("\"steps\":[\"read\\\"source\",\"derive\\\\value\"]"));
        assert_eq!(line.lines().count(), 1);
        assert!(writer.emit("x", &[("coreSeq", Value::Count(42))]).is_err());
        assert!(
            writer
                .emit("x", &[("number", Value::Number(f64::INFINITY))])
                .is_err()
        );
        writer.emit("end", &[]).unwrap();
        assert_eq!(writer.sequence, 2);
    }
}
