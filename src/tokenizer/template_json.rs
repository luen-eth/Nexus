//! HF chat templates use Python json.dumps, without Jinja's HTML escaping.
use minijinja::{
    value::{Kwargs, Rest, Value, ValueOrKwargs},
    Error, ErrorKind,
};
use serde_json::Value as Json;

pub(super) fn tojson(value: Value, rest: Rest<ValueOrKwargs>) -> Result<String, Error> {
    let rest = rest.into_values();
    let (positional, kwargs): (&[Value], Kwargs) = minijinja::value::from_args(&rest)?;
    if positional.len() > 4 {
        return Err(Error::from(ErrorKind::TooManyArguments));
    }
    let option = |positional: Option<Value>, name: &str| -> Result<Json, Error> {
        let named: Option<Value> = kwargs.get(name)?;
        if positional.is_some() && named.is_some() {
            return Err(invalid("Duplicate JSON option"));
        }
        serde_json::to_value(positional.or(named).unwrap_or(Value::UNDEFINED))
            .map_err(|e| invalid(&e.to_string()))
    };
    let ascii = option(positional.first().cloned(), "ensure_ascii")?
        .as_bool()
        .unwrap_or(false);
    let indent = match option(positional.get(1).cloned(), "indent")? {
        Json::Null => None,
        Json::String(s) if s.len() <= 128 => Some(s),
        Json::Number(n) if n.as_i64().is_some_and(|n| n <= 128) => {
            Some(" ".repeat(n.as_i64().unwrap().max(0) as usize))
        }
        _ => {
            return Err(invalid(
                "JSON indent must be a string or integer of at most 128",
            ))
        }
    };
    let separators = match option(positional.get(2).cloned(), "separators")? {
        Json::Null => (
            if indent.is_some() { "," } else { ", " }.into(),
            ": ".into(),
        ),
        Json::Array(items) if items.len() == 2 && items.iter().all(Json::is_string) => (
            items[0].as_str().unwrap().into(),
            items[1].as_str().unwrap().into(),
        ),
        _ => return Err(invalid("JSON separators must contain two strings")),
    };
    let sort = option(positional.get(3).cloned(), "sort_keys")?
        .as_bool()
        .unwrap_or(false);
    kwargs.assert_all_used()?;
    let json = serde_json::to_value(value).map_err(|e| invalid(&e.to_string()))?;
    let mut output = String::new();
    write_json(&json, &mut output, indent.as_deref(), &separators, sort, 0);
    if ascii {
        let mut escaped = String::new();
        for ch in output.chars() {
            if ch.is_ascii() {
                escaped.push(ch);
            } else {
                for unit in ch.encode_utf16(&mut [0; 2]) {
                    escaped.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
        Ok(escaped)
    } else {
        Ok(output)
    }
}
fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidOperation, message.to_owned())
}
fn write_json(
    value: &Json,
    out: &mut String,
    indent: Option<&str>,
    sep: &(String, String),
    sort: bool,
    depth: usize,
) {
    let newline = |out: &mut String, depth: usize| {
        if let Some(indent) = indent {
            out.push('\n');
            out.push_str(&indent.repeat(depth));
        }
    };
    match value {
        Json::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&sep.0);
                }
                newline(out, depth + 1);
                write_json(item, out, indent, sep, sort, depth + 1);
            }
            if !items.is_empty() {
                newline(out, depth);
            }
            out.push(']');
        }
        Json::Object(items) => {
            out.push('{');
            let mut entries: Vec<_> = items.iter().collect();
            if sort {
                entries.sort_by_key(|(key, _)| *key);
            }
            for (i, (key, value)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(&sep.0);
                }
                newline(out, depth + 1);
                out.push_str(&serde_json::to_string(key).expect("JSON string"));
                out.push_str(&sep.1);
                write_json(value, out, indent, sep, sort, depth + 1);
            }
            if !items.is_empty() {
                newline(out, depth);
            }
            out.push('}');
        }
        _ => out.push_str(&value.to_string()),
    }
}
