# Pronounce CLI

Standalone pronunciation scoring for English takes. Two engines (phoneme and acoustic) print one JSON object to **stdout** per invocation; logs and progress go to stderr. See `FIELDS.md` for the full field contract.

## Install

Shared interpreter with Mimora: `$MODELS_HOME/.venv` (Python 3.12). `mimora/.venv` and `pronounce/.venv` are symlinks to that directory. Install the CLI into it without pulling a second torch:

```bash
"$MODELS_HOME/.venv/bin/pip" install -e "$MODELS_HOME/pronounce" --no-deps
```

## Usage

```bash
"$MODELS_HOME/.venv/bin/python" -m pronounce score phoneme --text "..." --user take.wav [--ref ref.wav] [--lang en-us] [--device cpu]
"$MODELS_HOME/.venv/bin/python" -m pronounce score acoustic --text "..." --user take.wav --ref actor.wav [--device cpu]
"$MODELS_HOME/.venv/bin/python" -m pronounce schema
```

| Flag | Meaning |
|------|---------|
| `--text` | Expected English phrase (required) |
| `--user` | User take wav path (required) |
| `--ref` | Reference wav. Optional for phoneme; required for acoustic |
| `--lang` | espeak dialect, default `en-us` (phoneme only) |
| `--device` | `cpu` or `cuda`, default `cpu` |

## Model paths

Both engines load local Hugging Face snapshots under `$MODELS_HOME` (no download at score time). If the variable is unset, the CLI infers the directory that contains `llm/`.

| Engine | Path |
|--------|------|
| phoneme | `$MODELS_HOME/llm/wav2vec2/wav2vec2-xlsr-53-espeak-cv-ft` |
| acoustic | `$MODELS_HOME/llm/wav2vec2/wav2vec2-large-960h` |

Each directory must be a valid `from_pretrained` root (`config.json` + weights).

## espeak-ng

Phoneme scoring needs espeak for reference transcription. The `espeakng-loader` wheel ships the espeak-ng shared library and its data; `pronounce.common.espeak` registers them with phonemizer before the first phonemization. A system-installed espeak-ng remains a valid fallback when the wheel is absent.

## Output

Success: one JSON object on stdout, exit 0. Failure: `{"ok": false, "engine": "...", "error": "..."}` on stdout, exit 1. Run `"$MODELS_HOME/.venv/bin/python" -m pronounce schema` to print the full `FIELDS.md` contract.
