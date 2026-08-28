# Pronounce CLI

Standalone English pronunciation tools. Scoring engines (phoneme / acoustic) and Kokoro TTS / G2P each print one JSON object to **stdout**; logs go to stderr. Score fields: `FIELDS.md`.

## Install

Shared interpreter with mimora: `$MODELS_HOME/.venv` (Python 3.12). `mimora/.venv` and `pronounce/.venv` are symlinks to that directory. Install the CLI into it without pulling a second torch:

```bash
"$MODELS_HOME/.venv/bin/pip" install -e "$MODELS_HOME/pronounce" --no-deps
```

`tts` / `phonemes` need `kokoro` (already in the shared venv).

## Usage

```bash
"$MODELS_HOME/.venv/bin/python" -m pronounce score phoneme --text "..." --user take.wav [--ref ref.wav] [--lang en-us] [--device cpu]
"$MODELS_HOME/.venv/bin/python" -m pronounce score acoustic --text "..." --user take.wav --ref actor.wav [--device cpu]
"$MODELS_HOME/.venv/bin/python" -m pronounce tts --text "Hello." --out /tmp/hello.wav [--voice af_heart] [--lang en-us] [--device cpu]
"$MODELS_HOME/.venv/bin/python" -m pronounce phonemes --text "Hello, how are you?" [--lang en-us]
"$MODELS_HOME/.venv/bin/python" -m pronounce schema
```

| Flag | Meaning |
|------|---------|
| `--text` | English phrase (required for score / tts / phonemes) |
| `--user` | User take wav (score) |
| `--ref` | Reference wav. Optional for phoneme score; required for acoustic |
| `--out` | Output wav path (`tts`) |
| `--voice` | Kokoro voice id, default `af_heart` (`tts`) |
| `--lang` | `en-us` / `en-gb` (score phoneme, tts, phonemes) |
| `--device` | `cpu` or `cuda`, default `cpu` |

## Model paths

Loads local snapshots under `$MODELS_HOME` (no download at run time). If unset, the CLI infers the directory that contains `llm/`.

| 能力 | 路径 |
|------|------|
| score phoneme | `$MODELS_HOME/llm/wav2vec2/wav2vec2-xlsr-53-espeak-cv-ft` |
| score acoustic | `$MODELS_HOME/llm/wav2vec2/wav2vec2-large-960h` |
| tts | `$MODELS_HOME/llm/kokoro/Kokoro-82M` |
| phonemes / tts G2P | `$MODELS_HOME/llm/spacy`（`en_core_web_sm`） |

Wav2Vec2 directories must be valid `from_pretrained` roots. Kokoro needs `config.json`、`kokoro-v1_0.pth` 和 `voices/*.pt`。

## espeak-ng

Phoneme scoring and Kokoro G2P fallback need espeak. `espeakng-loader` ships the library; `pronounce.common.espeak` registers it. A system espeak-ng remains a valid fallback.

## Output

Success: one JSON object on stdout, exit 0. Failure: `{"ok": false, ... "error": "..."}` on stdout, exit 1. `tts` also writes a 24 kHz wav to `--out`. `schema` prints `FIELDS.md`.

