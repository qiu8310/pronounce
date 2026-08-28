# Pronounce CLI

Standalone English pronunciation tools. Scoring, TTS, and dictionary IPA each print one JSON object to **stdout**; logs go to stderr. Field contract: `FIELDS.md`.

## Install

Shared interpreter with mimora: `$MODELS_HOME/.venv` (Python 3.12). `mimora/.venv` and `pronounce/.venv` are symlinks to that directory. Install the CLI into it without pulling a second torch:

```bash
"$MODELS_HOME/.venv/bin/pip" install -e "$MODELS_HOME/pronounce" --no-deps
```

`tts` needs `kokoro` (already in the shared venv). `phonemes` needs espeak (`espeakng-loader`).

## Usage

```bash
"$MODELS_HOME/.venv/bin/python" -m pronounce score phoneme --text "..." --user take.wav [--ref ref.wav] [--lang en-us] [--device cpu] [--calibration cal.json]
"$MODELS_HOME/.venv/bin/python" -m pronounce score acoustic --text "..." --user take.wav [--ref actor.wav] [--voice af_heart] [--device cpu]
"$MODELS_HOME/.venv/bin/python" -m pronounce tts --text "Hello." --out /tmp/hello.wav [--voice af_heart] [--lang en-us] [--speed 0.8]
"$MODELS_HOME/.venv/bin/python" -m pronounce phonemes --text "Hello, how are you?" [--lang en-us]
"$MODELS_HOME/.venv/bin/python" -m pronounce schema
```

Copy-paste examples with the sample wavs in [`demo/`](demo/README.md).

| Flag | Meaning |
|------|---------|
| `--text` | English word, sentence, or paragraph |
| `--user` | User take wav (score) |
| `--ref` | Reference wav. Optional for phoneme. Optional for acoustic: if omitted, Kokoro synthesizes one (`ref_generated`) |
| `--out` | Output wav path (`tts`) |
| `--voice` | Kokoro voice id, default `af_heart` (`tts`; acoustic auto-ref) |
| `--speed` | Listen tempo for `tts`, default `1`. `0.8` is slower |
| `--lang` | `en-us` / `en-gb` |
| `--device` | `cpu` or `cuda`, default `cpu` |
| `--calibration` | Per-user `calibration.json` (score) |
| `--user-name` | Name stored with that calibration (score) |

`--text` may be a paragraph. IPA is one row per whitespace token. TTS concatenates Kokoro chunks. Scoring still works best on a short take that matches the text.

## Model paths

Loads local snapshots under `$MODELS_HOME` (no download at run time). If unset, the CLI infers the directory that contains `llm/`.

| 能力 | 路径 |
|------|------|
| score phoneme | `$MODELS_HOME/llm/wav2vec2/wav2vec2-xlsr-53-espeak-cv-ft` |
| score acoustic | `$MODELS_HOME/llm/wav2vec2/wav2vec2-large-960h` |
| tts (and acoustic auto-ref) | `$MODELS_HOME/llm/kokoro/Kokoro-82M` |
| tts G2P (internal) | `$MODELS_HOME/llm/spacy`（`en_core_web_sm`） |
| phonemes (dictionary IPA) | espeak-ng via `espeakng-loader` |

Wav2Vec2 directories must be valid `from_pretrained` roots. Kokoro needs `config.json`、`kokoro-v1_0.pth` 和 `voices/*.pt`。

## espeak-ng

Phoneme scoring, dictionary IPA, and Kokoro's unknown-word fallback need espeak. `espeakng-loader` ships the library; `pronounce.common.espeak` registers it. A system espeak-ng remains a valid fallback.

## Output

Success: one JSON object on stdout, exit 0. Failure: `{"ok": false, ... "error": "..."}` on stdout, exit 1. `tts` writes a wav to `--out` (24 kHz at `--speed 1`). Score includes `prosody` contours when audio is present. `schema` prints `FIELDS.md`.
