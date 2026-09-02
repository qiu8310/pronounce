# Pronounce CLI

Standalone English pronunciation tools. Scoring, TTS, and dictionary IPA each print one JSON object to **stdout**; logs go to stderr. Field contract: `FIELDS.md`.

## Install

Shared interpreter with mimora: `$MODELS_HOME/.venv` (Python 3.12). `mimora/.venv` and `pronounce/.venv` are symlinks to that directory. Install the CLI into it without pulling a second torch:

```bash
"$MODELS_HOME/.venv/bin/pip" install -e "$MODELS_HOME/pronounce" --no-deps
```

`tts` needs `kokoro` (already in the shared venv). `tts-zh` needs `melotts` plus Chinese G2P extras (`pypinyin`, `jieba`, `cn2an`); install those into the shared venv without pulling a second torch (`pip install ... --no-deps` for MeloTTS itself). `phonemes` needs espeak (`espeakng-loader`).

## Usage

```bash
"$MODELS_HOME/.venv/bin/python" -m pronounce score phoneme --text "..." --user take.wav [--ref ref.wav] [--lang en-us] [--device cpu] [--calibration cal.json]
"$MODELS_HOME/.venv/bin/python" -m pronounce score phoneme --ipa ɪ --user take.wav [--ref ref.wav] [--lang en-gb]
"$MODELS_HOME/.venv/bin/python" -m pronounce score acoustic --text "..." --user take.wav [--ref actor.wav] [--voice af_heart] [--device cpu]
"$MODELS_HOME/.venv/bin/python" -m pronounce tts --text "Hello." --out /tmp/hello.wav [--voice af_heart] [--lang en-us] [--speed 0.8]
"$MODELS_HOME/.venv/bin/python" -m pronounce tts --ipa ɪ --out /tmp/ih.wav [--lang en-gb]
"$MODELS_HOME/.venv/bin/python" -m pronounce tts-zh --text "你好。" --out /tmp/nihao.wav [--device cpu] [--speed 0.8]
"$MODELS_HOME/.venv/bin/python" -m pronounce phonemes --text "Hello, how are you?" [--lang en-us]
"$MODELS_HOME/.venv/bin/python" -m pronounce schema
"$MODELS_HOME/.venv/bin/python" -m pronounce serve --port 8787
```

Copy-paste examples with the sample wavs in [`demo/`](demo/README.md) (CLI and `serve`).

Resident HTTP (loopback only): see [Serve](#serve). Isolated phones use `--ipa` / HTTP `"ipa"` (Unicode, e.g. `"ɪ"`): TTS speaks them with espeak-ng (Kokoro cannot); phoneme score uses that phone as the expected sequence and skips G2P.

| Flag | Meaning |
|------|---------|
| `--text` | Target phrase (`tts` English; `tts-zh` Chinese; score/phonemes English). Optional for `tts` / `score phoneme` when `--ipa` is set |
| `--ipa` | Isolated IPA phone (`tts` and `score phoneme`). Speaks with espeak-ng; scoring skips G2P |
| `--user` | User take wav (score) |
| `--ref` | Reference wav. Optional for phoneme. Optional for acoustic: if omitted, Kokoro synthesizes one (`ref_generated`) |
| `--out` | Output wav path (`tts` / `tts-zh`) |
| `--voice` | Kokoro voice id, default `af_heart` (`tts`; acoustic auto-ref) |
| `--speed` | Listen tempo for `tts` / `tts-zh`, default `1`. `0.8` is slower |
| `--lang` | `en-us` / `en-gb` (`tts` and score; Chinese TTS is `tts-zh`, not `--lang zh`) |
| `--device` | `cpu` or `cuda`, default `cpu` |
| `--calibration` | Per-user `calibration.json` (score) |
| `--user-name` | Name stored with that calibration (score) |

`--text` may be a paragraph. Dictionary IPA (`phonemes`) is one row per whitespace token. Sentence TTS concatenates Kokoro chunks. Isolated-phone TTS (`--ipa` / HTTP `"ipa"`) is one espeak-ng phone, not Kokoro. Scoring still works best on a short take that matches the text (or a single `--ipa`).

## Model paths

Loads local snapshots under `$MODELS_HOME` (no download at run time). If unset, the CLI infers the directory that contains `llm/`.

| 能力 | 路径 |
|------|------|
| score phoneme | `$MODELS_HOME/llm/wav2vec2/wav2vec2-xlsr-53-espeak-cv-ft` |
| score acoustic | `$MODELS_HOME/llm/wav2vec2/wav2vec2-large-960h` |
| tts (and acoustic auto-ref) | `$MODELS_HOME/llm/kokoro/Kokoro-82M` |
| tts-zh | `$MODELS_HOME/llm/melo/MeloTTS-Chinese` + `chinese-roberta-wwm-ext-large` |
| tts G2P (internal) | `$MODELS_HOME/llm/spacy`（`en_core_web_sm`） |
| phonemes (dictionary IPA) | espeak-ng via `espeakng-loader` |
| isolated-phone HTTP TTS | system `espeak-ng` / `espeak` on `PATH` |

Wav2Vec2 directories must be valid `from_pretrained` roots. Kokoro needs `config.json`、`kokoro-v1_0.pth` 和 `voices/*.pt`。

## espeak-ng

Phoneme scoring, dictionary IPA, and Kokoro's unknown-word fallback need the espeak **library**. `espeakng-loader` ships it; `pronounce.common.espeak` registers it. A system espeak-ng remains a valid fallback.

Isolated-phone `tts --ipa` / `POST /tts` (`"ipa"`) needs the espeak **binary** on `PATH` (`espeak-ng` or `espeak`). That path does not use `espeakng-loader`.

## Output

Success: one JSON object on stdout, exit 0. Failure: `{"ok": false, ... "error": "..."}` on stdout, exit 1. CLI `tts --text` writes a wav to `--out` (24 kHz at `--speed 1`). Isolated-phone TTS (`--ipa` or HTTP `"ipa"`) writes espeak-ng's rate (22.05 kHz). `tts-zh` writes 44.1 kHz. Score includes `prosody` contours when audio is present. `schema` prints `FIELDS.md`. `serve` is a long-running process: JSON is in the HTTP body; access logs go to stderr.

## Serve

Loopback HTTP worker for Oral. Warms the phoneme engine and Kokoro once, then `POST /tts`, `POST /phonemes`, and `POST /score` call the same helpers as the CLI (`pronounce.tts.to_file`, `pronounce.score.jobs.score_phoneme`, dictionary IPA). Bind only `127.0.0.1`, `localhost`, or `::1`. Copy-paste curls: [`demo/`](demo/README.md). Request/response fields: [`FIELDS.md`](FIELDS.md#http-serve).

```bash
"$MODELS_HOME/.venv/bin/python" -m pronounce serve --port 8787
"$MODELS_HOME/.venv/bin/python" -m pronounce serve --host 127.0.0.1 --port 8787 --no-load
```

| Flag | Meaning |
|------|---------|
| `--host` | Loopback only (`127.0.0.1`, `localhost`, `::1`). Default `127.0.0.1` |
| `--port` | Default `8787` |
| `--no-load` | Skip model warmup (tests / dry start) |

| Method | Path | CLI equivalent |
|--------|------|----------------|
| `GET` | `/health` | — (`{"ok": true, "engine": "phoneme", "tts": "kokoro"}`) |
| `POST` | `/tts` | `tts` (no `--speed`; isolated phones via `"ipa"`) |
| `POST` | `/phonemes` | `phonemes` |
| `POST` | `/score` | `score phoneme` (`ref_wav` required) |

Not exposed: `score acoustic`, `tts-zh`, `--speed`, `--calibration`. A non-loopback `--host` prints `{"ok": false, "command": "serve", "error": "..."}` on stdout and exits 1. Ctrl+C prints `serve stopped` on stderr and exits 0.
