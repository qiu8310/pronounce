# Pronounce CLI JSON fields

Stdout is one JSON object per invocation.

- `score`: envelope below. Failure: `{"ok": false, "engine": "...", "error": "..."}`.
- `tts` / `phonemes`: see [TTS and IPA](#tts-and-ipa). Failure: `{"ok": false, "error": "..."}`.

## Always present (success)

| Field | Type | When empty | Meaning |
|-------|------|------------|---------|
| `ok` | bool | never | `true` on success |
| `engine` | `"phoneme"` \| `"acoustic"` | never | Which engine ran |
| `text` | string | never | `--text` value |
| `user_wav` | string | never | Absolute `--user` path |
| `ref_wav` | string \| null | `null` when `--ref` omitted | Absolute `--ref` path |
| `score` | number | never | 0–100 overall pronunciation score |
| `passed` | bool | never | Engine pass/fail |
| `scored` | bool | never | Always `true` for phoneme/acoustic |
| `transcription` | string | `""` if ASR produced nothing | What the recognizer heard |
| `feedback` | string | `""` when no summary | Short English summary |
| `word_errors` | array | `[]` when no mispronounced words | Per-word error records (engine-specific; see below) |
| `words_with_errors` | string[] | `[]` when none marked bad | Words marked bad |
| `word_diff` | object[] | `[]` when recognition matches the target | GUI-style expected/heard diffs |
| `reference_words` | object[] | `[]` when none | Per target-phrase word tags |
| `recognized_units` | object[] | `[]` when none | Heard units (phones or words) |
| `prosody` | object | `ref_*` are `[]` without `--ref` | Pitch/energy contours for charts (see below). Not part of the score |
| `phoneme` | object | `{}` when engine is acoustic | Phoneme-engine diagnostics |
| `acoustic` | object | `{}` when engine is phoneme | Acoustic-engine diagnostics |

## Nested shapes

### `word_errors` (phoneme)

Each bad word:

```json
{
  "word": "bandwidth",
  "expected": ["b", "æ", "n", "d", "w", "ɪ", "d", "θ"],
  "heard": ["b", "æ", "n", "d", "w", "ɪ", "t"],
  "level": "bad",
  "distance": 0.42
}
```

- `word` — whitespace token from the target phrase
- `expected` / `heard` — phone lists for that word
- `level` — severity (`"bad"` for entries in this list)
- `distance` — word-local phone distance

### `word_errors` (acoustic)

Per mispronounced target word from phoneme-alignment diagnostics (`differences["errors"]` in the acoustic engine). Not the same as `word_diff`.

Missing word (nothing aligned in the transcription):

```json
{ "position": 0, "expected": "time", "actual": "", "word": "time" }
```

Mispronounced word (phoneme edit distance over tolerance):

```json
{
  "position": 12,
  "expected": "taɪm",
  "actual": "taɪmz",
  "word": "time",
  "actual_word": "times"
}
```

- `position` — start index of the word in the expected-phoneme sequence
- `expected` — expected word text when missing; expected phoneme string when mispronounced
- `actual` — `""` when missing; actual phoneme string when mispronounced
- `word` — expected word text
- `actual_word` — optional; reconstructed heard word text when the word was present but wrong

### `word_diff`

Acoustic: one `{"expected", "heard"}` pair per diverging ASR word segment (substitution / deletion / insertion).
Phoneme: often `{"expected": "<word>"}` for each word in `words_with_errors`.

### `reference_words`

One `{"word", "correct"}` (and engine-specific extras as produced) per target-phrase word, in order.

### `recognized_units`

Each `{"unit", "correct"}`. Units are **words** for acoustic and **phonemes** for phoneme.

### `prosody`

Pitch and loudness time series so a UI can draw intonation and energy. They do **not** feed `score` / `passed`. Scoring engines leave `prosody` empty; the CLI fills it from the same 16 kHz mono waveforms used for scoring, after leading/trailing silence trim (`top_db=30`).

A successful `score` run always has these four keys. Lists are truncated in the examples; a real take is one sample per STFT hop (librosa default 512 samples at 16 kHz ≈ 32 ms).

Phoneme without `--ref` (user contours only):

```json
{
  "f0": [0.0, 118.4, 132.1, 125.0, 110.2],
  "energy": [12.4, 88.0, 140.2, 95.1, 40.0],
  "ref_f0": [],
  "ref_energy": []
}
```

With a reference wav (passed `--ref`, or acoustic auto-TTS when `--ref` is omitted):

```json
{
  "f0": [120.1, 135.8, 128.0, 122.4],
  "energy": [40.0, 180.2, 90.5, 55.0],
  "ref_f0": [180.4, 175.2, 160.0, 155.8],
  "ref_energy": [30.0, 200.1, 85.4, 48.2]
}
```

| Key | Type | Unit | When empty |
|-----|------|------|------------|
| `f0` | number[] | Hz | never on success; may be all `0` if the take is unvoiced |
| `energy` | number[] | display 0–250 | never on success |
| `ref_f0` | number[] | Hz | `[]` when there is no reference wav (phoneme, no `--ref`) |
| `ref_energy` | number[] | display 0–250 | `[]` when there is no reference wav (phoneme, no `--ref`) |

Acoustic without `--ref` still fills `ref_f0` / `ref_energy`: Kokoro synthesizes a native-speed reference and sets `ref_generated` to `true`.

How the series are built:

- **`f0` / `ref_f0`** — `librosa.pyin` with `fmin=50`, `fmax=450`. Unvoiced frames are 0, then linearly interpolated across neighboring voiced frames so the line is continuous. A fully unvoiced clip stays all zeros.
- **`energy` / `ref_energy`** — frame RMS, then `MinMaxScaler` to `[0, 250]` **per clip**. User and reference energy are not on a shared absolute scale; compare shape, not numeric values.
- User vs reference lists are usually **different lengths** (each wav is trimmed on its own). Overlay by normalizing each series onto 0–1 time; do not zip by index. `f0` and `energy` of the *same* clip are usually similar length, not guaranteed equal.

### `weak_phonemes` (inside `phoneme`)

Each `{"phoneme", "severity", "count"}` — worst reference phones, most-to-least severe. Empty for acoustic (`phoneme` is `{}`).

### `ipa_words` (inside `phoneme`)

Per whitespace token, word-local phone alignment for the IPA diagnosis panel:

| Key | Type | Notes |
|-----|------|-------|
| `word` | string | Token text |
| `expected` | string[] | Reference phones (parallel to `heard` / `ok`) |
| `heard` | string[] | Recognized phones |
| `ok` | bool[] | Per-phone match flags |
| `start_sec` | number \| null | Start into prepared user waveform, or null when untimed |
| `end_sec` | number \| null | End into prepared user waveform, or null when untimed |

Acoustic leaves `ipa_words` unused (`phoneme` is `{}`).

## `phoneme` object

Empty `{}` when `engine` is `acoustic`. Otherwise:

| Field | Type | When empty |
|-------|------|------------|
| `bucket` | int | `-1` if engine does not bucketize |
| `user_percent` | number | `0.0` when unused |
| `grade` | string | `""` when ungraded |
| `grade_value` | number | `-1.0` when ungraded |
| `expected_phonemes` | string[] | `[]` |
| `transcribed_phonemes` | string[] | `[]` |
| `weak_phonemes` | object[] | `[]` |
| `ipa_words` | object[] | `[]` |
| `per_phone_distance` | number | `0.0` |
| `bad_baseline` | number | `0.0` |
| `phoneme_score` | number | `0.0` |
| `recall` | number | `0.0` |
| `good_anchor` | number | `0.0` |

## `acoustic` object

Empty `{}` when `engine` is `phoneme`. Otherwise:

| Field | Type | When empty |
|-------|------|------------|
| `acoustic_distance` | int | `0` |
| `acoustic_per_step` | number | `0.0` |
| `acoustic_baseline` | number | `0.0` |

## Failure

```json
{ "ok": false, "engine": "acoustic", "error": "the acoustic engine requires --ref" }
```

`error` is a human-readable string (model path under `llm/wav2vec2/`, missing files, exception message). No traceback on stdout.

## Example: phoneme

```json
{
  "ok": true,
  "engine": "phoneme",
  "text": "I don't have the bandwidth.",
  "user_wav": "/abs/path/take.wav",
  "ref_wav": null,
  "score": 82.4,
  "passed": true,
  "scored": true,
  "transcription": "aɪ doʊnt hæv ðə bændwɪt",
  "feedback": "Good overall; watch the final /θ/ in bandwidth.",
  "word_errors": [
    {
      "word": "bandwidth",
      "expected": ["b", "æ", "n", "d", "w", "ɪ", "d", "θ"],
      "heard": ["b", "æ", "n", "d", "w", "ɪ", "t"],
      "level": "bad",
      "distance": 0.42
    }
  ],
  "words_with_errors": ["bandwidth"],
  "word_diff": [{"expected": "bandwidth"}],
  "reference_words": [
    {"word": "I", "correct": true},
    {"word": "don't", "correct": true},
    {"word": "have", "correct": true},
    {"word": "the", "correct": true},
    {"word": "bandwidth.", "correct": false}
  ],
  "recognized_units": [
    {"unit": "aɪ", "correct": true},
    {"unit": "t", "correct": false}
  ],
  "prosody": {
    "f0": [0.0, 118.4, 132.1, 125.0, 110.2],
    "energy": [12.4, 88.0, 140.2, 95.1, 40.0],
    "ref_f0": [],
    "ref_energy": []
  },
  "phoneme": {
    "bucket": 4,
    "user_percent": 85.0,
    "grade": "4",
    "grade_value": 4.0,
    "expected_phonemes": ["aɪ", "d", "oʊ", "n", "t"],
    "transcribed_phonemes": ["aɪ", "d", "oʊ", "n", "t"],
    "weak_phonemes": [{"phoneme": "θ", "severity": 0.9, "count": 1}],
    "ipa_words": [
      {
        "word": "bandwidth",
        "expected": ["b", "æ", "n", "d", "w", "ɪ", "d", "θ"],
        "heard": ["b", "æ", "n", "d", "w", "ɪ", "t"],
        "ok": [true, true, true, true, true, true, false, false],
        "start_sec": 1.2,
        "end_sec": 1.8
      }
    ],
    "per_phone_distance": 0.31,
    "bad_baseline": 1.2,
    "phoneme_score": 78.0,
    "recall": 0.91,
    "good_anchor": 0.18
  },
  "acoustic": {}
}
```

## Example: acoustic

```json
{
  "ok": true,
  "engine": "acoustic",
  "text": "I don't have the bandwidth.",
  "user_wav": "/abs/path/take.wav",
  "ref_wav": "/abs/path/ref.wav",
  "score": 74.0,
  "passed": true,
  "scored": true,
  "transcription": "i don't have the bandwidth",
  "feedback": "Close; word 'bandwidth' diverged.",
  "word_errors": [
    {
      "position": 18,
      "expected": "bændwɪdθ",
      "actual": "bændwɪt",
      "word": "bandwidth",
      "actual_word": "bandwith"
    }
  ],
  "words_with_errors": ["bandwidth"],
  "word_diff": [{"expected": "bandwidth", "heard": "bandwith"}],
  "reference_words": [
    {"word": "i", "correct": true},
    {"word": "don't", "correct": true},
    {"word": "have", "correct": true},
    {"word": "the", "correct": true},
    {"word": "bandwidth", "correct": false}
  ],
  "recognized_units": [
    {"unit": "i", "correct": true},
    {"unit": "don't", "correct": true},
    {"unit": "have", "correct": true},
    {"unit": "the", "correct": true},
    {"unit": "bandwith", "correct": false}
  ],
  "prosody": {
    "f0": [120.1, 135.8, 128.0, 122.4],
    "energy": [40.0, 180.2, 90.5, 55.0],
    "ref_f0": [180.4, 175.2, 160.0, 155.8],
    "ref_energy": [30.0, 200.1, 85.4, 48.2]
  },
  "phoneme": {},
  "acoustic": {
    "acoustic_distance": 12,
    "acoustic_per_step": 0.21,
    "acoustic_baseline": 0.55
  }
}
```

## TTS and IPA

These commands do not use the score envelope. Success still prints one JSON object.

Acoustic score without `--ref` synthesizes a native-speed Kokoro wav, sets `ref_wav` to that temp path, and sets `ref_generated` to `true`.

### `tts`

| Field | Type | Meaning |
|-------|------|---------|
| `ok` | bool | `true` |
| `command` | `"tts"` | Which subcommand ran |
| `text` | string | `--text` (word, sentence, or paragraph) |
| `voice` | string | `--voice` (default `af_heart`) |
| `lang` | string | `--lang` (`en-us` / `en-gb`) |
| `out` | string | Absolute `--out` path |
| `speed` | number | Playback tempo (`1` = native; `0.8` = slower) |
| `sample_rate` | number | Wav header rate (`native_rate * speed`) |
| `native_rate` | number | Always `24000` (synthesis rate) |

`--speed` does not change the mouth timing of the model; it writes a lower sample rate so a normal player plays the file slower (same trick mimora uses for slow listen). Scoring reference audio is always synthesized at `speed=1`.

### `tts-zh`

Chinese TTS via MeloTTS. Separate subcommand from English `tts` (not `--lang zh`).

| Field | Type | Meaning |
|-------|------|---------|
| `ok` | bool | `true` |
| `command` | `"tts-zh"` | Which subcommand ran |
| `text` | string | `--text` |
| `speaker` | `"ZH"` | Only speaker in MeloTTS-Chinese |
| `lang` | `"zh"` | Always Chinese |
| `out` | string | Absolute `--out` path |
| `speed` | number | Playback tempo (`1` = native; `0.8` = slower) |
| `sample_rate` | number | Wav header rate (`native_rate * speed`) |
| `native_rate` | number | Always `44100` |

`--speed` is the same tape-slow as English `tts`. Synthesis itself is always native rate.

### `phonemes`

Dictionary IPA for people to read (espeak-ng, stress kept). One `words[]` row per whitespace token, so a paragraph stays aligned with the page.

| Field | Type | Meaning |
|-------|------|---------|
| `ok` | bool | `true` |
| `command` | `"phonemes"` | Which subcommand ran |
| `text` | string | `--text` |
| `lang` | string | `--lang` |
| `ipa` | string | Space-joined IPA for the whole text |
| `words` | object[] | `{"word", "ipa"}` per `text.split()` token |
