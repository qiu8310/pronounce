# Pronounce demo

样例音频（Kokoro 合成的 “Hello”）：

| 文件 | 口音 | 音色 |
|------|------|------|
| `hello-en-us.wav` | 美式 `en-us` | `af_heart` |
| `hello-en-gb.wav` | 英式 `en-gb` | `bf_emma` |

成功时一条 JSON 打在 **stdout**；日志在 stderr。字段说明：`python -m pronounce schema` 或仓库根的 [`FIELDS.md`](../FIELDS.md)。

```bash
export MODELS_HOME="/Users/mora/Workspace/models"
PY="$MODELS_HOME/.venv/bin/python"
DEMO="$MODELS_HOME/pronounce/demo"
```

## 词典 IPA

```bash
"$PY" -m pronounce phonemes --text "Hello" --lang en-us
"$PY" -m pronounce phonemes --text "Hello" --lang en-gb
```

## 合成（复现这两段 wav）

```bash
"$PY" -m pronounce tts --text "Hello" --out "$DEMO/hello-en-us.wav" --voice af_heart --lang en-us
"$PY" -m pronounce tts --text "Hello" --out "$DEMO/hello-en-gb.wav" --voice bf_emma --lang en-gb
# 慢放：写出的 wav 采样率按 --speed 降低（0.8 = 磁带减速）
"$PY" -m pronounce tts --text "Hello" --out /tmp/hello-slow.wav --voice af_heart --lang en-us --speed 0.8
```

## 音素打分

`--ref` 可省略。下面用美音当用户跟读、英音当参考。

```bash
"$PY" -m pronounce score phoneme --text "Hello" --user "$DEMO/hello-en-us.wav" --lang en-us --device cpu
"$PY" -m pronounce score phoneme --text "Hello" --user "$DEMO/hello-en-us.wav" --ref "$DEMO/hello-en-gb.wav" --lang en-us --device cpu
"$PY" -m pronounce score phoneme --text "Hello" --user "$DEMO/hello-en-gb.wav" --lang en-gb --device cpu
```

## 声学打分

声学引擎需要参考音：给 `--ref`，或省略后由 Kokoro 按 `--voice` 合成（JSON 里会有 `"ref_generated": true`）。

```bash
"$PY" -m pronounce score acoustic --text "Hello" --user "$DEMO/hello-en-us.wav" --ref "$DEMO/hello-en-gb.wav" --device cpu
"$PY" -m pronounce score acoustic --text "Hello" --user "$DEMO/hello-en-us.wav" --voice af_heart --lang en-us --device cpu
```

## 字段约定

```bash
"$PY" -m pronounce schema
```
