# Pronounce demo

样例音频（Kokoro 合成的 “Hello”，以及 espeak-ng 合成的孤立音素 `/ɪ/`）：

| 文件 | 内容 | 口音 | 音色 |
|------|------|------|------|
| `hello-en-us.wav` | Hello | 美式 `en-us` | Kokoro `af_heart` |
| `hello-en-gb.wav` | Hello | 英式 `en-gb` | Kokoro `bf_emma` |
| `ih-en-gb.wav` | IPA `ɪ` | 英式 `en-gb` | espeak-ng |

成功时一条 JSON 打在 **stdout**；日志在 stderr。`serve` 的 JSON 在 HTTP 响应体里。字段说明：`python -m pronounce schema` 或仓库根的 [`FIELDS.md`](../FIELDS.md)。

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

## 孤立音素（`--ipa`）

Kokoro 不能念 IPA；把 `ɪ` 当 `--text` 还会被 G2P 读成字母名。`--ipa` 用 espeak-ng 合成单个音素，打分时跳过 G2P。需要系统 `espeak-ng`（或 `espeak`）在 `PATH` 上。HTTP 写法见下面 [常驻 HTTP serve](#常驻-http-serve)。

复现 `ih-en-gb.wav`：

```bash
"$PY" -m pronounce tts --ipa ɪ --out "$DEMO/ih-en-gb.wav"
```

同一条当用户跟读和参考（`--text` 可省略）：

```bash
"$PY" -m pronounce score phoneme --ipa ɪ --user "$DEMO/ih-en-gb.wav" --ref "$DEMO/ih-en-gb.wav" --lang en-gb --device cpu
```

## 中文合成（MeloTTS，`tts-zh`）

和英语 `tts` 分开的子命令，不是 `--lang zh`。

```bash
"$PY" -m pronounce tts-zh --text "你好，今天天气怎么样？" --out /tmp/nihao.wav
"$PY" -m pronounce tts-zh --text "你好" --out /tmp/nihao-slow.wav --speed 0.8
```

## 音素打分

`--ref` 可省略。下面用美音当用户跟读、英音当参考。

```bash
"$PY" -m pronounce score phoneme --text "Hello" --user "$DEMO/hello-en-us.wav" --lang en-us --device cpu
"$PY" -m pronounce score phoneme --text "Hello" --user "$DEMO/hello-en-us.wav" --ref "$DEMO/hello-en-gb.wav" --lang en-us --device cpu
"$PY" -m pronounce score phoneme --text "Hello" --user "$DEMO/hello-en-gb.wav" --lang en-gb --device cpu
```

## 声学打分

声学引擎需要参考音：给 `--ref`，或省略后由 Kokoro 按 `--voice` 合成（JSON 里会有 `"ref_generated": true`）。`serve` **没有**声学引擎，只有 CLI。

```bash
"$PY" -m pronounce score acoustic --text "Hello" --user "$DEMO/hello-en-us.wav" --ref "$DEMO/hello-en-gb.wav" --device cpu
"$PY" -m pronounce score acoustic --text "Hello" --user "$DEMO/hello-en-us.wav" --voice af_heart --lang en-us --device cpu
```

## 常驻 HTTP serve

给 Oral 用的 loopback worker：启动时预热 phoneme + Kokoro，之后 `POST /tts`、`POST /phonemes`、`POST /score` 和 CLI 走同一套函数。只绑 `127.0.0.1` / `localhost` / `::1`。JSON 在 HTTP 响应体里，访问日志在 stderr。`/score` 必须带 `ref_wav`（CLI phoneme 的 `--ref` 仍可省略）。没有 `score acoustic`、`tts-zh`、`--speed`。Ctrl+C 在 stderr 打 `serve stopped` 后退出。

另开一个终端：

```bash
"$PY" -m pronounce serve --port 8787
# 测试起进程、不加载权重： --no-load
```

```bash
curl -sS http://127.0.0.1:8787/health

curl -sS -X POST http://127.0.0.1:8787/phonemes \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello","lang":"en-us"}'

curl -sS -X POST http://127.0.0.1:8787/tts \
  -H 'Content-Type: application/json' \
  -d "{\"text\":\"Hello\",\"out\":\"$DEMO/hello-en-us.wav\",\"voice\":\"af_heart\",\"lang\":\"en-us\"}"

curl -sS -X POST http://127.0.0.1:8787/score \
  -H 'Content-Type: application/json' \
  -d "{\"text\":\"Hello\",\"user_wav\":\"$DEMO/hello-en-us.wav\",\"ref_wav\":\"$DEMO/hello-en-gb.wav\",\"lang\":\"en-us\"}"

# 孤立 ɪ（默认 lang=en-gb）
curl -sS -X POST http://127.0.0.1:8787/tts \
  -H 'Content-Type: application/json' \
  -d "{\"ipa\":\"ɪ\",\"out\":\"$DEMO/ih-en-gb.wav\"}"

curl -sS -X POST http://127.0.0.1:8787/score \
  -H 'Content-Type: application/json' \
  -d "{\"ipa\":\"ɪ\",\"user_wav\":\"$DEMO/ih-en-gb.wav\",\"ref_wav\":\"$DEMO/ih-en-gb.wav\",\"lang\":\"en-gb\"}"
```

字段见 [`FIELDS.md`](../FIELDS.md#http-serve)。

## 字段约定

```bash
"$PY" -m pronounce schema
```
