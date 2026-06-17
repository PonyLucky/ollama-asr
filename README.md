# ollama-asr

A small [Textual](https://textual.textualize.io/) TUI that records your voice from the system **default microphone**, saves it as an mp3, sends it to a local **Ollama** speech-to-text endpoint, then shows the recognized text and language and copies the text to the clipboard.

## Screenshot

Main view:
> ![Main view](./assets/screenshots/screenshot-1.png)

Recording:
> ![Recording](./assets/screenshots/screenshot-2.png)

Done:
> ![Done](./assets/screenshots/screenshot-3.png)

Show more info:
> ![Recording](./assets/screenshots/screenshot-4.png)

## How it works

1. Reads the default mic via `pactl get-default-source`.
2. **Enter** starts recording (`ffmpeg -f pulse … libmp3lame` → mp3).
3. **Enter** again stops and uploads the mp3 to Ollama:
   `POST /v1/audio/transcriptions` (OpenAI-compatible, `multipart/form-data`).
   When a specific language is selected (see **Tab** below) it's sent as the
   optional ISO-639-1 `language` field; `auto` omits it and lets the model detect.
4. The reply (`language French<asr_text>…`) is parsed into language + text, and the
   spelled-out language name is mapped to its ISO-639-1 code (10 main languages).
5. The text is copied to the clipboard with `wl-copy`.

Press **Tab** to cycle the transcription language through `auto` and every code in
`LANGUAGES` (e.g. `auto → fr → en → auto`).

## Requirements

System packages (install with your distro's package manager):

- `ffmpeg` (with `libmp3lame` + `pulse` input)
- `pulseaudio-utils` / `pipewire-pulse` for `pactl`
- `wl-clipboard` for `wl-copy`

On Fedora/Nobara:

```sh
sudo dnf install wl-clipboard ffmpeg pulseaudio-utils
```

## Usage

```sh
./run.sh
```

## Configuration

Settings are read from a `.env` next to the script (see `.env.example`); real shell
environment variables override it.

| Variable           | Default                                          | Notes                                            |
| ------------------ | ------------------------------------------------ | ------------------------------------------------ |
| `OLLAMA_URL`       | `http://localhost:11434/v1/audio/transcriptions` | [Transcription endpoint](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)                |
| `OLLAMA_API_KEY`   | `ollama`                                         | Sent as `Authorization: Bearer …`                |
| `OLLAMA_MODEL`     | `hf.co/ggml-org/Qwen3-ASR-1.7B-GGUF:Q8_0`        | ASR model                                        |
| `LANGUAGES`        | _(empty)_                                        | ISO-639-1 codes, comma-separated, e.g. `fr,en`   |
| `OLLAMA_ASR_FILE`  | `~/.cache/ollama-asr/recording.mp3`              | Where the recording is written                   |

## Running against Ollama on another machine

By default the app talks to Ollama on `localhost`. To run the app on one machine
while Ollama runs on another in your local network, point `OLLAMA_URL` at the
server's LAN address (keep the `/v1/audio/transcriptions` path):

```sh
OLLAMA_URL=http://192.168.1.50:11434/v1/audio/transcriptions
```

Only the audio file is sent over the network; mic recording (`ffmpeg`/`pactl`)
and clipboard (`wl-copy`) still run locally. Make sure the chosen `OLLAMA_MODEL`
is pulled **on the server**.

On the server, Ollama must listen on all interfaces instead of `localhost`. With
a systemd unit, edit `/etc/systemd/system/ollama.service` and add an
`Environment` line under `[Service]`:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Then reload and restart:

```sh
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Open port `11434` in the firewall. With **firewalld** (Fedora/Nobara/RHEL):

```sh
sudo firewall-cmd --add-port=11434/tcp --permanent && sudo firewall-cmd --reload
```

With **ufw** (other):

```sh
sudo ufw allow 11434/tcp
```

Verify from the client: `curl http://192.168.1.50:11434/api/tags` should list the
server's models.

> ⚠️ **Trusted networks only.** Binding to `0.0.0.0` exposes Ollama
> unauthenticated to everyone who can reach the port. Only do this on a trusted
> local network. To be stricter, restrict the firewall rule to the client's IP
> instead of opening the port to everyone:
>
> ```sh
> # firewalld
> sudo firewall-cmd --add-rich-rule='rule family="ipv4" source address="192.168.1.10" port port="11434" protocol="tcp" accept' --permanent && sudo firewall-cmd --reload
> # ufw
> sudo ufw allow from 192.168.1.10 to any port 11434 proto tcp
> ```

## License

See LICENSE file in `./LICENSE`, Project under MIT.
