#!/usr/bin/env python3
"""ollama-asr: a Textual TUI to record from the default mic and transcribe via Ollama.

Flow:
    Enter  -> start recording from the system default microphone (PipeWire/PulseAudio)
    Enter  -> stop recording, save to an mp3, POST it to Ollama's Whisper-compatible
              transcription endpoint, show the text + detected language, copy text to
              the Wayland clipboard with wl-copy.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import requests
from dotenv import load_dotenv
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Footer, Header, Static

# --- Configuration -----------------------------------------------------------
# Load a .env sitting next to this script (real environment variables win).
load_dotenv(Path(__file__).with_name(".env"))

# Reference: https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
# Each OLLAMA_URL is a server root; the transcription endpoint path is appended below.
TRANSCRIBE_ENDPOINT = "/v1/audio/transcriptions"
DEFAULT_MODEL = "hf.co/ggml-org/Qwen3-ASR-1.7B-GGUF:Q8_0"
# Ollama ignores the bearer token, so it's always the literal "ollama".
OLLAMA_TOKEN = "ollama"


@dataclass(frozen=True)
class Server:
    """An Ollama server: its root URL and the model to use with it."""

    root_url: str
    model: str

    @property
    def transcribe_url(self) -> str:
        """Full OpenAI-compatible transcription endpoint for this server."""
        return self.root_url.rstrip("/") + TRANSCRIBE_ENDPOINT


PRIMARY = Server(
    os.environ.get("OLLAMA_URL", "http://localhost:11434"),
    os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
)

# Optional fallback, used when the primary root URL doesn't answer at startup.
# Leave FALLBACK_OLLAMA_URL blank (the default) to disable it.
_fallback_url = os.environ.get("FALLBACK_OLLAMA_URL", "").strip()
FALLBACK: Server | None = (
    Server(
        _fallback_url,
        os.environ.get("FALLBACK_OLLAMA_MODEL", DEFAULT_MODEL),
    )
    if _fallback_url
    else None
)
# Expected languages as ISO-639-1 codes, e.g. "fr,en". Used to (a) hint the model
# when exactly one is given and (b) map the detected language name to its code.
LANGUAGES = [
    c.strip().lower() for c in os.environ.get("LANGUAGES", "").split(",") if c.strip()
]
# Where to keep the recording. Reused each time; the last take stays on disk.
RECORD_PATH = Path(
    os.environ.get(
        "OLLAMA_ASR_FILE", str(Path.home() / ".cache" / "ollama-asr" / "recording.mp3")
    )
)

# Manual mapping for the 10 main languages: spelled-out name -> ISO-639-1 code.
LANGUAGE_CODES = {
    "english": "en",
    "french": "fr",
    "spanish": "es",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "dutch": "nl",
    "russian": "ru",
    "chinese": "zh",
    "japanese": "ja",
}
CODE_TO_NAME = {code: name.capitalize() for name, code in LANGUAGE_CODES.items()}

# Languages the user can cycle through with Tab: "auto" (let the model detect)
# followed by every code listed in LANGUAGES.
LANGUAGE_CYCLE = ["auto"] + LANGUAGES


def code_to_name(code: str) -> str:
    """Human label for a cycle entry, e.g. 'fr' -> 'French', 'auto' -> 'auto-detect'."""
    if code == "auto":
        return "auto-detect"
    return CODE_TO_NAME.get(code, code)


class State(Enum):
    IDLE = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()
    RESULT = auto()
    ERROR = auto()


def get_default_source() -> str:
    """Return the PulseAudio/PipeWire default source (microphone) name."""
    try:
        out = subprocess.run(
            ["pactl", "get-default-source"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        name = out.stdout.strip()
        if name:
            return name
    except (OSError, subprocess.SubprocessError):
        pass
    return "default"


def parse_transcription(raw: str) -> tuple[str, str, str]:
    """Split Ollama's reply into (language_name, iso_code, text).

    The model returns e.g. ``language French<asr_text>Bonjour ...``. The spelled-out
    language is mapped to its ISO-639-1 code via the 10-language table above.
    Falls back gracefully when the prefix/marker is absent or the language is unknown.
    """
    language = "unknown"
    code = ""
    text = raw.strip()
    if "<asr_text>" in text:
        prefix, _, body = text.partition("<asr_text>")
        text = body.strip()
        prefix = prefix.strip()
        if prefix.lower().startswith("language"):
            prefix = prefix[len("language") :].strip()
        if prefix:
            language = prefix
            code = LANGUAGE_CODES.get(prefix.lower(), "")
    return language, code, text


def copy_to_clipboard(text: str) -> bool:
    """Copy text to the Wayland clipboard via wl-copy. Returns False if unavailable."""
    if not shutil.which("wl-copy"):
        return False
    try:
        subprocess.run(["wl-copy"], input=text.encode("utf-8"), check=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def server_is_up(server: Server, timeout: float = 1.0) -> bool:
    """Probe a server's root URL; True if it answers HTTP 200 within ``timeout`` seconds."""
    try:
        resp = requests.get(server.root_url.rstrip("/") + "/", timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def transcribe(server: Server, path: Path, language: str | None = None) -> str:
    """Blocking POST of the mp3 to ``server``; returns the raw ``text`` field.

    ``language`` is the optional ISO-639-1 hint sent as the OpenAI-compatible
    ``language`` form field. Pass ``None``/``"auto"`` to let the model auto-detect.
    """
    data = {"model": server.model}
    if language and language != "auto":
        data["language"] = language
    with open(path, "rb") as fh:
        resp = requests.post(
            server.transcribe_url,
            headers={"Authorization": f"Bearer {OLLAMA_TOKEN}"},
            files={"file": (path.name, fh, "audio/mpeg")},
            data=data,
            timeout=300,
        )
    resp.raise_for_status()
    return resp.json().get("text", "")


class Recorder:
    """Records the default mic to an mp3 using ffmpeg's pulse input."""

    def __init__(self, source: str, outfile: Path) -> None:
        self.source = source
        self.outfile = outfile
        self.proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        self.outfile.parent.mkdir(parents=True, exist_ok=True)
        self.proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "pulse",
            "-i",
            self.source,
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(self.outfile),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

    async def stop(self) -> None:
        """Ask ffmpeg to finalize the file cleanly, falling back to terminate."""
        if self.proc is None:
            return
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.write(b"q")
                await self.proc.stdin.drain()
                self.proc.stdin.close()
            await asyncio.wait_for(self.proc.wait(), timeout=5)
        except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError):
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
        finally:
            self.proc = None


class ASRApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    #panel {
        width: 80%;
        max-width: 90;
        height: auto;
        border: round $primary;
        padding: 1 2;
    }
    #status {
        text-align: center;
        text-style: bold;
        padding: 1 0;
    }
    #language {
        color: $accent;
        text-style: bold;
        padding-top: 1;
    }
    #text {
        padding-top: 1;
    }
    #info {
        color: $text-muted;
        border: round $surface;
        padding: 0 1;
        margin-top: 1;
    }
    #hint {
        color: $text-muted;
        text-align: center;
        padding-top: 1;
    }
    .recording { color: $error; }
    .working { color: $warning; }
    .ok { color: $success; }
    """

    BINDINGS = [
        ("enter", "toggle", "Start / Stop"),
        Binding("tab", "cycle_language", "Language", priority=True),
        ("c", "copy", "Copy"),
        ("n", "new", "New"),
        ("i", "info", "Info"),
        ("q", "quit", "Quit"),
    ]

    state: reactive[State] = reactive(State.IDLE)

    def __init__(self) -> None:
        super().__init__()
        self.source = get_default_source()
        self.server = PRIMARY  # may switch to FALLBACK after the startup probe
        self.recorder = Recorder(self.source, RECORD_PATH)
        self._record_start = 0.0
        self._timer: Timer | None = None
        self.lang_index = 0  # position in LANGUAGE_CYCLE
        self._last_text = ""  # most recent transcription, for re-copy

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="panel"):
            yield Static(f"🎙  Microphone: [b]{self.source}[/b]", id="device")
            yield Static("", id="langsel")
            yield Static("", id="status")
            yield Static("", id="language")
            yield Static("", id="text")
            yield Static("", id="info")
            yield Static("", id="hint")
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "dracula"
        self.render_info()
        self.query_one("#info", Static).display = False
        self.render_language()
        self.render_state()
        self.select_server()

    def render_info(self) -> None:
        """Refresh the details panel to reflect the currently active server."""
        self.query_one("#info", Static).update(
            f"[b]Endpoint[/b]  {self.server.transcribe_url}\n"
            f"[b]Model[/b]     {self.server.model}\n"
            f"[b]File[/b]      {RECORD_PATH}"
        )

    @work
    async def select_server(self) -> None:
        """Probe the primary server at startup; fall back if it's unreachable.

        The probe is a 1s GET of the server root, considered healthy on HTTP 200.
        Runs as a worker so the 1s timeout never blocks the UI.
        """
        if await asyncio.to_thread(server_is_up, PRIMARY):
            return  # primary is healthy; self.server already points at it
        if FALLBACK is None:
            self.notify(
                f"Ollama at {PRIMARY.root_url} is unreachable and no fallback is "
                "configured — transcription will fail.",
                title="Ollama unreachable",
                severity="warning",
                timeout=10,
            )
            return
        if await asyncio.to_thread(server_is_up, FALLBACK):
            self.server = FALLBACK
            self.render_info()
            self.notify(
                f"Primary Ollama ({PRIMARY.root_url}) did not respond — using "
                f"fallback {FALLBACK.root_url}.",
                title="Using fallback Ollama",
                severity="warning",
                timeout=10,
            )
        else:
            self.notify(
                f"Neither primary ({PRIMARY.root_url}) nor fallback "
                f"({FALLBACK.root_url}) responded — transcription will fail.",
                title="No Ollama reachable",
                severity="error",
                timeout=10,
            )

    def action_info(self) -> None:
        """Toggle the details panel (endpoint, model, recording path)."""
        info = self.query_one("#info", Static)
        info.display = not info.display

    # --- language selection --------------------------------------------------
    @property
    def selected_language(self) -> str:
        return LANGUAGE_CYCLE[self.lang_index]

    def render_language(self) -> None:
        parts = []
        for i, code in enumerate(LANGUAGE_CYCLE):
            if i == self.lang_index:
                parts.append(f"[reverse b] {code} [/]")
            else:
                parts.append(f" [dim]{code}[/dim] ")
        self.query_one("#langsel", Static).update(
            "🌐 Language [dim](Tab)[/dim]: " + "".join(parts)
        )

    def action_cycle_language(self) -> None:
        if len(LANGUAGE_CYCLE) <= 1:
            return  # nothing configured beyond auto-detect
        self.lang_index = (self.lang_index + 1) % len(LANGUAGE_CYCLE)
        self.render_language()

    # --- state rendering -----------------------------------------------------
    def watch_state(self, _old: State, _new: State) -> None:
        self.render_state()

    def render_state(self) -> None:
        status = self.query_one("#status", Static)
        hint = self.query_one("#hint", Static)
        status.remove_class("recording", "working", "ok")
        if self.state is State.IDLE:
            status.update("Ready")
            hint.update(
                "[b]Enter[/b] record  ·  [b]Tab[/b] language  ·  [b]i[/b] info  ·  [b]q[/b] quit"
            )
        elif self.state is State.RECORDING:
            status.update(f"● Recording…  {self._format_elapsed()}")
            status.add_class("recording")
            hint.update("Press [b]Enter[/b] to stop")
        elif self.state is State.TRANSCRIBING:
            status.update("Transcribing…")
            status.add_class("working")
            hint.update("Sending audio to Ollama, please wait")
        elif self.state is State.RESULT:
            status.update("Done")
            status.add_class("ok")
            hint.update("[b]Enter[/b] record again  ·  [b]c[/b] copy  ·  [b]n[/b] new")
        elif self.state is State.ERROR:
            status.add_class("recording")
            hint.update("Press [b]Enter[/b] to try again")

    # --- recording timer -----------------------------------------------------
    def _format_elapsed(self) -> str:
        elapsed = int(time.monotonic() - self._record_start)
        return f"{elapsed // 60}:{elapsed % 60:02d}"

    def _tick(self) -> None:
        self.query_one("#status", Static).update(
            f"● Recording…  {self._format_elapsed()}"
        )

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    # --- actions -------------------------------------------------------------
    def action_toggle(self) -> None:
        if self.state is State.IDLE or self.state in (State.RESULT, State.ERROR):
            self.start_recording()
        elif self.state is State.RECORDING:
            self.stop_and_transcribe()
        # ignore Enter while transcribing

    def action_copy(self) -> None:
        """Re-copy the most recent transcription to the clipboard."""
        if not self._last_text:
            self.notify("Nothing to copy yet", severity="warning")
            return
        if copy_to_clipboard(self._last_text):
            self.notify("Copied to clipboard")
        else:
            self.notify("wl-copy not found — install wl-clipboard", severity="warning")

    def action_new(self) -> None:
        """Clear the current output and return to the idle screen."""
        if self.state in (State.RECORDING, State.TRANSCRIBING):
            return  # don't interrupt an in-flight recording/transcription
        self._last_text = ""
        self.query_one("#language", Static).update("")
        self.query_one("#text", Static).update("")
        self.state = State.IDLE

    @work(exclusive=True)
    async def start_recording(self) -> None:
        self.query_one("#language", Static).update("")
        self.query_one("#text", Static).update("")
        try:
            await self.recorder.start()
        except OSError as exc:
            self.show_error(f"Could not start ffmpeg: {exc}")
            return
        self._record_start = time.monotonic()
        self.state = State.RECORDING
        self._timer = self.set_interval(0.25, self._tick)

    @work(exclusive=True)
    async def stop_and_transcribe(self) -> None:
        self._stop_timer()
        await self.recorder.stop()
        self.state = State.TRANSCRIBING
        try:
            raw = await asyncio.to_thread(
                transcribe, self.server, RECORD_PATH, self.selected_language
            )
        except requests.RequestException as exc:
            self.show_error(f"Transcription request failed: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - surface anything to the UI
            self.show_error(f"Unexpected error: {exc}")
            return

        language, code, text = parse_transcription(raw)
        self._last_text = text
        label = f"{language} ({code})" if code else language
        self.query_one("#language", Static).update(f"Detected language: {label}")
        self.query_one("#text", Static).update(text or "[i](no speech detected)[/i]")
        if text and copy_to_clipboard(text):
            self.query_one("#text", Static).update(
                text + "\n\n[green]✓ copied to clipboard[/green]"
            )
        elif text:
            self.query_one("#text", Static).update(
                text
                + "\n\n[yellow]wl-copy not found — install wl-clipboard to copy[/yellow]"
            )
        self.state = State.RESULT

    def show_error(self, message: str) -> None:
        self.query_one("#status", Static).update(f"Error: {message}")
        self.state = State.ERROR


if __name__ == "__main__":
    ASRApp().run()
