# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenShorts is an AI-powered vertical video generator that transforms long YouTube videos or local uploads into viral-ready short clips (9:16 format) for TikTok, Instagram Reels, and YouTube Shorts. Uses Google Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`, overridable with `GEMINI_MODEL`) for viral moment detection and title generation.

## Development Commands

### Local Development (Docker)
```bash
docker compose up --build   # Build and run full stack
```
- Backend: http://localhost:8000 (FastAPI/Uvicorn)
- Frontend: http://localhost:5175 (Vite proxies API calls to backend)

### Frontend Only (Dashboard)
```bash
cd dashboard
npm install
npm run dev       # Dev server with HMR (port 5173)
npm run build     # Production build
npm run lint      # ESLint (strict, --max-warnings 0)
```

### Backend Only
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Architecture

### Core Processing Pipeline
1. **Ingest** - YouTube download (yt-dlp) or local upload
2. **Transcription** - faster-whisper with word-level timestamps
3. **Scene Detection** - PySceneDetect for segment boundaries
4. **AI Analysis** - Gemini identifies 3-15 viral moments (15-60 sec each)
5. **FFmpeg Extraction** - Precise clip cutting
6. **AI Cropping** - Vertical reframing with subject tracking
7. **Effects/Subtitles** - Optional AI-generated FFmpeg filters
8. **Hook Overlay** - Text overlays with styled fonts
9. **Voice Dubbing** - Optional ElevenLabs AI translation (30+ languages)
10. **S3 Backup** - Silent background upload
11. **Social Distribution** - Upload-Post API (async upload)

### Key Files
| File | Purpose |
|------|---------|
| `main.py` | Core video processing: transcription, scene detection, clip extraction, vertical reframing |
| `app.py` | FastAPI server with async job queue and REST endpoints |
| `editor.py` | Gemini AI integration for dynamic video effects (FFmpeg filter generation) |
| `hooks.py` | Hook text overlay generation with font rendering |
| `s3_uploader.py` | AWS S3 upload with caching |
| `subtitles.py` | SRT generation, FFmpeg subtitle burning, and dubbed video transcription |
| `translate.py` | ElevenLabs dubbing API for AI voice translation |
| `dashboard/src/App.jsx` | Main React component with state management |
| `dashboard/src/components/TranslateModal.jsx` | Voice dubbing UI with language selection |
| `dashboard/vite-plugin-seo.js` | Build-time SEO surface: injects crawler-visible homepage content, emits static pages, sitemap.xml and llms.txt |
| `dashboard/seo/data.js` | Single source of truth for pricing, pipeline and competitor facts used by every generated page |

### SEO / AI-crawler surface

The dashboard is a client-rendered SPA with hash routing, so the HTML served for
`/` used to contain an empty `<div id="root">`. Googlebot renders JavaScript and
saw the real page; GPTBot, ClaudeBot and PerplexityBot do not and measured the
homepage as zero characters of text. `vite-plugin-seo.js` fixes that at build time:

- Injects the content of `seo/landing-fallback.js` into `#root`. React's
  `createRoot().render()` replaces it on mount, so users get the app and
  non-executing clients get the copy. **Keep it in sync with `Landing.jsx`.**
- Emits the standalone pages (the `/alternatives` cluster, the clip-generator,
  open-source, use-case and automation pages, and `/mcp`; the full list is
  `buildPages()` in `seo/pages.js`) as flat `.html` files.
  nginx resolves the clean URL through `try_files $uri $uri.html`; serving them as
  directories instead makes nginx 301 to a trailing slash and every canonical
  would then point at a redirect.
- Generates `sitemap.xml` and `llms.txt` from the same page list, so they cannot
  drift. Do not add a static `public/sitemap.xml` back.

When editing pricing anywhere, edit `seo/data.js` too. Nothing on the site should
say "OpenShorts is free" without naming the Cloud price in the same breath: both
are true of different editions and quoting only the first one is what makes AI
answers describe the paid product as free.

### Cómo se elige el layout

`POST /api/process` acepta `layouts`: una lista (JSON) o cadena separada por
comas con `auto`, `split`, `screencast`, `speaker_cut`, `punch_in` y `none`.
Cada nombre enciende su variable de entorno para **ese** trabajo
(`app.py:layout_env`); `none` apaga el picker aunque prod corra con
`AUTO_LAYOUT=1` (recorte simple y nada más). Sin `layouts` manda el env del
despliegue, que desde el 25-ago-2026 es `AUTO_LAYOUT=1`. El dashboard lo expone
en opciones avanzadas ("vertical layout": auto / split / screencast / none,
`MediaInput.jsx`, recordado en `localStorage.os_layout`).

`auto` activa `layout_picker.py`: **una** llamada a Gemini por vídeo de origen
(no por clip) que elige entre `none` / `screencast` / `split`. Medido sobre el
corpus de 48 contra etiquetas revisadas a mano: 94% / 92% / 96% en tres pasadas,
con 0-1 falsos positivos sobre los 28 clips que no deben tocarse, y solo 2 clips
que cambian de respuesta entre pasadas.

**Manda 12 fotogramas a 1024px, no el vídeo.** Gemini factura vídeo a ~300
tokens por segundo: una hora de fuente son ~1,08M de tokens (no cabe en una
ventana de 1M) y una subida de 1-2 GB para recibir una palabra. Doce fotogramas
cuestan ~3k tokens **dure lo que dure la fuente**, que es lo que hace viable
esto con los podcasts de una hora que entran de verdad. La resolución importa y
el número de fotogramas no: a 640px detecta 15 de 20 (una hoja de cálculo es
ilegible), a 1024px sube a 17, y pasar a 24 fotogramas lo empeora. A 1024px la
diferencia con mandar el vídeo entero cae dentro de la varianza que ya tiene el
propio modo vídeo, a 2,2 s por clip en vez de ~15 s.

Lo que hace que funcione, y que conviene no deshacer: se le pide una **decisión
entre opciones cerradas**, no una medida. Los cuatro intentos anteriores (Canny,
MSER, cobertura temporal, anchura) le pedían un número y ninguno separó una hoja
de cálculo de un marcador de esquina. La varianza que este repo atribuía a
Gemini era de las medidas continuas, no del modelo.

`layout_picker.apply()` sólo **añade**: una elección explícita del usuario nunca
se desactiva porque el modelo diga `none`.

### Transcription quality gate and adaptive retry (`asr_quality.py`)

ASR is auto-detecting and **original-language only** (`task="transcribe"`,
never `"translate"`): clip timing, captions and metadata all need the words
that were actually said. Nothing used to look at the transcript between the
decoder and the clip selector. A Hindi/Hinglish stand-up upload
(`_pbr5nLNKHM`) came back from whisper-`small` as unreadable text, the
meaningful selector rejected every candidate (correctly — the input was
unusable), and the job died with "Clip detection failed", which blames the
wrong stage. Re-run on the same audio, what `small` actually produces is
Devanagari-shaped **phonetic nonsense** — "दिसकनेक्टिक फ्लाइट ती आर श्पाइश
जेट" — at `avg_logprob` -0.75; the same audio through `large-v3-turbo` with
the language pinned gives real Hinglish at -0.29.

`asr_quality.evaluate_transcript()` is the missing check. It scores a
transcript 0-100 from the signals faster-whisper computes and the pipeline used
to **throw away** — `avg_logprob`, `no_speech_prob`, `compression_ratio`,
`language_probability` — plus language-independent structural signals
(repetition and n-gram loops, speech rate, timestamp sanity, missing word
timestamps, junk fragments). Verdict: `GOOD` / `RETRY` / `BAD`.
`transcribe_media_checked()` runs **one probe, one pass, one repair**:
`_probe_best_model` decodes ~75 seconds (three 25s slices spread across the
timeline, via faster-whisper's own `clip_timestamps`) and upgrades to
`WHISPER_RETRY_MODEL` when the base model comes back non-GOOD *or* merely
`uncertain`; the video is then transcribed once with whatever that chose; only
a `BAD` result is transcribed again, once. `BAD` twice raises
`TranscriptQualityError`, so the user is told the audio could not be
transcribed instead of that no clips were found.
Debug: `meaningful_debug/asr_{initial,retry}.json` and `asr_quality_*.json`;
`tools/validate_asr_semantic.py output/<job>` replays a job's saved transcript
through semantic validation, sentence units, the Candidate Finder, the
over-long repair, the critic and the opening guard **without re-transcribing
or rendering anything** — which is how a failing video should be investigated
before any GPU time is spent on it.

Deterministic signals measure **structure**, and that is not enough on its
own. Two measured failures, one after the other:

1. whisper-`small` transcribed this video as fluent Devanagari-shaped
   **nonsense** — no repetition, valid timestamps, 173 words/min — and it
   scored 87/100.
2. The retry, `large-v3-turbo` at `avg_logprob` **-0.29** and a structural
   score of **100/100**, was still a third invented. Nothing about it was
   `uncertain`, so the reader was never called and the clip selector got it
   as GOOD.

So the reader (`asr_semantic.py`) is not gated on decoder confidence alone.
It samples ~6 windows spread across the WHOLE timeline — anchored on segments,
not character offsets, because the corruption in the real job lived in the
MIDDLE, exactly where a first/last-1500-characters sample never looks — labels
them in ONE structured call, and returns a per-region verdict plus
`GOOD` / `PARTIAL` / `BAD`. **A retry caused by a semantic failure is always
read again**, however confident the decoder is: the retry has to prove it
fixed the thing it was run for. `ASR_QUALITY_GEMINI` is honoured literally —
`0` never, `auto` only when justified, `1` every attempt — and a clean video
in `auto` costs zero reads. An unavailable reader is never a guilty verdict.

**The reader is advisory. `PARTIAL` is an ACCEPT.** This is the correction to
the original design, and it was measured on the same stand-up
(`Qd4jGgu06dw`). A transcript does not have to be *correct* to locate a moment
in a video — it has to be good enough to find the words, which is why the
upstream tool, with no quality gate whatsoever, cuts this video fine. Worse,
on genuinely code-switched speech `GOOD` is close to unreachable by
construction: `aggregate()` needs 80% of sampled seconds GOOD, while the
reader marks a whole 25-second window PARTIAL for one oddly-spelled word —
and correct phonetic Devanagari for a code-switched English word
(`प्रवल्म` = "problem", `संसिटिव` = "sensitive") reads as exactly that. So
every Hinglish video escalated to the end of the ladder whatever the models
produced, and its last rung was OOM-killed. Only `BAD` — nothing usable in
there at all — is now a refusal; a structural `RETRY` still earns the one
repair pass, then is used.

Two things that used to act on a `PARTIAL` verdict are gone: the **dense**
second reader pass over the whole timeline, and
`meaningful_pipeline._drop_unreliable_candidates`, which deleted any candidate
with more than 20% of its runtime inside a BAD region. A sample of 25-second
windows is fair evidence about a *transcript* and weak evidence about any one
*clip*, and the Blind Context Critic already reads every candidate in full —
so that filter removed proposals from a rough transcript before the judge that
could actually assess them ever ran, and the pipeline then reported "no usable
semantic candidates", which is a different failure from the real one.
`_flag_unreliable_candidates` records the overlap for the debug JSON and drops
nothing. Silence about a stretch still means "not examined", never "fine" and
never "bad".

**Transcription never loops.** There used to be an escalation ladder —
`small` → `large-v3-turbo` → `large-v3`, each rung re-transcribing the WHOLE
video, walked until something came back GOOD. On a 9.5-minute Hinglish upload
that was ~16 minutes of CPU per pass; the third model load was OOM-killed
(`exit code -9`) with turbo still resident, and the job delivered nothing
after ~50 minutes. `subtitles.whisper_model_ladder` now returns at most one
model, and `_get_whisper_model` evicts before allocating
(`WHISPER_MODEL_CACHE` defaults to **1**) so two large models are never
resident at once.

The probe is what replaced the extra rungs, and it is a **quality** decision,
not a language one — there is still no list of languages that get the big
model. The signal that makes it work is `uncertain` (`avg_logprob` below
-0.60): whisper-`small`'s full transcript of that video scored **77.8/GOOD**
while being a fifth invented, so status alone would have kept it, but the
decoder's own confidence said it was guessing — and that flag is available
after 75 seconds just as well as after 17 minutes. The probe's language is
pinned onto the upgraded pass for the same reason `_retry_language` exists.

`large-v3` is no longer wired in, though it does measurably rescue audio turbo
mangles — on the 36 seconds of the real stand-up turbo made the biggest mess
of:

| | avg_logprob | reader |
|---|---|---|
| `large-v3-turbo` | -0.944 | BAD 10 (phonetic gibberish) |
| `large-v3` | **-0.187** | PARTIAL 60 (intelligible; the bit's punchline survives) |

Across three sampled slices that took the transcript from 30% BAD to **0%**
BAD. It costs ~1.8x turbo's decode time on CPU and was the pass that got
OOM-killed, so it is available as `WHISPER_RETRY_MODEL=large-v3` for a
deployment with the RAM, and is not a third automatic attempt.
`tools/compare_asr_models.py` is how that was measured — it decodes short
slices straight out of a source with `clip_timestamps` (no ffmpeg, nothing cut
to disk) and has the pipeline's own reader grade each one, so the same
question can be re-asked before any future model change.

### Judges rank clips; they do not end the job

Measured on two real runs of the same pipeline. An English stand-up set
(`Vy96iuq94Ko`) scored **95-100** on six of seven candidates and produced six
clips. A Hinglish one (`Qd4jGgu06dw`) scored 80/40, 30/40, 30/20 and 60/30 —
four candidates that each contained a real joke — and the job died with
"Clip detection failed — the AI model did not return usable clips for this
video", which is simply false: the model returned four.

Four judges run in series (Candidate Finder, duration gate, Blind Context
Critic, Opening Independence Guard) and **every one of them could return an
empty list**, at which point `get_meaningful_clips` returns None and `main.py`
raises. Three changes, all the same shape — a gate that should have been a
preference:

- **The critic has a shipping tier.** Its only path through was `verdict ==
  PASS` + four booleans + **85 on BOTH** standalone and completeness, on a
  scale whose own prompt says "90+ should mean genuinely excellent". That is
  near-excellence demanded twice of every clip.
  `meaningful_critic.accept_floor()` (`MEANINGFUL_ACCEPT_FLOOR`, default 70)
  is the real quality question — is this worth publishing — and everything
  between the two is decided by RANK, which is what a score is for.
  `strict_pass` still exists and still means flawless; it just no longer means
  "or delete it".
- **A repair that made a clip worse is discarded.** The repaired verification
  replaced the original unconditionally, so on the real job a 60/30 candidate
  was "repaired" to 30/20 and the repair stage turned a borderline clip into a
  certain rejection (`_is_worse`, which also restores `final_range` — not just
  the sentence IDs, or the record keeps the repaired timestamps).
- **`meaningful_pipeline._best_effort` is the floor.** When nothing clears the
  bar, the best of what was actually found is published, ranked by the
  critic's own scores and flagged `low_confidence`, instead of the job
  failing. A user can delete a mediocre clip; they can do nothing with an
  error, and cannot tell from it whether the video was unsuitable or the tool
  broke. `MEANINGFUL_ALWAYS_PRODUCE=0` restores the old behaviour;
  `MEANINGFUL_FALLBACK_MAX` (3) caps it.

Replaying both jobs' saved verdicts through this: the English run is
**unchanged** (same six clips, same order) and the Hinglish one publishes three
instead of nothing.

### The judges are told what they are actually looking at

`meaningful_selector.WHAT_YOU_ARE_JUDGING` is shared by the Blind Context
Critic, the boundary repairer and both Opening Guard prompts. The critic used
to open with *"You are seeing EXACTLY AND ONLY what the eventual viewer will
hear"*, which is false in two ways that each cost real clips:

- **The viewer hears speech, not a transcript of it.** On the Hinglish job
  three of five rejections were the critic marking a clip down for ASR noise —
  *"contains significant transcription errors ('पुगली उलूश शुष')"*,
  *"fragmented, repetitive, and nonsensical"* — about a video in which a
  comedian tells a joke and an audience laughs. Misrecognised words are a
  defect in the TRANSCRIPT, not in the clip.
- **The viewer can see.** A fourth rejection demanded to know who "Sharon"
  was, while she stood on a stage holding a microphone. The speaker and their
  surroundings are visible and never need saying aloud.

The second rule is deliberately narrow, or it would gut the Opening Guard: the
SPEAKER is visible, everyone and everything they refer to is not, so a pronoun
with no antecedent ("she told me…") is still a failure. The repairer gets the
block too — it moves boundaries from the critic's failure reason, and would
otherwise try to repair its way around a garbled stretch.

Measured by replaying both jobs' saved transcripts through the real models
(`tools/validate_asr_semantic.py`, no re-transcription, no rendering):

| | candidates | critic accepted | survived the guard |
|---|---|---|---|
| Hinglish, before | 5 | **0** | 0 (job failed) |
| Hinglish, after | 5 | **4** at 95/95 | **3** |
| English, before | 7 | 6 | 6 |
| English, after | 8 | 8 | **6** (unchanged) |

The critic's own wording afterwards: *"The narrative is easy to follow despite
some transcription noise."* That is the whole fix.

### Clip duration is a preference, not a gate

`15-90` was hardcoded in 8 enforcement points across 5 files plus 4 prompt
strings. It is now `MIN_CLIP_SECONDS` / `MAX_CLIP_SECONDS` in
`meaningful_selector.py` (**15-110**), imported by the critic, the opening
guard and the pipeline, with `duration_rule_text()` as the single sentence
every prompt uses — so the rule cannot drift between what is enforced and what
the model is told.

90s was never a platform limit (YouTube Shorts allows 3 minutes, TikTok 10,
Reels 3); it was an editorial preference enforced as a gate, and that gate
threw away the best moment of a real job. The lift bit's punchline landed at
**107s**, so the only repair that could include it measured 107s, and a bare
`> 90.0` deleted it — leaving the version cut off mid-sentence, which the
critic then correctly rejected as incomplete. Every prompt still asks for
25-60s and says a complete 100-second clip beats a truncated 60-second one.

`build_candidate_windows`'s overlap is tied to `MAX_CLIP_SECONDS` rather than
written out: the overlap must be at least as long as the longest allowed clip
or a long moment straddles two windows and is proposable in neither.
`tests/test_overlong_repair.py` derives its spans from the constant for the
same reason — it used to hardcode the arithmetic of a 90-second cap, and
raising the cap silently inverted three of its assertions.

An over-long Candidate Finder proposal is narrowed before it is rejected
(`meaningful_selector.repair_overlong_candidate`). The old code dropped
anything over 90s outright, which on this video discarded three of four
proposals (110.9s, 95.3s, 117.2s) and left the critic a single candidate. The
repair asks for the shortest COMPLETE section inside that same range and
re-validates every returned ID locally: it must exist, lie inside the original
proposal, be in order, and land in the 15-90s band. No clamping, no arbitrary
word trimming — boundaries stay on sentence units like every other boundary
here, and "no complete shorter section exists" is still a rejection.

**Quality drives the retry, never the language, and there is no language
blacklist.** A clean Spanish video costs exactly one pass; a noisy English one
retries like any other. **Mixed scripts are not a defect either**: "मुझे
actually ये approach better लगती है" is normal Hinglish, so the script signal
only fires on three-plus *unrelated* non-Latin families and is capped
(`SOFT_SCORE_FLOOR`) so it can reach `RETRY` but never `BAD` on its own. The
signals that can condemn a transcript are the decoder's own confidence and
structural damage.

**The retry keeps attempt 1's language** when attempt 1 was at least
`ASR_PIN_LANGUAGE_ABOVE` (0.5) sure of it. This is not a nicety: measured on
the Hinglish stand-up, whisper-`small` detected `hi` at 89% and large-v3-turbo,
re-detecting for itself, decided `en` at 96% and emitted an English
**translation** of the Hindi audio ("This was the disconnecting flight of Bali,
SpiceJet") — whisper does that once the language token is wrong, even though
the task is always `transcribe`. It scored **99/100**, because it is fluent
English; no quality signal can catch it, and clip timing, captions and metadata
would all have switched language behind the user's back. The retry's job is to
transcribe the same speech better, not different speech. Below that confidence
the detection may itself be the fault, so the stronger model decides again.

`WHISPER_LANG_DETECT_SEGMENTS` (4) is the other half of the fix: faster-whisper
samples ONE window by default, so the first 30 seconds — applause, music, an
English intro — pick the language for the whole file and a wrong pick there
garbles everything with no way back.

`count_speech_units()` replaces `str.split()` wherever speech volume is
measured (`main.speech_is_sparse` included): Chinese, Japanese, Thai and Khmer
are written without spaces, so a good minute of Japanese counted as six
"words" and was routed to the silent-video vision path.

The four meaningful-pipeline prompts share one `MULTILINGUAL_RULES` block
(`meaningful_selector.py`) so the language policy cannot drift between stages,
and `build_sentence_units` recognises `। ॥ 。 ？ ！ ؟ ۔` and friends — its
Latin-only `. ? !` rule turned a whole Hindi or Japanese transcript into a few
pause-split blobs. Punctuation is never trusted alone: when the ASR barely
punctuated a transcript the pause threshold drops to 0.70s, and a hard ceiling
splits continuous speech that has neither punctuation nor a gap.

Captions in non-Latin scripts need fonts: Anton and the Liberation faces are
Latin-only, so both images install `fonts-noto-core` + `fonts-noto-cjk` and
`fonts/openshorts-fontmap.conf` appends Noto as a `<default>` fallback (Latin
text still renders in Anton — fontconfig only reaches the fallback for glyphs
the chosen font lacks). `remotion/src/lib/fonts.ts` does the same for the
browser renderer.

Arabic and Hebrew needed one more fix. libass shapes them correctly but applies
bidi **per override-delimited run**, and the karaoke path wraps the active word
in `{\c…}…{\r}` — which split the line into three runs, each reordered on its
own, putting the first word of an Arabic caption at the far LEFT. Verified
against the same text through the SRT path (no overrides), which orders it
correctly; Unicode RLE/RLM controls do NOT help, because the split happens
above the bidi layer. `generate_ass` emits those three runs in visual order for
an RTL block — the runs, not the words: libass still bidi-orders *inside* a
run, so reversing the words as well cancels itself out. `\q2` disables
auto-wrap there, because the flip is only valid for a single visual line.

### Remote transcription on Kaggle (`kaggle_worker.py`, `kaggle-worker/`)

Transcription is the long pole of a job — ~16 minutes of CPU for a 9.5-minute
video on the dev box — and it is the **only** stage worth moving off the
machine: its input is a URL and its output is a JSON transcript. The render
needs the video file and would have to ship gigabytes back, so it stays here.

**Kaggle runs BATCH kernels. There is no endpoint and nothing can call a kernel
while it runs.** The only interface is: push a version → it queues → it runs →
poll `kernels_status` → download the output. So `kaggle_worker.transcribe_url`
is a dispatcher, not a client: it substitutes the job into a copy of
`kaggle-worker/worker.py` (a **script** kernel, not a notebook, so the source
stays diffable and runnable by hand), pushes it, polls, and pulls
`transcript.json` out of the kernel output. Budget 3-8 minutes wall clock,
dominated by queue wait and kernel boot rather than by the work.

**It is strictly an accelerator and can never fail a job.** Every path —
no credentials, push rejected, kernel ERROR, timeout, missing output, a
transcript the quality gate rejects — returns None, and `main.transcribe_video`
transcribes locally exactly as it did before the module existed. Kaggle's GPU
quota is weekly, its queue is shared and its egress IP is sometimes blocked by
YouTube; none of that may take a job down.

The **cache kernel** (`kaggle-worker/cache_builder.py`, pushed by
`tools/kaggle_setup.py cache`) holds exactly one thing: the ~1.6 GB
faster-whisper weights, which every job would otherwise pull from HuggingFace.
The worker lists it in `kernel_sources` and Kaggle mounts its output read-only
at `/kaggle/input/`. Deliberately a KERNEL output and not a Dataset: a dataset
would have to be built on the dev machine, i.e. download 1.6 GB at home to
upload it back to Google.

It deliberately does **not** cache the BgUtils `node_modules`, and that is a
measured decision. Deno keeps npm packages in its own global cache and only
symlinks them into `node_modules`, so a copied tree is incomplete and its
symlinks point at build-time absolute paths — the first attempt got as far as
starting the token server and died with *"Could not find package 'lru-cache'
from referrer .../proxy-agent/dist/index.js"*. Doing it properly means shipping
`DENO_DIR` too and forcing both builds onto identical absolute paths, which is
a lot of fragility to save the ~1-2 minutes `deno install` actually costs. The
worker installs it fresh each run.

The repo id behind each model name is **not** written in the cache builder:
`faster_whisper.download_model` owns that mapping. A copy of it there was
already wrong once — it said `Systran/faster-whisper-large-v3-turbo` while
faster-whisper 1.2.1 resolves `large-v3-turbo` to
`mobiuslabsgmbh/faster-whisper-large-v3-turbo`, and the build died on a 401.
The download passes `use_auth_token=False`, because a Kaggle image can carry a
stale `HF_TOKEN` and huggingface_hub then sends it and gets 401 on a public
repo.

The download recipe in the worker is **measured, not improvised**: mweb plus a
BgUtils PO token, which fetched a 10-minute video from Kaggle's egress in
**9.3 s** with no bot-check. Do not simplify it away.

Two contract rules:
- The worker returns **raw** whisper words; `merge_continuation_words` runs on
  the backend, so there is exactly one implementation and a remote transcript
  cannot drift from a local one in how words are joined.
- A remote transcript clears the **same** gate via
  `transcribe_backends.judge_remote_transcript` — "it came from the GPU box" is
  not a quality argument. It returns None instead of raising on BAD, because
  the caller still has a working local path.

`render_worker` writes the job with **`pprint`, not `json.dumps`** — the
destination is a Python source file and the two literal syntaxes differ exactly
where it hurts: `json.dumps(None)` is `null`, `True` is `true`. The first real
dispatch died 26 seconds into a kernel with `NameError: name 'null' is not
defined`, and the test meant to prevent it had passed, because `ast.parse`
checks SYNTAX and `{"language": null}` is perfectly valid Python that fails at
import. The test now `ast.literal_eval`s the rendered JOB block and compares it
to the dict that was sent.

It also passes a **function** as the `re.sub` replacement rather than a string:
a URL is user input, and `re.sub` interprets backslashes in a replacement, so a
URL containing `
` or `` would become a literal newline or a group
reference. Both have tests.

**The language is pinned from a LOCAL probe before dispatch, and that is the
most important rule of the remote path.** `main.transcribe_video` runs
`transcribe_backends.probe_language_for_remote` on the copy it already
downloaded — the same three-slice probe, with the small model — and passes the
result to `kaggle_worker.transcribe_url(language=...)`. Measured on a real
Kaggle run: left to detect for itself, large-v3-turbo called the Hindi stand-up
**English at 90%** and returned a fluent English TRANSLATION of it —
*"Yesterday, I went to a lift and I went to a couple"* for *"कल में एक लिफ्ट
में गुसी..."* — although the task is always `transcribe`. It is the same
failure `_retry_language` exists to stop locally, and the remote path shipped
without that protection until a test run produced it. Nothing downstream can
catch it: `judge_remote_transcript` scores it ~99/100 because it is genuinely
good English, and the clips, captions and metadata all change language behind
the user's back. Below `ASR_PIN_LANGUAGE_ABOVE` the detection may itself be
wrong, so the remote model decides after all. The probe costs ~15s of CPU
against a 16-minute local transcription.

**Fetching the output has two traps, and the second one is the real one.**

The kernel used to leave its BgUtils git clone in `/kaggle/working` —
`.git/hooks/*.sample`, devcontainer config, `node_modules`, thousands of
entries — and Kaggle publishes that whole directory as the kernel's output.
`worker.prune_output()` now leaves only `transcript.json`, `result.json` and
the token-server log, in BOTH the success and failure paths, so the download is
three small files rather than a source tree.

The blocker, though, was encoding. `kernels_output` follows its own output
pages (`while token and page_token is None` — so pass ONE call and never a
`page_token`, which switches that off) and then writes the kernel log with
`open(outfile, "w")`, no encoding. On a host whose default codepage is not
UTF-8 — a Windows dev box, cp1252 — a Devanagari transcript in that log raises
`UnicodeEncodeError` **after** the data files have been written, in binary, to
disk. Treating that exception as failure threw away a GPU transcription that
had already succeeded, twice, while the kernel logs plainly said
"Transcribed 134 segments in 16.6s (cuda)". `_fetch_transcript` now checks
whether `transcript.json` exists before believing the exception. The Linux
container is UTF-8 and never hit this; only the host-side
`tools/kaggle_setup.py test` did.

Ceiling worth knowing: Kaggle is a batch notebook platform being used as an
inference API, and ~30 GPU-hours/week will not support the paid product. Modal,
RunPod, fal and Replicate give a real HTTP endpoint with seconds of latency.
The seam is `main.transcribe_video(source_url=...)`, so swapping the backend is
one module.

### The x264 preset is a speed knob, not a quality knob

`ffmpeg_utils.QUALITY` shipped at `-preset medium -crf 18`, and every "burn a
filter over a finished clip" pass uses it: the hook overlay (`hooks.py`), the
caption burn (`subtitles.py`), editor effects and `finalize_clip_passthrough`.
A clip gets at least two of them.

At a fixed CRF the preset trades encoding SPEED against bitrate efficiency —
CRF is what holds perceptual quality. Measured on a real 55.7s 1080x1920 clip
out of this pipeline, re-encoding at crf 18:

| preset | time | size | SSIM vs source |
|---|---|---|---|
| medium | 144.3s | 28.8 MB | 0.99756 |
| fast | 115.2s | 30.3 MB | 0.99755 |
| **veryfast** | **55.0s** | **26.5 MB** | 0.99650 |

2.6x faster, a **smaller** file, and 0.001 of SSIM given up — far less than
YouTube and TikTok destroy re-encoding the upload. `QUALITY` is now `veryfast`,
overridable with `FFMPEG_PRESET_QUALITY`. On the two-clip Hinglish job that is
roughly **180 seconds per clip**.

**This replaced a worse plan.** The obvious target was the three encodes per
clip — reframe, then hook, then captions — collapsed into one. It is the wrong
change: those intermediate files are the ADDRESSING SCHEME, not waste.
`app.py` parses them in eight places — `_strip_burned_captions`
(`^subtitled_\d+_(.+)$`) walks back to the uncaptioned file so the modal can
RESTYLE captions instead of layering them, `_strip_burned_hook`
(`^(?:hooked_\d+_|hook_)(.+)$`) does the same for the hook, `_canonical_clip_file`
globs all three prefixes for downloads and social posting, and the dashboard
decides whether a clip "has captions" from the prefix. Collapse the encodes and
`hooked_<ts>_<clip>.mp4` never exists, so restyling captions walks back to a
missing file. Fixing that means redesigning how clips are addressed across
`app.py` and the dashboard — a large refactor of working code, to save less
time than one constant did.

Hardware encoding is not the answer on this box either: `h264_nvenc` is
compiled into the image but the dev machine is Intel Iris Xe, so there is no
NVIDIA GPU for the container to use.

### Docker build context and the model cache

A local `docker compose up --build` used to send **4.0 GB** of context and die
with a BuildKit EOF. `.gitignore` excluded `output/` and `.cache/`;
`.dockerignore` did not, so the faster-whisper model downloads (~2 GB in
`.cache/huggingface`) and every rendered job (~800 MB in `output/`) were being
uploaded to the daemon on every build. Both are produced at RUNTIME inside the
container and bind-mounted back in by compose, so neither may ever enter the
image. With them excluded the context is **5.1 MB**.

A named volume for `/app/.cache/huggingface` was considered and deliberately
NOT added: the dev compose bind-mounts `.:/app`, so that path already persists
across container recreation, and a named volume would shadow the existing
on-disk cache and force a multi-GB re-download for no gain. It only becomes
the right answer if the source bind mount goes away.

### Hook grounding for on-screen clips (`hook_grounding.py`)

The hook and title come from the detail pass, which only reads the
transcript, so on a clip whose meaning is on the screen (a settings dialog,
a spreadsheet) they summarise the video's topic instead of naming what is
shown. After the render, if the `<clip>.layout.json` sidecar says at least
25% of the clip is `screencast` / `wide` / `inset` (plus `general` when the
layout picker called the video a screencast: a face-less scene there is a
slide or a dialog, not a group shot), three frames from those
stretches at 1024px plus the clip's own words go to Gemini
(`GroundedHook`) and `viral_hook_text` / `video_title_for_youtube_short`
are rewritten in place before `auto_hook_clip` burns them; the originals
stay under `hook_grounding.before`. Gemini-only (frames): with just a local
LLM it logs one line and keeps the transcript hook. `HOOK_GROUNDING=0`
disables it. The detail prompt itself now carries the rule "about this
moment, not the video", which is the cheap half of the same fix.

### Local LLM for the moment picker (`llm_backend.py`)

`LLM_BASE_URL` (+ `LLM_MODEL`, `LLM_API_KEY`) routes the two transcript
passes of `get_viral_clips` to any OpenAI-compatible `/chat/completions`
instead of Gemini; the response is validated with the same pydantic schemas
Gemini enforces server-side, so `main.py` sees one shape. `main.score_batch_size`
drops to 3 windows per call there (local contexts are 4-8k; a truncated
prompt scores garbage silently). Self-host `/api/process` then accepts a
request without `X-Gemini-Key` and `/api/config.localLlm` tells the dashboard
not to demand one. Frame-based stages (`layout_picker`, `screencast_layout`,
`get_visual_clips`) stay on Gemini and degrade as they always did without a
key. Never wired in cloud mode: `BILLING_ENABLED` ignores it.

### Thumbnail Studio (`thumbnail.py`, `/api/thumbnail/*`)

Titles come from the transcript plus 10 frames at 1024px, never the whole
video (same reasoning as the layout picker: an hour of video is ~1M tokens for
a text task). Two calls: a 25-title brainstorm across fixed styles, then a
critic that scores, dedupes by angle and returns 10, each paired with a 1-4
word `thumbnail_text` that complements the title rather than repeating it.
Rules baked in: payoff inside 50 characters (phones cut there), keyword in the
first 3 words, same language as the transcript. Text model is
`GEMINI_MODEL_THUMBNAIL` (default `gemini-3.7-flash`), deliberately not
`GEMINI_MODEL`: flash-lite is fine for a closed-choice layout pick and visibly
worse at creative titles. Image model is `GEMINI_IMAGE_MODEL` (default
`gemini-3.1-flash-image`).

Thumbnails are `count` **different concepts**, not one prompt repeated: a text
call designs each (hook text, side for the text, palette, scene prompt), then
one image call per concept in parallel. By default (`burn_text=true`) the
image model is told to leave that side as negative space and PIL sets the text
in Anton with a black stroke, so accents and spelling are never wrong; the
`AI painted` toggle lets the model render the text itself. Every output is
cover-cropped to 1280x720 and saved under YouTube's 2 MB limit.
`GET /api/thumbnail/frames/{session}` scores sampled frames by face area and
sharpness (MediaPipe + Laplacian), keeps them spread across the runtime, and
the dashboard offers them as the person reference so the thumbnail shows the
creator instead of a stranger; an uploaded face photo still wins.

### Video Reframing Modes

**A source already shot vertical is passed through untouched.**
`reframe_v2.source_already_fits()` gates it: every layout below reorganises the
frame to buy back width the crop threw away, and on a 9:16 upload there is none
to buy. GENERAL was the visible failure — its 0.42 height ratio, which buys
presence on a landscape source by overflowing the sides, scaled a 1080x1920
source down to a 453px sliver floating over a blurred copy of itself, and the
scene classifier routes every face-less shot (a slide, a screen recording) there.
So the picker is skipped (one Gemini call saved per upload), the classifier is
skipped, and every scene renders TRACK, whose crop is the whole frame.
`general_filtergraph` additionally floors the foreground at the height where the
source fills the output width, so the editor's explicit GENERAL override on a
portrait clip cannot reproduce the shrink either.

- **TRACK Mode** (single subject): MediaPipe face detection + YOLOv8 fallback with "Heavy Tripod" stabilization
- **GENERAL Mode** (groups/landscapes): Blurred background layout preserving full width
- **SPLIT Mode** (two-shot conversation, `split_layout.py`): both speakers stacked
  in half-frames. Off by default (`SPLIT_LAYOUT=1`); v2 engine only, so a
  fallback to the v1 loop silently renders GENERAL instead. It upgrades scenes
  the classifier already sent to GENERAL, never TRACK ones, and needs both faces
  visible **in the same frame** for at least half the sampled frames — that is
  what separates a real two-shot from a plano/contraplano, where stacking would
  show the same person twice. `SPLIT_TIGHTNESS` (default 0.8) trades a little
  upscale for keeping the other speaker out of each half. Captions on a SPLIT
  stretch sit on the seam between the halves (`{\an5}` per word event in
  `subtitles.generate_ass`), the one place they cover nobody; the render
  records which stretches are stacked in a `<clip>.layout.json` sidecar
  (`layout_ranges.py`) and every metadata writer copies it into the clip's
  `layout_ranges`, so `/api/subtitle` finds it after a restyle too. The fast
  rerender (cut without reframe) carries the canonical clip's ranges through
  the new cut (`layout_ranges.remap`, in `recut.perform_recut`). Only the
  ASS path can do this; SRT burns keep one alignment for the whole file.
- **SCREENCAST / WIDE Modes** (`screencast_layout.py`, `SCREENCAST_LAYOUT=1`):
  for scenes whose meaning lives outside the centre. Gemini reports each range's
  **width_fraction**, and that is the gate — coverage was tried before and did
  not separate a spreadsheet from a corner ticker, while width does (a bug spans
  ~15% and survives any crop, a spreadsheet spans ~100% and cannot). Content
  narrower than 0.5 moves nothing. Between 0.5 and 0.85 there is room beside the
  content, so SCREENCAST stacks it over the presenter. Above 0.85 the presenter
  is composited **on top of** the content and stacking would show it twice, so
  those scenes get WIDE: the GENERAL layout with side-cropping disabled.
- **INSET Mode** (`camera_inset.py`): pantalla a ancho completo arriba, el
  recuadro de la webcam ampliado abajo. Para el caso de una sola fuente con la
  cámara compuesta en una esquina (OBS, VOD de stream). Se encadena detrás de
  la decisión `screencast`, **no** se le pregunta a Gemini: ofrecido como cuarta
  opción respondió `screencast` en los 5 clips que tienen recuadro, en dos
  pasadas, y la exactitud global cayó de 92% a 83-85%. El detector geométrico
  encuentra esos 5 sin falsos positivos. Los tres filtros que hacen falta, cada
  uno pagado con una iteración: sujeto **pequeño**, **descentrado en
  horizontal** (una cara de talking head está centrada aunque esté alta), y
  **quieto entre muestras** (3-11px frente a 316px de una persona real).
- **ALTERNATE Mode** (`active_speaker.py`, `SPEAKER_SIGNAL=1` + `SPEAKER_CUT=1`):
  hard cuts to whoever is talking, rendered through the TRACK path as a
  trajectory with jumps. `SPEAKER_SIGNAL=1` alone just gates SPLIT on both people
  actually speaking. Mouth activity **must** be normalised per speaker before
  comparing (`normalise_activity`): raw frame-difference magnitude scales with
  local contrast and lighting, and on a real two-shot it handed one speaker
  90-100% of the scene.
- **Punch-in** (`punch_in.py`, `PUNCH_IN=1`): not a layout. A ~12% push on the
  clip's beats, riding the TRACK path by widening its per-frame crop command
  from x-only to w/h/x/y. Beats currently come from the audio envelope;
  `emphasis_times` is a plain list of seconds so the transcript's hook words can
  replace it without touching the module.

### Key Classes
- `SmoothedCameraman` - Stabilized camera movement with safe zone logic (prevents jitter)
- `SpeakerTracker` - Prevents rapid speaker switching, handles temporary occlusions

### API Endpoints
| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/process` | Submit video for processing |
| GET | `/api/status/{job_id}` | Poll job status and logs |
| POST | `/api/edit` | Apply AI video effects |
| POST | `/api/subtitle` | Generate and apply subtitles (auto-transcribes dubbed videos) |
| POST | `/api/hook` | Add text hook overlays |
| POST | `/api/translate` | AI voice dubbing via ElevenLabs |
| GET | `/api/translate/languages` | List supported dubbing languages |
| POST | `/api/social/post` | Post to social media (async upload) |
| POST | `/mcp` | MCP server (JSON-RPC): the pipeline as agent tools |
| POST/GET/DELETE | `/api/keys` | User API keys (cloud mode, session JWT only) |
| DELETE | `/api/account` | Erase the account and everything in it (GDPR art. 17) |

### Agent access (MCP, API keys, webhooks)

- **API keys** (`cloud/api_keys.py`): `osk_...` tokens, sha256-stored, created in
  the dashboard account page. `cloud/auth.get_current_user_optional` accepts
  them (`Bearer osk_...` or `X-API-Key`) and resolves the owner, so metering,
  entitlement, plan priority and job ownership apply to agents with zero
  endpoint changes. Key management itself refuses API-key auth: a leaked key
  cannot mint replacements.
- **MCP server** (`mcp_server.py`, mounted always): stateless Streamable-HTTP
  JSON-RPC at `/mcp` — no SDK dependency, ~3 methods + 8 tools. Each tool calls
  back into this same app in-process (`httpx.ASGITransport`) forwarding the
  caller's auth headers, so it can never drift from the REST behavior. Cloud
  mode 401s without a resolvable user; self-host stays BYOK-open.
- **OAuth for MCP clients** (`cloud/mcp_oauth.py`, cloud mode only): claude.ai
  and ChatGPT connect by URL, so the server publishes RFC 9728/8414 metadata
  under `/.well-known/`, accepts dynamic client registration (`POST
  /oauth/register`, public clients, PKCE S256 mandatory) and bounces
  `GET /oauth/authorize` to the dashboard consent screen (`#/oauth/authorize`),
  because the session JWT lives in localStorage on the frontend host and a
  bare API GET cannot see it. `POST /api/oauth/authorize` (session auth) mints
  the code; `POST /oauth/token` redeems it by **minting an ordinary `osk_`
  key** named after the client and returning it as the access token. No new
  auth path, no refresh tokens: the key shows up in Account → API keys and
  revoking it disconnects the app. The `/mcp` 401 carries
  `WWW-Authenticate: Bearer resource_metadata=...` so clients find the flow.
  `oauth_codes` is in `USER_OWNED_TABLES`; `oauth_clients` deliberately not.
- **Webhooks**: `POST /api/process` takes `webhook_url` + optional
  `webhook_secret` (HMAC-SHA256, `X-OpenShorts-Signature`). Validated with
  `security_utils.assert_public_url` at submit AND at delivery (DNS rebinding).
  Fired once per job from `run_job_wrapper` after the R2 archive so the payload
  can carry durable download links; survives redeploys via the resume manifest.
  `PUBLIC_API_URL` env sets the absolute-URL base when behind a proxy.

### Account erasure (GDPR art. 17)

`DELETE /api/account` (`cloud/account.py`, dashboard: Account → Delete account)
is immediate and irreversible: there is no recovery window because after the
delete there is nothing left to authenticate a recovery request against. It
refuses API-key auth (a leaked `osk_` must not destroy its own account) and
requires the caller to retype the account email.

The order of the steps is the design, and each one is a failure mode:
**Stripe cancel first**, aborting the whole thing if it fails, so we never erase
a user we are still billing; **R2 before the database**, because those rows are
the only index of which objects are theirs and dropping them first turns a
failed purge into permanent orphans; the DB delete is **one transaction** over
an explicit table list (`USER_OWNED_TABLES`) rather than the declared ON DELETE
CASCADEs, since `create_all` never ALTERs an existing table and a constraint
added after a table shipped exists in the models but not in production.
`tests/test_account_erasure.py` fails if a new table references `users.id`
without joining that list.

`app.py` registers a callback for the local working files, which record
ownership three different ways: the `.owner` file clip jobs write (so jobs
recovered from disk after a restart count too), `saas_jobs`, and
`thumbnail_sessions`. That last one is the only thing that ever deletes
generated thumbnails: the hourly sweep skips their directory and they are
served publicly at `/thumbnails/`.

What deliberately survives: the Stripe customer and its invoices (6-year
retention, Spanish commercial law) and one `account_deletions` row holding a
sha256 of the email as proof the erasure happened, itself purged after 5 years.
The "why are you leaving" answer is a closed list (`DELETION_REASONS`), never
free text — anything the user could type would land in a row designed to
outlive them. Deleting users also made one webhook path reachable that never
was before: `_apply_topup` reads the user id from Stripe metadata, so it now
confirms the row still exists before inserting, or the FK violation makes
Stripe retry the same doomed event for three days.

### Concurrency Model
Async job queue with semaphore-based concurrency control. Configure via `MAX_CONCURRENT_JOBS` env var (default: 5). Jobs auto-cleanup after 1 hour.

### Paid proxy accounting (`cloud/proxy_ledger.py`)

Downloads go direct → static ISP proxies (flat rate) → DataImpulse (per GB),
and the duration probe (`cloud/metering.probe_url_minutes`) follows the same
order, with one extra free step before any per-GB attempt: the fallback
clients through a static (`fallback-static`). **The client list is explicit
and shared** (`yt_clients.py`: `default,mweb` + the bgutil PO token
provider): with account cookies yt-dlp's own defaults are `tv_downgraded` +
`web`, and on a share of videos both come back UNPLAYABLE / SABR-only, which
yt-dlp reports as "Video unavailable". That was mistaken for an IP ban for a
week (it happened on every static IP too) and fed ~26 downloads a week to
the per-GB proxy, which then fetched 360p through the same dead list.
Measured in the prod container on 6-sep-2026, same static, same video:
cookies + defaults → unavailable; cookies + `default,mweb` → 1080p; no
cookies → 1080p. `mweb` needs the PO token, and the token needs the
webpage: never put `player_skip: webpage` back. A fallback attempt runs
anonymously when an HD attempt already failed with the cookies on that
route, and every attempt asks for the 1080p format spec (the old
`best[ext=mp4]` fallback spec was itself the 360p progressive file).
Two rules keep the per-GB proxy at zero on a normal day: the probe
reaches it **only** when a static route failed for a reason another IP can
fix (`static_failure_warrants_paid`: bot-check, 403/429, proxy/network
errors), never for a private/removed/members-only video, an uploader's
country block (the residential pool failed identically in 5 of 6 paid
probes, 3-5 sep) or a live stream
with no duration (those failed the same on every IP and used to cost ~1.7 MB
× 2 extractors each), and **never for a non-YouTube URL** (the download
plan already excluded those; Twitch, Kick, Rumble and product pages were
reaching it through the probe). The probe also carries `YOUTUBE_COOKIES`,
like the download does: an anonymous probe from the static IPs gets "Sign in
to confirm you're not a bot" in bursts (4-sep-2026: ~10 probes in one hour,
1.8 MB each on the per-GB proxy) because a datacenter IP's anonymous rate
limit is low and we make ~400 YouTube hits a day from three of them, while
the authenticated download sails through the same IPs. `main.py` prints `PROXY_ROUTE=<json>` after
every download (winner, paid bytes across all attempts including failed
paid ones, each free attempt's error); `app.py` persists it as a
`proxy_usage` row at job end and pages Telegram when the paid proxy carried
bytes, folding a burst into one message per 5 min. The in-memory monthly
counter and the container log (rotates within the hour) cannot answer "what
cost $14 on the 28th"; the table can. `PAID_PROXY_DAILY_MB` (default
500) is the hard ceiling: past it the paid proxy is dropped from the probe
and from every new job's env until UTC midnight. The watcher probes the
static pool against a real YouTube watch page (playable markers), not
google.com — the 28th happened because YouTube refused the static IPs while
google kept answering 204. On the dev Mac, do not keep
`PROXY_URL` in `.env`: every local `main.py` run then bills DataImpulse.

### Deploys and running jobs (handover + drain)

Every push to `main` redeploys the API container. Coolify starts the NEW
container before stopping the old one (rolling update) and both share
`output/`, so `app.py` coordinates them instead of relying on a fast swap:

- Each instance writes its id to `output/.instance` at startup. An instance
  that sees another id there is the old one and **drains**: it finishes the
  jobs it is running, starts none, and leaves queued manifests on disk.
- A running job heartbeats its `.resume.json` every 10 s. The resume scan
  (startup + every 30 s) re-enqueues only manifests nobody heartbeated for
  60 s, so no job runs twice and none is lost. Max 2 resume attempts.
- SIGTERM (`docker stop`) drains too, up to `DRAIN_TIMEOUT_SECONDS` (840),
  then hands the signal to uvicorn. The app's Coolify stop grace period is
  900 s (`application_settings.stop_grace_period`); keep the timeout below it.
  After the drain hands the signal to uvicorn, `--timeout-graceful-shutdown 15`
  (Dockerfile) caps the wait for in-flight connections: uvicorn's default is
  unbounded, and one open range download kept a drained container alive for
  the full grace period while Traefik still routed half the traffic to its
  closed port.
- `/health/ready` + the Dockerfile `HEALTHCHECK` are what keep Traefik off a
  dying container: its docker provider only routes to `healthy` containers,
  so an instance answers 503 from the moment it gets SIGTERM (out of rotation
  within ~10 s, socket still open) and a booting one gets no traffic until it
  answers. Only SIGTERM flips it, not the marker drain: at that point the new
  container is still booting and nobody else would be routable. The Coolify
  app has its health check enabled on that path so it waits for the new
  container to be `healthy` before stopping the old one. With that option on,
  Coolify replaces the Dockerfile HEALTHCHECK with its own curl/wget command
  AND its own interval/retries (5 s × 3), so the image must ship `curl` or
  every deploy rolls back as unhealthy, and a stopping container takes 15 s
  to turn `unhealthy`. That is why the drain keeps serving for
  `PROXY_DRAIN_SECONDS` (20) after the jobs are done before it hands the
  signal to uvicorn: closing the socket earlier is 502s until Traefik
  notices (measured ~60 s per deploy with retries=12 and no grace). And
  `HARD_EXIT_SECONDS` (30) after that the process is ended outright: uvicorn
  finishing does not end the interpreter while an executor thread hangs in
  a network probe, and that kept a drained container alive for the full 900 s.
  `/health` stays a plain liveness probe for the external watcher.
- `/api/status` answers from disk for a job this instance never held, so a
  poll landing on either container during the handover is fine.
- `main.py` leaves `.transcript_checkpoint.json` in the job dir so a job that
  does get re-run skips the paid transcription (download and Gemini repeat).

Before pushing, still batch small commits (tests, docs) with the next real
change: every deploy is a ~5 min build plus a handover.
