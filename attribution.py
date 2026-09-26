"""Credit the creator whose video a clip was cut from.

yt-dlp already knows who uploaded a video — `uploader`, `uploader_id` (the
modern `@handle`), `uploader_url`, `channel_id` — and `main.py` used to take
`info['title']` and throw the rest away. This module keeps it, and turns it
into a line that can go on a post.

TWO THINGS MAKE THIS HARDER THAN IT LOOKS.

A YouTube handle is not an Instagram handle. `@sharonvermacomedy` on YouTube
may be someone else entirely on Instagram, and an @mention that names the
wrong person is worse than no credit at all — it publicly tags a stranger.
So the only cross-platform handles used here are ones the creator themselves
published in the video description, and the choice between them is made by a
model that is handed a CLOSED LIST and can only pick from it (`pick_handle`).
It cannot write a handle, so it cannot invent one. That is the same shape the
layout picker uses, and for the same reason: asking for a decision between
known options is reliable where asking for a value is not.

And a description is full of links that are not the creator's: the venue, the
editor, a sponsor, the channel's other show. Measured on two real videos, one
had a single Instagram link and the other had two — the comedian's and the
comedy club's. A regex alone cannot tell those apart; the surrounding words
can, which is why each candidate is stored with the text around it.

Credit is not a licence. A standard YouTube upload stays all-rights-reserved
whether or not a post names the author, and `license` is recorded here so that
question can at least be asked.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional

#: Written next to the downloaded video. A sidecar rather than a return value
#: because the attribution has to survive a resumed job: `main.py` may re-enter
#: after a restart with the video already on disk and no yt-dlp call to make.
SIDECAR = ".source_attribution.json"

#: Characters kept either side of a social link, as the evidence for whose
#: account it is. Enough for "Follow the comedian: <link>" or "Venue: <link>"
#: to be readable, short enough that a description full of links stays small.
CONTEXT_CHARS = 90

_PROFILE_PATTERNS = [
    # Instagram profile URLs. /p/, /reel/, /reels/, /tv/, /stories/ and
    # /explore/ are POSTS, not accounts — crediting one of those would link a
    # single video rather than the person, and /explore/ is not a person at all.
    ("instagram", re.compile(
        r"instagram\.com/(?!p/|reel/|reels/|tv/|stories/|explore/)"
        r"@?([A-Za-z0-9._]{1,30})", re.I)),
    ("tiktok", re.compile(r"tiktok\.com/@([A-Za-z0-9._]{1,24})", re.I)),
    ("twitter", re.compile(r"(?:twitter|x)\.com/(?!i/|home|share)"
                           r"([A-Za-z0-9_]{1,15})", re.I)),
]

#: Fields worth keeping out of yt-dlp's info dict. The rest of it is hundreds
#: of keys of formats and thumbnails that nothing here reads.
_INFO_FIELDS = ("uploader", "uploader_id", "uploader_url", "channel",
                "channel_id", "channel_url", "webpage_url", "license",
                "upload_date", "title")


def find_socials(description: str) -> List[Dict[str, str]]:
    """Profile links in a video description, each with the words around it.

    Deduplicated per (platform, handle): creators routinely paste the same
    Instagram link twice, and the same account listed twice is not two
    candidates.
    """
    found: Dict[str, Dict[str, str]] = {}
    text = description or ""
    for platform, pattern in _PROFILE_PATTERNS:
        for match in pattern.finditer(text):
            handle = match.group(1)
            if not handle or handle.lower() in ("www", "http", "https"):
                continue
            key = f"{platform}:{handle.lower()}"
            if key in found:
                continue
            start = max(0, match.start() - CONTEXT_CHARS)
            end = min(len(text), match.end() + CONTEXT_CHARS)
            found[key] = {
                "platform": platform,
                "handle": handle,
                "context": " ".join(text[start:end].split()),
            }
    return list(found.values())


def from_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """The attribution worth keeping from one yt-dlp info dict.

    The description is NOT stored — it can be thousands of words, and the only
    part that matters here is the profile links, which are extracted now while
    the description is in hand.
    """
    data = {key: info.get(key) for key in _INFO_FIELDS}
    data["socials"] = find_socials(info.get("description") or "")
    return data


def save(output_dir: str, data: Dict[str, Any]) -> Optional[str]:
    """Write the sidecar. Never raises — attribution must not fail a download."""
    path = os.path.join(output_dir, SIDECAR)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
    except Exception as e:
        print(f"⚠️ Could not record source attribution "
              f"({type(e).__name__}: {e}) — clips will publish uncredited.")
        return None


def load(output_dir: str) -> Optional[Dict[str, Any]]:
    """Read the sidecar, or None. Absent is normal: uploads have no uploader."""
    path = os.path.join(output_dir, SIDECAR)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def youtube_handle(attr: Dict[str, Any]) -> Optional[str]:
    """The uploader's YouTube handle, without the @.

    `uploader_id` is the handle on modern YouTube ("@tarunratnani8819") but was
    a channel name on older extractions and can still be a bare UC… id, which
    is not a handle and must not be printed as one.
    """
    raw = str((attr or {}).get("uploader_id") or "").strip()
    if raw.startswith("@"):
        return raw[1:]
    if raw.startswith("UC") and len(raw) > 20:
        return None
    return raw or None


def candidates_for(attr: Dict[str, Any], platform: str) -> List[Dict[str, str]]:
    """The profile links found for one platform."""
    return [s for s in (attr or {}).get("socials") or []
            if s.get("platform") == platform]


PICK_PROMPT = """A short video clip is about to be published, cut from someone
else's longer video. The post should credit the person who MADE that video.

Below are {platform} accounts found in the source video's description, each
with the text surrounding the link. Exactly one of them, at most, belongs to
the creator being credited. The others are typically the venue, the editor,
the videographer, a sponsor, a collaborator, or the channel's other shows.

CREATOR: {uploader}
YOUTUBE CHANNEL: {channel_url}

CANDIDATES:
{candidates}

Answer with the NUMBER of the account belonging to {uploader}, or 0 if none of
them clearly does. Do not guess: crediting the wrong account publicly tags a
stranger, and no credit is better than a wrong one.

Return only: {{"choice": <number>}}
"""


def pick_handle(attr: Dict[str, Any], platform: str,
                api_key: Optional[str] = None,
                model_name: Optional[str] = None) -> Optional[str]:
    """The creator's own handle on `platform`, or None.

    Returns immediately without a model call when there is nothing to decide:
    no candidates, or exactly one. The model is only asked to DISAMBIGUATE, and
    its answer is an index into the list it was shown — it never writes a
    handle, so it cannot produce an account that was not in the description.
    """
    candidates = candidates_for(attr, platform)
    if not candidates:
        return None
    if len(candidates) == 1:
        # Nothing to disambiguate, so nothing to ask. This is the common case:
        # of two real videos measured, one listed a single account.
        return candidates[0]["handle"]

    # Several accounts, and telling them apart needs to read the words around
    # them. That is a Gemini call, and it is OPT-IN: this pipeline already
    # spends heavily on Gemini per video (candidate finder, critic, opening
    # guard, metadata, transliteration), and one more call for a nicety is not
    # obviously worth a free-tier quota. Off, the credit names the creator in
    # words — which tags nobody and is never wrong.
    if os.environ.get("CREDIT_PICK_HANDLE", "").strip() not in ("1", "true", "yes"):
        print(f"   ℹ️ {len(candidates)} {platform} accounts in the description; "
              f"crediting by name (set CREDIT_PICK_HANDLE=1 to pick one).")
        return None

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Several candidates and no way to choose. Picking the first would be a
        # coin flip with someone else's name on it.
        print(f"   ℹ️ {len(candidates)} {platform} accounts in the description "
              f"and no Gemini key to tell them apart — crediting by name only.")
        return None

    from google import genai
    from google.genai import types as genai_types
    from pydantic import BaseModel

    class Choice(BaseModel):
        choice: int

    listing = "\n".join(
        f'{i}. @{c["handle"]}  —  "{c["context"]}"'
        for i, c in enumerate(candidates, start=1))
    prompt = PICK_PROMPT.format(
        platform=platform,
        uploader=attr.get("uploader") or attr.get("channel") or "the creator",
        channel_url=attr.get("channel_url") or attr.get("uploader_url") or "(unknown)",
        candidates=listing)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=(model_name or os.getenv("GEMINI_MODEL")
                   or "gemini-3.1-flash-lite"),
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Choice, candidate_count=1, temperature=0.0))
        choice = int(json.loads(response.text)["choice"])
    except Exception as e:
        print(f"   ⚠️ Could not pick a {platform} account "
              f"({type(e).__name__}: {e}) — crediting by name only.")
        return None

    # 0 means "none of these is the creator", which is a real answer. Anything
    # outside the list is a malformed one, and both end the same way.
    if not 1 <= choice <= len(candidates):
        return None
    return candidates[choice - 1]["handle"]


def credit_line(attr: Dict[str, Any], handle: Optional[str] = None,
                template: Optional[str] = None) -> str:
    """One line of credit, or "" when there is nothing to credit.

    `handle` is a platform handle already resolved by `pick_handle`; without
    one the creator is named in words, which is still attribution and still
    true. A YouTube handle is never substituted for a missing Instagram one —
    "@name" on Instagram means an Instagram account.
    """
    if not attr:
        return ""
    name = (attr.get("uploader") or attr.get("channel") or "").strip()
    if not name and not handle:
        return ""
    template = (template
                or os.environ.get("CREDIT_TEMPLATE", "").strip()
                or "🎥 Original: {credit}")
    return template.format(credit=f"@{handle}" if handle else name,
                           name=name or (handle or ""),
                           url=attr.get("webpage_url") or "")


def apply_to_caption(caption: str, credit: str, limit: int = 2200) -> str:
    """Put the credit into a caption, above any trailing hashtags.

    Hashtags belong at the end of an Instagram caption, so appending the credit
    after them buries it in the tag block where nobody reads it. It goes on its
    own line before them instead.

    The credit is what survives if the result is too long: a caption trimmed
    into a broken sentence is worse than a shorter one, and an attribution that
    got silently truncated away is worse than either.
    """
    credit = (credit or "").strip()
    if not credit:
        return caption
    body = (caption or "").rstrip()
    if credit in body:
        return body                     # re-queued clip; do not credit twice

    lines = body.split("\n")
    tail = []
    while lines and (not lines[-1].strip()
                     or lines[-1].strip().startswith("#")):
        tail.insert(0, lines.pop())
    head = "\n".join(lines).rstrip()
    tail_text = "\n".join(tail).strip()

    parts = [p for p in (head, credit, tail_text) if p]
    out = "\n\n".join(parts)
    if len(out) <= limit:
        return out

    # Too long: drop the hashtags first, then trim the body — never the credit.
    out = "\n\n".join(p for p in (head, credit) if p)
    if len(out) <= limit:
        return out
    room = limit - len(credit) - 2
    return (head[:max(0, room)].rstrip() + "\n\n" + credit) if room > 0 else credit
