"""yt-dlp client selection for YouTube, shared by the download (``main.py``)
and the duration probe (``cloud/metering.py``) so the two cannot drift.

Why the client list is explicit. With account cookies yt-dlp's defaults are
``tv_downgraded`` + ``web``, and for a share of videos both come back
UNPLAYABLE / SABR-only, which surfaces as "Video unavailable" with no
formats. That looked like an IP ban (it also happened on every static ISP IP)
and for a week sent ~26 downloads a week to the per-GB proxy, which then
downloaded 360p through the same dead client list. Measured on 6-sep-2026
inside the prod container, same static proxy, same video:

    cookies + default clients            -> Video unavailable
    cookies + mweb (+ PO token)          -> 1080p
    no cookies + default clients         -> 1080p
    cookies + default,mweb (+ PO token)  -> 1080p (also on a video that worked)

``mweb`` needs a GVS PO token for its https formats, which the bgutil
provider mints as long as the webpage is NOT skipped (``player_skip:
webpage`` drops the account's Data Sync ID and the token request fails).
The old fallback list (``tv_embed``, ``android``) is gone from yt-dlp: the
first is "unsupported", the second is skipped whenever cookies are present.
"""

HD_CLIENTS = ["default", "mweb"]


def pot_provider_args(bgutil_http, bgutil_script):
    """Extractor args selecting the PO token provider (bgutil http or script)."""
    if bgutil_http:
        return {"youtubepot-bgutilhttp": {"base_url": [bgutil_http]}}
    if bgutil_script:
        return {"youtubepot-bgutilscript": {"script_path": [bgutil_script]}}
    return {}


def hd_extractor_args(bgutil_http, bgutil_script):
    """Extractor args for the HD attempt, or None when no PO token provider
    is configured (the HD path needs one; the plan then skips it)."""
    provider = pot_provider_args(bgutil_http, bgutil_script)
    if not provider:
        return None
    return {**provider, "youtube": {"player_client": list(HD_CLIENTS)}}


def fallback_extractor_args(bgutil_http, bgutil_script):
    """Extractor args for the conservative attempt: same live client list,
    the provider when there is one, and never ``player_skip`` (see module
    docstring). Without a provider mweb still serves the progressive 360p
    format, which is enough for a probe and better than no download."""
    return {**pot_provider_args(bgutil_http, bgutil_script),
            "youtube": {"player_client": list(HD_CLIENTS)}}
