"""Crediting the creator whose video a clip was cut from.

The property that matters most is NEGATIVE: never name an account that is not
theirs. An @mention is a public claim about a real person, so a wrong one is
worse than no credit at all — and the failure is silent, because a plausible
handle looks exactly like a correct one.

Everything here is built from that: handles come only from the description the
creator wrote, the model picks an INDEX into a closed list rather than writing
a name, and every unresolved case degrades to crediting them by name.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import attribution as attr


# The two real descriptions this was measured on. One creator listed a single
# Instagram account; the other listed two — his own and the comedy club's —
# which is the case a regex alone cannot resolve.
SHARON = """Pressure | Stand-up comedy by Sharon Verma
Follow me on instagram.com/sharonverma for more
Shot at the studio."""

TARUN = """Ganje se Panga

I went bald at 25. Since then I've discovered...

Follow me: instagram.com/tarun_ratnani
Venue: instagram.com/undergroundcomedy.blr"""


# ---------------------------------------------------------------------------
# Finding accounts in a description
# ---------------------------------------------------------------------------

def test_finds_a_single_instagram_account():
    found = attr.find_socials(SHARON)
    assert [s["handle"] for s in found] == ["sharonverma"]
    assert found[0]["platform"] == "instagram"


def test_finds_both_accounts_and_keeps_the_words_around_them():
    """The context is the whole point: "Follow me" versus "Venue" is the only
    thing distinguishing the creator from the room he performed in."""
    found = attr.find_socials(TARUN)
    assert [s["handle"] for s in found] == ["tarun_ratnani", "undergroundcomedy.blr"]
    assert "Follow me" in found[0]["context"]
    assert "Venue" in found[1]["context"]


@pytest.mark.parametrize("url", [
    "instagram.com/p/Cabc123/",
    "instagram.com/reel/Cabc123/",
    "instagram.com/reels/Cabc123/",
    "instagram.com/tv/Cabc123/",
    "instagram.com/stories/someone/123",
    "instagram.com/explore/tags/comedy/",
])
def test_post_links_are_not_accounts(url):
    """Crediting instagram.com/p/... would link one video instead of a person,
    and /explore/ is not a person at all."""
    assert attr.find_socials(f"watch this {url} amazing") == []


def test_the_same_account_twice_is_one_candidate():
    """Creators paste the same link in the description and the pinned comment.
    Two copies must not look like a choice to be made."""
    text = "instagram.com/sharonverma ... later ... instagram.com/sharonverma"
    assert len(attr.find_socials(text)) == 1


def test_tiktok_and_x_are_recognised():
    found = attr.find_socials("tiktok.com/@someone and x.com/handle")
    assert {(s["platform"], s["handle"]) for s in found} == {
        ("tiktok", "someone"), ("twitter", "handle")}


def test_an_empty_description_finds_nothing():
    assert attr.find_socials("") == []
    assert attr.find_socials(None) == []


# ---------------------------------------------------------------------------
# What is kept from yt-dlp
# ---------------------------------------------------------------------------

def test_from_info_keeps_the_uploader_and_drops_the_description():
    """A description can be thousands of words and nothing downstream reads it
    — only the links in it, which are extracted here while it is in hand."""
    data = attr.from_info({
        "uploader": "Tarun Ratnani", "uploader_id": "@tarunratnani8819",
        "channel_url": "https://youtube.com/channel/UC123",
        "webpage_url": "https://youtu.be/x", "license": None,
        "description": TARUN, "formats": [{"url": "..."}] * 50,
    })
    assert data["uploader"] == "Tarun Ratnani"
    assert data["uploader_id"] == "@tarunratnani8819"
    assert "description" not in data
    assert "formats" not in data
    assert len(data["socials"]) == 2


def test_youtube_handle_strips_the_at():
    assert attr.youtube_handle({"uploader_id": "@sharonvermacomedy"}) == \
        "sharonvermacomedy"


def test_a_channel_id_is_not_a_handle():
    """Older extractions put a UC… id in uploader_id. Printing that as "@UC…"
    would be a credit to an account nobody can find."""
    assert attr.youtube_handle({"uploader_id": "UConP52HdpwYUgpiSnjyBnFw"}) is None


def test_no_uploader_id_is_no_handle():
    assert attr.youtube_handle({}) is None


# ---------------------------------------------------------------------------
# Choosing between accounts
# ---------------------------------------------------------------------------

def test_one_candidate_needs_no_model(monkeypatch):
    """Sharon listed one account. Spending a Gemini call to pick it out of a
    list of one would be paying to answer a question with no alternatives."""
    monkeypatch.setattr(attr, "PICK_PROMPT", None)  # would raise if formatted
    data = attr.from_info({"uploader": "Sharon Verma", "description": SHARON})
    assert attr.pick_handle(data, "instagram") == "sharonverma"


def test_no_candidates_is_no_handle():
    data = attr.from_info({"uploader": "Someone", "description": "no links"})
    assert attr.pick_handle(data, "instagram") is None


def test_two_candidates_without_a_key_credits_nobody(monkeypatch):
    """Picking the first would be a coin flip with a stranger's name on it."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    data = attr.from_info({"uploader": "Tarun Ratnani", "description": TARUN})
    assert attr.pick_handle(data, "instagram") is None


class FakeGemini:
    """Stands in for google.genai, returning a fixed choice index."""

    def __init__(self, choice):
        self.choice = choice
        self.prompt = None

    def install(self, monkeypatch):
        outer = self

        class Models:
            def generate_content(self, model, contents, config):
                outer.prompt = contents
                return type("R", (), {"text": json.dumps({"choice": outer.choice})})()

        class Client:
            def __init__(self, api_key=None):
                self.models = Models()

        import google.genai as genai
        monkeypatch.setattr(genai, "Client", Client)


def _fake_genai(monkeypatch, choice):
    pytest.importorskip("google.genai")
    fake = FakeGemini(choice)
    fake.install(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return fake


def test_the_model_picks_the_creator_not_the_venue(monkeypatch):
    fake = _fake_genai(monkeypatch, 1)
    data = attr.from_info({"uploader": "Tarun Ratnani", "description": TARUN})
    assert attr.pick_handle(data, "instagram") == "tarun_ratnani"
    # It was shown both, with the words that tell them apart.
    assert "tarun_ratnani" in fake.prompt and "undergroundcomedy.blr" in fake.prompt
    assert "Venue" in fake.prompt


def test_zero_means_none_of_them(monkeypatch):
    """A real answer, not a failure: a description can list a sponsor and an
    editor and no creator account at all."""
    _fake_genai(monkeypatch, 0)
    data = attr.from_info({"uploader": "Tarun Ratnani", "description": TARUN})
    assert attr.pick_handle(data, "instagram") is None


@pytest.mark.parametrize("choice", [3, -1, 99])
def test_an_index_outside_the_list_is_refused(monkeypatch, choice):
    """The model answers with a position, so it cannot name an account that was
    not in the description — but it can still answer out of range, and that is
    not a handle either."""
    _fake_genai(monkeypatch, choice)
    data = attr.from_info({"uploader": "Tarun Ratnani", "description": TARUN})
    assert attr.pick_handle(data, "instagram") is None


def test_a_model_error_credits_by_name_instead(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    import google.genai as genai

    class Boom:
        def __init__(self, api_key=None):
            raise RuntimeError("503")

    monkeypatch.setattr(genai, "Client", Boom)
    data = attr.from_info({"uploader": "Tarun Ratnani", "description": TARUN})
    assert attr.pick_handle(data, "instagram") is None


# ---------------------------------------------------------------------------
# The credit line
# ---------------------------------------------------------------------------

def test_a_handle_becomes_a_mention():
    assert attr.credit_line({"uploader": "Sharon Verma"}, "sharonverma") == \
        "🎥 Original: @sharonverma"


def test_without_a_handle_the_creator_is_named_in_words():
    """Still attribution, still true, and it tags nobody."""
    assert attr.credit_line({"uploader": "Sharon Verma"}) == \
        "🎥 Original: Sharon Verma"


def test_nothing_to_credit_is_an_empty_line():
    assert attr.credit_line({}) == ""
    assert attr.credit_line(None) == ""


def test_the_template_is_configurable(monkeypatch):
    monkeypatch.setenv("CREDIT_TEMPLATE", "Credit: {credit} · {url}")
    line = attr.credit_line({"uploader": "S", "webpage_url": "https://y/x"}, "s")
    assert line == "Credit: @s · https://y/x"


# ---------------------------------------------------------------------------
# Putting it in the caption
# ---------------------------------------------------------------------------

def test_credit_goes_above_trailing_hashtags():
    """Hashtags end an Instagram caption, so appending after them buries the
    credit in the tag block where nobody reads it."""
    out = attr.apply_to_caption(
        "A funny bit about ordering food.\n\n#comedy #standup", "🎥 Original: @x")
    assert out == ("A funny bit about ordering food.\n\n"
                   "🎥 Original: @x\n\n"
                   "#comedy #standup")


def test_a_caption_with_no_hashtags_just_gains_a_line():
    assert attr.apply_to_caption("Just a caption.", "🎥 Original: @x") == \
        "Just a caption.\n\n🎥 Original: @x"


def test_an_empty_credit_changes_nothing():
    assert attr.apply_to_caption("Untouched #tag", "") == "Untouched #tag"


def test_crediting_twice_does_not_duplicate():
    """Re-queueing a clip after editing its caption must not stack credits."""
    once = attr.apply_to_caption("Body\n\n#tag", "🎥 Original: @x")
    assert attr.apply_to_caption(once, "🎥 Original: @x") == once


def test_hashtags_are_dropped_before_the_credit_is():
    """Instagram's limit is 2200. What survives a squeeze is the attribution,
    not the tags — a credit silently truncated away is the failure this whole
    module exists to prevent."""
    body = "x" * 2100
    tags = " ".join(f"#tag{i}" for i in range(20))
    out = attr.apply_to_caption(f"{body}\n\n{tags}", "🎥 Original: @someone")
    assert len(out) <= 2200
    assert "🎥 Original: @someone" in out
    assert "#tag0" not in out


def test_an_oversized_body_is_trimmed_and_the_credit_kept():
    out = attr.apply_to_caption("y" * 5000, "🎥 Original: @someone")
    assert len(out) <= 2200
    assert out.endswith("🎥 Original: @someone")


# ---------------------------------------------------------------------------
# The sidecar
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip(tmp_path):
    data = attr.from_info({"uploader": "Sharon Verma", "description": SHARON})
    attr.save(str(tmp_path), data)
    assert attr.load(str(tmp_path)) == data


def test_load_without_a_sidecar_is_none(tmp_path):
    """The normal case for an upload: a local file has no uploader."""
    assert attr.load(str(tmp_path)) is None


def test_save_to_an_unwritable_place_never_raises(tmp_path):
    """Attribution must not be able to fail a download."""
    assert attr.save(str(tmp_path / "does" / "not" / "exist"), {"a": 1}) is None
