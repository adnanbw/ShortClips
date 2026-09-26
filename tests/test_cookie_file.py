"""The cookies file is a live session credential, and it used to live in /app.

Two failures came out of that. It sat untracked inside the git working tree,
one `git add -A` away from a public repo. And /app belongs to whoever owns the
checkout — once the container ran as the host's uid, the cookies.txt left by
the previous user could not be overwritten, the write failed, and yt-dlp ran
ANONYMOUSLY. The job then died on "Sign in to confirm you're not a bot", which
reads as YouTube blocking the server rather than as credentials never loaded.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()

#: Comments are stripped before matching, or this file's own account of the
#: old path would satisfy the test that the old path is gone.
CODE = "\n".join(line for line in MAIN.splitlines()
                 if not line.lstrip().startswith("#"))


def test_cookies_are_not_written_into_the_source_tree():
    """A session credential must not be created inside a git checkout."""
    assert "'/app/cookies.txt'" not in CODE
    assert '"/app/cookies.txt"' not in CODE


def test_cookies_go_to_a_private_temp_file():
    """mkstemp creates 0600 and /tmp is writable whatever uid the container
    runs as — the two properties the old path lacked."""
    block = MAIN[MAIN.index("cookies_env = os.environ.get"):]
    block = block[:block.index("_proxy = os.environ.get")]
    assert "tempfile.mkstemp" in block
    assert "ytcookies_" in block


def test_a_failed_write_says_what_it_will_cost():
    """Falling back to an anonymous download is the real consequence, and it
    surfaces later as a YouTube bot-check with nothing pointing back here."""
    block = MAIN[MAIN.index("Failed to write cookies file"):]
    block = block[:block.index("_proxy = os.environ.get")]
    assert "WITHOUT cookies" in block
    assert "bot" in block.lower()


def test_the_cookie_contents_are_never_logged():
    """A headerless cookies blob printed to the job log would hand a live
    YouTube session to anyone reading it."""
    block = MAIN[MAIN.index("cookies_env = os.environ.get"):]
    block = block[:block.index("_proxy = os.environ.get")]
    prints = re.findall(r"print\((.*?)\)\n", block, re.S)
    assert prints, "expected the cookie block to log something"
    for call in prints:
        assert "cookies_env" not in call


def test_cookie_files_are_gitignored():
    """Defence in depth: main.py no longer puts one here, but an older
    checkout may still have the file sitting in it."""
    ignore = open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read()
    assert "cookies.txt" in ignore
