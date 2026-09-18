"""A busy machine must not look like "YouTube blocked the request".

The bgutil yt-dlp plugin hardcodes 15 s for the `node ... --version`
availability probe it runs before every extraction, with no env var or
extractor-arg to change it. Measured on the dev box while a docker build was
running, that exact command took **56 seconds** and returned the correct
answer, 2.0.0 — so the probe timed out, the exception escaped through
is_available(), every download strategy died, and the job reported
"YouTube blocked the request or the download tooling is out of date".

That message is false and expensive: it points at proxies and cookies when the
real problem is CPU. It is not only a build-time hazard either —
MAX_CONCURRENT_JOBS defaults to 5 and a render is CPU-hungry.
"""
import sys
import types

import pytest

# Same guard the rest of the suite uses: main needs cv2/mediapipe, which the
# minimal test environment does not have.
main = pytest.importorskip("main")


@pytest.fixture
def fake_plugin(monkeypatch):
    """Stand in for yt_dlp_plugins.extractor.getpot_bgutil_script."""
    module = types.ModuleType("yt_dlp_plugins.extractor.getpot_bgutil_script")

    class Base:
        _GET_SCRIPT_VSN_TIMEOUT = 15.0
        _GETPOT_TIMEOUT = 20.0

    class Node(Base):
        _GET_SCRIPT_VSN_TIMEOUT = 15.0

    module.BgUtilScriptPTPBase = Base
    module.BgUtilScriptNodePTP = Node

    package = types.ModuleType("yt_dlp_plugins.extractor")
    package.getpot_bgutil_script = module
    root = types.ModuleType("yt_dlp_plugins")
    root.extractor = package

    monkeypatch.setitem(sys.modules, "yt_dlp_plugins", root)
    monkeypatch.setitem(sys.modules, "yt_dlp_plugins.extractor", package)
    monkeypatch.setitem(
        sys.modules, "yt_dlp_plugins.extractor.getpot_bgutil_script", module)
    return module


class TestTheProbeGetsRoomToBreathe:
    def test_the_version_probe_budget_is_raised(self, fake_plugin, monkeypatch):
        monkeypatch.delenv("BGUTIL_PROBE_TIMEOUT", raising=False)
        main._relax_bgutil_timeouts()
        assert fake_plugin.BgUtilScriptPTPBase._GET_SCRIPT_VSN_TIMEOUT >= 56.0

    def test_every_provider_class_is_patched(self, fake_plugin, monkeypatch):
        monkeypatch.delenv("BGUTIL_PROBE_TIMEOUT", raising=False)
        main._relax_bgutil_timeouts()
        assert fake_plugin.BgUtilScriptNodePTP._GET_SCRIPT_VSN_TIMEOUT >= 56.0

    def test_the_token_generation_budget_is_raised_too(self, fake_plugin,
                                                       monkeypatch):
        monkeypatch.delenv("BGUTIL_PROBE_TIMEOUT", raising=False)
        main._relax_bgutil_timeouts()
        assert fake_plugin.BgUtilScriptPTPBase._GETPOT_TIMEOUT >= 56.0

    def test_it_is_configurable(self, fake_plugin, monkeypatch):
        monkeypatch.setenv("BGUTIL_PROBE_TIMEOUT", "120")
        main._relax_bgutil_timeouts()
        assert fake_plugin.BgUtilScriptPTPBase._GET_SCRIPT_VSN_TIMEOUT == 120.0

    def test_a_junk_value_falls_back_to_the_default(self, fake_plugin,
                                                    monkeypatch):
        monkeypatch.setenv("BGUTIL_PROBE_TIMEOUT", "soon")
        main._relax_bgutil_timeouts()
        assert fake_plugin.BgUtilScriptPTPBase._GET_SCRIPT_VSN_TIMEOUT == 90.0

    def test_a_longer_existing_budget_is_left_alone(self, fake_plugin,
                                                    monkeypatch):
        fake_plugin.BgUtilScriptPTPBase._GET_SCRIPT_VSN_TIMEOUT = 300.0
        monkeypatch.delenv("BGUTIL_PROBE_TIMEOUT", raising=False)
        main._relax_bgutil_timeouts()
        assert fake_plugin.BgUtilScriptPTPBase._GET_SCRIPT_VSN_TIMEOUT == 300.0


class TestItCanNeverBreakADownload:
    def test_a_missing_plugin_is_not_an_error(self, monkeypatch):
        """Self-host installs without the plugin must still download."""
        monkeypatch.setitem(sys.modules, "yt_dlp_plugins", None)
        main._relax_bgutil_timeouts()

    def test_an_unpatchable_attribute_is_skipped(self, fake_plugin,
                                                 monkeypatch):
        class Frozen:
            __slots__ = ()
            _GET_SCRIPT_VSN_TIMEOUT = 15.0

        fake_plugin.Frozen = Frozen
        monkeypatch.delenv("BGUTIL_PROBE_TIMEOUT", raising=False)
        main._relax_bgutil_timeouts()  # must not raise
