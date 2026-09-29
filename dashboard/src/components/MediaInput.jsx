import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Link2, Upload, FileVideo, X, Info, Loader2, ChevronDown } from 'lucide-react';
import { getApiUrl } from '../config';

const SUPPORTED_PLATFORMS = [
    'YouTube', 'Vimeo', 'TikTok', 'X / Twitter', 'Twitch',
    'Facebook', 'Instagram', 'Dailymotion', 'Reddit', 'Streamable',
];

export default function MediaInput({ onProcess, isProcessing }) {
    const [youtubeUrlEnabled, setYoutubeUrlEnabled] = useState(true);
    // File upload is the primary path; the link is secondary.
    const [mode, setMode] = useState('file'); // 'file' | 'url'
    const [url, setUrl] = useState('');
    const [file, setFile] = useState(null);
    const [acknowledged, setAcknowledged] = useState(false);
    const [outputFormat, setOutputFormat] = useState('vertical'); // vertical | horizontal | square
    const [showInfo, setShowInfo] = useState(false);
    // Advanced generation controls — empty string means "let the AI decide",
    // which keeps the default pipeline behavior untouched.
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [targetClips, setTargetClips] = useState('');
    const [clipMinSeconds, setClipMinSeconds] = useState('');
    const [clipMaxSeconds, setClipMaxSeconds] = useState('');
    // Auto-hook: burn the AI hook text into every clip. On by default; the
    // choice persists so turning it off sticks across sessions.
    const [autoHook, setAutoHook] = useState(() => {
        try { return localStorage.getItem('os_auto_hook') !== '0'; } catch { return true; }
    });
    const [autoHookStyle, setAutoHookStyle] = useState(() => {
        try { return localStorage.getItem('os_auto_hook_style') || 'classic'; } catch { return 'classic'; }
    });
    // Layout: 'auto' lets the AI pick per video (server default); the others
    // force one on so a podcast host who knows what they uploaded doesn't
    // depend on the detector, and 'none' keeps the plain single crop.
    const [layout, setLayout] = useState(() => {
        try { return localStorage.getItem('os_layout') || 'auto'; } catch { return 'auto'; }
    });
    // Split mode: cut a chosen range into fixed-length pieces with no model at
    // all. A mode switch rather than a separate tab, because everything else
    // about the flow — source, output format, the clip grid, scheduling — is
    // identical and a tab would duplicate all of it.
    const [splitMode, setSplitMode] = useState(false);
    const [splitSeconds, setSplitSeconds] = useState('90');
    const [splitStart, setSplitStart] = useState('');
    const [splitEnd, setSplitEnd] = useState('');
    const [splitSnap, setSplitSnap] = useState(true);
    const infoRef = useRef(null);

    // Same spellings split_selector.parse_timecode accepts, because a user
    // reading times off a video player types "1:30", not "90".
    const readTimecode = (text) => {
        const value = String(text ?? '').trim();
        if (!value) return null;
        const parts = value.split(':').map(Number);
        if (parts.some((n) => Number.isNaN(n))) return null;
        return parts.reduce((total, n) => total * 60 + n, 0);
    };

    // What the chosen range will actually produce, while the form is still on
    // screen. "5 parts of a 30-minute range" is 6-minute clips that Instagram
    // refuses, and finding that out after a film has downloaded and rendered
    // is a bad way to learn it.
    const splitPlan = useMemo(() => {
        if (!splitMode) return '';
        const seconds = Number(splitSeconds) || 90;
        const from = readTimecode(splitStart);
        const to = readTimecode(splitEnd);
        if (seconds > 180) return `${seconds}s is over the 3-minute Reels and Shorts cap.`;
        if (from == null || to == null) return '';
        const span = to - from;
        if (span <= 0) return 'That range ends before it starts.';
        // Mirrors split_selector.plan_boundaries: a tail under 20s is absorbed
        // into the piece before it rather than shipped as its own clip.
        const whole = Math.floor(span / seconds);
        const tail = span - whole * seconds;
        const pieces = tail >= 20 ? whole + 1 : Math.max(1, whole);
        return `≈ ${pieces} piece${pieces === 1 ? '' : 's'} of ${seconds}s.`;
    }, [splitMode, splitSeconds, splitStart, splitEnd]);

    // Close the compatibility popover on any outside click.
    useEffect(() => {
        if (!showInfo) return;
        const onClick = (e) => {
            if (infoRef.current && !infoRef.current.contains(e.target)) setShowInfo(false);
        };
        document.addEventListener('mousedown', onClick);
        return () => document.removeEventListener('mousedown', onClick);
    }, [showInfo]);

    useEffect(() => {
        fetch(getApiUrl('/api/config'))
            .then((r) => r.ok ? r.json() : null)
            .then((cfg) => {
                if (cfg && cfg.youtubeUrlEnabled === false) {
                    setYoutubeUrlEnabled(false);
                    setMode('file');
                }
            })
            .catch(() => {});
    }, []);

    // A link pasted in the landing hero: preload it here so the user picks up
    // where they left off. Not auto-submitted — the rights attestation below
    // has to be ticked by the user.
    useEffect(() => {
        let pending = null;
        try {
            pending = localStorage.getItem('os_pending_url');
            if (pending) localStorage.removeItem('os_pending_url');
        } catch { /* ignore */ }
        if (pending) {
            setMode('url');
            setUrl(pending);
        }
    }, []);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!acknowledged) return;
        const advanced = splitMode ? {
            // Split mode deliberately drops the AI controls rather than
            // sending them to be ignored: there is no clip count to target and
            // no hook to write, and a request that carries them reads as if
            // they might still apply.
            layout,
            splitMode: true,
            splitSeconds: splitSeconds || '90',
            splitStart: splitStart || null,
            splitEnd: splitEnd || null,
            splitSnap,
        } : {
            targetClips: targetClips || null,
            clipMinSeconds: clipMinSeconds || null,
            clipMaxSeconds: clipMaxSeconds || null,
            autoHook,
            autoHookStyle,
            layout,
        };
        try {
            localStorage.setItem('os_auto_hook', autoHook ? '1' : '0');
            localStorage.setItem('os_auto_hook_style', autoHookStyle);
            localStorage.setItem('os_layout', layout);
        } catch { /* ignore */ }
        if (mode === 'url' && url) {
            onProcess({ type: 'url', payload: url, acknowledged: true, outputFormat, ...advanced });
        } else if (mode === 'file' && file) {
            onProcess({ type: 'file', payload: file, acknowledged: true, outputFormat, ...advanced });
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setMode('file');
        }
    };

    return (
        <div className="card p-4 sm:p-6 animate-fade">
            <div className="flex gap-4 sm:gap-6 mb-6 border-b border-rule" data-tutorial="source-tabs">
                <button
                    onClick={() => setMode('file')}
                    className={`flex items-center gap-2 pb-3 px-1 -mb-px border-b-2 text-sm lowercase whitespace-nowrap transition-colors ${mode === 'file'
                        ? 'text-ink border-brass'
                        : 'text-muted border-transparent hover:text-ink2'
                        }`}
                >
                    <Upload size={16} className={`hidden sm:block ${mode === 'file' ? 'text-brass' : ''}`} />
                    Upload File
                </button>
                {youtubeUrlEnabled && (
                    <button
                        onClick={() => setMode('url')}
                        className={`flex items-center gap-2 pb-3 px-1 -mb-px border-b-2 text-sm lowercase whitespace-nowrap transition-colors ${mode === 'url'
                            ? 'text-ink border-brass'
                            : 'text-muted border-transparent hover:text-ink2'
                            }`}
                    >
                        <Link2 size={16} className={`hidden sm:block ${mode === 'url' ? 'text-brass' : ''}`} />
                        Video URL
                    </button>
                )}
            </div>

            <form onSubmit={handleSubmit}>
                {mode === 'url' ? (
                    <div className="space-y-4" data-tutorial="drop-zone">
                        <div className="relative">
                            <input
                                type="url"
                                value={url}
                                onChange={(e) => setUrl(e.target.value)}
                                placeholder="https://... paste a video link"
                                className="input-field pr-11"
                                required
                            />
                            <div className="absolute inset-y-0 right-2 flex items-center" ref={infoRef}>
                                <button
                                    type="button"
                                    onClick={() => setShowInfo((v) => !v)}
                                    aria-label="Supported platforms"
                                    className="p-1.5 text-muted hover:text-brass transition-colors"
                                >
                                    <Info size={16} />
                                </button>
                                {showInfo && (
                                    <div className="absolute right-0 top-full mt-2 w-64 z-20 card p-4 text-left animate-fade">
                                        <p className="eyebrow mb-2">Paste a link from</p>
                                        <div className="flex flex-wrap gap-1.5">
                                            {SUPPORTED_PLATFORMS.map((p) => (
                                                <span key={p} className="text-xs px-2 py-0.5 rounded-full bg-paper3 text-ink2">
                                                    {p}
                                                </span>
                                            ))}
                                        </div>
                                        <p className="text-xs text-muted mt-2.5 leading-relaxed">
                                            …and 1,000+ more sites. If a link has a public video, we can usually fetch it.
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ) : (
                    <div
                        data-tutorial="drop-zone"
                        className={`border-2 border-dashed rounded-card p-6 sm:p-8 text-center transition-colors ${file ? 'border-brass' : 'border-rule2 hover:border-brass'
                            }`}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                    >
                        {file ? (
                            <div className="flex items-center justify-center gap-3 text-ok min-w-0">
                                <FileVideo size={18} className="shrink-0" />
                                <span className="font-medium truncate">{file.name}</span>
                                <button
                                    type="button"
                                    onClick={() => setFile(null)}
                                    className="p-1 text-muted hover:text-ink hover:bg-paper3 rounded-full transition-colors"
                                >
                                    <X size={16} />
                                </button>
                            </div>
                        ) : (
                            <label className="cursor-pointer block">
                                <input
                                    type="file"
                                    accept="video/*"
                                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                                    className="hidden"
                                />
                                <Upload className="mx-auto mb-3 text-muted" size={18} />
                                <p className="text-ink2 lowercase">Click to upload or drag and drop</p>
                                <p className="readout mt-2">MP4, MOV up to 500MB</p>
                            </label>
                        )}
                    </div>
                )}

                {/* How the clips are chosen. A mode switch rather than a second
                    tab: the source picker, the output format, the clip grid and
                    the scheduling modal are identical either way. */}
                <div className="mt-5">
                    <p className="eyebrow mb-2">How to cut</p>
                    <div className="grid grid-cols-2 gap-2">
                        {[
                            { value: false, label: 'find moments', hint: 'AI picks the good bits' },
                            { value: true, label: 'fixed split', hint: 'equal pieces · no AI' },
                        ].map((m) => (
                            <button
                                key={String(m.value)}
                                type="button"
                                onClick={() => setSplitMode(m.value)}
                                className={`p-3 rounded border text-left transition-colors ${splitMode === m.value
                                    ? 'border-brass bg-brass/5'
                                    : 'border-rule hover:border-ink2'
                                    }`}
                            >
                                <span className="block font-mono text-sm leading-none lowercase">{m.label}</span>
                                <span className="block mt-1 text-[11px] sm:text-[10px] leading-tight text-muted">{m.hint}</span>
                            </button>
                        ))}
                    </div>
                </div>

                {splitMode && (
                    <div className="mt-4 animate-fade">
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-2">
                            <div>
                                <p className="eyebrow mb-1.5">piece length (s)</p>
                                <input
                                    type="number" min="5" max="600" step="5"
                                    value={splitSeconds}
                                    onChange={(e) => setSplitSeconds(e.target.value)}
                                    placeholder="90"
                                    className="input-field"
                                />
                            </div>
                            <div>
                                <p className="eyebrow mb-1.5">from</p>
                                <input
                                    type="text"
                                    value={splitStart}
                                    onChange={(e) => setSplitStart(e.target.value)}
                                    placeholder="start"
                                    className="input-field"
                                />
                            </div>
                            <div>
                                <p className="eyebrow mb-1.5">to</p>
                                <input
                                    type="text"
                                    value={splitEnd}
                                    onChange={(e) => setSplitEnd(e.target.value)}
                                    placeholder="end"
                                    className="input-field"
                                />
                            </div>
                        </div>
                        <p className="mt-2 text-[11px] text-muted leading-snug">
                            Times as <span className="font-mono">1:30</span> or{' '}
                            <span className="font-mono">00:01:30</span> or seconds. Leave blank for the whole video.
                            {splitPlan && <span className="text-ink2"> {splitPlan}</span>}
                        </p>
                        <label className="mt-3 flex items-center gap-2 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={splitSnap}
                                onChange={(e) => setSplitSnap(e.target.checked)}
                                className="accent-brass"
                            />
                            <span className="text-xs text-ink2 lowercase">
                                snap cuts to quiet moments
                            </span>
                        </label>
                        <p className="mt-2 text-[11px] text-muted leading-snug">
                            No captions, no hook and no AI in this mode — it cuts where you say and keeps the whole picture.
                        </p>
                    </div>
                )}

                {/* Output format selector */}
                <div className="mt-5" data-tutorial="output-format">
                    <p className="eyebrow mb-2">Output format</p>
                    <div className="grid grid-cols-3 gap-2">
                        {[
                            { value: 'vertical', label: '9:16', hint: 'Shorts · Reels · TikTok', w: 18, h: 32 },
                            { value: 'square', label: '1:1', hint: 'Feed posts', w: 28, h: 28 },
                            { value: 'horizontal', label: '16:9', hint: 'Keep landscape · YouTube', w: 36, h: 20 },
                        ].map((f) => {
                            const active = outputFormat === f.value;
                            return (
                                <button
                                    key={f.value}
                                    type="button"
                                    onClick={() => setOutputFormat(f.value)}
                                    className={`py-3 px-2 rounded-input border flex flex-col items-center gap-2 transition-colors
                                        ${active ? 'border-[color:var(--color-accent)] text-ink' : 'border-rule2 text-muted hover:border-[color:var(--color-accent)]'}`}
                                >
                                    {/* Aspect-ratio glyph */}
                                    <span
                                        className="rounded-[3px] border-2 transition-colors"
                                        style={{
                                            width: `${f.w}px`,
                                            height: `${f.h}px`,
                                            borderColor: active ? 'var(--color-accent)' : 'var(--color-rule-2)',
                                            backgroundColor: active ? 'color-mix(in srgb, var(--color-accent) 22%, transparent)' : 'transparent',
                                        }}
                                    />
                                    <span className="block font-mono text-sm leading-none">{f.label}</span>
                                    <span className="block text-[11px] sm:text-[10px] leading-tight text-center text-muted">{f.hint}</span>
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* Advanced generation controls — collapsed by default; blank = AI
                    decides. Hidden entirely in split mode: there is no clip count
                    to target, no length band to set and no hook to write, and
                    showing dead controls reads as if they might still apply. */}
                <div className="mt-4" hidden={splitMode}>
                    <button
                        type="button"
                        onClick={() => setShowAdvanced((v) => !v)}
                        className="flex items-center gap-1.5 text-xs text-muted hover:text-ink2 lowercase transition-colors"
                    >
                        <ChevronDown size={14} className={`transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
                        advanced options
                        {(targetClips || clipMinSeconds || clipMaxSeconds || !autoHook) && (
                            <span className="text-brass">·</span>
                        )}
                    </button>
                    {showAdvanced && (
                        /* Stacked on a phone: three number fields side by side leaves
                           ~100px each, which crushes both label and value. */
                        <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-2 animate-fade">
                            <div>
                                <p className="eyebrow mb-1.5">clips to aim for</p>
                                <input
                                    type="number" min="1" max="15" step="1"
                                    value={targetClips}
                                    onChange={(e) => setTargetClips(e.target.value)}
                                    placeholder="auto"
                                    className="input-field"
                                />
                            </div>
                            <div>
                                <p className="eyebrow mb-1.5">min length (s)</p>
                                <input
                                    type="number" min="5" max="175" step="1"
                                    value={clipMinSeconds}
                                    onChange={(e) => setClipMinSeconds(e.target.value)}
                                    placeholder="15"
                                    className="input-field"
                                />
                            </div>
                            <div>
                                <p className="eyebrow mb-1.5">max length (s)</p>
                                <input
                                    type="number" min="10" max="180" step="1"
                                    value={clipMaxSeconds}
                                    onChange={(e) => setClipMaxSeconds(e.target.value)}
                                    placeholder="60"
                                    className="input-field"
                                />
                            </div>
                            <p className="col-span-1 sm:col-span-3 text-[11px] leading-relaxed text-muted">
                                Targets, not guarantees: the AI returns fewer clips when the
                                material doesn't hold them. Leave blank to let it decide.
                            </p>
                            <div className="col-span-1 sm:col-span-3 flex flex-wrap items-center justify-between gap-3 pt-3 sm:pt-1 border-t border-rule">
                                <span className="text-xs text-ink2">vertical layout</span>
                                <select
                                    value={layout}
                                    onChange={(e) => setLayout(e.target.value)}
                                    className="input-field !w-auto text-xs py-1.5"
                                    aria-label="vertical layout"
                                >
                                    <option value="auto">Auto (AI picks per video)</option>
                                    <option value="split">Two speakers stacked</option>
                                    <option value="screencast">Screen over presenter</option>
                                    <option value="none">Single crop only</option>
                                </select>
                            </div>
                            <div className="col-span-1 sm:col-span-3 flex flex-wrap items-center justify-between gap-3 pt-3 sm:pt-1 border-t border-rule">
                                <label className="flex items-center gap-2 text-xs text-ink2 cursor-pointer select-none">
                                    <input
                                        type="checkbox"
                                        checked={autoHook}
                                        onChange={(e) => setAutoHook(e.target.checked)}
                                        className="w-4 h-4 shrink-0 accent-[var(--color-accent)] cursor-pointer"
                                    />
                                    auto hook titles on clips
                                </label>
                                {autoHook && (
                                    <select
                                        value={autoHookStyle}
                                        onChange={(e) => setAutoHookStyle(e.target.value)}
                                        className="input-field !w-auto text-xs py-1.5"
                                    >
                                        <option value="classic">Classic</option>
                                        <option value="dark">Dark</option>
                                        <option value="yellow">Yellow</option>
                                        <option value="red">Red</option>
                                        <option value="outline">Outline</option>
                                        <option value="outline_yellow">Outline+</option>
                                    </select>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                <label className="flex items-start gap-2.5 mt-5 text-left text-[13px] sm:text-xs leading-relaxed text-muted cursor-pointer select-none">
                    <input
                        type="checkbox"
                        checked={acknowledged}
                        onChange={(e) => setAcknowledged(e.target.checked)}
                        className="mt-0.5 w-4 h-4 shrink-0 accent-[var(--color-accent)] cursor-pointer"
                    />
                    <span>
                        I confirm I own this content or have the rights to process it. I am responsible for any content I submit. See our <a href="/terms" target="_blank" rel="noopener noreferrer" className="text-ink2 underline underline-offset-2 hover:text-brass transition-colors" onClick={(e) => e.stopPropagation()}>Terms</a> and <a href="/privacy" target="_blank" rel="noopener noreferrer" className="text-ink2 underline underline-offset-2 hover:text-brass transition-colors" onClick={(e) => e.stopPropagation()}>Privacy Policy</a>.
                    </span>
                </label>

                <button
                    type="submit"
                    data-tutorial="generate"
                    disabled={isProcessing || !acknowledged || (mode === 'url' && !url) || (mode === 'file' && !file)}
                    className="w-full btn-primary mt-4"
                >
                    {isProcessing ? (
                        <>
                            <Loader2 size={16} className="animate-spin" />
                            Processing Video...
                        </>
                    ) : (
                        <>
                            Generate Clips
                        </>
                    )}
                </button>
            </form>
        </div>
    );
}
