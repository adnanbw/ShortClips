import React, { useState, useMemo } from 'react';
import { Loader2, Calendar, CheckCircle, AlertCircle, Video, Instagram, Youtube, ChevronLeft, ChevronRight, Circle, ExternalLink } from 'lucide-react';
import { apiFetch } from '../lib/api';
import Modal from './ui/Modal';
import SegmentedControl from './ui/SegmentedControl';
import TikTokDraftNotice from './TikTokDraftNotice';

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const TIMEZONES = [
    { value: 'Pacific/Midway', label: '(GMT-11:00) Midway' },
    { value: 'Pacific/Honolulu', label: '(GMT-10:00) Honolulu' },
    { value: 'America/Anchorage', label: '(GMT-09:00) Alaska' },
    { value: 'America/Los_Angeles', label: '(GMT-08:00) Los Angeles' },
    { value: 'America/Denver', label: '(GMT-07:00) Denver' },
    { value: 'America/Mexico_City', label: '(GMT-06:00) Mexico City' },
    { value: 'America/Chicago', label: '(GMT-06:00) Chicago' },
    { value: 'America/New_York', label: '(GMT-05:00) New York' },
    { value: 'America/Bogota', label: '(GMT-05:00) Bogota' },
    { value: 'America/Caracas', label: '(GMT-04:00) Caracas' },
    { value: 'America/Santiago', label: '(GMT-04:00) Santiago' },
    { value: 'America/Argentina/Buenos_Aires', label: '(GMT-03:00) Buenos Aires' },
    { value: 'America/Sao_Paulo', label: '(GMT-03:00) Sao Paulo' },
    { value: 'Atlantic/Azores', label: '(GMT-01:00) Azores' },
    { value: 'UTC', label: '(GMT+00:00) UTC' },
    { value: 'Europe/London', label: '(GMT+00:00) London' },
    { value: 'Europe/Madrid', label: '(GMT+01:00) Madrid' },
    { value: 'Europe/Paris', label: '(GMT+01:00) Paris' },
    { value: 'Europe/Berlin', label: '(GMT+01:00) Berlin' },
    { value: 'Europe/Rome', label: '(GMT+01:00) Rome' },
    { value: 'Africa/Lagos', label: '(GMT+01:00) Lagos' },
    { value: 'Europe/Istanbul', label: '(GMT+03:00) Istanbul' },
    { value: 'Asia/Dubai', label: '(GMT+04:00) Dubai' },
    { value: 'Asia/Kolkata', label: '(GMT+05:30) India' },
    { value: 'Asia/Bangkok', label: '(GMT+07:00) Bangkok' },
    { value: 'Asia/Shanghai', label: '(GMT+08:00) Shanghai' },
    { value: 'Asia/Tokyo', label: '(GMT+09:00) Tokyo' },
    { value: 'Australia/Sydney', label: '(GMT+10:00) Sydney' },
    { value: 'Pacific/Auckland', label: '(GMT+12:00) Auckland' },
];

const PLATFORM_OPTIONS = [
    { value: 'tiktok', label: 'TikTok', icon: <Video size={16} /> },
    { value: 'instagram', label: 'Instagram', icon: <Instagram size={16} /> },
    { value: 'youtube', label: 'YouTube', icon: <Youtube size={16} /> },
];

function getDayLabel(date) {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    const target = new Date(date);
    target.setHours(0, 0, 0, 0);

    if (target.getTime() === today.getTime()) return 'Today';
    if (target.getTime() === tomorrow.getTime()) return 'Tomorrow';
    return DAYS[target.getDay()];
}

function formatDate(date) {
    return `${MONTHS[date.getMonth()]} ${date.getDate()}`;
}

// Browsers report the IANA zone the OS gives them, and that is often a LEGACY
// ALIAS rather than the modern name: Windows in India resolves to
// Asia/Calcutta, not Asia/Kolkata. Both are valid and mean the same zone, but
// only the modern one is in TIMEZONES above.
const LEGACY_TZ_ALIASES = {
    'Asia/Calcutta': 'Asia/Kolkata',
    'Asia/Saigon': 'Asia/Ho_Chi_Minh',
    'Asia/Katmandu': 'Asia/Kathmandu',
    'Asia/Rangoon': 'Asia/Yangon',
    'Europe/Kiev': 'Europe/Kyiv',
    'America/Buenos_Aires': 'America/Argentina/Buenos_Aires',
    'US/Pacific': 'America/Los_Angeles',
    'US/Eastern': 'America/New_York',
    'US/Central': 'America/Chicago',
    'US/Mountain': 'America/Denver',
};

/**
 * The viewer's timezone, as an IANA name.
 *
 * This used to return 'UTC' for any zone missing from TIMEZONES, which made a
 * hardcoded DROPDOWN LIST silently decide what time the user's posts go out.
 * On a machine reporting Asia/Calcutta that shifted every scheduled post by
 * 5.5 hours — invisible on the Upload-Post path, which just echoes the zone
 * back, and plainly wrong on the Instagram path, where the poster resolves a
 * posting slot in it. The list is for the picker; it is not a validator.
 */
function detectTimezone() {
    try {
        const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
        if (!tz) return 'UTC';
        return LEGACY_TZ_ALIASES[tz] || tz;
    } catch {
        return 'UTC';
    }
}

export default function ScheduleWeekModal({ isOpen, onClose, clips, jobId, uploadPostKey, uploadUserId, isManaged }) {
    const [time, setTime] = useState('12:00');
    const [timezone, setTimezone] = useState(detectTimezone);
    const [platforms, setPlatforms] = useState({
        tiktok: true,
        instagram: true,
        youtube: true
    });
    const [startOffset, setStartOffset] = useState(1);

    // Where the clips go. 'uploadpost' is the original path and stays the
    // default; 'instagram' is the self-hosted poster (own Meta app, B2 +
    // Supabase), offered only when the server says it is configured.
    const [destination, setDestination] = useState('uploadpost');
    const [igAvailable, setIgAvailable] = useState(false);
    // 'slots' books the next free posting slots on the poster; 'explicit' uses
    // the time + start-day grid below, which is what Upload-Post always uses.
    const [timing, setTiming] = useState('slots');

    // A zone the picker does not list is still a real zone, and dropping it
    // would put the old 'UTC' fallback back by another route — a <select>
    // whose value matches no option renders blank and reads as "nothing
    // chosen". Offer it instead.
    const timezoneOptions = useMemo(() => (
        TIMEZONES.some(t => t.value === timezone)
            ? TIMEZONES
            : [{ value: timezone, label: timezone }, ...TIMEZONES]
    ), [timezone]);

    // Which clips actually get published. Everything is on by default, which
    // is the old behaviour of this modal — it published the whole job with no
    // say in it.
    const [selected, setSelected] = useState(() => new Set());

    // clips?.length, not clips: the array is rebuilt on every parent render,
    // so depending on its identity would reset the tick boxes under the user
    // mid-edit.
    React.useEffect(() => {
        if (!isOpen) return;
        setSelected(new Set(Array.from({ length: clips?.length || 0 }, (_, i) => i)));
    }, [isOpen, clips?.length]);

    const toggleClip = (index) => setSelected((prev) => {
        const next = new Set(prev);
        next.has(index) ? next.delete(index) : next.add(index);
        return next;
    });

    // EVERY clip is rendered, ticked or not — a list that hid the unticked ones
    // would give the user no way to put one back. Only the ticked ones take a
    // date or a slot, and `position` is their rank among those: deselecting
    // the third clip must close the gap, not leave a hole in the schedule.
    //
    // `index` (which clip) and `position` (where it lands) were the same number
    // until now, and the two uses have to stay apart: `index` is what the API
    // is told to publish, `position` is what indexes the progress results.
    const rows = useMemo(() => {
        let position = 0;
        return (clips || []).map((clip, index) => {
            if (!selected.has(index)) {
                return { clip, index, selected: false, position: null, date: null };
            }
            const date = new Date();
            date.setDate(date.getDate() + startOffset + position);
            date.setHours(0, 0, 0, 0);
            return { clip, index, selected: true, position: position++, date };
        });
    }, [clips, startOffset, selected]);

    const schedule = useMemo(() => rows.filter((r) => r.selected), [rows]);

    const [scheduling, setScheduling] = useState(false);
    const [progress, setProgress] = useState({ current: 0, total: 0, results: [] });
    const [done, setDone] = useState(false);

    // Reset state when modal reopens
    const prevOpen = React.useRef(false);
    React.useEffect(() => {
        if (isOpen && !prevOpen.current) {
            setScheduling(false);
            setDone(false);
            setProgress({ current: 0, total: 0, results: [] });
        }
        prevOpen.current = isOpen;
    }, [isOpen]);

    // Ask once per open whether the self-hosted poster is wired up. Showing a
    // destination the server cannot honour would just move the failure to the
    // schedule button.
    React.useEffect(() => {
        if (!isOpen) return;
        let cancelled = false;
        (async () => {
            try {
                const res = await apiFetch('/api/instagram/config');
                if (!res.ok) return;
                const cfg = await res.json();
                if (!cancelled) setIgAvailable(Boolean(cfg.configured));
            } catch { /* not configured is the normal case */ }
        })();
        return () => { cancelled = true; };
    }, [isOpen]);

    if (!isOpen) return null;

    const toInstagram = destination === 'instagram' && igAvailable;
    const selectedPlatforms = Object.keys(platforms).filter(k => platforms[k]);

    // Managed (cloud plan/trial) users post with the server-side key — no BYOK
    // needed. The self-hosted poster uses the server's own B2 + Supabase
    // credentials, so it needs no key from the browser at all.
    const canPost = toInstagram || isManaged || (uploadPostKey && uploadUserId);

    const pad = (n) => String(n).padStart(2, '0');
    const wallClock = (date) =>
        `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${time}:00`;

    // The self-hosted poster takes the whole batch in ONE request, unlike
    // Upload-Post which is one call per clip. That is what lets it hand out
    // consecutive posting slots: asking for "the next free slot" three times
    // in parallel would return the same instant three times.
    const handleScheduleInstagram = async () => {
        const total = schedule.length;
        setScheduling(true);
        setDone(false);
        setProgress({ current: 0, total, results: [] });

        const body = {
            job_id: jobId,
            timezone,
            post_now: timing === 'now',
            clips: schedule.map(({ clip, index, date }) => ({
                clip_index: index,
                caption: clip.video_description_for_instagram
                    || clip.video_description_for_tiktok || '',
                // Omitted in slot mode, so the poster picks the time.
                ...(timing === 'explicit' ? { scheduled_local: wallClock(date) } : {}),
            })),
        };

        try {
            const res = await apiFetch('/api/instagram/schedule', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!res.ok) throw new Error(await res.text());
            const started = await res.json();

            // The upload runs in the background because a clip is tens of
            // megabytes over a home upstream — minutes, not seconds. Polling is
            // what stops the modal looking hung, which is what previously made
            // people press the button again and start a second upload of the
            // same clips alongside the first.
            let answer = null;
            for (;;) {
                await new Promise(r => setTimeout(r, 2000));
                const poll = await apiFetch(`/api/instagram/schedule/${started.task_id}`);
                if (!poll.ok) throw new Error(await poll.text());
                const state = await poll.json();
                setProgress(p => ({ ...p, current: state.done || 0, total: state.total || total }));
                if (state.status === 'failed') throw new Error(state.error || 'upload failed');
                if (state.status === 'done') { answer = state.result; break; }
            }

            // Map the server's per-clip verdicts back onto the rows, which are
            // keyed by position in `schedule`, not by clip_index.
            const byIndex = new Map(
                (answer.results || []).map(r => [r.clip_index, r]));
            const results = schedule.map(({ index }, i) => {
                const r = byIndex.get(index);
                return {
                    index: i,
                    success: Boolean(r && r.ok),
                    error: r && !r.ok ? r.error : undefined,
                    scheduled_at: r?.scheduled_at,
                };
            });
            setProgress({ current: total, total, results });
        } catch (e) {
            setProgress({
                current: total, total,
                results: schedule.map((_, i) => ({ index: i, success: false, error: e.message })),
            });
        }

        setDone(true);
        setScheduling(false);
    };

    const handleScheduleAll = async () => {
        if (!canPost) return;
        if (toInstagram) return handleScheduleInstagram();
        if (selectedPlatforms.length === 0) return;

        setScheduling(true);
        setDone(false);
        const total = schedule.length;
        setProgress({ current: 0, total, results: [] });

        const results = [];
        for (let i = 0; i < schedule.length; i++) {
            const { clip, index, date } = schedule[i];

            // Build local datetime string: "2026-04-06T12:00:00"
            // Upload-Post accepts this + timezone IANA parameter
            const scheduledDate = wallClock(date);

            const payload = {
                job_id: jobId,
                clip_index: index,
                api_key: uploadPostKey,
                user_id: uploadUserId,
                platforms: selectedPlatforms,
                title: clip.video_title_for_youtube_short || 'Viral Short',
                description: clip.video_description_for_instagram || clip.video_description_for_tiktok || '',
                scheduled_date: scheduledDate,
                timezone
            };

            try {
                const res = await apiFetch('/api/social/post', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!res.ok) {
                    const errText = await res.text();
                    throw new Error(errText);
                }

                results.push({ index: i, success: true });
            } catch (e) {
                results.push({ index: i, success: false, error: e.message });
            }

            setProgress({ current: i + 1, total, results: [...results] });
        }

        setDone(true);
        setScheduling(false);
    };

    const successCount = progress.results.filter(r => r.success).length;
    const failCount = progress.results.filter(r => !r.success).length;

    const footer = (
        <div className="flex flex-col sm:flex-row gap-3">
            <button
                onClick={onClose}
                disabled={scheduling}
                className="btn-ghost flex-1"
            >
                {done ? 'close' : 'cancel'}
            </button>
            {!done ? (
                <button
                    onClick={handleScheduleAll}
                    disabled={scheduling || !canPost || schedule.length === 0
                        || (!toInstagram && selectedPlatforms.length === 0)}
                    className="btn-primary flex-1"
                >
                    {scheduling ? (
                        <>
                            <Loader2 size={16} className="animate-spin" />
                            scheduling...
                        </>
                    ) : (
                        <>
                            <Calendar size={16} />
                            {timing === 'now' && toInstagram ? 'post' : 'schedule'} {schedule.length} clip{schedule.length === 1 ? '' : 's'}
                        </>
                    )}
                </button>
            ) : (
                <a
                    href={toInstagram
                        ? (import.meta.env.VITE_IG_POSTER_URL || 'https://postschedule.netlify.app')
                        : 'https://app.upload-post.com/calendar'}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn-primary flex-1 no-underline"
                >
                    <ExternalLink size={16} />
                    {toInstagram ? 'view queue' : 'view calendar'}
                </a>
            )}
        </div>
    );

    return (
        <Modal
            isOpen={isOpen}
            onClose={scheduling ? undefined : onClose}
            eyebrow="PUBLISH · WEEK"
            title="schedule week"
            size="md"
            footer={footer}
        >
            <div className="mb-4 flex items-center justify-between gap-2">
                <p className="readout">
                    {schedule.length} OF {clips?.length || 0} CLIPS · {!toInstagram ? '1/DAY' : timing === 'now' ? 'PUBLISHING NOW' : timing === 'slots' ? 'NEXT FREE SLOTS' : '1/DAY'}
                </p>
                {!scheduling && !done && (clips?.length || 0) > 1 && (
                    <button
                        type="button"
                        onClick={() => setSelected(schedule.length === clips.length
                            ? new Set()
                            : new Set(clips.map((_, i) => i)))}
                        className="btn-quiet px-2 py-1 text-xs lowercase"
                    >
                        {schedule.length === clips.length ? 'select none' : 'select all'}
                    </button>
                )}
            </div>

            {/* Destination — only shown once there is a choice to make. */}
            {igAvailable && (
                <div className="mb-5">
                    <label className="eyebrow block mb-2">destination</label>
                    <SegmentedControl
                        options={[
                            { value: 'uploadpost', label: 'Upload-Post', disabled: scheduling },
                            { value: 'instagram', label: 'Instagram', icon: <Instagram size={16} />, disabled: scheduling },
                        ]}
                        value={destination}
                        onChange={setDestination}
                    />
                </div>
            )}

            {!canPost && (
                <div className="mb-4 p-3 bg-warn/10 text-warn text-xs rounded-input flex items-start gap-2">
                    <AlertCircle size={14} className="mt-0.5 shrink-0" />
                    <div>Set your Upload-Post API key in Settings first.</div>
                </div>
            )}

            {/* Timing — the poster can place clips on its own posting slots, so
                the time/day grid below is only one of two ways to schedule. */}
            {toInstagram && (
                <div className="mb-5">
                    <label className="eyebrow block mb-2">timing</label>
                    <SegmentedControl
                        options={[
                            { value: 'now', label: 'post now', disabled: scheduling },
                            { value: 'slots', label: 'next free slots', disabled: scheduling },
                            { value: 'explicit', label: 'pick a time', disabled: scheduling },
                        ]}
                        value={timing}
                        onChange={setTiming}
                    />
                    {timing === 'slots' && (
                        <p className="text-xs text-muted mt-2 lowercase">
                            times come from the posting slots set in your instagram poster.
                        </p>
                    )}
                    {timing === 'now' && (
                        schedule.length > 1 ? (
                            /* Instagram allows 50 posts a day, so this is not a
                               limit problem — it is a feed problem. Every clip
                               goes out on the poster's next cron pass, so N
                               clips means N reels within a few minutes. */
                            <div className="mt-2 p-3 bg-warn/10 text-warn text-xs rounded-input flex items-start gap-2">
                                <AlertCircle size={14} className="mt-0.5 shrink-0" />
                                <div>
                                    all {schedule.length} clips publish within a few minutes of
                                    each other. use slots to space them out.
                                </div>
                            </div>
                        ) : (
                            <p className="text-xs text-muted mt-2 lowercase">
                                publishes on the poster's next check, within a minute.
                            </p>
                        )
                    )}
                </div>
            )}

            {/* Time + Timezone */}
            <div className={`mb-5 grid grid-cols-2 gap-3${toInstagram && timing !== 'explicit' ? ' hidden' : ''}`}>
                <div>
                    <label className="eyebrow block mb-2">time</label>
                    <input
                        type="time"
                        value={time}
                        onChange={(e) => setTime(e.target.value)}
                        disabled={scheduling}
                        className="input-field [color-scheme:dark]"
                    />
                </div>
                <div>
                    <label className="eyebrow block mb-2">timezone</label>
                    <select
                        value={timezone}
                        onChange={(e) => setTimezone(e.target.value)}
                        disabled={scheduling}
                        className="input-field appearance-none cursor-pointer"
                    >
                        {timezoneOptions.map(tz => (
                            <option key={tz.value} value={tz.value}>{tz.label}</option>
                        ))}
                    </select>
                </div>
            </div>

            {/* Start day offset */}
            <div className={`mb-5 flex flex-wrap items-center justify-between gap-2${toInstagram && timing !== 'explicit' ? ' hidden' : ''}`}>
                <span className="eyebrow">start from</span>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setStartOffset(Math.max(1, startOffset - 1))}
                        disabled={startOffset <= 1 || scheduling}
                        className="btn-quiet px-2 py-2 disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                        <ChevronLeft size={16} />
                    </button>
                    <span className="readout text-ink2 min-w-[110px] text-center">
                        {(() => {
                            const d = new Date();
                            d.setDate(d.getDate() + startOffset);
                            return `${getDayLabel(d)} · ${formatDate(d)}`;
                        })()}
                    </span>
                    <button
                        onClick={() => setStartOffset(startOffset + 1)}
                        disabled={scheduling}
                        className="btn-quiet px-2 py-2 disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                        <ChevronRight size={16} />
                    </button>
                </div>
            </div>

            {/* Calendar list */}
            <div className="mb-5 border-y border-rule divide-y divide-rule">
                {rows.map(({ clip, index, date, selected: isOn, position }) => {
                    // Results are positional over the SELECTED clips, so an
                    // unticked row has no result and a ticked one looks its own
                    // up by `position`, never by `index`.
                    const outcome = position === null ? undefined : progress.results[position];
                    const idle = !scheduling && !done;
                    return (
                    <div
                        key={index}
                        onClick={idle ? () => toggleClip(index) : undefined}
                        className={`flex items-center gap-3 py-2.5${idle ? ' cursor-pointer' : ''}${isOn ? '' : ' opacity-40'}`}
                    >
                        <div className="w-24 shrink-0">
                            <span className="readout">
                                {!isOn ? 'skipped'
                                    : toInstagram && timing !== 'explicit'
                                        ? (outcome?.scheduled_at
                                            ? `${getDayLabel(new Date(outcome.scheduled_at))} · ${formatDate(new Date(outcome.scheduled_at))}`
                                            : (timing === 'now' ? 'now' : `slot ${position + 1}`))
                                        : `${getDayLabel(date)} · ${formatDate(date)}`}
                            </span>
                        </div>

                        <div className="flex-1 min-w-0">
                            <div className="text-xs text-ink truncate">
                                {clip.video_title_for_youtube_short || 'Viral Short'}
                            </div>
                            <div className="readout mt-0.5 truncate">
                                {!isOn ? 'not scheduled'
                                    : toInstagram && timing !== 'explicit'
                                        ? (outcome?.scheduled_at
                                            ? new Date(outcome.scheduled_at)
                                                .toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                                            : (timing === 'now' ? 'on the next check' : 'next free slot'))
                                        : `${time} · ${timezoneOptions.find(t => t.value === timezone)?.label || timezone}`}
                            </div>
                        </div>

                        <div className="shrink-0">
                            {/* Idle: a tick box. Running: the status of this
                                clip. The old build showed a hollow circle while
                                idle, which read as an empty checkbox and is
                                exactly the control this now is. */}
                            {idle ? (
                                isOn
                                    ? <CheckCircle size={16} className="text-brass" />
                                    : <Circle size={16} className="text-muted" />
                            ) : (
                                <>
                                    {outcome?.success === true && (
                                        <CheckCircle size={16} className="text-ok" />
                                    )}
                                    {outcome?.success === false && (
                                        <AlertCircle size={16} className="text-danger" />
                                    )}
                                    {scheduling && progress.current === position && (
                                        <Loader2 size={16} className="text-brass animate-spin" />
                                    )}
                                    {isOn && outcome === undefined && !scheduling && (
                                        <Circle size={16} className="text-muted" />
                                    )}
                                </>
                            )}
                        </div>
                    </div>
                    );
                })}
            </div>

            {/* Platforms — the self-hosted poster publishes to one Instagram
                account and nothing else, so there is nothing to choose. */}
            <div className={`mb-5${toInstagram ? ' hidden' : ''}`}>
                <label className="eyebrow block mb-2">platforms</label>
                <SegmentedControl
                    multi
                    options={PLATFORM_OPTIONS.map(opt => ({ ...opt, disabled: scheduling }))}
                    value={selectedPlatforms}
                    onChange={(arr) => setPlatforms({
                        tiktok: arr.includes('tiktok'),
                        instagram: arr.includes('instagram'),
                        youtube: arr.includes('youtube')
                    })}
                />
            </div>

            {/* A whole week of tiktok posts is a whole week of silent drafts:
                this modal writes no captions of its own either (it sends the
                clip's generated title/description), and tiktok keeps none of
                them on a draft. Say it before the button, not after. */}
            {!toInstagram && platforms.tiktok && !scheduling && !done && <TikTokDraftNotice />}

            {/* Progress bar */}
            {(scheduling || done) && (
                <div className="mb-1">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-xs text-muted lowercase">
                            {!scheduling ? 'complete'
                                : toInstagram ? `uploading clip ${Math.min(progress.current + 1, progress.total)} of ${progress.total}...`
                                : 'scheduling...'}
                        </span>
                        <span className="readout">{progress.current}/{progress.total}</span>
                    </div>
                    <div className="w-full h-1.5 bg-paper3 rounded-full overflow-hidden">
                        <div
                            className={`h-full rounded-full transition-all duration-500 ${done && failCount === 0 ? 'bg-ok' : done && failCount > 0 ? 'bg-danger' : 'bg-brass'}`}
                            style={{ width: `${(progress.current / progress.total) * 100}%` }}
                        />
                    </div>
                    {done && (
                        <div className="mt-3 text-xs text-center lowercase">
                            {failCount === 0 ? (
                                <span className="text-ok">all clips scheduled</span>
                            ) : (
                                <span className="text-danger">{successCount} scheduled, {failCount} failed</span>
                            )}
                        </div>
                    )}
                </div>
            )}
        </Modal>
    );
}
