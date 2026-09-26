import React, { useState, useEffect, useMemo } from 'react';
import { Loader2, Download, Film, FolderOpen } from 'lucide-react';
import { apiJson } from '../lib/api';

// Past work, from whichever store this deployment has.
//
// CLOUD reads /api/history — the signed-in user's library in R2, private signed
// links, grouped by project, with a "reopen project" action that re-hydrates
// the job for further editing.
//
// SELF-HOST has no library and those endpoints do not exist (they live in
// cloud/videos.py). It reads /api/jobs instead, which enumerates the job
// directories still on disk. Nothing is re-hydrated there because nothing left:
// opening a job just points the results view at a job_id /api/status already
// serves. Jobs disappear from this list when JOB_RETENTION_SECONDS sweeps them,
// which is the truth rather than a bug.
export default function HistoryTab({ onReopenProject, onOpenJob }) {
  const [videos, setVideos] = useState(null);
  const [jobs, setJobs] = useState(null);
  const [projects, setProjects] = useState({});
  const [reopening, setReopening] = useState(null);
  const [reopenError, setReopenError] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    apiJson('/api/history')
      .then((d) => { if (!cancelled) setVideos(d.videos || []); })
      .catch(() => {
        // Not an error here — it simply means this is a self-host deployment.
        apiJson('/api/jobs')
          .then((d) => { if (!cancelled) { setJobs(d.jobs || []); setVideos([]); } })
          .catch(() => { if (!cancelled) setError('Could not load your history.'); });
      });
    apiJson('/api/projects')
      .then((d) => {
        if (cancelled) return;
        const map = {};
        for (const p of d.projects || []) map[p.job_id] = p;
        setProjects(map);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  // Group videos by job, preserving the newest-first order of /api/history.
  const groups = useMemo(() => {
    const byJob = new Map();
    for (const v of videos || []) {
      const key = v.job_id || v.id;
      if (!byJob.has(key)) byJob.set(key, []);
      byJob.get(key).push(v);
    }
    return [...byJob.entries()];
  }, [videos]);

  const handleReopen = async (jobId) => {
    if (!onReopenProject || reopening) return;
    setReopening(jobId);
    setReopenError('');
    try {
      await onReopenProject(jobId);
    } catch (e) {
      setReopenError('Could not reopen this project. Please try again.');
      setReopening(null);
    }
  };

  const fmtDate = (iso) => (iso ? new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) : '');

  if (videos === null && jobs === null && !error) {
    return <div className="flex justify-center py-20"><Loader2 className="animate-spin text-brass" /></div>;
  }

  // Self-host: one row per job, opened straight from disk.
  if (jobs !== null) {
    return (
      <div className="h-full overflow-y-auto p-8 max-w-5xl mx-auto animate-fade">
        <p className="eyebrow mb-1.5">06 · HISTORY</p>
        <h1 className="font-display lowercase text-2xl text-ink mb-2">Past jobs</h1>
        <p className="text-muted text-sm mb-8 lowercase">
          Every finished job still on this machine. Open one to get its clips back, with
          captions, hooks and scheduling exactly as they were.
        </p>

        {error && <p className="text-danger text-sm">{error}</p>}

        {jobs.length === 0 && (
          <div className="text-center py-20 text-muted">
            <Film size={40} className="mx-auto mb-4 text-muted" />
            <p className="lowercase">No finished jobs on disk. Generate one from the Clip Generator.</p>
          </div>
        )}

        <div className="border-y border-rule divide-y divide-rule">
          {jobs.map((j) => (
            <button
              key={j.job_id}
              onClick={() => onOpenJob && onOpenJob(j.job_id)}
              disabled={!onOpenJob}
              className="w-full flex items-center gap-4 py-3 text-left hover:bg-paper3 transition-colors disabled:cursor-default px-2"
            >
              <div className="w-16 shrink-0 aspect-[9/16] bg-black rounded-input overflow-hidden">
                {j.preview_url
                  ? <video src={j.preview_url} preload="metadata" muted className="w-full h-full object-cover" />
                  : <div className="w-full h-full flex items-center justify-center"><Film size={14} className="text-muted" /></div>}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm text-ink truncate" title={j.title}>{j.title || j.job_id}</p>
                <p className="readout mt-0.5">
                  {fmtDate(j.created_at)} · {j.clip_count} clip{j.clip_count === 1 ? '' : 's'}
                  {j.language ? ` · ${j.language}` : ''}
                </p>
              </div>
              <FolderOpen size={16} className="text-muted shrink-0" />
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-8 max-w-5xl mx-auto animate-fade">
      <p className="eyebrow mb-1.5">06 · HISTORY</p>
      <h1 className="font-display lowercase text-2xl text-ink mb-2">Your library</h1>
      <p className="text-muted text-sm mb-8 lowercase">
        All the shorts you've generated, saved while your plan is active. Kept for 7 days after your plan ends. Reopen a project to keep editing its clips.
      </p>

      {error && <p className="text-danger text-sm">{error}</p>}
      {reopenError && <p className="text-danger text-sm mb-4">{reopenError}</p>}

      {videos && videos.length === 0 && (
        <div className="text-center py-20 text-muted">
          <Film size={40} className="mx-auto mb-4 text-muted" />
          <p className="lowercase">No videos yet. Generate your first short from the Clip Generator.</p>
        </div>
      )}

      <div className="space-y-10">
        {groups.map(([jobId, vids]) => {
          const project = projects[jobId];
          return (
            <section key={jobId}>
              <div className="flex flex-wrap items-center justify-between gap-3 mb-4 pb-2 border-b border-rule">
                <div className="min-w-0">
                  <p className="text-sm text-ink font-medium truncate" title={project?.title || vids[0]?.title}>
                    {project?.title || vids[0]?.title || 'Project'}
                  </p>
                  <p className="readout mt-0.5">
                    {fmtDate(vids[0]?.created_at)} · {vids.length} clip{vids.length === 1 ? '' : 's'}
                  </p>
                </div>
                {project && onReopenProject && (
                  <button
                    onClick={() => handleReopen(jobId)}
                    disabled={!!reopening}
                    className="btn-ghost px-3 py-2 text-xs shrink-0"
                    title="Restore this project in the Clip Generator to keep editing subtitles, hooks, effects and dubbing"
                  >
                    {reopening === jobId
                      ? <><Loader2 size={14} className="animate-spin" /> reopening…</>
                      : <><FolderOpen size={14} /> reopen project</>}
                  </button>
                )}
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-5">
                {vids.map((v) => (
                  <div key={v.id} className="card card-hover overflow-hidden group">
                    <div className="aspect-[9/16] bg-black">
                      <video src={v.view_url} controls preload="metadata" className="w-full h-full object-contain" />
                    </div>
                    <div className="p-3">
                      <p className="text-sm text-ink font-medium line-clamp-2 mb-1" title={v.title}>{v.title || 'Short'}</p>
                      <div className="flex items-center justify-between">
                        <span className="readout">{fmtDate(v.created_at)}</span>
                        <a href={v.download_url} className="text-micro font-mono uppercase text-brass hover:text-ink flex items-center gap-1 transition-colors" title="Download">
                          <Download size={14} /> Download
                        </a>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}
