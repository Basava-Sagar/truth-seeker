/**
 * ExplainabilityView
 * ------------------
 * 1. Sync seismograph — the signature visual. Per-frame audio-visual
 *    disagreement over the WHOLE video as an SVG trace, with the
 *    analyzed (most suspicious) window shaded and its peak marked.
 * 2. Heatmap frames — Grad-CAM projected onto original video frames.
 * 3. Attention map + audio spectrogram CAM, side by side.
 */

function SyncSeismograph({ timeline, window: win }) {
  if (!timeline?.length) return null;

  const W = 760, H = 180, PAD = 8;
  const tMax = timeline[timeline.length - 1].time || 1;
  const vMax = Math.max(...timeline.map((p) => p.value), 0.01);

  const x = (t) => PAD + (t / tMax) * (W - 2 * PAD);
  const y = (v) => H - PAD - (v / vMax) * (H - 2 * PAD);

  const path = timeline
    .map((p, i) => `${i === 0 ? "M" : "L"}${x(p.time).toFixed(1)},${y(p.value).toFixed(1)}`)
    .join(" ");

  const peak = timeline.reduce((a, b) => (b.value > a.value ? b : a));

  return (
    <div className="panel seismo">
      <h2>Audio–visual disagreement over time</h2>
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Sync disagreement timeline">
        {/* shaded region = the window the deep-dive explanation used */}
        <rect
          x={x(win.start)} y={PAD}
          width={Math.max(x(win.end) - x(win.start), 2)}
          height={H - 2 * PAD}
          fill="var(--amber-dim)"
        />
        {/* baseline grid */}
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD}
              stroke="var(--line)" strokeWidth="1" />
        {/* the trace */}
        <path d={path} fill="none" stroke="var(--amber)" strokeWidth="1.8" />
        {/* peak marker */}
        <circle cx={x(peak.time)} cy={y(peak.value)} r="4"
                fill="var(--amber)" />
        <text x={Math.min(x(peak.time) + 8, W - 90)} y={y(peak.value) - 6}
              fontSize="11" fill="var(--ink)"
              fontFamily="var(--font-mono)">
          peak @ {peak.time}s
        </text>
      </svg>
      <div className="axis-label">
        <span>0s</span>
        <span>higher = lips and speech disagree more</span>
        <span>{tMax.toFixed(1)}s</span>
      </div>
      <p className="caption">
        Shaded band: the {win.start}–{win.end}s window examined in depth below
        (scored {(win.fake_probability * 100).toFixed(0)}% manipulated).
      </p>
    </div>
  );
}

function HeatmapFrames({ frames }) {
  if (!frames?.length) return null;
  return (
    <div className="panel">
      <h2>Where the model looked — mouth region heatmaps</h2>
      <div className="frames">
        {frames.map((f) => (
          <figure className="frame-card" key={f.time}>
            <img
              src={`data:image/jpeg;base64,${f.image_b64}`}
              alt={`Grad-CAM heatmap on the mouth region at ${f.time} seconds`}
            />
            <figcaption>t = {f.time}s</figcaption>
          </figure>
        ))}
      </div>
      <p className="caption">
        Warm colors mark the mouth pixels that most increased the
        manipulation score in the flagged window.
      </p>
    </div>
  );
}

function VisualHeatmap({ heatmapB64, time, probability }) {
  if (!heatmapB64) return null;
  return (
    <div className="panel">
      <h2>Where the visual model looked — full-face heatmap</h2>
      <div className="frames">
        <figure className="frame-card">
          <img
            src={`data:image/png;base64,${heatmapB64}`}
            alt={`Grad-CAM heatmap over the full face at ${time} seconds`}
          />
          <figcaption>t = {time}s</figcaption>
        </figure>
      </div>
      <p className="caption">
        Warm colors mark the face regions (blending seams, texture, edges —
        not just the mouth) that most increased this frame's manipulation
        score ({(probability * 100).toFixed(0)}%). A separate detector from
        the mouth-region heatmaps above — this one checks for visual
        face-swap artifacts, independent of audio-visual sync.
      </p>
    </div>
  );
}

function Spectra({ explanation }) {
  return (
    <div className="panel">
      <h2>Cross-modal evidence</h2>
      <div className="spectra">
        <div>
          <img
            src={`data:image/png;base64,${explanation.attention_map_b64}`}
            alt="Cross-modal attention matrix between audio and video frames"
          />
          <p className="caption">
            Attention between audio (rows) and lip frames (columns).
            Genuine speech concentrates on the diagonal; blurred or shifted
            mass indicates desynchronization. Diagonality here:{" "}
            {(explanation.attention_diagonality * 100).toFixed(0)}%.
          </p>
        </div>
        <div>
          <img
            src={`data:image/png;base64,${explanation.audio_cam_b64}`}
            alt="Grad-CAM heatmap over the audio mel-spectrogram"
          />
          <p className="caption">
            Mel-spectrogram of the flagged second with audio Grad-CAM overlay —
            warm regions are the time–frequency areas that drove the score.
          </p>
        </div>
      </div>
    </div>
  );
}

export default function ExplainabilityView({ result }) {
  const {
    sync_timeline, explanation,
    visual_heatmap_b64, visual_heatmap_time, visual_probability,
  } = result;
  return (
    <div className="stack">
      <SyncSeismograph timeline={sync_timeline} window={explanation.window} />
      <HeatmapFrames frames={explanation.heatmap_frames} />
      <VisualHeatmap
        heatmapB64={visual_heatmap_b64}
        time={visual_heatmap_time}
        probability={visual_probability}
      />
      <Spectra explanation={explanation} />
    </div>
  );
}
