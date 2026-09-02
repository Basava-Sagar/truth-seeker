export default function VerdictCard({ result }) {
  const {
    verdict, fake_probability, model_trained, video, explanation,
    lipsync_probability, visual_probability, fusion_method,
  } = result;
  const isFake = verdict === "fake";
  const hasFusion = fusion_method != null;

  return (
    <div className="panel">
      <h2>Verdict</h2>

      <div className="verdict">
        <span className={`stamp ${isFake ? "fake" : "real"}`}>
          {isFake ? "LIKELY MANIPULATED" : "NO MANIPULATION FOUND"}
        </span>
        <div className="prob">
          {(fake_probability * 100).toFixed(1)}%
          <small>{hasFusion ? "combined manipulation probability" : "manipulation probability"}</small>
        </div>
      </div>

      {hasFusion && (
        <div className="detector-breakdown">
          <div>
            <b>{(lipsync_probability * 100).toFixed(1)}%</b>
            <small>lip-sync desync</small>
          </div>
          <div>
            {visual_probability != null ? (
              <>
                <b>{(visual_probability * 100).toFixed(1)}%</b>
                <small>visual manipulation</small>
              </>
            ) : (
              <>
                <b>—</b>
                <small>visual model unavailable for this clip</small>
              </>
            )}
          </div>
        </div>
      )}

      <div className="meta-row">
        <div><b>{video.duration}s</b>duration</div>
        <div><b>{video.n_windows}</b>windows scored</div>
        <div><b>{(video.face_detection_rate * 100).toFixed(0)}%</b>face tracked</div>
        <div>
          <b>{(explanation.attention_diagonality * 100).toFixed(0)}%</b>
          attention diagonality
        </div>
      </div>

      {!model_trained && (
        <div className="untrained-note">
          No trained weights loaded — this verdict comes from a randomly
          initialized model and is not meaningful. Train the model and place
          the checkpoint at <code>models/weights/truth_seeker.pt</code>.
        </div>
      )}

      <p className="summary">{explanation.summary}</p>
    </div>
  );
}
