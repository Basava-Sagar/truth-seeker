import { useEffect, useState } from "react";
import { analyzeVideo, checkHealth } from "./api.js";
import VideoUploader from "./components/VideoUploader.jsx";
import VerdictCard from "./components/VerdictCard.jsx";
import ExplainabilityView from "./components/ExplainabilityView.jsx";

export default function App() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    checkHealth().then(setHealth).catch(() => setHealth(false));
  }, []);

  const runAnalysis = async (file) => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await analyzeVideo(file));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <h1>TRUTH<span className="tick">·</span>SEEKER</h1>
          <p className="sub">cross-modal deepfake analysis — audio × lips</p>
        </div>
        {health === false && <span className="badge warn">API offline</span>}
        {health && !health.model_trained && (
          <span className="badge warn">untrained weights</span>
        )}
        {health && health.model_trained && (
          <span className="badge">
            model ready{health.visual_model_trained ? " · fusion active" : " · lip-sync only"}
          </span>
        )}
      </header>

      <div className="grid">
        <VideoUploader onAnalyze={runAnalysis} busy={busy} error={error} />

        <div className="stack">
          {result ? (
            <>
              <VerdictCard result={result} />
              <ExplainabilityView result={result} />
            </>
          ) : (
            <div className="panel">
              <h2>How it works</h2>
              <p className="caption" style={{ fontSize: "14px" }}>
                Truth-Seeker tracks the speaker's mouth at 25 fps and converts
                the audio to a mel-spectrogram, then a dual-stream network
                checks whether lip motion and speech sounds stay synchronized.
                Every verdict ships with its evidence: a disagreement timeline,
                heatmaps of the exact mouth regions that raised suspicion, and
                the cross-modal attention pattern. Upload a video with a
                visible, audible speaker to begin.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
