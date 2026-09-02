import { useRef, useState } from "react";

const ACCEPTED = [".mp4", ".mov", ".webm", ".mkv", ".avi"];

export default function VideoUploader({ onAnalyze, busy, error }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [drag, setDrag] = useState(false);

  const pick = (f) => {
    if (!f) return;
    const ok = ACCEPTED.some((ext) => f.name.toLowerCase().endsWith(ext));
    if (!ok) return;
    setFile(f);
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(f);
    });
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDrag(false);
    pick(e.dataTransfer.files?.[0]);
  };

  return (
    <div className="panel">
      <h2>Submit evidence</h2>

      <div
        className={`dropzone ${drag ? "drag" : ""}`}
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
      >
        <strong>{file ? file.name : "Drop a video here"}</strong>
        <p>{file
          ? `${(file.size / 1048576).toFixed(1)} MB — ready to analyze`
          : "or click to browse · mp4, mov, webm · max 100 MB"}</p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          hidden
          onChange={(e) => pick(e.target.files?.[0])}
        />
      </div>

      {previewUrl && (
        <video className="preview" src={previewUrl} controls muted />
      )}

      <button
        className="btn"
        disabled={!file || busy}
        onClick={() => onAnalyze(file)}
      >
        {busy ? "Analyzing…" : "Run cross-modal analysis"}
      </button>

      {busy && (
        <>
          <div className="pulse-track"><div className="pulse" /></div>
          <p className="status">
            Extracting frames, tracking lips, computing spectrograms…
          </p>
        </>
      )}

      {error && <p className="status error">{error}</p>}
    </div>
  );
}
