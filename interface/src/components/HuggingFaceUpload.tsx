import { useState } from 'react';
import { ExternalLink, CheckCircle, Upload } from 'lucide-react';
import { uploadToHuggingFace, type UploadModelResponse } from '../lib/api';

interface Props {
  uploadId: string;
  apiBaseUrl: string;
}

export default function HuggingFaceUpload({ uploadId, apiBaseUrl }: Props) {
  const [hfToken, setHfToken] = useState('');
  const [hfUsername, setHfUsername] = useState('');
  const [datasetName, setDatasetName] = useState('');
  const [isPrivate, setIsPrivate] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadModelResponse | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!hfToken.trim() || !hfUsername.trim()) {
      setError('HuggingFace token and username are required.');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await uploadToHuggingFace(
        {
          upload_id: uploadId,
          hf_token: hfToken,
          hf_username: hfUsername,
          dataset_name: datasetName || undefined,
          private: isPrivate,
        },
        apiBaseUrl || undefined
      );
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed.');
    } finally {
      setLoading(false);
    }
  };

  if (result) {
    return (
      <div
        className="rounded-2xl p-6 page-animate-in"
        style={{
          background: 'rgba(30,50,40,0.3)',
          border: '1px solid rgba(80,180,110,0.25)',
          boxShadow: '0 4px 40px rgba(0,0,0,0.5), 0 0 60px rgba(60,160,90,0.04)',
        }}
      >
        <div className="flex items-center gap-3 mb-5">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{
              background: 'rgba(70,180,110,0.15)',
              border: '1px solid rgba(80,200,120,0.2)',
            }}
          >
            <CheckCircle size={20} style={{ color: 'rgba(100,220,140,0.9)' }} />
          </div>
          <div>
            <p style={{ color: 'rgba(160,230,180,0.95)', fontWeight: 600, fontSize: '0.9rem' }}>
              Uploaded to HuggingFace Hub
            </p>
            <p style={{ color: 'rgba(100,160,120,0.6)', fontSize: '0.73rem' }}>
              Your model is now live
            </p>
          </div>
        </div>

        <div className="space-y-3">
          <div
            className="rounded-xl px-4 py-3 flex items-center justify-between"
            style={{ background: 'rgba(12,20,15,0.7)', border: '1px solid rgba(80,180,100,0.12)' }}
          >
            <span style={{ color: 'rgba(120,160,130,0.6)', fontSize: '0.75rem' }}>Repository</span>
            <span style={{ fontFamily: 'monospace', color: 'rgba(170,220,185,0.85)', fontSize: '0.8rem' }}>
              {result.repo_id}
            </span>
          </div>

          <a
            href={result.repo_url}
            target="_blank"
            rel="noopener noreferrer"
            className="chrome-btn-primary flex items-center justify-center gap-2.5 w-full rounded-xl py-3.5 transition-all duration-200"
            style={{
              color: 'rgba(200,230,210,0.9)',
              fontSize: '0.8rem',
              fontWeight: 500,
              textDecoration: 'none',
            }}
          >
            <ExternalLink size={15} />
            View on HuggingFace Hub
          </a>
        </div>
      </div>
    );
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-2xl p-6 space-y-5 page-animate-in"
      style={{
        background: 'rgba(16,16,26,0.85)',
        border: '1px solid rgba(180,185,240,0.12)',
        boxShadow: '0 4px 40px rgba(0,0,0,0.55)',
      }}
    >
      <div className="flex items-center gap-2.5 mb-2">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{
            background: 'rgba(120,130,190,0.15)',
            border: '1px solid rgba(180,185,240,0.12)',
          }}
        >
          <Upload size={14} style={{ color: 'rgba(180,185,230,0.7)' }} />
        </div>
        <h3
          style={{
            fontFamily: 'Orbitron, sans-serif',
            fontSize: '0.7rem',
            letterSpacing: '0.12em',
            color: 'rgba(190,195,230,0.85)',
            textTransform: 'uppercase',
          }}
        >
          Upload to HuggingFace Hub
        </h3>
      </div>

      <div
        className="rounded-lg px-4 py-2.5 flex items-center gap-2"
        style={{
          background: 'rgba(180,185,220,0.04)',
          border: '1px solid rgba(180,185,220,0.08)',
        }}
      >
        <span style={{ color: 'rgba(130,135,170,0.55)', fontSize: '0.72rem' }}>Experiment ID:</span>
        <span
          style={{
            fontFamily: 'monospace',
            fontSize: '0.78rem',
            color: 'rgba(180,185,230,0.75)',
          }}
        >
          {uploadId}
        </span>
      </div>

      <div>
        <label className="chrome-label block mb-2">HuggingFace Token *</label>
        <input
          type="password"
          className="chrome-input w-full rounded-lg px-4 py-2.5 text-sm"
          placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
          value={hfToken}
          onChange={(e) => setHfToken(e.target.value)}
          autoComplete="off"
        />
        <p style={{ color: 'rgba(120,125,160,0.45)', fontSize: '0.68rem', marginTop: 5 }}>
          Your token is sent directly to the API and never stored.
        </p>
      </div>

      <div>
        <label className="chrome-label block mb-2">HuggingFace Username *</label>
        <input
          type="text"
          className="chrome-input w-full rounded-lg px-4 py-2.5 text-sm"
          placeholder="your-hf-username"
          value={hfUsername}
          onChange={(e) => setHfUsername(e.target.value)}
        />
      </div>

      <div>
        <label className="chrome-label block mb-2">Dataset / Repo Name (optional)</label>
        <input
          type="text"
          className="chrome-input w-full rounded-lg px-4 py-2.5 text-sm"
          placeholder="my-evalora-model"
          value={datasetName}
          onChange={(e) => setDatasetName(e.target.value)}
        />
      </div>

      <label className="flex items-center gap-3 cursor-pointer select-none">
        <div
          onClick={() => setIsPrivate(!isPrivate)}
          className="relative w-9 h-5 rounded-full transition-all duration-200 flex-shrink-0"
          style={{
            background: isPrivate
              ? 'linear-gradient(135deg, rgba(160,170,220,0.4), rgba(120,130,190,0.3))'
              : 'rgba(40,40,55,0.8)',
            border: isPrivate
              ? '1px solid rgba(180,185,240,0.35)'
              : '1px solid rgba(100,100,130,0.2)',
            cursor: 'pointer',
          }}
        >
          <div
            className="absolute top-0.5 transition-all duration-200 w-4 h-4 rounded-full"
            style={{
              left: isPrivate ? '18px' : '2px',
              background: isPrivate
                ? 'linear-gradient(135deg, #ddd, #fff)'
                : 'rgba(100,100,130,0.5)',
              boxShadow: isPrivate ? '0 1px 4px rgba(0,0,0,0.4)' : 'none',
            }}
          />
        </div>
        <span style={{ color: 'rgba(180,185,215,0.75)', fontSize: '0.82rem' }}>
          Private repository
        </span>
      </label>

      {error && (
        <div
          className="rounded-xl px-5 py-3 text-sm"
          style={{
            background: 'rgba(200,60,60,0.08)',
            border: '1px solid rgba(200,80,80,0.2)',
            color: 'rgba(230,160,160,0.9)',
          }}
        >
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={loading}
        className="chrome-btn-primary w-full rounded-xl py-3.5 text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
        style={{
          fontFamily: 'Orbitron, sans-serif',
          letterSpacing: '0.1em',
          fontSize: '0.7rem',
          color: 'rgba(210,215,250,0.9)',
        }}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2.5">
            <svg className="animate-spin" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
            Uploading...
          </span>
        ) : (
          'Upload to HuggingFace'
        )}
      </button>
    </form>
  );
}
