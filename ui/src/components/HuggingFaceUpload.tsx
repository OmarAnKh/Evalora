import { useState } from 'react';
import { ExternalLink, CheckCircle, Upload } from 'lucide-react';
import { uploadToHuggingFace, type UploadModelResponse } from '../lib/api';

interface Props {
  uploadId: string;
}

export default function HuggingFaceUpload({ uploadId }: Props) {
  const [token, setToken] = useState('');
  const [username, setUsername] = useState('');
  const [repoName, setRepoName] = useState('');
  const [isPrivate, setIsPrivate] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadModelResponse | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token.trim() || !username.trim()) { setError('Token and username are required.'); return; }
    setError(null);
    setLoading(true);
    try {
      const res = await uploadToHuggingFace({
        upload_id: uploadId,
        hf_token: token,
        hf_username: username,
        dataset_name: repoName.trim() || undefined,
        private: isPrivate,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed.');
    } finally {
      setLoading(false);
    }
  };

  if (result) {
    return (
      <div className="card-solid p-5 space-y-4 animate-fade-up" style={{ borderColor: 'var(--success-border)' }}>
        <div className="flex items-center gap-3">
          <div className="section-icon" style={{ background: 'var(--success-dim)', border: '1px solid var(--success-border)' }}>
            <CheckCircle size={16} style={{ color: 'var(--success)' }} />
          </div>
          <div>
            <p style={{ color: 'var(--success)', fontWeight: 600, fontSize: '0.88rem' }}>Uploaded to HuggingFace Hub</p>
            <p className="mono" style={{ color: 'var(--text-secondary)', fontSize: '0.75rem' }}>{result.repo_id}</p>
          </div>
        </div>
        <a href={result.repo_url} target="_blank" rel="noopener noreferrer"
          className="btn btn-success w-full py-3" style={{ textDecoration: 'none' }}>
          <ExternalLink size={14} /> View on HuggingFace Hub
        </a>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="card-solid p-5 space-y-4 animate-fade-up">
      <div className="flex items-center gap-2.5">
        <div className="section-icon" style={{ background: 'var(--accent-dim)', border: '1px solid var(--border-focus)' }}>
          <Upload size={13} style={{ color: 'var(--accent)', opacity: 0.8 }} />
        </div>
        <h3 className="font-brand text-xs tracking-widest uppercase" style={{ color: 'var(--text-secondary)' }}>Upload to HuggingFace</h3>
      </div>

      <div className="flex items-center gap-2 rounded-lg px-3 py-2" style={{ background: 'var(--accent-dim)', border: '1px solid var(--border-focus)' }}>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>Experiment:</span>
        <span className="mono" style={{ color: 'var(--accent)', fontSize: '0.75rem' }}>{uploadId}</span>
      </div>

      <div>
        <label className="label">HuggingFace Token *</label>
        <input type="password" className="input" placeholder="hf_xxxxxx" value={token} onChange={e => setToken(e.target.value)} autoComplete="off" />
        <p style={{ color: 'var(--text-muted)', fontSize: '0.65rem', marginTop: 4 }}>Sent directly to the API, never stored.</p>
      </div>
      <div>
        <label className="label">Username *</label>
        <input className="input" placeholder="your-hf-username" value={username} onChange={e => setUsername(e.target.value)} />
      </div>
      <div>
        <label className="label">Repo Name</label>
        <input className="input" placeholder="optional" value={repoName} onChange={e => setRepoName(e.target.value)} />
      </div>

      <label className="flex items-center gap-2.5 cursor-pointer select-none">
        <div className={`toggle-track ${isPrivate ? 'active' : ''}`} onClick={() => setIsPrivate(!isPrivate)}>
          <div className="toggle-knob" />
        </div>
        <span style={{ color: 'var(--text-secondary)', fontSize: '0.78rem' }}>Private repository</span>
      </label>

      {error && <div className="error-banner">{error}</div>}

      <button type="submit" disabled={loading} className="btn btn-primary w-full py-3 font-brand tracking-wider">
        {loading ? (<><div className="spinner" /> Uploading...</>) : 'Upload to HuggingFace'}
      </button>
    </form>
  );
}
