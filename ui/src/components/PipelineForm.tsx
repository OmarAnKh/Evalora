import { useState, useRef } from 'react';
import { Upload, ChevronDown, ChevronUp, Cpu, Zap, X } from 'lucide-react';
import { runFullPipeline, type FullRunParams, type PipelineTrainEvaluateResponse } from '../lib/api';

interface Props {
  onResult: (result: PipelineTrainEvaluateResponse) => void;
}

export default function PipelineForm({ onResult }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [trainingConfig, setTrainingConfig] = useState<File | null>(null);
  const [baseModel, setBaseModel] = useState('unsloth/mistral-7b-instruct-v0.2-bnb-4bit');
  const [experimentName, setExperimentName] = useState('');
  const [trainRatio, setTrainRatio] = useState(0.8);
  const [valRatio, setValRatio] = useState(0.1);
  const [testRatio, setTestRatio] = useState(0.1);
  const [seed, setSeed] = useState(42);
  const [epochs, setEpochs] = useState(3);
  const [lr, setLr] = useState(0.0002);
  const [batchSize, setBatchSize] = useState(1);
  const [gradAccum, setGradAccum] = useState(4);
  const [useKappa, setUseKappa] = useState(true);
  const [useBert, setUseBert] = useState(true);
  const [advanced, setAdvanced] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const configRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) { setError('Upload a dataset file to continue.'); return; }
    setError(null);
    setLoading(true);
    try {
      const result = await runFullPipeline({
        file,
        training_config: trainingConfig || null,
        base_model_name: baseModel,
        train_ratio: trainRatio,
        val_ratio: valRatio,
        test_ratio: testRatio,
        seed,
        experiment_name: experimentName || undefined,
        num_train_epochs: epochs,
        learning_rate: lr,
        per_device_train_batch_size: batchSize,
        gradient_accumulation_steps: gradAccum,
        use_cohen_kappa: useKappa,
        use_bertscore: useBert,
      });
      onResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit} className="w-full max-w-4xl mx-auto space-y-5 animate-fade-up">
      {/* Title */}
      <div className="text-center mb-6">
        <div className="flex items-center justify-center gap-2 mb-2">
          <Cpu size={16} style={{ color: 'var(--accent)', opacity: 0.7 }} />
          <span className="font-brand text-xs tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
            Pipeline · Full Run
          </span>
        </div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
          Upload your dataset, configure training, and let Evalora handle the rest.
        </p>
      </div>

      {/* Dataset Upload */}
      <div className="card p-5">
        <span className="label">Dataset File *</span>
        <div
          className={`dropzone ${dragOver ? 'dragover' : ''} ${file ? 'has-file' : ''} flex flex-col items-center justify-center py-8 px-4`}
          onClick={() => fileRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <input ref={fileRef} type="file" className="hidden" accept=".csv,.json,.jsonl,.parquet,.xlsx"
            onChange={e => e.target.files?.[0] && setFile(e.target.files[0])} />
          {file ? (
            <div className="flex items-center gap-3">
              <div className="section-icon" style={{ background: 'var(--success-dim)', border: '1px solid var(--success)' }}>
                <Zap size={14} style={{ color: 'var(--success)' }} />
              </div>
              <div className="text-left">
                <p style={{ color: 'var(--text-primary)', fontSize: '0.85rem', fontWeight: 500 }}>{file.name}</p>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>{(file.size / 1024).toFixed(1)} KB</p>
              </div>
              <button type="button" onClick={e => { e.stopPropagation(); setFile(null); }}
                className="ml-2 p-1 rounded-md hover:bg-white/5" style={{ color: 'var(--text-muted)' }}>
                <X size={14} />
              </button>
            </div>
          ) : (
            <>
              <Upload size={20} style={{ color: 'var(--text-muted)', marginBottom: 8 }} />
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
                Drop file or <span style={{ color: 'var(--accent)' }}>browse</span>
              </p>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.7rem', marginTop: 3 }}>CSV, JSON, JSONL, Parquet, XLSX</p>
            </>
          )}
        </div>
      </div>

      {/* Core Config */}
      <div className="card p-5 space-y-4">
        <span className="label">Model Configuration</span>
        <div>
          <label className="label">Base Model</label>
          <input className="input" value={baseModel} onChange={e => setBaseModel(e.target.value)} />
        </div>
        <div>
          <label className="label">Experiment Name</label>
          <input className="input" placeholder="optional" value={experimentName} onChange={e => setExperimentName(e.target.value)} />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="label">Epochs</label>
            <input type="number" min={1} className="input" value={epochs} onChange={e => setEpochs(Number(e.target.value))} />
          </div>
          <div>
            <label className="label">Learning Rate</label>
            <input type="number" step="0.00001" className="input" value={lr} onChange={e => setLr(Number(e.target.value))} />
          </div>
        </div>

        <div className="divider" />
        <span className="label">Evaluation Metrics</span>
        <div className="flex gap-5">
          <label className="flex items-center gap-2.5 cursor-pointer select-none">
            <div className={`toggle-track ${useKappa ? 'active' : ''}`} onClick={() => setUseKappa(!useKappa)}>
              <div className="toggle-knob" />
            </div>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.78rem' }}>Cohen's Kappa</span>
          </label>
          <label className="flex items-center gap-2.5 cursor-pointer select-none">
            <div className={`toggle-track ${useBert ? 'active' : ''}`} onClick={() => setUseBert(!useBert)}>
              <div className="toggle-knob" />
            </div>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.78rem' }}>BERTScore</span>
          </label>
        </div>
      </div>

      {/* Advanced */}
      <div className="card overflow-hidden">
        <button type="button" className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-white/[0.015] transition-colors"
          onClick={() => setAdvanced(!advanced)}>
          <span className="label" style={{ marginBottom: 0 }}>Advanced Settings</span>
          {advanced ? <ChevronUp size={14} style={{ color: 'var(--text-muted)' }} /> : <ChevronDown size={14} style={{ color: 'var(--text-muted)' }} />}
        </button>
        {advanced && (
          <div className="px-5 pb-5 space-y-4" style={{ borderTop: '1px solid var(--border)' }}>
            <div className="grid grid-cols-3 gap-3 pt-4">
              <div><label className="label">Train</label><input type="number" step={0.05} className="input" value={trainRatio} onChange={e => setTrainRatio(Number(e.target.value))} /></div>
              <div><label className="label">Val</label><input type="number" step={0.05} className="input" value={valRatio} onChange={e => setValRatio(Number(e.target.value))} /></div>
              <div><label className="label">Test</label><input type="number" step={0.05} className="input" value={testRatio} onChange={e => setTestRatio(Number(e.target.value))} /></div>
            </div>
            <div className="grid grid-cols-3 gap-3">
              <div><label className="label">Seed</label><input type="number" className="input" value={seed} onChange={e => setSeed(Number(e.target.value))} /></div>
              <div><label className="label">Batch Size</label><input type="number" min={1} className="input" value={batchSize} onChange={e => setBatchSize(Number(e.target.value))} /></div>
              <div><label className="label">Grad Accum</label><input type="number" min={1} className="input" value={gradAccum} onChange={e => setGradAccum(Number(e.target.value))} /></div>
            </div>
            <div>
              <label className="label">Training Config File</label>
              <div className="flex items-center gap-2 input cursor-pointer" style={{ padding: '8px 12px' }}
                onClick={() => configRef.current?.click()}>
                <Upload size={12} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                <span style={{ color: trainingConfig ? 'var(--text-primary)' : 'var(--text-muted)', fontSize: '0.78rem' }}>
                  {trainingConfig ? trainingConfig.name : 'Optional JSON/YAML'}
                </span>
                {trainingConfig && (
                  <button type="button" onClick={e => { e.stopPropagation(); setTrainingConfig(null); }} className="ml-auto p-0.5 rounded hover:bg-white/5" style={{ color: 'var(--text-muted)' }}>
                    <X size={12} />
                  </button>
                )}
                <input ref={configRef} type="file" className="hidden" accept=".json,.yaml,.yml"
                  onChange={e => e.target.files?.[0] && setTrainingConfig(e.target.files[0])} />
              </div>
            </div>
          </div>
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}

      <button type="submit" disabled={loading || !file}
        className={`btn btn-primary w-full py-3.5 font-brand tracking-wider ${!file ? 'opacity-40 cursor-not-allowed' : ''}`}>
        {loading ? (<><div className="spinner" /> Running Pipeline...</>) : 'Build'}
      </button>
    </form>
  );
}