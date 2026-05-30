import { useState, useRef } from 'react';
import { Upload, Settings, ChevronDown, ChevronUp, Cpu, Zap } from 'lucide-react';
import { runFullPipeline, type FullRunParams, type PipelineTrainEvaluateResponse } from '../lib/api';

interface Props {
  onResult: (result: PipelineTrainEvaluateResponse) => void;
  apiBaseUrl: string;
}

export default function PipelineForm({ onResult, apiBaseUrl }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [trainingConfig, setTrainingConfig] = useState<File | null>(null);
  const [baseModel, setBaseModel] = useState('unsloth/mistral-7b-instruct-v0.2-bnb-4bit');
  const [experimentName, setExperimentName] = useState('');
  const [trainRatio, setTrainRatio] = useState(0.8);
  const [valRatio, setValRatio] = useState(0.1);
  const [testRatio, setTestRatio] = useState(0.1);
  const [seed, setSeed] = useState(42);
  const [epochs, setEpochs] = useState(3);
  const [learningRate, setLearningRate] = useState(0.0002);
  const [batchSize, setBatchSize] = useState(1);
  const [gradAccum, setGradAccum] = useState(4);
  const [useCohenKappa, setUseCohenKappa] = useState(true);
  const [useBertScore, setUseBertScore] = useState(true);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const configInputRef = useRef<HTMLInputElement>(null);

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) { setError('Please upload a dataset file.'); return; }
    setError(null);
    setLoading(true);
    try {
      const params: FullRunParams = {
        file,
        training_config: trainingConfig,
        base_model_name: baseModel,
        train_ratio: trainRatio,
        val_ratio: valRatio,
        test_ratio: testRatio,
        seed,
        experiment_name: experimentName || undefined,
        num_train_epochs: epochs,
        learning_rate: learningRate,
        per_device_train_batch_size: batchSize,
        gradient_accumulation_steps: gradAccum,
        use_cohen_kappa: useCohenKappa,
        use_bertscore: useBertScore,
      };
      const result = await runFullPipeline(params, apiBaseUrl || undefined);
      onResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-3xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="flex items-center justify-center gap-3 mb-3">
          <Cpu size={20} style={{ color: 'rgba(180,185,220,0.7)' }} />
          <span
            style={{
              fontFamily: 'Orbitron, sans-serif',
              fontSize: '0.65rem',
              letterSpacing: '0.2em',
              color: 'rgba(160,165,200,0.6)',
              textTransform: 'uppercase',
            }}
          >
            Pipeline · Full Run
          </span>
          <Cpu size={20} style={{ color: 'rgba(180,185,220,0.7)' }} />
        </div>
        <p style={{ color: 'rgba(160,165,200,0.5)', fontSize: '0.82rem' }}>
          Upload your dataset, configure training, and let Evalora do the rest.
        </p>
      </div>

      {/* Dataset Upload */}
      <div className="glass-card rounded-2xl p-6">
        <p className="chrome-label mb-3">Dataset File *</p>
        <div
          className={`relative rounded-xl border-2 border-dashed transition-all duration-200 cursor-pointer flex flex-col items-center justify-center py-10 px-6 ${
            dragOver
              ? 'border-white/30 bg-white/5'
              : file
              ? 'border-white/20 bg-white/[0.03]'
              : 'border-white/10 bg-black/20 hover:border-white/20 hover:bg-white/[0.02]'
          }`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleFileDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".csv,.json,.jsonl,.parquet,.xlsx"
            onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])}
          />
          {file ? (
            <>
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center mb-3"
                style={{
                  background: 'rgba(100,200,140,0.12)',
                  border: '1px solid rgba(100,200,140,0.2)',
                }}
              >
                <Zap size={18} style={{ color: 'rgba(100,200,140,0.8)' }} />
              </div>
              <p style={{ color: 'rgba(200,220,200,0.9)', fontSize: '0.875rem', fontWeight: 500 }}>
                {file.name}
              </p>
              <p style={{ color: 'rgba(150,165,150,0.5)', fontSize: '0.75rem', marginTop: 4 }}>
                {(file.size / 1024).toFixed(1)} KB &nbsp;·&nbsp; Click to replace
              </p>
            </>
          ) : (
            <>
              <div
                className="w-12 h-12 rounded-xl flex items-center justify-center mb-4"
                style={{
                  background: 'rgba(180,185,220,0.06)',
                  border: '1px solid rgba(180,185,220,0.1)',
                }}
              >
                <Upload size={22} style={{ color: 'rgba(180,185,220,0.5)' }} />
              </div>
              <p style={{ color: 'rgba(180,185,210,0.7)', fontSize: '0.875rem' }}>
                Drop your dataset here or <span style={{ color: 'rgba(210,215,240,0.9)' }}>click to browse</span>
              </p>
              <p style={{ color: 'rgba(130,135,165,0.45)', fontSize: '0.75rem', marginTop: 4 }}>
                CSV, JSON, JSONL, Parquet, XLSX
              </p>
            </>
          )}
        </div>
        <div className="mt-4">
          <p style={{ color: 'rgba(150,155,185,0.6)', fontSize: '0.7rem', marginBottom: 8 }}>
            Example JSONL record (one per line):
          </p>
          <pre
            style={{
              background: 'rgba(10,10,16,0.7)',
              border: '1px solid rgba(180,180,220,0.08)',
              borderRadius: 12,
              padding: '12px 14px',
              fontSize: '0.68rem',
              color: 'rgba(170,175,205,0.75)',
              lineHeight: 1.45,
              overflowX: 'auto',
            }}
          >{`{
  "task": "Grade the response to the customer email.",
  "reference_answer": "We apologize for the delay and offer a replacement within 3 business days.",
  "answer": "Sorry for the wait. We can send a replacement this week.",
  "rubric": [
    {"criterion": "Apology", "description": "Must apologize clearly", "weight": 0.5},
    {"criterion": "Resolution", "description": "Offer replacement with a timeline", "weight": 0.5}
  ],
  "score": 3,
  "reasoning": "Includes an apology and a replacement, but the timeline is less specific than 3 business days."
}`}</pre>
        </div>
      </div>

      {/* Core Config */}
      <div className="glass-card rounded-2xl p-6 space-y-5">
        <p className="chrome-label mb-1">Model Configuration</p>

        <div>
          <label className="chrome-label block mb-2">Base Model</label>
          <input
            className="chrome-input w-full rounded-lg px-4 py-2.5 text-sm"
            value={baseModel}
            onChange={(e) => setBaseModel(e.target.value)}
            placeholder="unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
          />
        </div>

        <div>
          <label className="chrome-label block mb-2">Experiment Name (optional)</label>
          <input
            className="chrome-input w-full rounded-lg px-4 py-2.5 text-sm"
            value={experimentName}
            onChange={(e) => setExperimentName(e.target.value)}
            placeholder="my-finetune-run"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="chrome-label block mb-2">Training Epochs</label>
            <input
              type="number"
              min={1}
              step={1}
              className="chrome-input w-full rounded-lg px-4 py-2.5 text-sm"
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
            />
          </div>
          <div>
            <label className="chrome-label block mb-2">Learning Rate</label>
            <input
              type="number"
              step="0.00001"
              className="chrome-input w-full rounded-lg px-4 py-2.5 text-sm"
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
            />
          </div>
        </div>

        {/* Evaluation Metrics */}
        <div className="chrome-divider" />
        <p className="chrome-label">Evaluation Metrics</p>
        <div className="flex gap-6 mt-2">
          <label className="flex items-center gap-2.5 cursor-pointer select-none">
            <div
              onClick={() => setUseCohenKappa(!useCohenKappa)}
              className="relative w-9 h-5 rounded-full transition-all duration-200 flex-shrink-0"
              style={{
                background: useCohenKappa
                  ? 'linear-gradient(135deg, rgba(160,170,220,0.4), rgba(120,130,190,0.3))'
                  : 'rgba(40,40,55,0.8)',
                border: useCohenKappa
                  ? '1px solid rgba(180,185,240,0.35)'
                  : '1px solid rgba(100,100,130,0.2)',
                cursor: 'pointer',
              }}
            >
              <div
                className="absolute top-0.5 transition-all duration-200 w-4 h-4 rounded-full"
                style={{
                  left: useCohenKappa ? '18px' : '2px',
                  background: useCohenKappa
                    ? 'linear-gradient(135deg, #ddd, #fff)'
                    : 'rgba(100,100,130,0.5)',
                  boxShadow: useCohenKappa ? '0 1px 4px rgba(0,0,0,0.4)' : 'none',
                }}
              />
            </div>
            <span style={{ color: 'rgba(180,185,210,0.8)', fontSize: '0.8rem' }}>Cohen's Kappa</span>
          </label>

          <label className="flex items-center gap-2.5 cursor-pointer select-none">
            <div
              onClick={() => setUseBertScore(!useBertScore)}
              className="relative w-9 h-5 rounded-full transition-all duration-200 flex-shrink-0"
              style={{
                background: useBertScore
                  ? 'linear-gradient(135deg, rgba(160,170,220,0.4), rgba(120,130,190,0.3))'
                  : 'rgba(40,40,55,0.8)',
                border: useBertScore
                  ? '1px solid rgba(180,185,240,0.35)'
                  : '1px solid rgba(100,100,130,0.2)',
                cursor: 'pointer',
              }}
            >
              <div
                className="absolute top-0.5 transition-all duration-200 w-4 h-4 rounded-full"
                style={{
                  left: useBertScore ? '18px' : '2px',
                  background: useBertScore
                    ? 'linear-gradient(135deg, #ddd, #fff)'
                    : 'rgba(100,100,130,0.5)',
                  boxShadow: useBertScore ? '0 1px 4px rgba(0,0,0,0.4)' : 'none',
                }}
              />
            </div>
            <span style={{ color: 'rgba(180,185,210,0.8)', fontSize: '0.8rem' }}>BERTScore</span>
          </label>
        </div>
      </div>

      {/* Advanced Section */}
      <div className="glass-card rounded-2xl overflow-hidden">
        <button
          type="button"
          className="w-full flex items-center justify-between px-6 py-4 transition-all duration-150 hover:bg-white/[0.02]"
          onClick={() => setAdvancedOpen(!advancedOpen)}
        >
          <div className="flex items-center gap-2">
            <Settings size={14} style={{ color: 'rgba(160,165,200,0.6)' }} />
            <span className="chrome-label">Advanced Settings</span>
          </div>
          {advancedOpen ? (
            <ChevronUp size={14} style={{ color: 'rgba(160,165,200,0.5)' }} />
          ) : (
            <ChevronDown size={14} style={{ color: 'rgba(160,165,200,0.5)' }} />
          )}
        </button>

        {advancedOpen && (
          <div className="px-6 pb-6 space-y-5 border-t" style={{ borderColor: 'rgba(180,180,220,0.08)' }}>
            <div className="grid grid-cols-3 gap-4 mt-5">
              <div>
                <label className="chrome-label block mb-2">Train Ratio</label>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  className="chrome-input w-full rounded-lg px-4 py-2.5 text-sm"
                  value={trainRatio}
                  onChange={(e) => setTrainRatio(Number(e.target.value))}
                />
              </div>
              <div>
                <label className="chrome-label block mb-2">Val Ratio</label>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  className="chrome-input w-full rounded-lg px-4 py-2.5 text-sm"
                  value={valRatio}
                  onChange={(e) => setValRatio(Number(e.target.value))}
                />
              </div>
              <div>
                <label className="chrome-label block mb-2">Test Ratio</label>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  className="chrome-input w-full rounded-lg px-4 py-2.5 text-sm"
                  value={testRatio}
                  onChange={(e) => setTestRatio(Number(e.target.value))}
                />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="chrome-label block mb-2">Seed</label>
                <input
                  type="number"
                  className="chrome-input w-full rounded-lg px-4 py-2.5 text-sm"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                />
              </div>
              <div>
                <label className="chrome-label block mb-2">Batch Size</label>
                <input
                  type="number"
                  min={1}
                  className="chrome-input w-full rounded-lg px-4 py-2.5 text-sm"
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                />
              </div>
              <div>
                <label className="chrome-label block mb-2">Grad. Accum. Steps</label>
                <input
                  type="number"
                  min={1}
                  className="chrome-input w-full rounded-lg px-4 py-2.5 text-sm"
                  value={gradAccum}
                  onChange={(e) => setGradAccum(Number(e.target.value))}
                />
              </div>
            </div>

            <div>
              <label className="chrome-label block mb-2">Training Config File (optional)</label>
              <div
                className="flex items-center gap-3 rounded-lg px-4 py-2.5 cursor-pointer transition-all duration-150"
                style={{
                  background: 'rgba(12,12,18,0.8)',
                  border: '1px solid rgba(180,180,200,0.15)',
                }}
                onClick={() => configInputRef.current?.click()}
              >
                <Upload size={14} style={{ color: 'rgba(160,165,200,0.5)', flexShrink: 0 }} />
                <span style={{ color: 'rgba(160,165,200,0.6)', fontSize: '0.8rem' }}>
                  {trainingConfig ? trainingConfig.name : 'Upload config file (JSON/YAML)'}
                </span>
                <input
                  ref={configInputRef}
                  type="file"
                  className="hidden"
                  accept=".json,.yaml,.yml"
                  onChange={(e) => e.target.files?.[0] && setTrainingConfig(e.target.files[0])}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div
          className="rounded-xl px-5 py-3.5 text-sm"
          style={{
            background: 'rgba(200,60,60,0.08)',
            border: '1px solid rgba(200,80,80,0.2)',
            color: 'rgba(230,160,160,0.9)',
          }}
        >
          {error}
        </div>
      )}

      {/* Submit */}
      <button
        type="submit"
        disabled={loading}
        className="chrome-btn-primary w-full rounded-xl py-4 text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
        style={{
          fontFamily: 'Orbitron, sans-serif',
          letterSpacing: '0.12em',
          fontSize: '0.75rem',
          color: 'rgba(220,225,255,0.9)',
        }}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-3">
            <svg
              className="animate-spin"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
            Running Pipeline...
          </span>
        ) : (
          'Build'
        )}
      </button>
    </form>
  );
}
