import { CheckCircle, Hash, BarChart2, Brain, Database } from 'lucide-react';
import type { PipelineTrainEvaluateResponse } from '../lib/api';

interface Props {
  result: PipelineTrainEvaluateResponse;
}

type EvaluationMetrics = {
  baseline?: {
    model_name?: string;
    metrics?: {
      score?: Record<string, unknown>;
      rationale?: Record<string, unknown>;
    };
  };
  finetuned?: {
    model_name?: string;
    metrics?: {
      score?: Record<string, unknown>;
      rationale?: Record<string, unknown>;
    };
  };
};

function formatValue(val: unknown): string {
  if (typeof val === 'number') {
    return Number.isInteger(val) ? String(val) : val.toFixed(4);
  }
  if (typeof val === 'boolean') return val ? 'Yes' : 'No';
  if (val === null || val === undefined) return '—';
  return String(val);
}

function isNumeric(val: unknown): val is number {
  return typeof val === 'number';
}

function getMetricComparison(
  evaluation: EvaluationMetrics | undefined,
  group: 'score' | 'rationale'
) {
  const baseline = evaluation?.baseline?.metrics?.[group] || {};
  const finetuned = evaluation?.finetuned?.metrics?.[group] || {};
  const keys = Array.from(new Set([...Object.keys(baseline), ...Object.keys(finetuned)]));

  if (keys.length === 0) {
    return null;
  }

  return {
    baseline,
    finetuned,
    keys,
  };
}

function MetricCard({ label, value, fullWidth = false }: { label: string; value: unknown; fullWidth?: boolean }) {
  const numeric = isNumeric(value);
  const pct = numeric && (label.toLowerCase().includes('score') || label.toLowerCase().includes('accuracy') || label.toLowerCase().includes('kappa'));
  const displayVal = formatValue(value);

  return (
    <div
      className="metric-card rounded-xl p-4 flex flex-col gap-1.5"
      style={{ minWidth: 0, gridColumn: fullWidth ? '1 / -1' : undefined }}
    >
      <p style={{ color: 'rgba(140,145,175,0.65)', fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
        {label.replace(/_/g, ' ')}
      </p>
      <p
        style={{
          fontFamily: numeric ? 'Orbitron, sans-serif' : 'Inter, sans-serif',
          fontSize: numeric ? '1.25rem' : '0.875rem',
          fontWeight: 600,
          color: numeric ? 'rgba(210,215,240,0.95)' : 'rgba(190,195,220,0.8)',
          letterSpacing: numeric ? '0.02em' : 0,
        }}
      >
        {displayVal}{pct && numeric && Number(value) <= 1 ? '' : ''}
      </p>
    </div>
  );
}

function shouldSpanFullRow(label: string): boolean {
  return label.toLowerCase().includes('output_dir');
}

function SectionBlock({
  icon,
  title,
  data,
  accentColor = 'rgba(180,185,220,0.15)',
  excludeKeys = [],
}: {
  icon: React.ReactNode;
  title: string;
  data: Record<string, unknown>;
  accentColor?: string;
  excludeKeys?: string[];
}) {
  const exclude = new Set(excludeKeys.map((key) => key.toLowerCase()));
  const shouldInclude = (key: string) => !exclude.has(key.toLowerCase());
  const entries = Object.entries(data).filter(
    ([k, v]) => shouldInclude(k) && v !== null && v !== undefined && typeof v !== 'object'
  );
  const nested = Object.entries(data).filter(
    ([k, v]) => shouldInclude(k) && v !== null && typeof v === 'object' && !Array.isArray(v)
  );

  if (entries.length === 0 && nested.length === 0) {
    return null;
  }

  return (
    <div
      className="rounded-2xl p-5 space-y-4"
      style={{
        background: 'rgba(16,16,24,0.7)',
        border: '1px solid rgba(180,180,220,0.1)',
        boxShadow: '0 2px 20px rgba(0,0,0,0.4)',
      }}
    >
      <div className="flex items-center gap-2.5">
        <div
          className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
          style={{ background: accentColor, border: '1px solid rgba(255,255,255,0.06)' }}
        >
          {icon}
        </div>
        <h3
          style={{
            fontFamily: 'Orbitron, sans-serif',
            fontSize: '0.7rem',
            letterSpacing: '0.12em',
            color: 'rgba(190,195,225,0.85)',
            textTransform: 'uppercase',
          }}
        >
          {title}
        </h3>
      </div>

      {entries.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {entries.map(([k, v]) => (
            <MetricCard key={k} label={k} value={v} fullWidth={shouldSpanFullRow(k)} />
          ))}
        </div>
      )}

      {nested.map(([groupKey, groupVal]) => (
        <div key={groupKey} className="space-y-2">
          <p
            style={{
              color: 'rgba(150,155,185,0.55)',
              fontSize: '0.65rem',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
            }}
          >
            {groupKey.replace(/_/g, ' ')}
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {Object.entries(groupVal as Record<string, unknown>)
              .filter(([k, v]) => shouldInclude(k) && v !== null && v !== undefined && typeof v !== 'object')
              .map(([k, v]) => (
                <MetricCard key={k} label={k} value={v} fullWidth={shouldSpanFullRow(k)} />
              ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function ResultsDisplay({ result }: Props) {
  const evaluation = result.evaluation as EvaluationMetrics | undefined;
  const scoreComparison = getMetricComparison(evaluation, 'score');
  const rationaleComparison = getMetricComparison(evaluation, 'rationale');
  const modelName = evaluation?.finetuned?.model_name || evaluation?.baseline?.model_name;

  return (
    <div className="w-full max-w-3xl mx-auto space-y-5 page-animate-in">
      {/* Success banner */}
      <div
        className="rounded-2xl px-6 py-5 flex items-center gap-4"
        style={{
          background: 'rgba(40,80,55,0.2)',
          border: '1px solid rgba(80,180,110,0.2)',
          boxShadow: '0 4px 30px rgba(0,0,0,0.5), 0 0 40px rgba(60,160,90,0.04)',
        }}
      >
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
          style={{
            background: 'rgba(70,180,110,0.15)',
            border: '1px solid rgba(80,200,120,0.2)',
          }}
        >
          <CheckCircle size={20} style={{ color: 'rgba(100,220,140,0.85)' }} />
        </div>
        <div className="min-w-0">
          <p style={{ color: 'rgba(160,230,180,0.9)', fontSize: '0.9rem', fontWeight: 600 }}>
            Pipeline Completed Successfully
          </p>
          <p style={{ color: 'rgba(100,160,120,0.6)', fontSize: '0.75rem', marginTop: 2 }}>
            Experiment ID &nbsp;
            <span
              style={{
                fontFamily: 'monospace',
                color: 'rgba(140,200,160,0.7)',
                background: 'rgba(60,120,80,0.15)',
                padding: '1px 6px',
                borderRadius: 4,
              }}
            >
              {result.upload_id}
            </span>
          </p>
        </div>
      </div>

      {/* Pipeline info */}
      <div
        className="rounded-2xl p-5"
        style={{
          background: 'rgba(16,16,24,0.7)',
          border: '1px solid rgba(180,180,220,0.1)',
          boxShadow: '0 2px 20px rgba(0,0,0,0.4)',
        }}
      >
        <div className="flex items-center gap-2.5 mb-4">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center"
            style={{
              background: 'rgba(120,130,180,0.15)',
              border: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            <Database size={13} style={{ color: 'rgba(170,180,220,0.7)' }} />
          </div>
          <h3
            style={{
              fontFamily: 'Orbitron, sans-serif',
              fontSize: '0.7rem',
              letterSpacing: '0.12em',
              color: 'rgba(190,195,225,0.85)',
              textTransform: 'uppercase',
            }}
          >
            Pipeline Info
          </h3>
        </div>
        <div className="space-y-2">
          <div className="flex items-center justify-between py-2" style={{ borderBottom: '1px solid rgba(180,180,220,0.06)' }}>
            <span style={{ color: 'rgba(130,135,170,0.6)', fontSize: '0.78rem' }}>Upload ID</span>
            <span style={{ fontFamily: 'monospace', fontSize: '0.78rem', color: 'rgba(200,205,235,0.8)' }}>
              {result.pipeline.upload_id}
            </span>
          </div>
          <div className="flex items-center justify-between py-2">
            <span style={{ color: 'rgba(130,135,170,0.6)', fontSize: '0.78rem' }}>Result File</span>
            <span style={{ fontFamily: 'monospace', fontSize: '0.75rem', color: 'rgba(170,175,210,0.7)', wordBreak: 'break-all', textAlign: 'right', maxWidth: '60%' }}>
              {result.pipeline.result_file_path}
            </span>
          </div>
        </div>
      </div>

      {/* Training Results */}
      {result.training && Object.keys(result.training).length > 0 && (
        <SectionBlock
          icon={<Brain size={13} style={{ color: 'rgba(170,175,220,0.7)' }} />}
          title="Training Results"
          data={result.training}
          accentColor="rgba(100,110,180,0.2)"
          excludeKeys={["status", "config_path", "total_flos"]}
        />
      )}

      {/* Evaluation Results */}
      {(scoreComparison || rationaleComparison) && (
        <div
          className="rounded-2xl p-5 space-y-5"
          style={{
            background: 'rgba(16,16,24,0.7)',
            border: '1px solid rgba(180,180,220,0.1)',
            boxShadow: '0 2px 20px rgba(0,0,0,0.4)',
          }}
        >
          <div className="flex items-center gap-2.5">
            <div
              className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ background: 'rgba(80,160,100,0.15)', border: '1px solid rgba(255,255,255,0.06)' }}
            >
              <BarChart2 size={13} style={{ color: 'rgba(170,220,175,0.7)' }} />
            </div>
            <h3
              style={{
                fontFamily: 'Orbitron, sans-serif',
                fontSize: '0.7rem',
                letterSpacing: '0.12em',
                color: 'rgba(190,195,225,0.85)',
                textTransform: 'uppercase',
              }}
            >
              Evaluation Comparison
            </h3>
          </div>

          {modelName && (
            <div
              className="rounded-xl px-4 py-2"
              style={{
                background: 'rgba(12,12,18,0.6)',
                border: '1px solid rgba(180,180,220,0.08)',
                color: 'rgba(180,185,220,0.8)',
                fontSize: '0.78rem',
                fontFamily: 'Inter, sans-serif',
              }}
            >
              Model: <span style={{ color: 'rgba(210,215,240,0.9)' }}>{modelName}</span>
            </div>
          )}

          {([
            { title: 'Score Metrics', comparison: scoreComparison },
            { title: 'Rationale Metrics', comparison: rationaleComparison },
          ] as const)
            .filter(({ comparison }) => comparison)
            .map(({ title, comparison }) => (
              <div key={title} className="space-y-3">
                <p
                  style={{
                    color: 'rgba(150,155,185,0.55)',
                    fontSize: '0.65rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.1em',
                  }}
                >
                  {title}
                </p>

                <div className="grid grid-cols-3 gap-3 text-xs" style={{ color: 'rgba(130,135,170,0.6)' }}>
                  <span>Metric</span>
                  <span>Baseline</span>
                  <span>Finetuned</span>
                </div>

                <div className="space-y-2">
                  {comparison?.keys.map((key) => (
                    <div
                      key={key}
                      className="grid grid-cols-3 gap-3 items-center rounded-xl px-3 py-2"
                      style={{
                        background: 'rgba(12,12,18,0.55)',
                        border: '1px solid rgba(180,180,220,0.06)',
                      }}
                    >
                      <span style={{ color: 'rgba(170,175,210,0.7)', fontSize: '0.78rem' }}>
                        {key.replace(/_/g, ' ')}
                      </span>
                      <span style={{ color: 'rgba(200,205,235,0.8)', fontSize: '0.78rem' }}>
                        {formatValue(comparison?.baseline[key])}
                      </span>
                      <span style={{ color: 'rgba(200,225,210,0.85)', fontSize: '0.78rem' }}>
                        {formatValue(comparison?.finetuned[key])}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
        </div>
      )}

      {/* Upload ID highlight */}
      <div
        className="rounded-2xl px-5 py-4 flex items-center gap-3"
        style={{
          background: 'rgba(20,20,30,0.8)',
          border: '1px solid rgba(180,180,220,0.12)',
        }}
      >
        <Hash size={14} style={{ color: 'rgba(160,165,200,0.5)', flexShrink: 0 }} />
        <span style={{ color: 'rgba(140,145,180,0.6)', fontSize: '0.78rem' }}>Experiment ID for HuggingFace upload:</span>
        <span
          style={{
            fontFamily: 'monospace',
            fontSize: '0.82rem',
            color: 'rgba(200,205,240,0.9)',
            background: 'rgba(180,185,240,0.06)',
            padding: '2px 8px',
            borderRadius: 6,
            border: '1px solid rgba(180,185,240,0.1)',
          }}
        >
          {result.upload_id}
        </span>
      </div>
    </div>
  );
}
