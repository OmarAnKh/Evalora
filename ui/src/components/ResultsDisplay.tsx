import { useState, type ReactNode } from 'react';
import { BarChart2, Brain, CheckCircle, Database, GitCompare, Layers, Route, TrendingUp } from 'lucide-react';
import type { PipelineTrainEvaluateResponse } from '../lib/api';

interface Props {
  result: PipelineTrainEvaluateResponse;
}

type MetricGroup = 'score' | 'rationale';

type EvaluationVariant = {
  model_name?: string;
  model_variant?: string;
  split?: string;
  num_examples?: number;
  metrics?: Partial<Record<MetricGroup, Record<string, unknown>>>;
};

type EvaluationComparison = {
  upload_id?: string;
  split?: string;
  baseline?: EvaluationVariant;
  finetuned?: EvaluationVariant;
};

function fmt(v: unknown): string {
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(4);
  if (typeof v === 'boolean') return v ? 'Yes' : 'No';
  if (v == null) return '--';
  return String(v);
}

function isNumeric(value: unknown): value is number {
  return typeof value === 'number';
}

function scoreLabel(metric: string): string {
  return metric.replace(/_/g, ' ');
}

function getMetricGroups(evaluation: EvaluationComparison | undefined): Array<{
  key: MetricGroup;
  title: string;
  metrics: Array<{ key: string; baseline: unknown; finetuned: unknown }>;
}> {
  const groups: Array<{ key: MetricGroup; title: string }> = [
    { key: 'score', title: 'Score Metrics' },
    { key: 'rationale', title: 'Rationale Metrics' },
  ];

  return groups
    .map(({ key, title }) => {
      const baseline = evaluation?.baseline?.metrics?.[key] ?? {};
      const finetuned = evaluation?.finetuned?.metrics?.[key] ?? {};
      const keys = Array.from(new Set([...Object.keys(baseline), ...Object.keys(finetuned)]));

      return {
        key,
        title,
        metrics: keys
          .map((metricKey) => ({
            key: metricKey,
            baseline: baseline[metricKey],
            finetuned: finetuned[metricKey],
          }))
          .filter(({ baseline: base, finetuned: fine }) => isNumeric(base) || isNumeric(fine)),
      };
    })
    .filter(({ metrics }) => metrics.length > 0);
}

function formatChartValue(value: unknown): string {
  return fmt(value);
}

function formatDelta(baseline: unknown, finetuned: unknown): string {
  if (!isNumeric(baseline) || !isNumeric(finetuned)) return '--';
  const delta = finetuned - baseline;
  const prefix = delta > 0 ? '+' : '';
  return `${prefix}${delta.toFixed(4)}`;
}

function metricId(groupKey: string, metricKey: string): string {
  return `${groupKey}:${metricKey}`;
}

function SectionHeader({
  icon,
  title,
  tone = 'accent',
}: {
  icon: ReactNode;
  title: string;
  tone?: 'accent' | 'success';
}) {
  return (
    <div className="flex items-center gap-2.5">
      <div
        className="section-icon"
        style={{
          background: tone === 'success' ? 'var(--success-dim)' : 'var(--accent-dim)',
          border: '1px solid var(--panel-border)',
          color: tone === 'success' ? 'var(--success)' : 'var(--accent)',
        }}
      >
        {icon}
      </div>
      <h3 className="font-brand text-xs tracking-widest uppercase" style={{ color: 'var(--text-secondary)' }}>{title}</h3>
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="result-detail-row">
      <span>{label}</span>
      <strong className="mono">{fmt(value)}</strong>
    </div>
  );
}

function ComparisonBarChart({
  evaluation,
  activeMetricId,
  onSelectMetric,
}: {
  evaluation: EvaluationComparison;
  activeMetricId: string | null;
  onSelectMetric: (metricId: string) => void;
}) {
  const groups = getMetricGroups(evaluation);
  const metrics = groups.flatMap((group) => group.metrics.map((metric) => ({ ...metric, group: group.title })));


  if (!metrics.length) return null;

  return (
    <div className="chart-card w-full lg:col-span-2 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="font-brand text-xs tracking-widest uppercase" style={{ color: 'var(--text-secondary)' }}>
            Model Comparison
          </p>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: 3 }}>
            Baseline and finetuned scores across selected evaluation metrics.
          </p>
        </div>
        <div className="flex items-center gap-3 text-[0.64rem]" style={{ color: 'var(--text-muted)' }}>
          <span className="flex items-center gap-1.5">
            <span className="chart-legend-swatch" style={{ background: 'var(--baseline-gradient)' }} />
            Baseline
          </span>
          <span className="flex items-center gap-1.5">
            <span className="chart-legend-swatch" style={{ background: 'var(--finetuned-gradient)' }} />
            Finetuned
          </span>
        </div>
      </div>

      <div className="space-y-4">
        {groups.map(({ key: groupKey, title, metrics: groupMetrics }) => (
          <div key={groupKey} className="space-y-2">
            <p style={{ color: 'var(--text-muted)', fontSize: '0.62rem', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              {title}
            </p>
            <div className="space-y-3">
              {groupMetrics.map(({ key, baseline, finetuned }) => {
               const metricMax = Math.max(
                  1,
                  ...(isNumeric(baseline) ? [Math.abs(baseline)] : []),
                  ...(isNumeric(finetuned) ? [Math.abs(finetuned)] : [])
                );

                const baselineWidth = isNumeric(baseline)
                  ? `${(Math.abs(baseline) / metricMax) * 100}%`
                  : '0%';

                const finetunedWidth = isNumeric(finetuned)
                  ? `${(Math.abs(finetuned) / metricMax) * 100}%`
                  : '0%';
                const delta = isNumeric(baseline) && isNumeric(finetuned) ? finetuned - baseline : null;
                const active = activeMetricId === metricId(groupKey, key);

                return (
                  <button
                    key={`${groupKey}-${key}`}
                    type="button"
                    className={`chart-compare-row ${active ? 'is-active' : ''}`}
                    onClick={() => onSelectMetric(metricId(groupKey, key))}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <span className="chart-label" title={`${title}: ${scoreLabel(key)}`}>{scoreLabel(key)}</span>
                      <span className={`chart-delta ${delta != null && delta >= 0 ? 'is-positive' : 'is-negative'}`}>
                        {formatDelta(baseline, finetuned)}
                      </span>
                    </div>
                    <div className="chart-score-row">
                      <div className="chart-score-pill chart-score-pill-baseline">
                        <span className="chart-score-pill-label">Baseline</span>
                        <span className="chart-score-pill-value">{formatChartValue(baseline)}</span>
                      </div>
                      <div className="chart-score-pill chart-score-pill-finetuned">
                        <span className="chart-score-pill-label">Finetuned</span>
                        <span className="chart-score-pill-value">{formatChartValue(finetuned)}</span>
                      </div>
                    </div>
                    <div className="space-y-1.5">
                      <div className="chart-track">
                        <div className="chart-fill chart-fill-baseline" style={{ width: baselineWidth }} />
                      </div>
                      <div className="chart-track">
                        <div className="chart-fill chart-fill-finetuned" style={{ width: finetunedWidth }} />
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EvaluationSection({ evaluation }: { evaluation: EvaluationComparison }) {
  const [activeMetricId, setActiveMetricId] = useState<string | null>(null);
  const modelName = evaluation.baseline?.model_name || evaluation.finetuned?.model_name;
  const metricGroups = getMetricGroups(evaluation);
  const metricCount = metricGroups.reduce((total, group) => total + group.metrics.length, 0);
  const numericMetrics = metricGroups.flatMap((group) => group.metrics);
  const improvedCount = numericMetrics.filter(({ baseline, finetuned }) => isNumeric(baseline) && isNumeric(finetuned) && finetuned >= baseline).length;
  const bestLift = numericMetrics.reduce<{ label: string; value: number } | null>((best, metric) => {
    if (!isNumeric(metric.baseline) || !isNumeric(metric.finetuned)) return best;
    const value = metric.finetuned - metric.baseline;
    if (!best || value > best.value) return { label: scoreLabel(metric.key), value };
    return best;
  }, null);
  const activeMetric = activeMetricId
    ? metricGroups
        .flatMap((group) => group.metrics.map((metric) => ({ ...metric, groupKey: group.key, groupTitle: group.title })))
        .find((metric) => metricId(metric.groupKey, metric.key) === activeMetricId)
    : null;

  if (!metricGroups.length && !modelName) return null;

  return (
    <div className="card-solid result-section space-y-5">
      <SectionHeader icon={<BarChart2 size={13} />} title="Evaluation Metrics" tone="success" />

      <div className="evaluation-overview">
        <div className="evaluation-model-card">
          <span>Model</span>
          <strong className="mono">{modelName || 'Evaluation model'}</strong>
        </div>
        <div className="evaluation-kpi-grid">
          <div className="evaluation-kpi">
            <span>Metrics</span>
            <strong className="mono">{metricCount}</strong>
          </div>
          <div className="evaluation-kpi">
            <span>Non-decreasing</span>
            <strong className="mono">{improvedCount}</strong>
          </div>
          <div className="evaluation-kpi">
            <span>Best Lift</span>
            <strong className="mono">{bestLift ? `${bestLift.label} ${bestLift.value >= 0 ? '+' : ''}${bestLift.value.toFixed(4)}` : '--'}</strong>
          </div>
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <ComparisonBarChart evaluation={evaluation} activeMetricId={activeMetricId} onSelectMetric={setActiveMetricId} />
      </div>

      {activeMetric && (
        <div className="selected-metric-panel">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
            <div className="selected-metric-title">
              <TrendingUp size={14} />
              <div>
                <p>Selected Metric</p>
                <strong className="font-brand">{scoreLabel(activeMetric.key)}</strong>
              </div>
            </div>
            <button type="button" className="btn" style={{ padding: '6px 10px', fontSize: '0.72rem' }} onClick={() => setActiveMetricId(null)}>
              Clear
            </button>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <div className="metric">
              <p className="metric-label">Baseline</p>
              <p className={`metric-value ${typeof activeMetric.baseline === 'number' ? 'mono' : ''}`}>{fmt(activeMetric.baseline)}</p>
            </div>
            <div className="metric">
              <p className="metric-label">Finetuned</p>
              <p className={`metric-value ${typeof activeMetric.finetuned === 'number' ? 'mono' : ''}`}>{fmt(activeMetric.finetuned)}</p>
            </div>
            <div className="metric">
              <p className="metric-label">Delta</p>
              <p className="metric-value mono">{formatDelta(activeMetric.baseline, activeMetric.finetuned)}</p>
            </div>
            <div className="metric sm:col-span-3">
              <p className="metric-label">Group</p>
              <p className="metric-value" style={{ fontSize: '0.88rem' }}>{activeMetric.groupTitle}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function TrainingSection({ data }: { data: Record<string, unknown> }) {
  const excluded = new Set(['status', 'output_dir', 'config_path']);
  const entries = Object.entries(data).filter(([key, value]) => !excluded.has(key) && value != null && typeof value !== 'object');
  const nested = Object.entries(data).filter(([key, value]) => !excluded.has(key) && value != null && typeof value === 'object' && !Array.isArray(value));

  if (!entries.length && !nested.length) return null;

  return (
    <div className="card-solid result-section space-y-5">
      <SectionHeader icon={<Brain size={13} />} title="Training Metrics" />

      {entries.length > 0 && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {entries.map(([key, value]) => (
            <div key={key} className="metric metric-elevated">
              <p className="metric-label">{scoreLabel(key)}</p>
              <p className={`metric-value ${typeof value === 'number' ? 'mono' : ''}`}>{fmt(value)}</p>
            </div>
          ))}
        </div>
      )}

      {nested.map(([groupKey, groupValue]) => {
        const inner = Object.entries(groupValue as Record<string, unknown>).filter(([, value]) => value != null && typeof value !== 'object');
        if (!inner.length) return null;

        return (
          <div key={groupKey} className="space-y-3">
            <p style={{ color: 'var(--text-muted)', fontSize: '0.62rem', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              {scoreLabel(groupKey)}
            </p>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {inner.map(([key, value]) => (
                <div key={key} className="metric metric-elevated">
                  <p className="metric-label">{scoreLabel(key)}</p>
                  <p className={`metric-value ${typeof value === 'number' ? 'mono' : ''}`}>{fmt(value)}</p>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ResultHero({ result, evaluation }: { result: PipelineTrainEvaluateResponse; evaluation?: EvaluationComparison }) {
  const modelName = evaluation?.baseline?.model_name || evaluation?.finetuned?.model_name || 'Fine-tuned evaluator';
  const split = evaluation?.split || evaluation?.baseline?.split || evaluation?.finetuned?.split || 'test';
  const examples = evaluation?.finetuned?.num_examples ?? evaluation?.baseline?.num_examples;

  return (
    <div className="result-hero">
      <div className="result-hero-main">
        <div className="result-status-chip">
          <CheckCircle size={13} /> Pipeline completed
        </div>
        <h2 className="font-brand">Run results are ready</h2>
        <p>{modelName}</p>
      </div>
      <div className="result-hero-stats">
        <div>
          <span>Upload</span>
          <strong className="mono">{result.upload_id}</strong>
        </div>
        <div>
          <span>Split</span>
          <strong className="mono">{split}</strong>
        </div>
        <div>
          <span>Examples</span>
          <strong className="mono">{fmt(examples)}</strong>
        </div>
      </div>
    </div>
  );
}

export default function ResultsDisplay({ result }: Props) {
  const evaluation = result.evaluation as EvaluationComparison | undefined;

  return (
    <div className="w-full max-w-4xl mx-auto space-y-5 animate-fade-up">
      <ResultHero result={result} evaluation={evaluation} />

      <div className="grid gap-4 lg:grid-cols-[1.05fr_0.95fr]">
        <div className="card-solid result-section space-y-4">
          <SectionHeader icon={<Database size={13} />} title="Pipeline Info" />
          <div className="space-y-2">
            <DetailRow label="Result Path" value={result.pipeline.result_file_path} />
          </div>
        </div>

        <div className="card-solid result-section space-y-4">
          <SectionHeader icon={<GitCompare size={13} />} title="Run Summary" tone="success" />
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
            <div className="result-summary-tile">
              <Layers size={14} />
              <div>
                <span>Baseline</span>
                <strong className="mono">{evaluation?.baseline?.model_variant || 'base'}</strong>
              </div>
            </div>
            <div className="result-summary-tile">
              <Route size={14} />
              <div>
                <span>Finetuned</span>
                <strong className="mono">{evaluation?.finetuned?.model_variant || 'adapter'}</strong>
              </div>
            </div>
          </div>
        </div>
      </div>

      {evaluation && Object.keys(evaluation).length > 0 && <EvaluationSection evaluation={evaluation} />}

      {result.training && Object.keys(result.training).length > 0 && (
        <TrainingSection data={result.training} />
      )}
    </div>
  );
}
