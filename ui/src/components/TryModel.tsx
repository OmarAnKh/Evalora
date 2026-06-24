import { useState } from 'react';
import { MessageSquare, Plus, Trash2, Sparkles } from 'lucide-react';
import { tryModel, type ModelInferenceResponse, type RubricItem } from '../lib/api';

interface Props {
  uploadId: string;
  modelName?: string;
}

export default function TryModel({ uploadId, modelName}: Props) {
  const [referenceAnswer, setReferenceAnswer] = useState('');
  const [answer, setAnswer] = useState('');
  const [task, setTask] = useState('');
  const [rubric, setRubric] = useState<RubricItem[]>([
    { criterion: '', description: '', weight: 1 },
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ModelInferenceResponse | null>(null);

  const addRubricRow = () => setRubric([...rubric, { criterion: '', description: '', weight: 1 }]);
  const removeRubricRow = (i: number) => {
    if (rubric.length > 1) setRubric(rubric.filter((_, idx) => idx !== i));
  };
  const updateRubric = (i: number, field: keyof RubricItem, val: string | number) => {
    const next = [...rubric];
    (next[i] as Record<string, string | number>)[field] = val;
    setRubric(next);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!referenceAnswer.trim() || !answer.trim()) {
      setError('Reference answer and answer are required.');
      return;
    }
    const validRubric = rubric.filter((r) => r.criterion.trim() && r.description.trim());
    if (!validRubric.length) {
      setError('Add at least one rubric criterion.');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await tryModel({
        upload_id: uploadId,
        reference_answer: referenceAnswer,
        answer,
        rubric: validRubric,
        task: task.trim() || undefined,
        model_name: modelName || undefined,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Inference failed.');
      console.log(uploadId);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto animate-fade-up">
      <div className="card-solid p-5 space-y-5">
        <div className="flex items-center gap-2.5">
          <div
            className="section-icon"
            style={{ background: 'var(--accent-dim)', border: '1px solid var(--border)' }}
          >
            <MessageSquare size={13} style={{ color: 'var(--accent)', opacity: 0.8 }} />
          </div>
          <h3
            className="font-brand text-xs tracking-widest uppercase"
            style={{ color: 'var(--text-secondary)' }}
          >
            Try Model
          </h3>
        </div>

        <p style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>
          Test your fine-tuned model with a single prediction. Provide a reference answer, a
          student answer, and a scoring rubric.
        </p>

        {result ? (
          <div className="space-y-4">
            <div className="card p-4 space-y-3" style={{ borderColor: 'var(--primary)' }}>
              <div className="flex items-center gap-2">
                <Sparkles size={14} style={{ color: 'var(--accent)' }} />
                <span style={{ color: 'var(--accent)', fontWeight: 500, fontSize: '0.82rem' }}>
                  Model Response
                </span>
              </div>
              {result.score != null && (
                <div className="metric">
                  <p className="metric-label">Score</p>
                  <p
                    className="metric-value mono"
                    style={{ color: 'var(--accent)', fontSize: '1.8rem' }}
                  >
                    {result.score}
                  </p>
                </div>
              )}
              <div>
                <p className="metric-label" style={{ marginBottom: 6 }}>
                  Reasoning
                </p>
                <p
                  style={{
                    color: 'var(--text-secondary)',
                    fontSize: '0.82rem',
                    lineHeight: 1.6,
                    whiteSpace: 'pre-wrap',
                  }}
                >
                  {result.reasoning}
                </p>
              </div>
            </div>
            <button onClick={() => setResult(null)} className="btn w-full">
              Try Another
            </button>
          </div>
        ) : (
          <form onSubmit={submit} className="space-y-4">
            <div>
              <label className="label">Task Prompt (optional)</label>
              <input
                className="input"
                placeholder="Evaluation task description..."
                value={task}
                onChange={(e) => setTask(e.target.value)}
              />
            </div>
            <div>
              <label className="label">Reference Answer *</label>
              <textarea
                className="input"
                rows={3}
                placeholder="The correct / reference answer..."
                value={referenceAnswer}
                onChange={(e) => setReferenceAnswer(e.target.value)}
                style={{ resize: 'vertical' }}
              />
            </div>
            <div>
              <label className="label">Answer to Evaluate *</label>
              <textarea
                className="input"
                rows={3}
                placeholder="The student / model answer..."
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                style={{ resize: 'vertical' }}
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="label" style={{ marginBottom: 0 }}>
                  Scoring Rubric *
                </span>
                <button
                  type="button"
                  onClick={addRubricRow}
                  className="btn"
                  style={{ padding: '4px 10px', fontSize: '0.7rem' }}
                >
                  <Plus size={12} /> Add
                </button>
              </div>
              <div className="space-y-2.5">
                {rubric.map((r, i) => (
                  <div
                    key={i}
                    className="grid gap-2"
                    style={{ gridTemplateColumns: '1fr 1.5fr 64px 28px' }}
                  >
                    <input
                      className="input"
                      placeholder="Criterion"
                      value={r.criterion}
                      onChange={(e) => updateRubric(i, 'criterion', e.target.value)}
                      style={{ fontSize: '0.78rem' }}
                    />
                    <input
                      className="input"
                      placeholder="Description"
                      value={r.description}
                      onChange={(e) => updateRubric(i, 'description', e.target.value)}
                      style={{ fontSize: '0.78rem' }}
                    />
                    <input
                      type="number"
                      min={0}
                      step={0.1}
                      className="input"
                      value={r.weight}
                      onChange={(e) => updateRubric(i, 'weight', Number(e.target.value))}
                      style={{ fontSize: '0.78rem', textAlign: 'center', padding: '4px 8px' }}
                    />
                    <button
                      type="button"
                      onClick={() => removeRubricRow(i)}
                      className="flex items-center justify-center rounded-lg hover:bg-white/5 transition-colors"
                      style={{ color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {error && <div className="error-banner">{error}</div>}

            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary w-full py-3 font-brand tracking-wider"
            >
              {loading ? (
                <>
                  <div className="spinner" /> Evaluating...
                </>
              ) : (
                <>
                  <Sparkles size={14} /> Run Inference
                </>
              )}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}