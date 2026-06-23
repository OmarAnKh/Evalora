import { useState, useCallback, useEffect } from 'react';
import { RotateCcw, ChevronRight, MessageSquare, Upload, Sun, Moon } from 'lucide-react';
import SplashScreen from './components/SplashScreen';
import PipelineForm from './components/PipelineForm';
import ResultsDisplay from './components/ResultsDisplay';
import TryModel from './components/TryModel';
import HuggingFaceUpload from './components/HuggingFaceUpload';
import type { PipelineTrainEvaluateResponse } from './lib/api';

type View = 'pipeline' | 'results';
type NextStep = 'none' | 'try' | 'upload';
type Theme = 'light' | 'dark';

export default function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [view, setView] = useState<View>('pipeline');
  const [result, setResult] = useState<PipelineTrainEvaluateResponse | null>(null);
  const [nextStep, setNextStep] = useState<NextStep>('none');
  const [theme, setTheme] = useState<Theme>(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('evalora-theme');
      if (stored === 'light' || stored === 'dark') return stored;
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    return 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('evalora-theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  }, []);

  const handleResult = useCallback((res: PipelineTrainEvaluateResponse) => {
    setResult(res);
    setNextStep('none');
    setView('results');
  }, []);

  const dismissSplash = useCallback(() => setShowSplash(false), []);

  const handleReset = useCallback(() => {
    setResult(null);
    setNextStep('none');
    setView('pipeline');
  }, []);

  if (showSplash) return <SplashScreen onFinish={dismissSplash} />;

  return (
    <div className="min-h-screen flex flex-col relative" style={{ background: 'var(--bg-deep)' }}>
      {/* Gradient orbs background */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden" style={{ zIndex: 0 }}>
        {/* Primary coral orb */}
        <div
          style={{
            position: 'absolute',
            top: '-20%',
            right: '-10%',
            width: '50vw',
            height: '50vw',
            borderRadius: '50%',
            background: 'radial-gradient(circle, var(--primary-glow) 0%, transparent 70%)',
            filter: 'blur(80px)',
          }}
        />
        {/* Secondary teal orb */}
        <div
          style={{
            position: 'absolute',
            bottom: '10%',
            left: '-15%',
            width: '60vw',
            height: '60vw',
            borderRadius: '50%',
            background: 'radial-gradient(circle, var(--secondary-dim) 0%, transparent 70%)',
            filter: 'blur(100px)',
          }}
        />
        {/* Accent purple orb */}
        <div
          style={{
            position: 'absolute',
            top: '40%',
            right: '20%',
            width: '40vw',
            height: '40vw',
            borderRadius: '50%',
            background: 'radial-gradient(circle, var(--accent-dim) 0%, transparent 60%)',
            filter: 'blur(60px)',
          }}
        />
      </div>

      <header
        className="sticky top-0 flex items-center justify-between px-6 py-4"
        style={{
          background: 'var(--bg-surface)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          borderBottom: '1px solid var(--border)',
          zIndex: 50,
        }}
      >
        <div className="flex items-center gap-4">
          <div
            className="section-icon"
            style={{
              background: 'linear-gradient(135deg, var(--primary-dim), var(--secondary-dim))',
              border: '1px solid var(--border)',
            }}
          >
            <span className="font-brand" style={{ fontSize: '0.9rem', fontWeight: 700, background: 'linear-gradient(135deg, var(--primary), var(--secondary))', WebkitBackgroundClip: 'text', backgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              E
            </span>
          </div>
          <h1
            className="font-brand"
            style={{
              fontSize: '1.25rem',
              fontWeight: 700,
              letterSpacing: '0.02em',
              background: 'linear-gradient(135deg, var(--primary), var(--secondary), var(--accent))',
              WebkitBackgroundClip: 'text',
              backgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Evalora
          </h1>
          {view === 'results' && (
            <div className="flex items-center gap-2">
              <ChevronRight size={12} style={{ color: 'var(--text-muted)' }} />
              <span className="font-brand text-xs tracking-wider uppercase" style={{ color: 'var(--text-muted)' }}>
                Results
              </span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={toggleTheme}
            className="btn"
            style={{ padding: '8px 12px', minWidth: 36 }}
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? <Sun size={16} style={{ color: 'var(--warning)' }} /> : <Moon size={16} style={{ color: 'var(--accent)' }} />}
          </button>
          {view === 'results' && (
            <button onClick={handleReset} className="btn" style={{ padding: '8px 16px', fontSize: '0.8rem' }}>
              <RotateCcw size={12} /> New Run
            </button>
          )}
        </div>
      </header>

      <main className="flex-1 px-6 py-10" style={{ position: 'relative', zIndex: 1 }}>
        {view === 'pipeline' && <PipelineForm onResult={handleResult} />}

        {view === 'results' && result && (
          <div className="max-w-3xl mx-auto space-y-8">
            <ResultsDisplay result={result} />

            <div className="divider" />

            {nextStep === 'none' && (
              <div className="card p-8 text-center space-y-6 animate-fade-up">
                <p className="font-brand text-sm tracking-widest uppercase" style={{ color: 'var(--text-secondary)' }}>
                  What's Next?
                </p>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', maxWidth: 400, margin: '0 auto' }}>
                  Test your model with a custom prompt or upload it to HuggingFace Hub.
                </p>
                <div className="flex gap-4 justify-center flex-wrap">
                  <button onClick={() => setNextStep('try')} className="btn btn-primary" style={{ padding: '12px 28px' }}>
                    <MessageSquare size={16} /> Try Model
                  </button>
                  <button onClick={() => setNextStep('upload')} className="btn btn-secondary" style={{ padding: '12px 28px' }}>
                    <Upload size={16} /> Upload to HuggingFace
                  </button>
                </div>
              </div>
            )}

            {nextStep === 'try' && <TryModel uploadId={result.upload_id} />}
            {nextStep === 'upload' && <HuggingFaceUpload uploadId={result.upload_id} />}

            {nextStep !== 'none' && (
              <div className="text-center">
                <button
                  onClick={() => setNextStep(nextStep === 'try' ? 'upload' : 'try')}
                  className="btn"
                  style={{ fontSize: '0.85rem' }}
                >
                  {nextStep === 'try' ? 'Upload to HuggingFace instead' : 'Try Model instead'}
                </button>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="text-center py-5 mt-auto" style={{ borderTop: '1px solid var(--border)', position: 'relative', zIndex: 1 }}>
        <span className="font-brand" style={{ color: 'var(--text-muted)', fontSize: '0.65rem', letterSpacing: '0.18em' }}>
          EVALORA &middot; TEACH AI HOW TO EVALUATE
        </span>
      </footer>
    </div>
  );
}