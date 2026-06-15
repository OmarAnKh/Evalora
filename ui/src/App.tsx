import { useState, useCallback, useEffect } from 'react';
import { Moon, RotateCcw, ChevronRight, MessageSquare, SunMedium, Upload } from 'lucide-react';
import SplashScreen from './components/SplashScreen';
import PipelineForm from './components/PipelineForm';
import ResultsDisplay from './components/ResultsDisplay';
import TryModel from './components/TryModel';
import HuggingFaceUpload from './components/HuggingFaceUpload';
import type { PipelineTrainEvaluateResponse } from './lib/api';

type View = 'pipeline' | 'results';
type NextStep = 'none' | 'try' | 'upload';
type Theme = 'dark' | 'light';

const THEME_STORAGE_KEY = 'evalora-theme';

function getInitialTheme(): Theme {
  if (typeof window === 'undefined') return 'dark';

  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (storedTheme === 'dark' || storedTheme === 'light') return storedTheme;

  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [view, setView] = useState<View>('pipeline');
  const [result, setResult] = useState<PipelineTrainEvaluateResponse | null>(null);
  const [nextStep, setNextStep] = useState<NextStep>('none');
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    const root = document.documentElement;
    root.dataset.theme = theme;
    root.style.colorScheme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

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

  const toggleTheme = useCallback(() => {
    setTheme((currentTheme) => (currentTheme === 'dark' ? 'light' : 'dark'));
  }, []);

  // All hooks are above this line. Early return is safe now.
  if (showSplash) return <SplashScreen onFinish={dismissSplash} />;

  return (
    <div className="min-h-screen flex flex-col" style={{ background: 'var(--page-bg)' }}>
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse 70% 30% at 50% -5%, var(--page-glow) 0%, transparent 70%)',
        }}
      />

      <header
        className="sticky top-0 flex items-center justify-between px-5 py-3"
        style={{
          background: 'var(--header-bg)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
          borderBottom: '1px solid var(--border)',
          zIndex: 50,
        }}
      >
        <div className="flex items-center gap-3">
          <div className="section-icon" style={{ background: 'var(--accent-dim)', border: '1px solid var(--border-focus)' }}>
            <span className="font-brand" style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent)' }}>E</span>
          </div>
          <h1
            className="font-brand"
            style={{
              fontSize: '1.05rem',
              fontWeight: 700,
              letterSpacing: '0.04em',
              background: 'var(--brand-gradient)',
              WebkitBackgroundClip: 'text',
              backgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Evalora
          </h1>
          {view === 'results' && (
            <div className="flex items-center gap-1.5">
              <ChevronRight size={10} style={{ color: 'var(--text-muted)' }} />
              <span className="font-brand text-xs tracking-wider uppercase" style={{ color: 'var(--text-muted)' }}>
                Results
              </span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={toggleTheme}
            className="icon-btn"
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            aria-pressed={theme === 'light'}
            style={{ fontSize: '0.72rem' }}
          >
            {theme === 'dark' ? <SunMedium size={13} /> : <Moon size={13} />}
            <span className={`toggle-track ${theme === 'light' ? 'active' : ''}`} aria-hidden="true" style={{ marginLeft: 6 }}>
              <span className="toggle-knob" />
            </span>
          </button>

          {view === 'results' && (
            <button onClick={handleReset} className="btn" style={{ padding: '6px 12px', fontSize: '0.72rem' }}>
              <RotateCcw size={11} /> New Run
            </button>
          )}
        </div>
      </header>

      <main className="flex-1 px-4 py-8">
        {view === 'pipeline' && <PipelineForm onResult={handleResult} />}

        {view === 'results' && result && (
          <div className="max-w-4xl mx-auto space-y-6">
            <ResultsDisplay result={result} />

            <div className="divider" />

            {nextStep === 'none' && (
              <div className="card p-5 text-center space-y-4 animate-fade-up">
                <p className="font-brand text-xs tracking-widest uppercase" style={{ color: 'var(--text-secondary)' }}>
                  What's Next?
                </p>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                  Test your model with a custom prompt or upload it to HuggingFace Hub.
                </p>
                <div className="flex gap-3 justify-center flex-wrap">
                  <button onClick={() => setNextStep('try')} className="btn btn-primary" style={{ padding: '10px 24px' }}>
                    <MessageSquare size={14} /> Try Model
                  </button>
                  <button onClick={() => setNextStep('upload')} className="btn" style={{ padding: '10px 24px' }}>
                    <Upload size={14} /> Upload to HuggingFace
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
                  style={{ fontSize: '0.75rem' }}
                >
                  {nextStep === 'try' ? 'Upload to HuggingFace instead' : 'Try Model instead'}
                </button>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="text-center py-4 mt-auto" style={{ borderTop: '1px solid var(--border)' }}>
        <span className="font-brand" style={{ color: 'var(--text-muted)', fontSize: '0.6rem', letterSpacing: '0.15em' }}>
          EVALORA &middot; TEACH AI HOW TO EVALUATE
        </span>
      </footer>
    </div>
  );
}
