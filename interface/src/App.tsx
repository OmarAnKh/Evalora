import { useState, useCallback } from 'react';
import SplashScreen from './components/SplashScreen';
import PipelineForm from './components/PipelineForm';
import ResultsDisplay from './components/ResultsDisplay';
import HuggingFaceUpload from './components/HuggingFaceUpload';
import type { PipelineTrainEvaluateResponse } from './lib/api';
import { ArrowLeft, Server, RotateCcw, ChevronRight } from 'lucide-react';

type View = 'pipeline' | 'results';
type HFChoice = 'pending' | 'yes' | 'no';

export default function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [view, setView] = useState<View>('pipeline');
  const [result, setResult] = useState<PipelineTrainEvaluateResponse | null>(null);
  const [apiBaseUrl, setApiBaseUrl] = useState('');
  const [hfChoice, setHfChoice] = useState<HFChoice>('pending');

  const handleSplashDone = useCallback(() => setShowSplash(false), []);

  const handleResult = useCallback((res: PipelineTrainEvaluateResponse) => {
    setResult(res);
    setHfChoice('pending');
    setView('results');
  }, []);

  const handleReset = () => {
    setResult(null);
    setHfChoice('pending');
    setView('pipeline');
  };

  if (showSplash) {
    return <SplashScreen onFinish={handleSplashDone} />;
  }

  return (
    <div
      className="min-h-screen bg-grid page-animate-in"
      style={{ background: '#060608' }}
    >
      {/* Ambient background glow */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background:
            'radial-gradient(ellipse 80% 60% at 50% -10%, rgba(140,145,200,0.06) 0%, transparent 65%)',
          zIndex: 0,
        }}
      />

      {/* Header */}
      <header
        className="relative z-30 flex items-center justify-between px-6 py-4 border-b"
        style={{
          borderColor: 'rgba(180,180,220,0.07)',
          background: 'rgba(6,6,10,0.85)',
          backdropFilter: 'blur(20px)',
          position: 'sticky',
          top: 0,
        }}
      >
        <div className="flex items-center gap-3">
          {/* Logo mark */}
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{
              background: 'linear-gradient(145deg, rgba(60,60,75,0.9), rgba(25,25,38,0.95))',
              border: '1px solid rgba(200,200,230,0.15)',
              boxShadow: '0 2px 12px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1)',
            }}
          >
            <span
              style={{
                fontFamily: 'Orbitron, sans-serif',
                fontSize: '0.9rem',
                fontWeight: 700,
                background: 'linear-gradient(135deg, #888, #eee, #fff, #bbb)',
                WebkitBackgroundClip: 'text',
                backgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              E
            </span>
          </div>

          <h1
            style={{
              fontFamily: 'Orbitron, sans-serif',
              fontSize: '1.1rem',
              fontWeight: 700,
              letterSpacing: '0.06em',
              background:
                'linear-gradient(135deg, #7a7a7a 0%, #c8c8c8 25%, #ffffff 40%, #d8d8d8 55%, #a0a0a0 75%, #c0c0c0 100%)',
              WebkitBackgroundClip: 'text',
              backgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Evalora
          </h1>

          {view === 'results' && (
            <>
              <ChevronRight size={12} style={{ color: 'rgba(140,145,180,0.4)' }} />
              <span
                style={{
                  fontFamily: 'Orbitron, sans-serif',
                  fontSize: '0.6rem',
                  letterSpacing: '0.12em',
                  color: 'rgba(140,145,180,0.5)',
                  textTransform: 'uppercase',
                }}
              >
                Results
              </span>
            </>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* API URL input */}
          <div className="flex items-center gap-2 hidden sm:flex">
            <Server size={12} style={{ color: 'rgba(130,135,170,0.45)', flexShrink: 0 }} />
            <input
              className="chrome-input rounded-lg px-3 py-1.5 text-xs"
              style={{ width: 220, fontSize: '0.72rem' }}
              placeholder="http://localhost:8000"
              value={apiBaseUrl}
              onChange={(e) => setApiBaseUrl(e.target.value)}
            />
          </div>

          {view === 'results' && (
            <button
              onClick={handleReset}
              className="chrome-btn flex items-center gap-1.5 rounded-lg px-3 py-1.5"
              style={{ fontSize: '0.72rem', color: 'rgba(170,175,210,0.7)' }}
            >
              <RotateCcw size={11} />
              New Run
            </button>
          )}
        </div>
      </header>

      {/* Main content */}
      <main className="relative z-10 px-4 py-10">
        {view === 'pipeline' && (
          <PipelineForm onResult={handleResult} apiBaseUrl={apiBaseUrl} />
        )}

        {view === 'results' && result && (
          <div className="max-w-3xl mx-auto space-y-8">
            {/* Back button */}
            <button
              onClick={handleReset}
              className="flex items-center gap-1.5 text-xs transition-colors duration-150"
              style={{ color: 'rgba(140,145,185,0.5)' }}
            >
              <ArrowLeft size={12} />
              Back to pipeline
            </button>

            {/* Results */}
            <ResultsDisplay result={result} />

            {/* HuggingFace upload decision */}
            <div className="chrome-divider" />

            {hfChoice === 'pending' && (
              <div
                className="glass-card rounded-2xl p-6 space-y-4 page-animate-in"
                style={{ textAlign: 'center' }}
              >
                <p
                  style={{
                    fontFamily: 'Orbitron, sans-serif',
                    fontSize: '0.7rem',
                    letterSpacing: '0.12em',
                    color: 'rgba(180,185,220,0.7)',
                    textTransform: 'uppercase',
                    marginBottom: 8,
                  }}
                >
                  Upload Your Model
                </p>
                <p style={{ color: 'rgba(150,155,190,0.6)', fontSize: '0.82rem', marginBottom: 20 }}>
                  Would you like to publish this fine-tuned model to HuggingFace Hub?
                </p>
                <div className="flex gap-3 justify-center">
                  <button
                    onClick={() => setHfChoice('yes')}
                    className="chrome-btn-primary rounded-xl px-8 py-3 text-sm font-medium"
                    style={{
                      fontFamily: 'Orbitron, sans-serif',
                      fontSize: '0.68rem',
                      letterSpacing: '0.1em',
                      color: 'rgba(210,215,250,0.9)',
                    }}
                  >
                    Yes, Upload
                  </button>
                  <button
                    onClick={() => setHfChoice('no')}
                    className="chrome-btn rounded-xl px-8 py-3 text-sm"
                    style={{
                      fontSize: '0.68rem',
                      letterSpacing: '0.06em',
                      color: 'rgba(150,155,190,0.65)',
                    }}
                  >
                    Not Now
                  </button>
                </div>
              </div>
            )}

            {hfChoice === 'yes' && (
              <HuggingFaceUpload uploadId={result.upload_id} apiBaseUrl={apiBaseUrl} />
            )}

            {hfChoice === 'no' && (
              <div
                className="rounded-2xl px-5 py-4 text-center page-animate-in"
                style={{
                  background: 'rgba(16,16,24,0.6)',
                  border: '1px solid rgba(180,180,220,0.07)',
                }}
              >
                <p style={{ color: 'rgba(120,125,160,0.5)', fontSize: '0.8rem' }}>
                  Model saved locally. You can always upload it later.
                </p>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer
        className="relative z-10 text-center py-6 mt-8 border-t"
        style={{
          borderColor: 'rgba(180,180,220,0.05)',
          color: 'rgba(100,105,140,0.35)',
          fontSize: '0.68rem',
          letterSpacing: '0.08em',
          fontFamily: 'Orbitron, sans-serif',
        }}
      >
        EVALORA &nbsp;·&nbsp; TEACH AI HOW TO EVALUATE
      </footer>
    </div>
  );
}
