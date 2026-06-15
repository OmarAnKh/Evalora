import { useEffect, useState } from 'react';

interface Props {
  onFinish: () => void;
}

export default function SplashScreen({ onFinish }: Props) {
  const [exiting, setExiting] = useState(false);

  useEffect(() => {
    const t1 = setTimeout(() => setExiting(true), 2400);
    const t2 = setTimeout(() => onFinish(), 2800);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [onFinish]);

  return (
    <div
      className={`fixed inset-0 flex flex-col items-center justify-center z-50 ${exiting ? 'animate-fade-out' : 'animate-fade-in'}`}
      style={{ background: 'var(--page-bg)' }}
    >
      <div
        className="absolute inset-0 pointer-events-none"
        style={{ background: 'radial-gradient(ellipse 50% 40% at 50% 45%, var(--page-glow) 0%, transparent 70%)' }}
      />

      <h1
        className="font-brand"
        style={{
          fontSize: 'clamp(2.8rem, 7vw, 4.5rem)',
          fontWeight: 700,
          letterSpacing: '0.04em',
          lineHeight: 1,
          background: 'var(--brand-gradient)',
          WebkitBackgroundClip: 'text',
          backgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: 12,
        }}
      >
        Evalora
      </h1>

      <p
        className="animate-fade-in"
        style={{
          animationDelay: '0.4s',
          fontSize: 'clamp(0.7rem, 1.8vw, 0.85rem)',
          fontWeight: 400,
          letterSpacing: '0.18em',
          textTransform: 'uppercase',
          color: 'var(--text-muted)',
        }}
      >
        Teach AI How To Evaluate
      </p>

      <div
        className="absolute"
        style={{ bottom: '18%', width: 80, height: 2, borderRadius: 1, background: 'var(--accent-dim)', overflow: 'hidden' }}
      >
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background: 'linear-gradient(90deg, transparent, var(--accent), transparent)',
            animation: 'scan-line 1.2s ease-in-out infinite',
          }}
        />
      </div>
    </div>
  );
}
