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
      style={{ background: 'var(--bg-deep)' }}
    >
      {/* Animated gradient orbs */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div
          style={{
            position: 'absolute',
            top: '30%',
            left: '20%',
            width: '40vw',
            height: '40vw',
            borderRadius: '50%',
            background: 'radial-gradient(circle, var(--primary-glow) 0%, transparent 60%)',
            filter: 'blur(60px)',
            animation: 'pulse-glow 3s ease-in-out infinite',
          }}
        />
        <div
          style={{
            position: 'absolute',
            bottom: '20%',
            right: '15%',
            width: '35vw',
            height: '35vw',
            borderRadius: '50%',
            background: 'radial-gradient(circle, var(--secondary-dim) 0%, transparent 60%)',
            filter: 'blur(50px)',
            animation: 'pulse-glow 3s ease-in-out infinite 1s',
          }}
        />
        <div
          style={{
            position: 'absolute',
            top: '60%',
            left: '50%',
            width: '30vw',
            height: '30vw',
            borderRadius: '50%',
            background: 'radial-gradient(circle, var(--accent-dim) 0%, transparent 60%)',
            filter: 'blur(40px)',
            animation: 'pulse-glow 3s ease-in-out infinite 2s',
          }}
        />
      </div>

      <h1
        className="font-brand relative"
        style={{
          fontSize: 'clamp(3rem, 8vw, 5rem)',
          fontWeight: 700,
          letterSpacing: '0.02em',
          lineHeight: 1,
          background: 'linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%)',
          WebkitBackgroundClip: 'text',
          backgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: 16,
        }}
      >
        Evalora
      </h1>

      <p
        className="animate-fade-in relative"
        style={{
          animationDelay: '0.4s',
          fontSize: 'clamp(0.7rem, 2vw, 0.9rem)',
          fontWeight: 500,
          letterSpacing: '0.2em',
          textTransform: 'uppercase',
          color: 'var(--text-muted)',
        }}
      >
        Teach AI How To Evaluate
      </p>

      {/* Animated loading line - sweep animation */}
      <div
        className="animate-fade-in"
        style={{
          animationDelay: '0.6s',
          marginTop: 32,
          width: 120,
          height: 4,
          borderRadius: 2,
          background: 'var(--bg-elevated)',
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        <div
          style={{
            position: 'absolute',
            width: '50%',
            height: '100%',
            background: 'linear-gradient(90deg, transparent, var(--primary), var(--secondary), transparent)',
            borderRadius: 2,
            animation: 'loading-sweep 1.2s ease-in-out infinite',
          }}
        />
      </div>

    </div>
  );
}