import { useEffect, useState } from 'react';

interface Props {
  onFinish: () => void;
}

export default function SplashScreen({ onFinish }: Props) {
  const [phase, setPhase] = useState<'in' | 'hold' | 'out'>('in');

  useEffect(() => {
    const holdTimer = setTimeout(() => setPhase('hold'), 900);
    const outTimer = setTimeout(() => setPhase('out'), 2600);
    const doneTimer = setTimeout(() => onFinish(), 3200);
    return () => {
      clearTimeout(holdTimer);
      clearTimeout(outTimer);
      clearTimeout(doneTimer);
    };
  }, [onFinish]);

  return (
    <div
      className={`fixed inset-0 flex flex-col items-center justify-center bg-grid z-50 ${
        phase === 'out' ? 'splash-animate-out' : 'splash-animate-in'
      }`}
      style={{ background: '#060608' }}
    >
      {/* Background radial glow */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            'radial-gradient(ellipse 60% 50% at 50% 50%, rgba(160,165,210,0.07) 0%, transparent 70%)',
        }}
      />

      {/* Outer ring */}
      <div className="relative flex items-center justify-center mb-8">
        <div
          className="absolute rounded-full"
          style={{
            width: 280,
            height: 280,
            border: '1px solid rgba(200,200,230,0.06)',
            animation: 'pulse-ring 2.8s ease-out infinite',
          }}
        />
        <div
          className="absolute rounded-full"
          style={{
            width: 220,
            height: 220,
            border: '1px solid rgba(200,200,230,0.08)',
            animation: 'pulse-ring 2.8s ease-out 0.4s infinite',
          }}
        />
        <div
          className="absolute rounded-full"
          style={{
            width: 160,
            height: 160,
            border: '1px solid rgba(200,200,230,0.1)',
          }}
        />

        {/* Logo mark — a simple hexagonal chrome badge */}
        <div
          className="relative flex items-center justify-center"
          style={{ width: 120, height: 120 }}
        >
          <div
            style={{
              width: 96,
              height: 96,
              background:
                'linear-gradient(145deg, rgba(60,60,75,0.9) 0%, rgba(30,30,42,0.95) 100%)',
              border: '1px solid rgba(200,200,230,0.2)',
              borderRadius: 20,
              boxShadow:
                '0 8px 40px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.12), inset 0 -1px 0 rgba(0,0,0,0.3)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {/* "E" mark */}
            <span
              style={{
                fontFamily: 'Orbitron, sans-serif',
                fontSize: '2.5rem',
                fontWeight: 700,
                background:
                  'linear-gradient(135deg, #888 0%, #ddd 30%, #fff 50%, #ccc 70%, #999 100%)',
                WebkitBackgroundClip: 'text',
                backgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                filter: 'drop-shadow(0 0 12px rgba(200,205,255,0.3))',
              }}
            >
              E
            </span>
          </div>
        </div>
      </div>

      {/* Brand name */}
      <h1
        className="float-animation"
        style={{
          fontFamily: 'Orbitron, sans-serif',
          fontSize: 'clamp(3rem, 8vw, 5.5rem)',
          fontWeight: 900,
          letterSpacing: '0.05em',
          lineHeight: 1,
          background:
            'linear-gradient(135deg, #6e6e6e 0%, #c8c8c8 18%, #ffffff 28%, #e0e0e0 38%, #b0b0b0 50%, #d8d8d8 62%, #ffffff 72%, #c0c0c0 82%, #888888 100%)',
          WebkitBackgroundClip: 'text',
          backgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          filter: 'drop-shadow(0 0 30px rgba(200,210,255,0.25))',
          marginBottom: '1.2rem',
        }}
      >
        Evalora
      </h1>

      {/* Slogan */}
      <p
        className="slogan-animate"
        style={{
          fontFamily: 'Inter, sans-serif',
          fontSize: 'clamp(0.75rem, 2vw, 0.9rem)',
          fontWeight: 300,
          letterSpacing: '0.15em',
          textTransform: 'uppercase',
          color: 'rgba(180, 185, 210, 0.6)',
          marginTop: '0.5rem',
        }}
      >
        Teach AI How To Evaluate
      </p>

      {/* Bottom loader bar */}
      <div
        className="absolute bottom-12"
        style={{ width: 120, height: 2, background: 'rgba(255,255,255,0.05)', borderRadius: 1 }}
      >
        <div
          style={{
            height: '100%',
            borderRadius: 1,
            background:
              'linear-gradient(90deg, rgba(180,185,220,0.6), rgba(255,255,255,0.9), rgba(180,185,220,0.6))',
            animation: 'shimmer 1.8s linear infinite',
            backgroundSize: '200% auto',
          }}
        />
      </div>
    </div>
  );
}
