import React, { useState } from 'react';
import { Lock, LogIn, AlertCircle } from 'lucide-react';

interface AuthProps {
  onAuthSuccess: (token: string) => void;
}

const Auth = ({ onAuthSuccess }: AuthProps) => {
  const [pin, setPin] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const validateInputs = () => {
    if (!pin) {
      setError('PIN is required');
      return false;
    }

    if (pin.length < 4) {
      setError('PIN must be at least 4 digits');
      return false;
    }

    if (!/^\d+$/.test(pin)) {
      setError('PIN must contain only numbers');
      return false;
    }

    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!validateInputs()) {
      return;
    }

    setIsLoading(true);

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ pin }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Authentication failed');
      }

      // Store token in localStorage
      localStorage.setItem('yourdaddy-token', data.access_token);
      localStorage.setItem('yourdaddy-username', 'assistant_user');

      // Call success callback
      onAuthSuccess(data.access_token);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Authentication failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#1a1a2e] via-[#16213e] to-[#0f0f1e] p-4" role="main" aria-labelledby="auth-title">
      <div className="glass-strong p-8 rounded-2xl w-full max-w-md shadow-2xl animate-bounce-in">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-[#6C5CE7] to-[#A855F7] rounded-full mb-4 animate-glow" aria-hidden="true">
            <Lock className="w-8 h-8 text-white" />
          </div>
          <h1 id="auth-title" className="text-3xl font-bold gradient-text mb-2">
            PIN Authentication
          </h1>
          <p className="text-[#AAAAAA]">
            Enter your PIN to access YourDaddy Assistant
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/50 rounded-lg flex items-start gap-3 animate-slide-up" role="alert" aria-live="polite">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" aria-hidden="true" />
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-5" aria-label="PIN authentication form">
          {/* PIN Input */}
          <div>
            <label htmlFor="pin" className="block text-sm font-medium text-[#DDDDDD] mb-2">
              PIN
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none" aria-hidden="true">
                <Lock className="w-5 h-5 text-[#888888]" />
              </div>
              <input
                id="pin"
                type="password"
                value={pin}
                onChange={(e) => setPin(e.target.value)}
                className="glass-input pl-10 w-full text-center tracking-widest"
                placeholder="Enter your PIN"
                disabled={isLoading}
                autoComplete="off"
                required
                aria-required="true"
                maxLength={10}
                pattern="[0-9]*"
                inputMode="numeric"
                aria-invalid={!!error}
                aria-describedby={error ? 'pin-error' : undefined}
              />
            </div>
            <p className="mt-1 text-xs text-[#888888]">
              Enter your 4+ digit security PIN
            </p>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className="w-full btn-primary py-3 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Authenticating...</span>
              </>
            ) : (
              <>
                <LogIn className="w-5 h-5" />
                <span>Access Assistant</span>
              </>
            )}
          </button>
        </form>

        {/* Security Info */}
        <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
          <p className="text-xs text-blue-400 text-center">
            <strong>🔐 Secure:</strong> PIN authentication provides quick & secure access
          </p>
        </div>
      </div>
    </div>
  );
};

export default Auth;
