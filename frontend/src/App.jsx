import React, { useState, useEffect } from 'react';
import {
  Shield,
  CheckCircle,
  XCircle,
  History,
  BarChart3,
  AlertCircle,
  Loader,
  TrendingUp,
  Database,
  Zap,
  Activity,
  PieChart,
  Clock,
  Sparkles,
  Loader2,
  Brain,
  Target,
  RefreshCw,
} from 'lucide-react';
import './App.css';

const API_URL = 'http://localhost:5000/api';

function App() {
  const [newsText, setNewsText] = useState('');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);
  const [activeTab, setActiveTab] = useState('detect');
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState({ backend: false, ml: false, db: false });
  const [showResults, setShowResults] = useState(false);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (activeTab === 'history') fetchHistory();
    else if (activeTab === 'stats') fetchStats();
  }, [activeTab]);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_URL}/health`);
      const data = await res.json();
      setApiStatus({
        backend: data.status === 'OK',
        ml: data.services?.mlApi === 'connected',
        db: data.services?.database === 'connected',
      });
    } catch {
      setApiStatus({ backend: false, ml: false, db: false });
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_URL}/history?limit=20`);
      const data = await res.json();
      if (data.success) setHistory(data.data || []);
    } catch {
      setError('Failed to fetch history');
    }
  };

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_URL}/stats`);
      const data = await res.json();
      if (data.success) setStats(data.stats);
    } catch {
      setError('Failed to fetch statistics');
    }
  };

  const handleSubmit = async () => {
    if (!newsText.trim()) return setError('Please enter some news text');
    if (newsText.trim().length < 20) return setError('Please enter at least 20 characters');

    setLoading(true);
    setPrediction(null);
    setShowResults(false);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: newsText }),
      });
      const data = await res.json();

      if (data.success) {
        setPrediction({
          predictions: data.predictions,
          confidence: data.confidence,
          bert: data.bert,
        });
        // Delay to show results with animation
        setTimeout(() => setShowResults(true), 300);
      } else {
        setError(data.error || 'Prediction failed');
      }
    } catch {
      setError('Failed to connect to server. Make sure the backend is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setNewsText('');
    setPrediction(null);
    setShowResults(false);
    setError(null);
  };

  const getResultClass = (result) => result === 'Fake' ? 'result-fake' : 'result-real';

  const ResultCard = ({ title, result, confidence, icon: Icon, delay = 0 }) => (
    <div 
      className={`result-card ${getResultClass(result)}`}
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="result-card-header">
        <span className="result-card-title">
          {Icon && <Icon className="icon-small" />}
          {title}
        </span>
        {result === 'Fake' ? (
          <XCircle className="icon-medium icon-red" />
        ) : (
          <CheckCircle className="icon-medium icon-green" />
        )}
      </div>
      <div className="result-card-body">
        <span className={`result-value ${result === 'Fake' ? 'text-red' : 'text-green'}`}>
          {result}
        </span>
        {confidence && (
          <div className="confidence-box">
            <div className="confidence-label">Confidence</div>
            <div className="confidence-value">{confidence.toFixed(2)}%</div>
          </div>
        )}
      </div>
      <div className="progress-bar">
        <div 
          className={`progress-fill ${result === 'Fake' ? 'progress-red' : 'progress-green'}`}
          style={{ width: showResults ? `${confidence || 0}%` : '0%' }}
        />
      </div>
    </div>
  );

  const StatusBadge = ({ status, label, icon: Icon }) => (
    <div className={`status-badge ${status ? 'status-active' : 'status-inactive'}`}>
      <div className={`status-dot ${status ? 'dot-pulse' : ''}`} />
      {Icon && <Icon className="icon-tiny" />}
      {label}
    </div>
  );

  return (
    <div className="app-container">
      {/* Animated Background */}
      <div className="bg-animation">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
        <div className="blob blob-3"></div>
      </div>

      <div className="content-wrapper">
        {/* Header */}
        <header className="header">
          <div className="header-title">
            <div className="shield-container">
              <Shield className="shield-icon" />
              <div className="shield-glow"></div>
            </div>
            <h1 className="main-title">Fake News Detector</h1>
          </div>
          <p className="subtitle">
            <Brain className="icon-small" style={{ display: 'inline', marginRight: '8px' }} />
            AI-Powered News Verification System with BERT
          </p>

          <div className="status-badges">
            <StatusBadge status={apiStatus.backend} label="Backend" icon={Database} />
            <StatusBadge status={apiStatus.ml} label="ML API" icon={Brain} />
            <StatusBadge status={apiStatus.db} label="Database" icon={Activity} />
          </div>

          <div className="features">
            <div className="feature-item">
              <Zap className="icon-small" />
              <span>Real-time Analysis</span>
            </div>
            <div className="feature-item">
              <Target className="icon-small" />
              <span>BERT Model</span>
            </div>
            <div className="feature-item">
              <Database className="icon-small" />
              <span>Cloud Powered</span>
            </div>
          </div>
        </header>

        {/* Tabs */}
        <nav className="tabs">
          {[
            { key: 'detect', icon: Shield, label: 'Detect' },
            { key: 'history', icon: History, label: 'History' },
            { key: 'stats', icon: BarChart3, label: 'Stats' },
          ].map(({ key, icon: Icon, label }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className={`tab-button ${activeTab === key ? 'tab-active' : ''}`}
            >
              <Icon className="icon-medium" />
              {label}
            </button>
          ))}
        </nav>

        {/* Content */}
        <main className="main-content">
          {/* Detect Tab */}
          {activeTab === 'detect' && (
            <section className="card">
              {error && (
                <div className="error-alert">
                  <AlertCircle className="icon-medium icon-red" />
                  <div>
                    <div className="error-title">Error</div>
                    <div className="error-message">{error}</div>
                  </div>
                </div>
              )}

              <div className="form-group">
                <label className="form-label">
                  <PieChart className="icon-medium" />
                  Enter News Article:
                </label>
                <textarea
                  value={newsText}
                  onChange={(e) => {
                    setNewsText(e.target.value);
                    setError(null);
                  }}
                  className="textarea"
                  rows={10}
                  placeholder="Paste your news article here to check if it's fake or real...

Example: 'Scientists discover new planet with potential for human colonization. The planet, located 40 light-years away, has conditions similar to Earth including water and atmosphere.'"
                />
                <div className="textarea-footer">
                  <span className={newsText.length >= 20 ? 'text-green' : 'text-gray'}>
                    {newsText.length} characters {newsText.length >= 20 ? '✓' : '(minimum 20)'}
                  </span>
                  <div className="feature-item">
                    <Clock className="icon-tiny" />
                    <span>Analysis takes ~2-5 seconds</span>
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '1rem' }}>
                <button
                  onClick={handleSubmit}
                  disabled={loading || !newsText.trim() || newsText.length < 20}
                  className="submit-button"
                  style={{ flex: 1 }}
                >
                  {loading ? (
                    <>
                      <Loader2 className="icon-medium icon-spin" />
                      Analyzing with BERT AI...
                    </>
                  ) : (
                    <>
                      <Shield className="icon-medium" />
                      Analyze News Article
                    </>
                  )}
                </button>
                
                {newsText && (
                  <button
                    onClick={handleClear}
                    className="submit-button"
                    style={{ 
                      flex: 0,
                      minWidth: '120px',
                      background: 'rgba(239, 68, 68, 0.2)',
                      border: '2px solid var(--danger-red)'
                    }}
                  >
                    <RefreshCw className="icon-medium" />
                    Clear
                  </button>
                )}
              </div>

              {prediction && (
                <div className="results-section">
                  <h2 className="section-title">
                    <TrendingUp className="icon-large" />
                    Prediction Results
                  </h2>

                  {/* BERT Result - Primary */}
                  {prediction.bert && (
                    <div style={{ marginBottom: '2rem' }}>
                      <h3 style={{ 
                        fontSize: '1.25rem', 
                        marginBottom: '1rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        color: 'var(--accent-cyan)'
                      }}>
                        <Brain className="icon-medium" />
                        BERT Model (Primary)
                      </h3>
                      <ResultCard
                        title="BERT Tiny Fine-tuned"
                        result={prediction.bert.prediction}
                        confidence={prediction.bert.confidence}
                        icon={Sparkles}
                        delay={0}
                      />
                      {prediction.bert.probabilities && (
                        <div style={{
                          display: 'flex',
                          gap: '1rem',
                          marginTop: '1rem',
                          padding: '1rem',
                          background: 'rgba(0,0,0,0.2)',
                          borderRadius: '0.75rem'
                        }}>
                          <div style={{ flex: 1, textAlign: 'center' }}>
                            <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                              Fake Probability
                            </div>
                            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: 'var(--danger-red)' }}>
                              {prediction.bert.probabilities.fake}%
                            </div>
                          </div>
                          <div style={{ flex: 1, textAlign: 'center' }}>
                            <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                              Real Probability
                            </div>
                            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: 'var(--success-green)' }}>
                              {prediction.bert.probabilities.real}%
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Traditional ML Models */}
                  <h3 style={{ 
                    fontSize: '1.25rem', 
                    marginBottom: '1rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    color: 'var(--text-secondary)'
                  }}>
                    <Activity className="icon-medium" />
                    Traditional ML Models
                  </h3>
                  <ResultCard
                    title="Logistic Regression"
                    result={prediction.predictions.logisticRegression}
                    confidence={prediction.confidence?.logisticRegression}
                    icon={Target}
                    delay={100}
                  />
                  <ResultCard
                    title="Gradient Boosting"
                    result={prediction.predictions.gradientBoosting}
                    confidence={prediction.confidence?.gradientBoosting}
                    icon={TrendingUp}
                    delay={200}
                  />
                  <ResultCard
                    title="Random Forest"
                    result={prediction.predictions.randomForest}
                    confidence={prediction.confidence?.randomForest}
                    icon={Database}
                    delay={300}
                  />
                </div>
              )}
            </section>
          )}

          {/* History Tab */}
          {activeTab === 'history' && (
            <section className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                <h2 className="section-title" style={{ marginBottom: 0 }}>
                  <History className="icon-large" />
                  Recent Predictions
                </h2>
                <button
                  onClick={fetchHistory}
                  className="tab-button"
                  style={{ padding: '0.5rem 1rem' }}
                >
                  <RefreshCw className="icon-small" />
                  Refresh
                </button>
              </div>

              {history.length === 0 ? (
                <div className="empty-state">
                  <Database className="icon-xlarge" />
                  <p className="empty-title">No history available yet</p>
                  <p className="empty-subtitle">Start analyzing news to see your history here</p>
                </div>
              ) : (
                <div className="history-list">
                  {history.map((item, idx) => (
                    <div key={item._id} className="history-item">
                      <div className="history-header">
                        <span className="history-date">
                          <Clock className="icon-tiny" />
                          {new Date(item.timestamp).toLocaleString()}
                        </span>
                        <span className="history-badge">#{idx + 1}</span>
                      </div>
                      <p className="history-text">
                        {item.text.length > 200 ? `${item.text.substring(0, 200)}...` : item.text}
                      </p>
                      <div className="history-predictions">
                        <span className={`prediction-badge ${getResultClass(item.predictions.lr)}`}>
                          LR: {item.predictions.lr}
                        </span>
                        <span className={`prediction-badge ${getResultClass(item.predictions.gbc)}`}>
                          GBC: {item.predictions.gbc}
                        </span>
                        <span className={`prediction-badge ${getResultClass(item.predictions.rfc)}`}>
                          RFC: {item.predictions.rfc}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* Stats Tab */}
          {activeTab === 'stats' && (
            <section className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                <h2 className="section-title" style={{ marginBottom: 0 }}>
                  <BarChart3 className="icon-large" />
                  Statistics Dashboard
                </h2>
                <button
                  onClick={fetchStats}
                  className="tab-button"
                  style={{ padding: '0.5rem 1rem' }}
                >
                  <RefreshCw className="icon-small" />
                  Refresh
                </button>
              </div>

              {stats && stats.total > 0 ? (
                <div className="stats-grid">
                  <div className="stat-card stat-blue">
                    <Activity className="icon-xlarge" />
                    <div className="stat-value">{stats.total}</div>
                    <div className="stat-label">Total Predictions</div>
                  </div>
                  <div className="stat-card stat-green">
                    <CheckCircle className="icon-xlarge" />
                    <div className="stat-value">{stats.real}</div>
                    <div className="stat-label">Real News</div>
                  </div>
                  <div className="stat-card stat-red">
                    <XCircle className="icon-xlarge" />
                    <div className="stat-value">{stats.fake}</div>
                    <div className="stat-label">Fake News</div>
                  </div>
                  <div className="stat-card stat-purple">
                    <TrendingUp className="icon-xlarge" />
                    <div className="stat-value">{stats.fakePercentage}%</div>
                    <div className="stat-label">Fake Rate</div>
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <BarChart3 className="icon-xlarge" />
                  <p className="empty-title">No statistics available yet</p>
                  <p className="empty-subtitle">Start analyzing news to see statistics here</p>
                </div>
              )}
            </section>
          )}
        </main>

        {/* Footer */}
        <footer className="footer">
          <div className="footer-content">
            <p className="footer-title">
              <Sparkles className="icon-small" />
              Powered by BERT & Machine Learning | MERN Stack
            </p>
            <p className="footer-subtitle">
              Developed by Vaibhav Hiremath | © 2025
            </p>
            <p className="footer-subtitle" style={{ marginTop: '0.5rem', fontSize: '0.75rem' }}>
              Model: mrm8488/bert-tiny-finetuned-fake-news
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;