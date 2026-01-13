import { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Cpu,
  HardDrive,
  Database,
  CheckCircle,
  AlertCircle,
  Loader2,
  Package,
  Trash2,
  Sparkles,
  Clock,
  Download
} from 'lucide-react';
import { useSystemStore } from '../store';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * ModelLoader page - load HuggingFace models with progress tracking
 */
export default function ModelLoader() {
  const systemInfo = useSystemStore((state) => state.systemInfo);

  const [modelName, setModelName] = useState('');
  const [exampleModels, setExampleModels] = useState(null);
  const [loadResult, setLoadResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(null);
  const [cachedModels, setCachedModels] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);

  const progressPollRef = useRef(null);

  // Fetch example models and cache info on mount
  useEffect(() => {
    // Optimistic load from cache
    const cachedExamples = localStorage.getItem('example_models');
    if (cachedExamples) {
      try {
        setExampleModels(JSON.parse(cachedExamples));
      } catch (e) { }
    }

    fetch('/api/models/examples')
      .then(res => res.json())
      .then(data => {
        setExampleModels(data);
        localStorage.setItem('example_models', JSON.stringify(data));
      })
      .catch(() => { });

    fetchCacheInfo();
    fetchModelInfo();
  }, []);

  const fetchCacheInfo = async () => {
    try {
      const res = await fetch('/api/models/cache');
      const data = await res.json();
      setCachedModels(data.models || []);
    } catch (e) { }
  };

  const fetchModelInfo = async () => {
    try {
      const res = await fetch('/api/models/info');
      const data = await res.json();
      if (data.loaded) {
        setModelInfo(data);
      }
    } catch (e) { }
  };

  const pollProgress = (name) => {
    if (progressPollRef.current) {
      clearInterval(progressPollRef.current);
    }

    progressPollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/api/models/progress/${encodeURIComponent(name)}`);
        const data = await res.json();
        if (data.downloading) {
          setProgress(data);
        }
      } catch (e) { }
    }, 500);
  };

  const stopPolling = () => {
    if (progressPollRef.current) {
      clearInterval(progressPollRef.current);
      progressPollRef.current = null;
    }
  };

  const handleLoadModel = async () => {
    if (!modelName.trim() || isLoading) return;

    setIsLoading(true);
    setLoadResult(null);
    setProgress({ status: 'starting', percent: 0, message: 'Starting download...' });

    // Start polling for progress
    pollProgress(modelName.trim());

    try {
      const response = await fetch('/api/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName.trim(),
          dtype: 'auto',
          device: 'auto',
          trust_remote_code: true
        })
      });

      const data = await response.json();
      setLoadResult(data);

      if (data.success) {
        setModelInfo(data.model_info);
        setProgress({ status: 'complete', percent: 100, message: 'Model loaded!' });
        fetchCacheInfo();
      } else {
        setProgress(null);
      }
    } catch (err) {
      setLoadResult({ success: false, error: err.message });
      setProgress(null);
    } finally {
      setIsLoading(false);
      stopPolling();
    }
  };

  const handleQuickLoad = (modelId) => {
    setModelName(modelId);
  };

  const handleUnload = async () => {
    try {
      await fetch('/api/models/unload', { method: 'POST' });
      setModelInfo(null);
      setLoadResult(null);
      setProgress(null);
    } catch (e) { }
  };

  const handleDeleteFromCache = async (name) => {
    try {
      await fetch(`/api/models/cache/${encodeURIComponent(name)}`, { method: 'DELETE' });
      fetchCacheInfo();
    } catch (e) { }
  };

  const handleCleanup = async () => {
    try {
      const res = await fetch('/api/models/cache/cleanup', { method: 'POST' });
      const data = await res.json();
      fetchCacheInfo();
      alert(`Cleaned up ${data.deleted_count} models`);
    } catch (e) { }
  };

  return (
    <div className="model-loader">
      {/* Header */}
      <div className="page-header">
        <h1 className="page-title">Load HuggingFace Model</h1>
        <p className="page-subtitle">
          Download and analyze models directly from HuggingFace Hub
        </p>
      </div>

      {/* Main Content */}
      <div className="loader-grid">
        {/* Load Model Card */}
        <motion.div
          className="glass-card load-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="card-header">
            <Package size={24} />
            <h2>Load Model</h2>
          </div>

          <div className="input-section">
            <label className="input-label">Model ID</label>
            <input
              type="text"
              className="input"
              placeholder="e.g. gpt2, bert-base-uncased, prajjwal1/bert-tiny"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleLoadModel()}
              disabled={isLoading}
            />
            <p className="input-hint">
              Enter the HuggingFace model identifier (organization/model-name)
            </p>
          </div>

          <button
            className="btn btn-primary btn-lg w-full"
            onClick={handleLoadModel}
            disabled={isLoading || !modelName.trim()}
          >
            {isLoading ? (
              <>
                <Loader2 size={20} className="spinning" />
                Loading...
              </>
            ) : (
              <>
                <Download size={20} />
                Download & Load Model
              </>
            )}
          </button>

          {/* Progress Bar */}
          <AnimatePresence>
            {progress && (
              <motion.div
                className="progress-container"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                <div className="progress-header">
                  <span className="progress-status">{progress.message || progress.status}</span>
                  <span className="progress-percent">{progress.percent || 0}%</span>
                </div>
                <div className="progress-bar">
                  <motion.div
                    className="progress-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress.percent || 0}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
                {progress.speed_mbps && (
                  <div className="progress-details">
                    <span>{progress.speed_mbps} MB/s</span>
                    {progress.eta_seconds && <span>ETA: {progress.eta_seconds}s</span>}
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Result Message */}
          <AnimatePresence>
            {loadResult && !isLoading && (
              <motion.div
                className={`result-message ${loadResult.success ? 'success' : 'error'}`}
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                {loadResult.success ? (
                  <>
                    <CheckCircle size={20} />
                    <div>
                      <strong>Model loaded successfully!</strong>
                      <p>{loadResult.model_info?.architecture} - {loadResult.model_info?.num_params_millions}M params</p>
                    </div>
                  </>
                ) : (
                  <>
                    <AlertCircle size={20} />
                    <div>
                      <strong>Failed to load model</strong>
                      <p>{loadResult.error}</p>
                      {loadResult.suggestion && <p className="suggestion">{loadResult.suggestion}</p>}
                    </div>
                  </>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* Currently Loaded Model */}
        {modelInfo && (
          <motion.div
            className="glass-card loaded-model-card"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
          >
            <div className="card-header">
              <CheckCircle size={24} className="text-success" />
              <h2>Loaded Model</h2>
              <button className="btn btn-ghost btn-sm ml-auto" onClick={handleUnload}>
                <Trash2 size={16} />
                Unload
              </button>
            </div>

            <div className="model-details">
              <div className="detail-item">
                <span className="label">Name</span>
                <span className="value">{modelInfo.name}</span>
              </div>
              <div className="detail-item">
                <span className="label">Parameters</span>
                <span className="value">{modelInfo.num_params_millions}M</span>
              </div>
              <div className="detail-item">
                <span className="label">Memory</span>
                <span className="value">{modelInfo.memory_mb?.toFixed(1)} MB</span>
              </div>
              <div className="detail-item">
                <span className="label">Device</span>
                <span className="value">{modelInfo.device}</span>
              </div>
              <div className="detail-item">
                <span className="label">Quantizable Layers</span>
                <span className="value highlight">{modelInfo.num_quantizable_layers}</span>
              </div>
            </div>
          </motion.div>
        )}

        {/* Quick Start */}
        <motion.div
          className="glass-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="card-header">
            <Sparkles size={24} />
            <h2>Quick Start</h2>
          </div>

          <p className="text-sm text-muted mb-md">Click to select a model:</p>

          {exampleModels ? (
            <>
              {exampleModels.sample_models?.length > 0 && (
                <div className="model-group">
                  <h4 className="group-title">⭐ Sample Models (Pre-cached)</h4>
                  <div className="model-list">
                    {exampleModels.sample_models.map((model) => (
                      <button
                        key={model.id}
                        className={`model-chip sample ${modelName === model.id ? 'selected' : ''}`}
                        onClick={() => handleQuickLoad(model.id)}
                      >
                        <span className="model-id">{model.id}</span>
                        <span className="model-desc">Instant load</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="model-group">
                <h4 className="group-title">Small Models</h4>
                <div className="model-list">
                  {exampleModels.small_models?.map((model) => (
                    <button
                      key={model.id}
                      className={`model-chip ${modelName === model.id ? 'selected' : ''}`}
                      onClick={() => handleQuickLoad(model.id)}
                    >
                      <span className="model-id">{model.id}</span>
                      <span className="model-size">{model.size}</span>
                    </button>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="loading-placeholder">
              <Loader2 size={20} className="spinning" />
              <span>Loading examples...</span>
            </div>
          )}
        </motion.div>

        {/* System Status */}
        <motion.div
          className="glass-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="card-header">
            <Cpu size={24} />
            <h2>System</h2>
          </div>

          {systemInfo ? (
            <div className="status-list">
              <div className="status-item">
                <span className="status-label">Device</span>
                <span className="status-value">
                  {systemInfo.cuda_available ? '🟢 CUDA GPU' :
                    systemInfo.mps_available ? '🟢 Apple MPS' : '🟡 CPU'}
                </span>
              </div>

              {systemInfo.gpus?.length > 0 && (
                <div className="status-item">
                  <span className="status-label">GPU</span>
                  <span className="status-value">{systemInfo.gpus[0].name}</span>
                </div>
              )}

              <div className="status-item">
                <span className="status-label">RAM</span>
                <span className="status-value">{systemInfo.ram_available_gb?.toFixed(1)} GB</span>
              </div>
            </div>
          ) : (
            <p className="text-muted">Loading...</p>
          )}
        </motion.div>

        {/* Cached Models */}
        <motion.div
          className="glass-card cache-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="card-header">
            <Database size={24} />
            <h2>Model Cache</h2>
            <button className="btn btn-ghost btn-sm ml-auto" onClick={handleCleanup}>
              <Clock size={16} />
              Cleanup
            </button>
          </div>

          <p className="text-xs text-muted mb-sm">
            Models auto-delete after 4 hours (except samples)
          </p>

          {cachedModels.length > 0 ? (
            <div className="cache-list">
              {cachedModels.map((model) => (
                <div key={model.name} className={`cache-item ${model.is_sample ? 'sample' : ''}`}>
                  <div className="cache-info">
                    <span className="cache-name">
                      {model.is_sample && '⭐ '}
                      {model.name}
                    </span>
                    <span className="cache-size">{model.size_mb} MB</span>
                  </div>
                  {!model.is_sample && (
                    <button
                      className="btn btn-ghost btn-xs"
                      onClick={() => handleDeleteFromCache(model.name)}
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted text-sm">No models cached</p>
          )}
        </motion.div>
      </div>

      <style>{`
        .loader-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: var(--space-lg);
        }
        
        @media (max-width: 1024px) {
          .loader-grid {
            grid-template-columns: 1fr;
          }
        }
        
        .load-card {
          grid-column: span 2;
        }
        
        @media (max-width: 1024px) {
          .load-card {
            grid-column: span 1;
          }
        }
        
        .loaded-model-card {
          grid-column: span 2;
          background: rgba(16, 185, 129, 0.05);
          border-color: rgba(16, 185, 129, 0.3);
        }
        
        .cache-card {
          grid-column: span 2;
        }
        
        .card-header {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
          margin-bottom: var(--space-lg);
          color: var(--text-primary);
        }
        
        .card-header h2 {
          font-size: var(--text-lg);
          font-weight: 600;
          margin: 0;
        }
        
        .input-section {
          margin-bottom: var(--space-lg);
        }
        
        .input-hint {
          font-size: var(--text-xs);
          color: var(--text-tertiary);
          margin-top: var(--space-xs);
        }
        
        /* Progress Bar */
        .progress-container {
          margin-top: var(--space-lg);
          padding: var(--space-md);
          background: var(--glass-bg);
          border-radius: var(--radius-md);
        }
        
        .progress-header {
          display: flex;
          justify-content: space-between;
          margin-bottom: var(--space-sm);
          font-size: var(--text-sm);
        }
        
        .progress-status {
          color: var(--text-secondary);
        }
        
        .progress-percent {
          color: var(--color-accent-primary);
          font-weight: 600;
        }
        
        .progress-bar {
          height: 8px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 4px;
          overflow: hidden;
        }
        
        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--color-accent-primary), var(--color-accent-secondary));
          border-radius: 4px;
        }
        
        .progress-details {
          display: flex;
          justify-content: space-between;
          margin-top: var(--space-xs);
          font-size: var(--text-xs);
          color: var(--text-tertiary);
        }
        
        .result-message {
          display: flex;
          align-items: flex-start;
          gap: var(--space-md);
          padding: var(--space-md);
          border-radius: var(--radius-md);
          margin-top: var(--space-md);
        }
        
        .result-message.success {
          background: rgba(16, 185, 129, 0.1);
          border: 1px solid rgba(16, 185, 129, 0.3);
          color: var(--color-success);
        }
        
        .result-message.error {
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          color: var(--color-error);
        }
        
        .result-message strong {
          display: block;
        }
        
        .result-message p {
          margin: var(--space-xs) 0 0 0;
          font-size: var(--text-sm);
          opacity: 0.9;
        }
        
        .model-details {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: var(--space-sm);
        }
        
        .detail-item {
          display: flex;
          flex-direction: column;
          padding: var(--space-sm);
          background: var(--glass-bg);
          border-radius: var(--radius-md);
        }
        
        .detail-item .label {
          font-size: var(--text-xs);
          color: var(--text-tertiary);
        }
        
        .detail-item .value {
          font-size: var(--text-base);
          font-weight: 500;
          color: var(--text-primary);
        }
        
        .detail-item .value.highlight {
          color: var(--color-accent-primary);
        }
        
        .model-group {
          margin-bottom: var(--space-lg);
        }
        
        .group-title {
          font-size: var(--text-xs);
          font-weight: 600;
          color: var(--text-secondary);
          text-transform: uppercase;
          margin-bottom: var(--space-sm);
        }
        
        .model-list {
          display: flex;
          flex-wrap: wrap;
          gap: var(--space-sm);
        }
        
        .model-chip {
          display: flex;
          flex-direction: column;
          padding: var(--space-sm) var(--space-md);
          background: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-md);
          cursor: pointer;
          transition: all var(--transition-fast);
          text-align: left;
        }
        
        .model-chip:hover {
          border-color: var(--glass-border-hover);
          transform: translateY(-1px);
        }
        
        .model-chip.selected {
          border-color: var(--color-accent-primary);
          background: rgba(99, 102, 241, 0.1);
        }
        
        .model-chip.sample {
          border-color: rgba(16, 185, 129, 0.4);
          background: rgba(16, 185, 129, 0.1);
        }
        
        .model-id {
          font-size: var(--text-sm);
          font-weight: 500;
          color: var(--text-primary);
        }
        
        .model-size, .model-desc {
          font-size: var(--text-xs);
          color: var(--text-tertiary);
        }
        
        .status-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-xs);
        }
        
        .status-item {
          display: flex;
          justify-content: space-between;
          padding: var(--space-xs) 0;
          border-bottom: 1px solid var(--glass-border);
        }
        
        .status-item:last-child {
          border-bottom: none;
        }
        
        .status-label {
          font-size: var(--text-sm);
          color: var(--text-secondary);
        }
        
        .status-value {
          font-size: var(--text-sm);
          font-weight: 500;
          color: var(--text-primary);
        }
        
        .cache-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-xs);
        }
        
        .cache-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: var(--space-sm);
          background: var(--glass-bg);
          border-radius: var(--radius-md);
        }
        
        .cache-item.sample {
          background: rgba(16, 185, 129, 0.05);
        }
        
        .cache-info {
          display: flex;
          flex-direction: column;
        }
        
        .cache-name {
          font-size: var(--text-sm);
          color: var(--text-primary);
        }
        
        .cache-size {
          font-size: var(--text-xs);
          color: var(--text-tertiary);
        }
        
        .ml-auto {
          margin-left: auto;
        }
        
        .text-success {
          color: var(--color-success);
        }
        
        .spinning {
          animation: spin 1s linear infinite;
        }
        
        .loading-placeholder {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
          color: var(--text-tertiary);
        }
      `}</style>
    </div>
  );
}
