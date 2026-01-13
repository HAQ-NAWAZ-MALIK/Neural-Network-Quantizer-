import { useState, useEffect } from 'react';
import {
    Layers,
    Play,
    Settings2,
    Zap,
    RefreshCw,
    Download
} from 'lucide-react';
import { useQuantizationStore, useModelStore } from '../store';
import { motion, AnimatePresence } from 'framer-motion';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Cell, AreaChart, Area
} from 'recharts';

/**
 * Quantizer page - main quantization interface
 */
export default function Quantizer() {
    const { result, isQuantizing, quantizeWeights, quantizeLayer, quantizeModel, clearResult } = useQuantizationStore();
    const { modelInfo, layers, fetchLayers } = useModelStore();

    // Configuration state
    const [config, setConfig] = useState({
        inFeatures: 64,
        outFeatures: 128,
        bits: 8,
        method: 'int8',
        mode: 'symmetric',
        groupSize: null,
        pattern: 'random',
        dtype: 'float32'
    });

    const [activeTab, setActiveTab] = useState('heatmaps');
    const [source, setSource] = useState('custom'); // 'custom' | 'layer'
    const [target, setTarget] = useState('single'); // 'single' | 'full'
    const [selectedLayer, setSelectedLayer] = useState('');

    // Switch to layer mode if model is loaded
    useEffect(() => {
        if (modelInfo) {
            setSource('layer');
            if (layers.length === 0) fetchLayers();
        }
    }, [modelInfo]);

    const handleQuantize = async () => {
        if (source === 'layer') {
            if (target === 'full') {
                await quantizeModel(config);
            } else if (selectedLayer) {
                await quantizeLayer(selectedLayer, config);
            }
        } else {
            await quantizeWeights(config);
        }
    };

    const updateConfig = (key, value) => {
        setConfig((prev) => ({ ...prev, [key]: value }));

        // Auto-update method based on bits
        if (key === 'bits') {
            if (value === 4) {
                setConfig((prev) => ({ ...prev, bits: value, method: 'int4', groupSize: 128 }));
            } else {
                setConfig((prev) => ({ ...prev, bits: value, method: 'int8', groupSize: null }));
            }
        }
    };

    // Convert histogram data for Recharts
    const getHistogramData = (viz) => {
        if (!viz?.data) return [];
        const { x, y } = viz.data;
        if (!x || !y) return [];
        return x.map((val, i) => ({
            value: typeof val === 'number' ? val.toFixed(3) : val,
            count: y[i]
        }));
    };

    // Generate heatmap as a simple statistical summary
    const getHeatmapStats = (viz) => {
        if (!viz?.data?.z) return null;
        const z = viz.data.z;
        const flat = z.flat();
        return {
            min: Math.min(...flat).toFixed(4),
            max: Math.max(...flat).toFixed(4),
            mean: (flat.reduce((a, b) => a + b, 0) / flat.length).toFixed(4),
            rows: z.length,
            cols: z[0]?.length || 0
        };
    };

    return (
        <div className="quantizer">
            {/* Header */}
            <div className="page-header">
                <h1 className="page-title">Weight Quantizer</h1>
                <p className="page-subtitle">
                    Quantize neural network weights to lower precision formats
                </p>
                {modelInfo && (
                    <div className="model-badge" style={{ marginTop: '0.5rem', display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '4px 12px', background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', color: 'var(--color-accent-primary)', borderRadius: 'var(--radius-full)', fontSize: '0.875rem' }}>
                        <span style={{ opacity: 0.7 }}>Active Model:</span>
                        <strong>{modelInfo.name}</strong>
                    </div>
                )}
            </div>

            <div className="quantizer-layout">
                {/* Configuration Panel */}
                <motion.div
                    className="glass-card config-panel no-hover"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                >
                    <div className="panel-header">
                        <Settings2 size={20} />
                        <h2>Configuration</h2>
                    </div>

                    {/* Source Selection */}
                    {modelInfo && (
                        <div className="config-section">
                            <h3 className="config-section-title">Data Source</h3>
                            <div className="btn-group mb-md">
                                <button
                                    className={`btn ${source === 'layer' ? 'btn-primary' : 'btn-secondary'}`}
                                    onClick={() => setSource('layer')}
                                >
                                    Loaded Model
                                </button>
                                <button
                                    className={`btn ${source === 'custom' ? 'btn-primary' : 'btn-secondary'}`}
                                    onClick={() => setSource('custom')}
                                >
                                    Custom Weights
                                </button>
                            </div>

                            {source === 'layer' && (
                                <>
                                    <h3 className="config-section-title mt-md">Scope</h3>
                                    <div className="btn-group">
                                        <button
                                            className={`btn ${target === 'single' ? 'btn-primary' : 'btn-secondary'}`}
                                            onClick={() => setTarget('single')}
                                        >
                                            Single Layer
                                        </button>
                                        <button
                                            className={`btn ${target === 'full' ? 'btn-primary' : 'btn-secondary'}`}
                                            onClick={() => setTarget('full')}
                                        >
                                            All Layers
                                        </button>
                                    </div>
                                </>
                            )}
                        </div>
                    )}

                    {/* Weight Selection */}
                    {(source === 'custom' || target === 'single') && (
                        <div className="config-section">
                            <h3 className="config-section-title">
                                {source === 'layer' ? 'Select Layer' : 'Weight Dimensions'}
                            </h3>

                            {source === 'layer' ? (
                                <div className="input-group">
                                    <label className="input-label">Layer</label>
                                    <select
                                        className="input select"
                                        value={selectedLayer}
                                        onChange={(e) => setSelectedLayer(e.target.value)}
                                    >
                                        <option value="">Select a layer...</option>
                                        {layers.map((layer) => (
                                            <option key={layer} value={layer}>
                                                {layer}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            ) : (
                                <>
                                    <div className="input-group">
                                        <label className="input-label">Output Features</label>
                                        <input
                                            type="range"
                                            className="slider"
                                            min={8}
                                            max={512}
                                            step={8}
                                            value={config.outFeatures}
                                            onChange={(e) => updateConfig('outFeatures', parseInt(e.target.value))}
                                        />
                                        <span className="slider-value">{config.outFeatures}</span>
                                    </div>

                                    <div className="input-group">
                                        <label className="input-label">Input Features</label>
                                        <input
                                            type="range"
                                            className="slider"
                                            min={8}
                                            max={512}
                                            step={8}
                                            value={config.inFeatures}
                                            onChange={(e) => updateConfig('inFeatures', parseInt(e.target.value))}
                                        />
                                        <span className="slider-value">{config.inFeatures}</span>
                                    </div>
                                </>
                            )}
                        </div>
                    )}

                    {/* Quantization Settings */}
                    <div className="config-section">
                        <h3 className="config-section-title">Quantization</h3>

                        <div className="input-group">
                            <label className="input-label">Precision (bits)</label>
                            <div className="btn-group">
                                <button
                                    className={`btn ${config.bits === 8 ? 'btn-primary' : 'btn-secondary'}`}
                                    onClick={() => updateConfig('bits', 8)}
                                >
                                    8-bit
                                </button>
                                <button
                                    className={`btn ${config.bits === 4 ? 'btn-primary' : 'btn-secondary'}`}
                                    onClick={() => updateConfig('bits', 4)}
                                >
                                    4-bit
                                </button>
                            </div>
                        </div>

                        <div className="input-group">
                            <label className="input-label">Method</label>
                            <select
                                className="input select"
                                value={config.method}
                                onChange={(e) => updateConfig('method', e.target.value)}
                            >
                                <option value="int8">INT8 (Per-Channel)</option>
                                <option value="int4">INT4 (Grouped)</option>
                                <option value="nf4">NF4 (QLoRA Style)</option>
                            </select>
                        </div>

                        <div className="input-group">
                            <label className="input-label">Mode</label>
                            <select
                                className="input select"
                                value={config.mode}
                                onChange={(e) => updateConfig('mode', e.target.value)}
                            >
                                <option value="symmetric">Symmetric</option>
                                <option value="asymmetric">Asymmetric</option>
                            </select>
                        </div>

                        {config.bits === 4 && (
                            <div className="input-group">
                                <label className="input-label">Group Size</label>
                                <select
                                    className="input select"
                                    value={config.groupSize || 128}
                                    onChange={(e) => updateConfig('groupSize', parseInt(e.target.value))}
                                >
                                    <option value={32}>32</option>
                                    <option value={64}>64</option>
                                    <option value={128}>128</option>
                                    <option value={256}>256</option>
                                </select>
                            </div>
                        )}
                    </div>

                    {/* Weight Pattern (Custom Only) */}
                    {source === 'custom' && (
                        <div className="config-section">
                            <h3 className="config-section-title">Weight Pattern</h3>

                            <div className="pattern-grid">
                                {['random', 'gradient', 'ones', 'alternating', 'eye'].map((pattern) => (
                                    <button
                                        key={pattern}
                                        className={`pattern-btn ${config.pattern === pattern ? 'active' : ''}`}
                                        onClick={() => updateConfig('pattern', pattern)}
                                    >
                                        {pattern}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Actions */}
                    <div className="config-actions">
                        <button
                            className={`btn btn-accent btn-lg w-full ${isQuantizing ? 'loading' : ''}`}
                            onClick={handleQuantize}
                            disabled={isQuantizing || (source === 'layer' && target === 'single' && !selectedLayer)}
                        >
                            {isQuantizing ? (
                                <>
                                    <RefreshCw className="spin" size={20} />
                                    <span>Processing...</span>
                                </>
                            ) : (
                                <>
                                    <Zap size={20} />
                                    <span>{target === 'full' ? 'Quantize Entire Model' : 'Quantize'}</span>
                                </>
                            )}
                        </button>

                        {result && (
                            <button
                                className="btn btn-secondary w-full"
                                onClick={clearResult}
                            >
                                Clear Results
                            </button>
                        )}
                    </div>
                </motion.div>

                {/* Results Panel */}
                <div className="results-panel">
                    <AnimatePresence mode="wait">
                        {result ? (
                            result.summary ? (
                                <FullModelResults result={result} />
                            ) : (
                                <motion.div
                                    key="results"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -20 }}
                                >
                                    {/* Stats */}
                                    <div className="glass-card stats-summary">
                                        <div className="stats-grid">
                                            <div className="stat-item">
                                                <div className="stat-value accent">{result.stats.memory_savings_percent.toFixed(1)}%</div>
                                                <div className="stat-label">Memory Saved</div>
                                            </div>
                                            <div className="stat-item">
                                                <div className="stat-value">{result.stats.max_error.toFixed(6)}</div>
                                                <div className="stat-label">Max Error</div>
                                            </div>
                                            <div className="stat-item">
                                                <div className="stat-value">{result.stats.mean_error.toFixed(6)}</div>
                                                <div className="stat-label">Mean Error</div>
                                            </div>
                                            <div className="stat-item">
                                                <div className="stat-value">{config.bits}-bit</div>
                                                <div className="stat-label">Precision</div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Visualization Tabs */}
                                    <div className="tabs">
                                        <button
                                            className={`tab ${activeTab === 'heatmaps' ? 'active' : ''}`}
                                            onClick={() => setActiveTab('heatmaps')}
                                        >
                                            Statistics
                                        </button>
                                        <button
                                            className={`tab ${activeTab === 'distributions' ? 'active' : ''}`}
                                            onClick={() => setActiveTab('distributions')}
                                        >
                                            Distributions
                                        </button>
                                        <button
                                            className={`tab ${activeTab === 'error' ? 'active' : ''}`}
                                            onClick={() => setActiveTab('error')}
                                        >
                                            Details
                                        </button>
                                    </div>

                                    {/* Charts */}
                                    <div className="charts-container">
                                        {activeTab === 'heatmaps' && (
                                            <div className="grid grid-2">
                                                {['original_heatmap', 'quantized_heatmap', 'dequantized_heatmap', 'error_heatmap'].map((key) => {
                                                    const viz = result.visualizations[key];
                                                    const stats = getHeatmapStats(viz);
                                                    const title = viz?.layout?.title || key.replace('_', ' ');
                                                    return (
                                                        <div key={key} className="glass-card chart-card">
                                                            <h4 className="chart-title">{title}</h4>
                                                            {stats && (
                                                                <div className="heatmap-stats">
                                                                    <div className="heatmap-stat">
                                                                        <span className="label">Shape:</span>
                                                                        <span className="value">{stats.rows} × {stats.cols}</span>
                                                                    </div>
                                                                    <div className="heatmap-stat">
                                                                        <span className="label">Min:</span>
                                                                        <span className="value">{stats.min}</span>
                                                                    </div>
                                                                    <div className="heatmap-stat">
                                                                        <span className="label">Max:</span>
                                                                        <span className="value">{stats.max}</span>
                                                                    </div>
                                                                    <div className="heatmap-stat">
                                                                        <span className="label">Mean:</span>
                                                                        <span className="value">{stats.mean}</span>
                                                                    </div>
                                                                </div>
                                                            )}
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        )}

                                        {activeTab === 'distributions' && (
                                            <div className="grid grid-2">
                                                <div className="glass-card chart-card">
                                                    <h4 className="chart-title">Original Weight Distribution</h4>
                                                    <ResponsiveContainer width="100%" height={250}>
                                                        <BarChart data={getHistogramData(result.visualizations.original_histogram)}>
                                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                                            <XAxis dataKey="value" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                                                            <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                                                            <Tooltip
                                                                contentStyle={{
                                                                    backgroundColor: '#1a1a25',
                                                                    border: '1px solid rgba(255,255,255,0.1)',
                                                                    borderRadius: '8px'
                                                                }}
                                                            />
                                                            <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
                                                        </BarChart>
                                                    </ResponsiveContainer>
                                                </div>

                                                <div className="glass-card chart-card">
                                                    <h4 className="chart-title">Quantized Weight Distribution</h4>
                                                    <ResponsiveContainer width="100%" height={250}>
                                                        <BarChart data={getHistogramData(result.visualizations.quantized_histogram)}>
                                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                                            <XAxis dataKey="value" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                                                            <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                                                            <Tooltip
                                                                contentStyle={{
                                                                    backgroundColor: '#1a1a25',
                                                                    border: '1px solid rgba(255,255,255,0.1)',
                                                                    borderRadius: '8px'
                                                                }}
                                                            />
                                                            <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
                                                        </BarChart>
                                                    </ResponsiveContainer>
                                                </div>

                                                <div className="glass-card chart-card" style={{ gridColumn: 'span 2' }}>
                                                    <h4 className="chart-title">Quantization Scales Distribution</h4>
                                                    <ResponsiveContainer width="100%" height={250}>
                                                        <AreaChart data={getHistogramData(result.visualizations.scales_histogram)}>
                                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                                            <XAxis dataKey="value" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                                                            <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                                                            <Tooltip
                                                                contentStyle={{
                                                                    backgroundColor: '#1a1a25',
                                                                    border: '1px solid rgba(255,255,255,0.1)',
                                                                    borderRadius: '8px'
                                                                }}
                                                            />
                                                            <Area
                                                                type="monotone"
                                                                dataKey="count"
                                                                stroke="#8b5cf6"
                                                                fill="rgba(139, 92, 246, 0.3)"
                                                            />
                                                        </AreaChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            </div>
                                        )}

                                        {activeTab === 'error' && (
                                            <div className="glass-card error-stats">
                                                <h3>Quantization Details</h3>
                                                <div className="error-details">
                                                    {result.layer_name && (
                                                        <div className="detail-row">
                                                            <span>Analyzed Layer:</span>
                                                            <code style={{ color: 'var(--color-accent-primary)' }}>{result.layer_name}</code>
                                                        </div>
                                                    )}
                                                    <div className="detail-row">
                                                        <span>Original Shape:</span>
                                                        <code>{JSON.stringify(result.stats.original_shape)}</code>
                                                    </div>
                                                    <div className="detail-row">
                                                        <span>Quantized Shape:</span>
                                                        <code>{JSON.stringify(result.stats.quantized_shape)}</code>
                                                    </div>
                                                    <div className="detail-row">
                                                        <span>Scales Shape:</span>
                                                        <code>{JSON.stringify(result.stats.scales_shape)}</code>
                                                    </div>
                                                    <div className="detail-row">
                                                        <span>Original Dtype:</span>
                                                        <code>{result.stats.original_dtype}</code>
                                                    </div>
                                                    <div className="detail-row">
                                                        <span>Quantized Dtype:</span>
                                                        <code>{result.stats.quantized_dtype}</code>
                                                    </div>
                                                    <div className="detail-row">
                                                        <span>Max Error:</span>
                                                        <code>{result.stats.max_error.toExponential(4)}</code>
                                                    </div>
                                                    <div className="detail-row">
                                                        <span>Mean Error:</span>
                                                        <code>{result.stats.mean_error.toExponential(4)}</code>
                                                    </div>
                                                    <div className="detail-row">
                                                        <span>Memory Savings:</span>
                                                        <code className="success">{result.stats.memory_savings_percent.toFixed(2)}%</code>
                                                    </div>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </motion.div>
                            )
                        ) : (
                            <motion.div
                                key="placeholder"
                                className="results-placeholder glass-card"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                            >
                                <Layers size={64} />
                                <h3>No Results Yet</h3>
                                <p>Configure your quantization settings and click "Quantize" to see the results.</p>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>

            <style>{`
        .quantizer-layout {
          display: grid;
          grid-template-columns: 320px 1fr;
          gap: var(--space-xl);
          align-items: start;
        }
        
        @media (max-width: 1024px) {
          .quantizer-layout {
            grid-template-columns: 1fr;
          }
        }
        
        .config-panel {
          position: sticky;
          top: var(--space-xl);
        }
        
        .panel-header {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
          margin-bottom: var(--space-lg);
          color: var(--text-primary);
        }
        
        .panel-header h2 {
          font-size: var(--text-lg);
          font-weight: 600;
          margin: 0;
        }
        
        .config-section {
          margin-bottom: var(--space-lg);
          padding-bottom: var(--space-lg);
          border-bottom: 1px solid var(--glass-border);
        }
        
        .config-section:last-of-type {
          border-bottom: none;
        }
        
        .config-section-title {
          font-size: var(--text-sm);
          font-weight: 600;
          color: var(--text-secondary);
          margin-bottom: var(--space-md);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        
        .input-group {
          margin-bottom: var(--space-md);
        }
        
        .slider-value {
          display: block;
          text-align: right;
          font-size: var(--text-sm);
          color: var(--text-secondary);
          font-family: var(--font-mono);
          margin-top: var(--space-xs);
        }
        
        .btn-group {
          display: flex;
          gap: var(--space-xs);
        }
        
        .btn-group .btn {
          flex: 1;
        }
        
        .pattern-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: var(--space-xs);
        }
        
        .pattern-btn {
          padding: var(--space-sm);
          background: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-md);
          color: var(--text-secondary);
          font-size: var(--text-xs);
          cursor: pointer;
          transition: all var(--transition-fast);
          text-transform: capitalize;
        }
        
        .pattern-btn:hover {
          border-color: var(--glass-border-hover);
          color: var(--text-primary);
        }
        
        .pattern-btn.active {
          background: var(--color-accent-primary);
          border-color: var(--color-accent-primary);
          color: white;
        }
        
        .config-actions {
          display: flex;
          flex-direction: column;
          gap: var(--space-sm);
          margin-top: var(--space-lg);
        }
        
        .spinning {
          animation: spin 1s linear infinite;
        }
        
        .results-panel {
          min-height: 500px;
        }
        
        .results-placeholder {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          padding: var(--space-3xl);
          min-height: 400px;
          color: var(--text-tertiary);
        }
        
        .results-placeholder h3 {
          margin-top: var(--space-lg);
          color: var(--text-secondary);
        }
        
        .results-placeholder p {
          max-width: 300px;
        }
        
        .stats-summary {
          margin-bottom: var(--space-lg);
        }
        
        .stats-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: var(--space-lg);
          text-align: center;
        }
        
        .stat-item .stat-value {
          font-size: var(--text-2xl);
          font-weight: 700;
          color: var(--text-primary);
        }
        
        .stat-item .stat-value.accent {
          background: var(--gradient-primary);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        
        .stat-item .stat-label {
          font-size: var(--text-xs);
          color: var(--text-secondary);
          margin-top: var(--space-xs);
        }
        
        .charts-container {
          margin-top: var(--space-lg);
        }
        
        .chart-card {
          padding: var(--space-lg);
        }
        
        .chart-title {
          font-size: var(--text-sm);
          font-weight: 600;
          color: var(--text-primary);
          margin-bottom: var(--space-md);
        }
        
        .heatmap-stats {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: var(--space-sm);
        }
        
        .heatmap-stat {
          display: flex;
          justify-content: space-between;
          font-size: var(--text-sm);
        }
        
        .heatmap-stat .label {
          color: var(--text-secondary);
        }
        
        .heatmap-stat .value {
          color: var(--text-primary);
          font-family: var(--font-mono);
        }
        
        .error-stats {
          padding: var(--space-lg);
        }
        
        .error-stats h3 {
          font-size: var(--text-base);
          margin-bottom: var(--space-md);
        }
        
        .error-details {
          display: flex;
          flex-direction: column;
          gap: var(--space-sm);
        }
        
        .detail-row {
          display: flex;
          justify-content: space-between;
          font-size: var(--text-sm);
          padding: var(--space-sm);
          background: var(--glass-bg);
          border-radius: var(--radius-md);
        }
        
        .detail-row span {
          color: var(--text-secondary);
        }
        
        .detail-row code {
          color: var(--color-accent-primary);
        }
        
        .detail-row code.success {
          color: var(--color-success);
        }
        
        @media (max-width: 768px) {
          .stats-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }
      `}</style>
        </div>
    );
}

function FullModelResults({ result }) {
    if (!result || !result.summary) return null;

    return (
        <motion.div
            key="full-results"
            className="glass-card full-results"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
        >
            <h3>Model Quantization Summary</h3>

            <div className="stats-grid mb-lg">
                <div className="stat-item">
                    <div className="stat-value accent">
                        {result.summary.total_memory_saved_mb.toFixed(1)} MB
                    </div>
                    <div className="stat-label">Total Saved</div>
                </div>
                <div className="stat-item">
                    <div className="stat-value">
                        {result.summary.average_error.toExponential(2)}
                    </div>
                    <div className="stat-label">Avg Error</div>
                </div>
                <div className="stat-item">
                    <div className="stat-value">
                        {result.summary.layers_quantized}/{result.summary.total_layers}
                    </div>
                    <div className="stat-label">Layers</div>
                </div>
            </div>

            <h4 style={{ margin: 'var(--space-md) 0' }}>Layer Details</h4>
            <div className="table-responsive" style={{ maxHeight: '400px', overflowY: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ color: 'var(--text-secondary)', fontSize: 'var(--text-xs)', textAlign: 'left' }}>
                            <th style={{ padding: '8px' }}>Layer</th>
                            <th style={{ padding: '8px' }}>Shape</th>
                            <th style={{ padding: '8px', textAlign: 'right' }}>Error</th>
                            <th style={{ padding: '8px', textAlign: 'right' }}>Saved</th>
                        </tr>
                    </thead>
                    <tbody>
                        {result.layers.map((layer, i) => (
                            <tr key={i} style={{ borderTop: '1px solid var(--glass-border)' }}>
                                <td style={{ padding: '8px', fontFamily: 'var(--font-mono)', fontSize: '11px' }} title={layer.layer_name}>
                                    {layer.layer_name.split('.').slice(-2).join('.')}
                                </td>
                                <td style={{ padding: '8px', fontSize: '11px', color: 'var(--text-secondary)' }}>
                                    {JSON.stringify(layer.shape)}
                                </td>
                                <td style={{ padding: '8px', textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
                                    {layer.error?.toExponential(2) || 'N/A'}
                                </td>
                                <td style={{ padding: '8px', textAlign: 'right', color: 'var(--color-success)', fontSize: '11px' }}>
                                    {layer.memory_savings_percent?.toFixed(1)}%
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </motion.div>
    );
}
