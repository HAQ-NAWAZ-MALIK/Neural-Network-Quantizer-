import { useState, useEffect } from 'react';
import {
    BarChart3,
    Layers,
    TrendingUp,
    RefreshCw,
    AlertTriangle
} from 'lucide-react';
import { useQuantizationStore, useModelStore } from '../store';
import { motion } from 'framer-motion';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Cell, Legend
} from 'recharts';

/**
 * Analysis page - compare quantization methods and analyze weights
 */
export default function Analysis() {
    const { compareMethod } = useQuantizationStore();
    const { modelInfo, layers, fetchLayers } = useModelStore();

    const [comparison, setComparison] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [selectedMethods, setSelectedMethods] = useState(['int8', 'int4', 'nf4']);
    const [source, setSource] = useState('random'); // 'random' | 'layer'
    const [selectedLayer, setSelectedLayer] = useState('');

    // Switch to layer mode if model is loaded
    // Switch to layer mode if model is loaded
    useEffect(() => {
        if (modelInfo) {
            setSource('layer');
            if (layers.length === 0) fetchLayers();
        }
    }, [modelInfo]);

    const runComparison = async () => {
        setIsLoading(true);
        const layerToCompare = source === 'layer' ? selectedLayer : null;
        const result = await compareMethod(selectedMethods, layerToCompare);
        setComparison(result);
        setIsLoading(false);
    };

    const toggleMethod = (method) => {
        setSelectedMethods((prev) =>
            prev.includes(method)
                ? prev.filter(m => m !== method)
                : [...prev, method]
        );
    };

    // Prepare chart data
    const getComparisonData = () => {
        if (!comparison?.comparison) return [];
        return comparison.comparison
            .filter(c => !c.error)
            .map(c => ({
                method: c.method.toUpperCase(),
                meanError: c.mean_error,
                maxError: c.max_error,
                memorySavings: c.memory_savings_percent
            }));
    };

    const COLORS = ['#6366f1', '#8b5cf6', '#a855f7'];

    return (
        <div className="analysis">
            {/* Header */}
            <div className="page-header">
                <h1 className="page-title">Analysis</h1>
                <p className="page-subtitle">
                    Compare quantization methods and analyze weight distributions
                </p>
                {modelInfo && (
                    <div className="model-badge" style={{ marginTop: '0.5rem', display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '4px 12px', background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', color: 'var(--color-accent-primary)', borderRadius: 'var(--radius-full)', fontSize: '0.875rem' }}>
                        <span style={{ opacity: 0.7 }}>Active Model:</span>
                        <strong>{modelInfo.name}</strong>
                    </div>
                )}
            </div>

            {/* Method Comparison */}
            <section className="section">
                <div className="section-header">
                    <h2 className="section-title">
                        <BarChart3 size={20} />
                        Method Comparison
                        {comparison && (
                            <span className="source-badge">
                                Source: {comparison.source.startsWith('layer:') ? comparison.source.replace('layer:', '') : 'Random Weights'}
                            </span>
                        )}
                    </h2>
                    <button
                        className="btn btn-primary"
                        onClick={runComparison}
                        disabled={isLoading || selectedMethods.length === 0}
                    >
                        {isLoading ? (
                            <>
                                <RefreshCw size={16} className="spinning" />
                                Comparing...
                            </>
                        ) : (
                            <>
                                <TrendingUp size={16} />
                                Run Comparison
                            </>
                        )}
                    </button>
                </div>

                {/* Data Source Selection */}
                <div className="glass-card mb-lg">
                    <p className="text-sm text-muted mb-md">Select data source:</p>

                    <div className="source-selection mb-md">
                        <div className="btn-group">
                            {modelInfo && (
                                <button
                                    className={`btn ${source === 'layer' ? 'btn-primary' : 'btn-secondary'}`}
                                    onClick={() => setSource('layer')}
                                >
                                    Loaded Model Layer
                                </button>
                            )}
                            <button
                                className={`btn ${source === 'random' ? 'btn-primary' : 'btn-secondary'}`}
                                onClick={() => setSource('random')}
                            >
                                Random Weights
                            </button>
                        </div>
                    </div>

                    {source === 'layer' && (
                        <div className="layer-selection">
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
                    )}
                </div>

                {/* Method Selection */}
                <div className="glass-card">
                    <p className="text-sm text-muted mb-md">Select methods to compare:</p>
                    <div className="method-selection">
                        {['int8', 'int4', 'nf4'].map((method) => (
                            <button
                                key={method}
                                className={`method-btn ${selectedMethods.includes(method) ? 'active' : ''}`}
                                onClick={() => toggleMethod(method)}
                            >
                                <div className="method-check">
                                    {selectedMethods.includes(method) && '✓'}
                                </div>
                                <div className="method-info">
                                    <span className="method-name">{method.toUpperCase()}</span>
                                    <span className="method-desc">
                                        {method === 'int8' && '8-bit integer quantization'}
                                        {method === 'int4' && '4-bit integer with grouping'}
                                        {method === 'nf4' && 'Normal Float 4-bit (QLoRA)'}
                                    </span>
                                </div>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Comparison Results */}
                {comparison && (
                    <motion.div
                        className="comparison-results mt-lg"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                    >
                        <div className="grid grid-2">
                            {/* Error Chart */}
                            <div className="glass-card chart-card">
                                <h4 className="chart-title">Quantization Error by Method</h4>
                                <ResponsiveContainer width="100%" height={300}>
                                    <BarChart data={getComparisonData()}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                        <XAxis dataKey="method" tick={{ fill: '#94a3b8' }} />
                                        <YAxis tick={{ fill: '#94a3b8' }} />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: '#1a1a25',
                                                border: '1px solid rgba(255,255,255,0.1)',
                                                borderRadius: '8px'
                                            }}
                                        />
                                        <Bar dataKey="meanError" name="Mean Error" radius={[4, 4, 0, 0]}>
                                            {getComparisonData().map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Memory Savings Chart */}
                            <div className="glass-card chart-card">
                                <h4 className="chart-title">Memory Savings by Method</h4>
                                <ResponsiveContainer width="100%" height={300}>
                                    <BarChart data={getComparisonData()}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                        <XAxis dataKey="method" tick={{ fill: '#94a3b8' }} />
                                        <YAxis tick={{ fill: '#94a3b8' }} unit="%" />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: '#1a1a25',
                                                border: '1px solid rgba(255,255,255,0.1)',
                                                borderRadius: '8px'
                                            }}
                                            formatter={(value) => [`${value.toFixed(1)}%`, 'Savings']}
                                        />
                                        <Bar dataKey="memorySavings" name="Memory Savings" radius={[4, 4, 0, 0]}>
                                            {getComparisonData().map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Results Table */}
                        <div className="glass-card mt-lg">
                            <table className="results-table">
                                <thead>
                                    <tr>
                                        <th>Method</th>
                                        <th>Bits</th>
                                        <th>Max Error</th>
                                        <th>Mean Error</th>
                                        <th>Memory Savings</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {comparison.comparison?.filter(c => !c.error).map((result) => (
                                        <tr key={result.method}>
                                            <td><strong>{result.method.toUpperCase()}</strong></td>
                                            <td>{result.bits}</td>
                                            <td>{result.max_error?.toFixed(6)}</td>
                                            <td>{result.mean_error?.toFixed(6)}</td>
                                            <td>
                                                <span className="badge badge-success">
                                                    {result.memory_savings_percent?.toFixed(1)}%
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </motion.div>
                )}
            </section>

            {/* Model Analysis (if model loaded) */}
            {modelInfo && (
                <section className="section">
                    <h2 className="section-title">
                        <Layers size={20} />
                        Model Analysis
                    </h2>

                    <div className="glass-card">
                        <p>
                            Model <strong>{modelInfo.name}</strong> is loaded with{' '}
                            <strong>{modelInfo.num_quantizable_layers}</strong> quantizable layers.
                        </p>
                        <p className="text-sm text-muted mt-md">
                            Use the Models page to analyze individual layer weights and detect outliers.
                        </p>
                    </div>
                </section>
            )}

            {/* Info Section */}
            <section className="section">
                <div className="glass-card info-card">
                    <AlertTriangle size={24} className="text-warning" />
                    <div>
                        <h3>Understanding Quantization Trade-offs</h3>
                        <p>
                            Lower bit precision (4-bit) provides better memory savings but introduces more error.
                            8-bit quantization offers a good balance between compression and accuracy for most models.
                            NF4 uses a codebook optimized for normally distributed weights, ideal for LLMs.
                        </p>
                    </div>
                </div>
            </section>

            <style>{`
        .section {
          margin-top: var(--space-2xl);
        }
        
        .section-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: var(--space-lg);
        }
        
        .section-title {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
          font-size: var(--text-xl);
          font-weight: 600;
          margin: 0;
        }
        
        .method-selection {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: var(--space-md);
        }
        
        .method-btn {
          display: flex;
          align-items: flex-start;
          gap: var(--space-md);
          padding: var(--space-md);
          background: var(--glass-bg);
          border: 2px solid var(--glass-border);
          border-radius: var(--radius-lg);
          cursor: pointer;
          transition: all var(--transition-fast);
          text-align: left;
        }
        
        .method-btn:hover {
          border-color: var(--glass-border-hover);
        }
        
        .method-btn.active {
          border-color: var(--color-accent-primary);
          background: rgba(99, 102, 241, 0.1);
        }
        
        .method-check {
          width: 24px;
          height: 24px;
          display: flex;
          align-items: center;
          justify-content: center;
          border: 2px solid var(--glass-border);
          border-radius: var(--radius-md);
          font-size: var(--text-sm);
          color: var(--color-accent-primary);
          flex-shrink: 0;
        }
        
        .method-btn.active .method-check {
          background: var(--color-accent-primary);
          border-color: var(--color-accent-primary);
          color: white;
        }
        
        .method-info {
          display: flex;
          flex-direction: column;
        }
        
        .method-name {
          font-weight: 600;
          color: var(--text-primary);
        }
        
        .method-desc {
          font-size: var(--text-xs);
          color: var(--text-secondary);
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
        
        .results-table {
          width: 100%;
          border-collapse: collapse;
        }
        
        .results-table th,
        .results-table td {
          padding: var(--space-sm) var(--space-md);
          text-align: left;
          border-bottom: 1px solid var(--glass-border);
        }
        
        .results-table th {
          font-size: var(--text-xs);
          font-weight: 600;
          color: var(--text-secondary);
          text-transform: uppercase;
        }
        
        .results-table td {
          font-size: var(--text-sm);
          color: var(--text-primary);
        }
        
        .info-card {
          display: flex;
          gap: var(--space-lg);
          padding: var(--space-lg);
        }
        
        .info-card h3 {
          font-size: var(--text-base);
          margin-bottom: var(--space-sm);
        }
        
        .info-card p {
          margin: 0;
          font-size: var(--text-sm);
        }
        
        .text-warning {
          color: var(--color-warning);
          flex-shrink: 0;
        }
        
        .spinning {
          animation: spin 1s linear infinite;
        }
        
          .method-selection {
            grid-template-columns: 1fr;
          }
        }

        .btn-group {
          display: flex;
          gap: var(--space-xs);
          max-width: 400px;
        }
        
        .source-badge {
          font-size: var(--text-xs);
          font-weight: 500;
          padding: 4px 8px;
          background: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-full);
          color: var(--text-secondary);
          margin-left: var(--space-md);
        }
        
        .btn-group .btn {
          flex: 1;
        }

        .mb-lg { margin-bottom: var(--space-lg); }
        .mb-md { margin-bottom: var(--space-md); }
      `}</style>
        </div>
    );
}
