import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
    Zap,
    Cpu,
    HardDrive,
    TrendingUp,
    ArrowRight,
    Layers,
    Activity,
    MemoryStick
} from 'lucide-react';
import { useSystemStore, useQuantizationStore, useModelStore } from '../store';
import { motion } from 'framer-motion';

/**
 * Dashboard page - overview of system and recent activity
 */
export default function Dashboard() {
    const systemInfo = useSystemStore((state) => state.systemInfo);
    const fetchSystemInfo = useSystemStore((state) => state.fetchSystemInfo);
    const quantizationHistory = useQuantizationStore((state) => state.history);
    const modelInfo = useModelStore((state) => state.modelInfo);

    useEffect(() => {
        if (!systemInfo) {
            fetchSystemInfo();
        }
    }, [systemInfo, fetchSystemInfo]);

    const stats = [
        {
            label: 'GPU Status',
            value: systemInfo?.cuda_available ? 'CUDA Ready' : systemInfo?.mps_available ? 'MPS Ready' : 'CPU Only',
            icon: Cpu,
            color: systemInfo?.cuda_available ? 'success' : 'warning',
            detail: systemInfo?.gpus?.[0]?.name || 'No GPU detected'
        },
        {
            label: 'Available RAM',
            value: `${systemInfo?.ram_available_gb?.toFixed(1) || '?'}GB`,
            icon: MemoryStick,
            color: 'info',
            detail: `of ${systemInfo?.ram_total_gb?.toFixed(1) || '?'}GB total`
        },
        {
            label: 'Max Model Size',
            value: systemInfo?.max_model_size || 'Unknown',
            icon: Layers,
            color: 'accent',
            detail: 'Recommended limit'
        },
        {
            label: 'Quantizations',
            value: quantizationHistory.length,
            icon: Activity,
            color: 'success',
            detail: 'This session'
        }
    ];

    const quickActions = [
        {
            title: 'Quick Quantize',
            description: 'Test quantization on random weights',
            path: '/quantize',
            icon: Zap,
            gradient: 'var(--gradient-primary)'
        },
        {
            title: 'Load Model',
            description: 'Load a HuggingFace model',
            path: '/models',
            icon: HardDrive,
            gradient: 'var(--gradient-secondary)'
        },
        {
            title: 'Analyze Weights',
            description: 'Deep dive into weight distributions',
            path: '/analysis',
            icon: TrendingUp,
            gradient: 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)'
        }
    ];

    return (
        <div className="dashboard">
            {/* Header */}
            <div className="page-header">
                <h1 className="page-title">Dashboard</h1>
                <p className="page-subtitle">
                    Neural Network Weight Quantization Tool
                </p>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-4 stagger">
                {stats.map((stat, index) => (
                    <motion.div
                        key={stat.label}
                        className="glass-card stat-card"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                    >
                        <div className={`stat-icon ${stat.color}`}>
                            <stat.icon size={20} />
                        </div>
                        <div className="stat-content">
                            <div className="stat-value">{stat.value}</div>
                            <div className="stat-label">{stat.label}</div>
                            <div className="stat-detail">{stat.detail}</div>
                        </div>
                    </motion.div>
                ))}
            </div>

            {/* Quick Actions */}
            <section className="section">
                <h2 className="section-title">Quick Actions</h2>
                <div className="grid grid-3">
                    {quickActions.map((action, index) => (
                        <motion.div
                            key={action.path}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.4 + index * 0.1 }}
                        >
                            <Link to={action.path} className="action-card glass-card">
                                <div className="action-icon" style={{ background: action.gradient }}>
                                    <action.icon size={24} />
                                </div>
                                <div className="action-content">
                                    <h3 className="action-title">{action.title}</h3>
                                    <p className="action-description">{action.description}</p>
                                </div>
                                <ArrowRight size={20} className="action-arrow" />
                            </Link>
                        </motion.div>
                    ))}
                </div>
            </section>

            {/* Current Model */}
            {modelInfo && (
                <section className="section">
                    <h2 className="section-title">Loaded Model</h2>
                    <div className="glass-card model-info">
                        <div className="model-header">
                            <HardDrive size={24} />
                            <div>
                                <h3 className="model-name">{modelInfo.name}</h3>
                                <p className="model-arch">{modelInfo.architecture}</p>
                            </div>
                        </div>
                        <div className="model-stats">
                            <div className="model-stat">
                                <span className="stat-value">{modelInfo.num_params_billions?.toFixed(2)}B</span>
                                <span className="stat-label">Parameters</span>
                            </div>
                            <div className="model-stat">
                                <span className="stat-value">{modelInfo.num_quantizable_layers}</span>
                                <span className="stat-label">Quantizable Layers</span>
                            </div>
                            <div className="model-stat">
                                <span className="stat-value">{modelInfo.memory_footprint_gb}GB</span>
                                <span className="stat-label">Memory</span>
                            </div>
                        </div>
                    </div>
                </section>
            )}

            {/* Getting Started */}
            {!modelInfo && quantizationHistory.length === 0 && (
                <section className="section">
                    <div className="glass-card getting-started">
                        <div className="getting-started-content">
                            <Zap size={48} className="getting-started-icon" />
                            <h2>Get Started</h2>
                            <p>
                                Welcome to the Neural Network Quantizer! You can either test quantization
                                on random weights or load a real HuggingFace model for production use.
                            </p>
                            <div className="getting-started-actions">
                                <Link to="/quantize" className="btn btn-primary btn-lg">
                                    <Layers size={20} />
                                    Try Quantization
                                </Link>
                                <Link to="/models" className="btn btn-secondary btn-lg">
                                    <HardDrive size={20} />
                                    Load Model
                                </Link>
                            </div>
                        </div>
                    </div>
                </section>
            )}

            {/* System Warnings */}
            {systemInfo?.warnings?.length > 0 && (
                <section className="section">
                    <h2 className="section-title">System Warnings</h2>
                    <div className="warnings-list">
                        {systemInfo.warnings.map((warning, index) => (
                            <div key={index} className="warning-item glass-card">
                                <span className="badge badge-warning">Warning</span>
                                <span>{warning}</span>
                            </div>
                        ))}
                    </div>
                </section>
            )}

            <style>{`
        .dashboard {
          max-width: 1400px;
        }
        
        .section {
          margin-top: var(--space-2xl);
        }
        
        .section-title {
          font-size: var(--text-xl);
          font-weight: 600;
          margin-bottom: var(--space-lg);
          color: var(--text-primary);
        }
        
        .stat-card {
          display: flex;
          align-items: flex-start;
          gap: var(--space-md);
        }
        
        .stat-icon {
          width: 44px;
          height: 44px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: var(--radius-lg);
          flex-shrink: 0;
        }
        
        .stat-icon.success {
          background: var(--color-success-bg);
          color: var(--color-success);
        }
        
        .stat-icon.warning {
          background: var(--color-warning-bg);
          color: var(--color-warning);
        }
        
        .stat-icon.info {
          background: var(--color-info-bg);
          color: var(--color-info);
        }
        
        .stat-icon.accent {
          background: rgba(99, 102, 241, 0.1);
          color: var(--color-accent-primary);
        }
        
        .stat-content {
          flex: 1;
        }
        
        .stat-card .stat-value {
          font-size: var(--text-xl);
          font-weight: 700;
          color: var(--text-primary);
          line-height: 1.2;
        }
        
        .stat-card .stat-label {
          font-size: var(--text-sm);
          color: var(--text-secondary);
        }
        
        .stat-detail {
          font-size: var(--text-xs);
          color: var(--text-tertiary);
          margin-top: var(--space-xs);
        }
        
        .action-card {
          display: flex;
          align-items: center;
          gap: var(--space-md);
          text-decoration: none;
          transition: all var(--transition-base);
        }
        
        .action-card:hover {
          transform: translateY(-4px);
        }
        
        .action-card:hover .action-arrow {
          transform: translateX(4px);
        }
        
        .action-icon {
          width: 48px;
          height: 48px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: var(--radius-lg);
          color: white;
          flex-shrink: 0;
        }
        
        .action-content {
          flex: 1;
        }
        
        .action-title {
          font-size: var(--text-base);
          font-weight: 600;
          color: var(--text-primary);
          margin-bottom: var(--space-xs);
        }
        
        .action-description {
          font-size: var(--text-sm);
          color: var(--text-secondary);
          margin: 0;
        }
        
        .action-arrow {
          color: var(--text-tertiary);
          transition: transform var(--transition-fast);
        }
        
        .model-info {
          padding: var(--space-xl);
        }
        
        .model-header {
          display: flex;
          align-items: center;
          gap: var(--space-md);
          margin-bottom: var(--space-lg);
          color: var(--color-accent-primary);
        }
        
        .model-name {
          font-size: var(--text-lg);
          font-weight: 600;
          color: var(--text-primary);
        }
        
        .model-arch {
          font-size: var(--text-sm);
          color: var(--text-secondary);
          margin: 0;
        }
        
        .model-stats {
          display: flex;
          gap: var(--space-2xl);
        }
        
        .model-stat {
          display: flex;
          flex-direction: column;
        }
        
        .getting-started {
          text-align: center;
          padding: var(--space-3xl);
        }
        
        .getting-started-icon {
          color: var(--color-accent-primary);
          margin-bottom: var(--space-lg);
        }
        
        .getting-started h2 {
          margin-bottom: var(--space-md);
        }
        
        .getting-started p {
          max-width: 500px;
          margin: 0 auto var(--space-xl);
        }
        
        .getting-started-actions {
          display: flex;
          gap: var(--space-md);
          justify-content: center;
        }
        
        .warnings-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-sm);
        }
        
        .warning-item {
          display: flex;
          align-items: center;
          gap: var(--space-md);
          padding: var(--space-md);
        }
      `}</style>
        </div>
    );
}
