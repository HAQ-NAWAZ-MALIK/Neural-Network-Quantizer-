import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { useEffect } from 'react';
import {
  LayoutDashboard,
  Layers,
  BarChart3,
  Settings,
  Cpu,
  HardDrive,
  Zap,
  Github,
  Menu,
  X
} from 'lucide-react';
import { useSystemStore, useUIStore, useModelStore } from '../store';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * Main application layout with sidebar navigation
 */
export default function Layout() {
  const { sidebarOpen, toggleSidebar } = useUIStore();
  const systemInfo = useSystemStore((state) => state.systemInfo);
  const checkLoadedModel = useModelStore((state) => state.checkLoadedModel);
  const location = useLocation();

  // Sync model state on mount
  useEffect(() => {
    checkLoadedModel();
  }, []);

  const navItems = [
    { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/quantize', label: 'Quantizer', icon: Layers },
    { path: '/analysis', label: 'Analysis', icon: BarChart3 },
    { path: '/models', label: 'Models', icon: HardDrive },
  ];

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? '' : 'closed'}`}>
        {/* Logo */}
        <div className="sidebar-header">
          <div className="logo">
            <div className="logo-icon">
              <Zap size={24} />
            </div>
            <div className="logo-text">
              <span className="logo-title">Quantizer</span>
              <span className="logo-subtitle">Neural Network</span>
            </div>
          </div>
          <button className="btn btn-ghost btn-icon mobile-menu" onClick={toggleSidebar}>
            <X size={20} />
          </button>
        </div>

        {/* Navigation */}
        <nav className="sidebar-nav">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
            >
              <item.icon size={20} />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        {/* System Status */}
        <div className="sidebar-footer">
          <div className="system-status glass-card no-hover">
            <div className="status-header">
              <Cpu size={16} />
              <span>System Status</span>
            </div>
            {systemInfo ? (
              <div className="status-details">
                <div className="status-item">
                  <span className="status-label">GPU</span>
                  <span className={`badge ${systemInfo.cuda_available ? 'badge-success' : 'badge-warning'}`}>
                    {systemInfo.cuda_available ? 'CUDA' : systemInfo.mps_available ? 'MPS' : 'CPU'}
                  </span>
                </div>
                {systemInfo.gpus?.length > 0 && (
                  <div className="status-item">
                    <span className="status-label">{systemInfo.gpus[0].name}</span>
                    <span className="text-xs text-muted">{systemInfo.gpus[0].total_memory_gb}GB</span>
                  </div>
                )}
                <div className="status-item">
                  <span className="status-label">RAM</span>
                  <span className="text-xs text-muted">
                    {systemInfo.ram_available_gb?.toFixed(1)}GB / {systemInfo.ram_total_gb?.toFixed(1)}GB
                  </span>
                </div>
              </div>
            ) : (
              <div className="status-loading">
                <div className="spinner"></div>
                <span className="text-xs text-muted">Detecting...</span>
              </div>
            )}
          </div>

          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="nav-item github-link"
          >
            <Github size={20} />
            <span>GitHub</span>
          </a>
        </div>
      </aside>

      {/* Mobile menu button */}
      <button className="mobile-menu-btn btn btn-secondary btn-icon" onClick={toggleSidebar}>
        <Menu size={20} />
      </button>

      {/* Main Content */}
      <main className="main-content">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>

      <style>{`
        .sidebar {
          display: flex;
          flex-direction: column;
        }
        
        .sidebar-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: var(--space-xl);
        }
        
        .logo {
          display: flex;
          align-items: center;
          gap: var(--space-md);
        }
        
        .logo-icon {
          width: 40px;
          height: 40px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--gradient-primary);
          border-radius: var(--radius-lg);
          color: white;
        }
        
        .logo-text {
          display: flex;
          flex-direction: column;
        }
        
        .logo-title {
          font-size: var(--text-lg);
          font-weight: 700;
          color: var(--text-primary);
          line-height: 1.2;
        }
        
        .logo-subtitle {
          font-size: var(--text-xs);
          color: var(--text-tertiary);
        }
        
        .mobile-menu {
          display: none;
        }
        
        .mobile-menu-btn {
          display: none;
          position: fixed;
          top: var(--space-md);
          left: var(--space-md);
          z-index: 99;
        }
        
        .sidebar-nav {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: var(--space-xs);
        }
        
        .nav-item {
          display: flex;
          align-items: center;
          gap: var(--space-md);
          padding: var(--space-sm) var(--space-md);
          border-radius: var(--radius-lg);
          color: var(--text-secondary);
          text-decoration: none;
          transition: all var(--transition-fast);
        }
        
        .nav-item:hover {
          background: var(--glass-bg);
          color: var(--text-primary);
        }
        
        .nav-item.active {
          background: var(--gradient-primary);
          color: white;
          box-shadow: var(--shadow-md);
        }
        
        .sidebar-footer {
          margin-top: auto;
          display: flex;
          flex-direction: column;
          gap: var(--space-md);
        }
        
        .system-status {
          padding: var(--space-md);
        }
        
        .status-header {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
          font-size: var(--text-sm);
          font-weight: 500;
          color: var(--text-primary);
          margin-bottom: var(--space-sm);
        }
        
        .status-details {
          display: flex;
          flex-direction: column;
          gap: var(--space-xs);
        }
        
        .status-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          font-size: var(--text-xs);
        }
        
        .status-label {
          color: var(--text-secondary);
        }
        
        .status-loading {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
        }
        
        .github-link {
          opacity: 0.7;
        }
        
        .github-link:hover {
          opacity: 1;
        }
        
        @media (max-width: 768px) {
          .mobile-menu {
            display: flex;
          }
          
          .mobile-menu-btn {
            display: flex;
          }
          
          .sidebar.closed {
            transform: translateX(-100%);
          }
        }
      `}</style>
    </div>
  );
}
