import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster, toast } from 'react-hot-toast';
import { useEffect } from 'react';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Quantizer from './pages/Quantizer';
import Analysis from './pages/Analysis';
import ModelLoader from './pages/ModelLoader';
import { useSystemStore } from './store';
import './index.css';

function App() {
  const fetchSystemInfo = useSystemStore((state) => state.fetchSystemInfo);

  useEffect(() => {
    // Fetch system info on app load
    fetchSystemInfo();

    const handleOffline = () => toast.error("Internet connection lost");
    const handleOnline = () => toast.success("Internet connection restored");

    window.addEventListener('offline', handleOffline);
    window.addEventListener('online', handleOnline);

    return () => {
      window.removeEventListener('offline', handleOffline);
      window.removeEventListener('online', handleOnline);
    };
  }, [fetchSystemInfo]);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="quantize" element={<Quantizer />} />
          <Route path="analysis" element={<Analysis />} />
          <Route path="models" element={<ModelLoader />} />
        </Route>
      </Routes>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: 'rgba(15, 23, 42, 0.8)',
            color: '#e2e8f0',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            padding: '12px 24px',
            borderRadius: '12px',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
            fontSize: '0.95rem'
          },
          success: {
            iconTheme: {
              primary: '#6366f1',
              secondary: '#fff',
            },
            style: {
              border: '1px solid rgba(99, 102, 241, 0.2)',
              background: 'rgba(99, 102, 241, 0.1)',
            }
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#fff',
            },
            style: {
              border: '1px solid rgba(239, 68, 68, 0.2)',
              background: 'rgba(239, 68, 68, 0.1)',
            }
          }
        }}
      />
    </BrowserRouter>
  );
}

export default App;
