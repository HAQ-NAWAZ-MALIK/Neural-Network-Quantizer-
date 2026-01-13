import { create } from 'zustand';
import { toast } from 'react-hot-toast';

/**
 * Main application store using Zustand
 * Manages global state: system info, model, quantization results
 */

// API base URL - use relative path for Vite proxy in development
const API_BASE = import.meta.env.VITE_API_URL || '/api';

// System store - hardware info and capabilities
export const useSystemStore = create((set, get) => ({
    systemInfo: null,
    isLoading: false,
    error: null,

    fetchSystemInfo: async () => {
        set({ isLoading: true, error: null });

        // Optimistic load from cache
        const cachedSystem = localStorage.getItem('system_info');
        if (cachedSystem) {
            try {
                set({ systemInfo: JSON.parse(cachedSystem) });
            } catch (e) { }
        }

        try {
            const response = await fetch(`${API_BASE}/system/info`);
            if (!response.ok) throw new Error('Failed to fetch system info');
            const data = await response.json();
            set({ systemInfo: data, isLoading: false });
            localStorage.setItem('system_info', JSON.stringify(data));
            return data;
        } catch (error) {
            set({ error: error.message, isLoading: false });
            return null;
        }
    },

    checkModelRequirements: async (paramsB, dtype = 'fp16') => {
        try {
            const response = await fetch(
                `${API_BASE}/system/check-model?model_params_billions=${paramsB}&dtype=${dtype}`,
                { method: 'POST' }
            );
            return await response.json();
        } catch (error) {
            return { can_load: false, error: error.message };
        }
    }
}));

// Model store - loaded model info
export const useModelStore = create((set, get) => ({
    modelInfo: null,
    layers: [],
    isLoading: false,
    loadingProgress: 0,
    error: null,

    loadModel: async (modelName, options = {}) => {
        set({ isLoading: true, loadingProgress: 0, error: null });

        try {
            const response = await fetch(`${API_BASE}/models/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: modelName,
                    model_type: options.modelType || 'generic',
                    dtype: options.dtype || 'auto',
                    device: options.device || 'auto',
                    low_memory: options.lowMemory || false,
                    trust_remote_code: true
                })
            });

            const data = await response.json();

            if (data.success) {
                set({
                    modelInfo: data.model_info || { name: data.name || data.model_id },
                    isLoading: false,
                    loadingProgress: 100
                });
                toast.success(`Model loaded: ${data.model_info?.name || modelName}`);
                return data;
            } else {
                const errMsg = data.error || 'Failed to load model';
                set({ error: errMsg, isLoading: false });
                toast.error(errMsg);
                return data;
            }
        } catch (error) {
            set({ error: error.message, isLoading: false });
            toast.error(`Connection failed: ${error.message}`);
            return { success: false, error: error.message };
        }
    },

    fetchModelInfo: async (modelName) => {
        set({ isLoading: true, error: null });
        // Start streaming load... handled by component usually or separate action
    },

    checkLoadedModel: async () => {
        try {
            const response = await fetch(`${API_BASE}/models/info`);
            const data = await response.json();
            if (data && data.name) {
                set({
                    modelInfo: {
                        name: data.name,
                        num_params: data.num_params,
                        memory_mb: data.memory_mb,
                        device: data.device,
                        dtype: data.dtype
                    },
                    error: null
                });
                // Also fetch layers
                get().fetchLayers();
            } else {
                set({ modelInfo: null, layers: [] });
            }
        } catch (error) {
            console.error('Failed to check loaded model:', error);
        }
    },

    fetchLayers: async () => {
        try {
            const response = await fetch(`${API_BASE}/models/layers`);
            const data = await response.json();
            // Use quantizable_layers (strings) for the dropdown
            if (data.quantizable_layers) {
                set({ layers: data.quantizable_layers });
            } else if (data.layers) {
                // Fallback if structure is different
                set({ layers: data.layers.map(l => l.name) });
            }
        } catch (error) {
            console.error('Failed to fetch layers:', error);
        }
    },

    unloadModel: async () => {
        try {
            await fetch(`${API_BASE}/models/unload`, { method: 'POST' });
            set({ modelInfo: null, layers: [], error: null });
            toast.success('Model unloaded');
        } catch (error) {
            console.error('Failed to unload model:', error);
        }
    },

    clearError: () => set({ error: null })
}));

// Quantization store - quantization operations and results
export const useQuantizationStore = create((set, get) => ({
    result: null,
    isQuantizing: false,
    progress: 0,
    error: null,
    history: [],

    quantizeWeights: async (config) => {
        set({ isQuantizing: true, progress: 0, error: null });

        try {
            const response = await fetch(`${API_BASE}/quantize/weights`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    in_features: config.inFeatures || 64,
                    out_features: config.outFeatures || 128,
                    bits: config.bits || 8,
                    method: config.method || 'int8',
                    mode: config.mode || 'symmetric',
                    group_size: config.groupSize || null,
                    weight_pattern: config.pattern || 'random',
                    dtype: config.dtype || 'float32'
                })
            });

            const data = await response.json();

            if (data.success) {
                set({
                    result: data,
                    isQuantizing: false,
                    progress: 100,
                    history: [...get().history, {
                        timestamp: new Date().toISOString(),
                        config,
                        stats: data.stats
                    }]
                });
                toast.success('Custom weights quantized');
                return data;
            } else {
                set({ error: data.error || 'Quantization failed', isQuantizing: false });
                toast.error(data.error || 'Quantization failed');
                return data;
            }
        } catch (error) {
            set({ error: error.message, isQuantizing: false });
            toast.error(error.message);
            return { success: false, error: error.message };
        }
    },

    quantizeLayer: async (layerName, config) => {
        set({ isQuantizing: true, progress: 0, error: null });

        try {
            const response = await fetch(`${API_BASE}/quantize/layer`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    layer_name: layerName,
                    bits: config.bits || 8,
                    method: config.method || 'int8',
                    mode: config.mode || 'symmetric',
                    group_size: config.groupSize || null
                })
            });

            const data = await response.json();

            if (data.success) {
                set({
                    result: data,
                    error: null,
                    isQuantizing: false,
                    progress: 100
                });
                toast.success(`Layer ${layerName} quantized`);
            } else {
                const errMsg = data.error || 'Quantization failed';
                set({
                    result: null,
                    error: errMsg,
                    isQuantizing: false
                });
                toast.error(errMsg);
            }
            return data;
        } catch (error) {
            set({ error: error.message, isQuantizing: false });
            toast.error(error.message);
            return { success: false, error: error.message };
        }
    },

    quantizeModel: async (config) => {
        set({ isQuantizing: true, progress: 0, error: null });

        try {
            const response = await fetch(`${API_BASE}/quantize/model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    bits: config.bits || 8,
                    method: config.method || 'int8',
                    mode: config.mode || 'symmetric',
                    group_size: config.groupSize || null
                })
            });

            if (data.success) {
                set({
                    result: data,
                    error: null,
                    isQuantizing: false,
                    progress: 100
                });
                toast.success(`Full Model quantized! Saved ${data.summary?.total_memory_saved_mb?.toFixed(2)} MB`);
            } else {
                const errMsg = data.error || 'Optimization interrupted';
                set({
                    result: null,
                    error: errMsg,
                    isQuantizing: false
                });
                toast.error(errMsg);
            }
            return data;
        } catch (error) {
            set({ error: error.message, isQuantizing: false });
            toast.error(error.message);
            return { success: false, error: error.message };
        }
    },

    compareMethod: async (methods = ['int8', 'int4', 'nf4'], layerName = null) => {
        try {
            const body = { methods };
            if (layerName) {
                body.layer_name = layerName;
            }

            const response = await fetch(`${API_BASE}/analysis/compare`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            return await response.json();
        } catch (error) {
            toast.error(error.message);
            return { error: error.message };
        }
    },

    clearResult: () => set({ result: null, error: null }),
    clearHistory: () => set({ history: [] })
}));

// UI store - navigation, theme, etc.
export const useUIStore = create((set) => ({
    sidebarOpen: true,
    activeTab: 'quantize',
    theme: localStorage.getItem('theme') || 'dark',

    toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
    setActiveTab: (tab) => set({ activeTab: tab }),
    toggleTheme: () => set((state) => {
        const newTheme = state.theme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        return { theme: newTheme };
    })
}));
