# System Architecture

## Overview
The **PyTorch 8-Bit Weight Quantizer** is a full-stack application designed to analyze, quantize, and visualize neural network weights. It consists of a **FastAPI** backend that handles the heavy lifting of PyTorch model manipulation and quantization, and a **React** frontend that provides an interactive dashboard for users.

## High-Level Architecture (C4 Context)

```mermaid
graph TD
    User((User))
    
    subgraph "PyTorch 8-Bit Quantizer System"
        Frontend[("Frontend UI
        (React + Vite)")]
        Backend[("Backend API
        (FastAPI)")]
        Core[("Quantization Engine
        (PyTorch)")]
    end
    
    User -->|Interacts with| Frontend
    Frontend -->|HTTP Requests| Backend
    Backend -->|Invokes| Core
    Core -->|Loads/Saves| ModelFiles[("Model Files")]
```

## Detailed Component Architecture (C4 Container)

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI_Dashboard[("Dashboard Page")]
        UI_ModelLoader[("Model Loader Page")]
        UI_Quantizer[("Quantizer Interface")]
        UI_Analysis[("Analysis Visualization")]
        Store[("Zustand Store
        (State Management)")]
    end

    subgraph "Backend Layer"
        API_Main[("Main API Gateway
        (FastAPI)")]
        
        subgraph "Routes"
            Route_Models["/api/models"]
            Route_Quant["/api/quantize"]
            Route_Analysis["/api/analysis"]
            Route_System["/api/system"]
        end
        
        subgraph "Core Modules"
            Mod_Loader[("Model Loader")]
            Mod_Quantizer[("Quantization Engine")]
            Mod_Vis[("Visualization Engine")]
            Mod_System[("System Checker")]
        end
    end

    UI_Dashboard --> Store
    UI_ModelLoader --> Store
    UI_Quantizer --> Store
    
    Store -->|Axios Calls| API_Main
    
    API_Main --> Route_Models
    API_Main --> Route_Quant
    API_Main --> Route_Analysis
    API_Main --> Route_System
    
    Route_Models --> Mod_Loader
    Route_Quant --> Mod_Loader
    Route_Quant --> Mod_Quantizer
    Route_Analysis --> Mod_Vis
    Route_System --> Mod_System
```

## Quantization Core Class Diagram

The core logic resides in the `backend/core/quantizer.py` module, which implements a flexible strategy pattern for different quantization methods.

```mermaid
classDiagram
    class QuantizationMethod {
        <<Enum>>
        INT8
        INT4
        NF4
        GPTQ
    }
    
    class QuantizationConfig {
        +int bits
        +QuantizationMethod method
        +QuantizationMode mode
        +int group_size
        +bool use_double_quant
        +to_dict()
    }
    
    class QuantizationResult {
        +Tensor quantized_weights
        +Tensor scales
        +Tensor zero_points
        +Tuple original_shape
        +float max_error
        +float memory_savings_percent
    }
    
    class BaseQuantizer {
        +QuantizationConfig config
        +quantize(weights)
        +dequantize(result)
        #_calculate_error()
        #_calculate_memory_savings()
    }
    
    class INT8Quantizer {
        +quantize(weights)
        -_quantize_grouped()
        +dequantize_weights()
    }
    
    class INT4Quantizer {
        +quantize(weights)
        -_pack_int4()
        -_unpack_int4()
    }
    
    class NF4Quantizer {
        +Tensor codebook
        +quantize(weights)
        +dequantize_weights()
    }
    
    BaseQuantizer <|-- INT8Quantizer
    BaseQuantizer <|-- INT4Quantizer
    BaseQuantizer <|-- NF4Quantizer
    BaseQuantizer *-- QuantizationConfig
    BaseQuantizer ..> QuantizationResult : Produces
```

## Quantization Workflow

Sequence of events when a user triggers quantization:

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as Backend API
    participant Q as Quantizer Engine
    
    U->>FE: Select Parameters (Method, Bits, Group Size)
    U->>FE: Click "Quantize"
    FE->>API: POST /api/quantize/run (config)
    
    activate API
    API->>Q: Initialize Quantizer(config)
    API->>Q: ModelLoader.load_layer(layer_name)
    
    activate Q
    Q->>Q: Calculate Scales (Max/AbsMax)
    
    alt Grouped Quantization
        Q->>Q: Reshape to (Groups, GroupSize)
        Q->>Q: Calculate Scale per Group
    end
    
    Q->>Q: Round(Weights / Scales)
    Q->>Q: Clamp to Range (e.g. -128, 127)
    
    if INT4 or NF4
        Q->>Q: Pack 4-bit values into INT8
    end
    
    Q->>Q: Dequantize (Recover FP32)
    Q->>Q: Calculate Error (MSE, L1)
    Q-->>API: Return QuantizationResult
    deactivate Q
    
    API-->>FE: Return Result + Metrics
    deactivate API
    
    FE->>U: Display Error Heatmap & Memory Savings
    FE->>U: Show Before/After Weight Distribution
```

## Tech Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Frontend** | React 19 | UI Library |
| | Vite | Build Tool & Dev Server |
| | Zustand | State Management |
| | Recharts | Data Visualization (Charts) |
| | Tailwind CSS | Utility-first CSS Framework (inferred) |
| **Backend** | Python 3.10+ | Core Language |
| | FastAPI | Web Framework |
| | PyTorch | Deep Learning Framework |
| | NumPy | Numerical Computing |
| | Uvicorn | ASGI Server |
