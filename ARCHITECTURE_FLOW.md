# OrchestrAIte: System Architecture and Data Flow

This document provides detailed visual representations of the OrchestrAIte system architecture, showing both the embedding process (how musical data is stored) and the generation flow (how music is created).

## Table of Contents
- [RAG Embedding Flow](#rag-embedding-flow)
- [Music Generation Flow](#music-generation-flow)
- [Agent Interaction Diagram](#agent-interaction-diagram)
- [Data Processing Pipeline](#data-processing-pipeline)

---

## RAG Embedding Flow

This diagram shows how MIDI files are processed and stored in the vector database for later retrieval during music generation.

```mermaid
flowchart TD
    A[MIDI File Collection] --> B[MIDI Parser]
    B --> C[Musical Pattern Extraction]
    C --> D[Segment Extraction]
    C --> E[Chord Progression Analysis]
    C --> F[Melody Pattern Analysis]
    
    D --> G[Note Sequences]
    E --> H[Harmonic Progressions]
    F --> I[Melodic Intervals]
    
    G --> J[Feature Vectorization]
    H --> J
    I --> J
    
    J --> K[Gemini LLM Embedding]
    K --> L[Vector Representation]
    L --> M[(ChromaDB Vector Store)]
    
    M --> N[Indexed by Genre]
    M --> O[Indexed by Style]
    M --> P[Indexed by Instruments]
    
    style A fill:#e1f5fe
    style M fill:#f3e5f5
    style K fill:#fff3e0
```

### Embedding Process Details

1. **MIDI Collection**: Raw MIDI files from various sources
2. **Pattern Extraction**: Musical segments, chord progressions, and melodic patterns
3. **Vectorization**: Convert musical features to numerical representations
4. **LLM Embedding**: Use Gemini to create semantic embeddings
5. **Storage**: Index and store in ChromaDB with metadata

---

## Music Generation Flow

This diagram illustrates the complete process from user input to final music output.

### Input Processing & RAG Retrieval
```mermaid
flowchart TD
    A[User Input] --> B{Input Type}
    B -->|Natural Language| C[Description Parser Agent]
    B -->|Parameters| D[Direct Parameters]
    
    C --> E[Musical Parameters]
    D --> E
    E --> F[MIDI Reference Agent]
    
    F --> G[(ChromaDB Vector Store)]
    G --> H[Pattern Retrieval]
    H --> I[Similarity Scoring]
    I --> J[Top Patterns Selected]
    
    style A fill:#e8f5e8
    style G fill:#f3e5f5
    style F fill:#e1f5fe
```

### Agent Orchestration & Generation
```mermaid
flowchart TD
    A[Selected Patterns] --> B[Artist Context Agent]
    B --> C[Musical Director Agent]
    
    C --> D[Chord Agent]
    C --> E[Melody Agent] 
    C --> F[Instrument Agent]
    C --> G[Drum Agent]
    C --> H[Vocal Agent]
    
    D --> I[Harmonic Structure]
    E --> J[Melodic Lines]
    F --> K[Instrumental Parts] 
    G --> L[Rhythmic Foundation]
    H --> M[Vocal Elements]
    
    style C fill:#fff3e0
    style A fill:#f3e5f5
```

### Synthesis & Output Generation
```mermaid
flowchart TD
    A[Harmonic Structure] --> E[MIDI Synthesis Agent]
    B[Melodic Lines] --> E
    C[Instrumental Parts] --> E
    D[Rhythmic Foundation] --> E
    
    E --> F[Combined MIDI File]
    F --> G[Audio Renderer Agent]
    G --> H[MP3 Output]
    
    F --> I[Export Agent]
    H --> I
    I --> J[Final Output Files]
    
    style E fill:#fff3e0
    style J fill:#ffebee
```

### Generation Process Steps

1. **Input Processing**: Parse user requirements into musical parameters
2. **RAG Retrieval**: Query vector database for similar musical patterns
3. **Context Application**: Apply artist-specific styling and characteristics
4. **Multi-Agent Generation**: Parallel creation of musical elements by specialized agents
5. **Synthesis**: Combine all elements into cohesive musical composition
6. **Output**: Generate final MIDI and MP3 files with quality validation

---

## Agent Interaction Diagram

This diagram shows how different agents interact and share information during the generation process.

### Core System Layers
```mermaid
graph TD
    subgraph "Input Layer"
        UI[User Input]
        DP[Description Parser]
        UI --> DP
    end
    
    subgraph "RAG Layer"
        MRA[MIDI Reference Agent]
        VDB[(Vector Database)]
        MRA <--> VDB
        DP --> MRA
    end
    
    subgraph "Context Layer"
        ACA[Artist Context Agent]
        MDA[Musical Director Agent]
        MRA --> ACA
        ACA --> MDA
    end
    
    style UI fill:#e8f5e8
    style VDB fill:#f3e5f5
    style MDA fill:#fff3e0
```

### Generation Agents Network
```mermaid
graph TD
    A[Musical Director] --> B[Chord Agent]
    A --> C[Melody Agent]
    A --> D[Instrument Agent]
    A --> E[Drum Agent]
    A --> F[Vocal Agent]
    
    B <--> C
    C <--> D
    B <--> D
    E <--> B
    F <--> C
    
    subgraph "Shared Context"
        MC[Musical Context]
        B -.-> MC
        C -.-> MC
        D -.-> MC
        E -.-> MC
        F -.-> MC
    end
    
    style A fill:#fff3e0
    style MC fill:#f3e5f5
```

### Output Processing Chain
```mermaid
graph TD
    A[Chord Agent] --> D[MIDI Synth Agent]
    B[Melody Agent] --> D
    C[Instrument Agent] --> D
    E[Drum Agent] --> D
    F[Vocal Agent] --> D
    
    D --> G[Audio Renderer Agent]
    D --> H[Export Agent]
    G --> H
    
    H --> I[Final Output]
    
    style D fill:#fff3e0
    style I fill:#ffebee
```

### Agent Communication Patterns

- **Solid arrows**: Direct data flow between components
- **Dotted arrows**: Shared context access for coordination
- **Bidirectional arrows**: Collaborative interaction and feedback loops
- **Subgraphs**: Logical grouping of related system components

---

## Data Processing Pipeline

This diagram shows the detailed data transformation pipeline from input to output.

### Input to RAG Processing
```mermaid
flowchart LR
    subgraph "Input Processing"
        A[Raw Input] --> B[Parameter Extraction]
        B --> C[Validation & Normalization]
    end
    
    subgraph "RAG Processing"
        C --> D[Query Construction]
        D --> E[Vector Search]
        E --> F[Pattern Filtering]
        F --> G[Similarity Ranking]
    end
    
    style A fill:#e8f5e8
    style E fill:#f3e5f5
```

### Musical Intelligence & Generation
```mermaid
flowchart LR
    subgraph "Musical Intelligence"
        A[Pattern Analysis] --> B[Harmonic Analysis]
        B --> C[Melodic Analysis]
        C --> D[Style Adaptation]
    end
    
    subgraph "Generation Pipeline"
        D --> E[Chord Generation]
        E --> F[Melody Generation]
        F --> G[Instrument Arrangement]
        G --> H[Drum Programming]
    end
    
    style A fill:#f3e5f5
    style D fill:#fff3e0
```

### Post-Processing & Output
```mermaid
flowchart LR
    subgraph "Post-Processing"
        A[Musical Validation] --> B[Coverage Extension]
        B --> C[Quality Checking]
        C --> D[Format Conversion]
    end
    
    subgraph "Output Generation"
        D --> E[MIDI Creation]
        E --> F[Audio Synthesis]
        F --> G[MP3 Encoding]
        G --> H[File Export]
    end
    
    style A fill:#fff3e0
    style H fill:#ffebee
```

### Pipeline Stages

1. **Input Processing**: Clean and validate user input parameters
2. **RAG Processing**: Retrieve and rank similar musical patterns from vector database
3. **Musical Intelligence**: Analyze and adapt patterns for generation context
4. **Generation Pipeline**: Create musical elements using specialized AI agents
5. **Post-Processing**: Validate, extend, and optimize generated musical content
6. **Output Generation**: Convert to final MIDI and MP3 formats with quality assurance

---

## RAG Query and Response Flow

This diagram specifically focuses on how the RAG system processes queries and returns relevant musical patterns.

```mermaid
sequenceDiagram
    participant User
    participant DP as Description Parser
    participant MRA as MIDI Reference Agent
    participant VDB as Vector Database
    participant LLM as Gemini LLM
    participant Agents as Generation Agents
    
    User->>DP: Input Description/Parameters
    DP->>MRA: Parsed Musical Requirements
    
    Note over MRA: Construct RAG Query
    MRA->>LLM: Generate Query Embedding
    LLM-->>MRA: Query Vector
    
    MRA->>VDB: Vector Similarity Search
    VDB-->>MRA: Top K Similar Patterns
    
    Note over MRA: Pattern Analysis & Filtering
    MRA->>MRA: Score & Rank Patterns
    MRA->>MRA: Apply Fallback if Needed
    
    Note over MRA: Logging RAG Usage
    MRA->>MRA: Log Request Details
    MRA->>MRA: Log Response Patterns
    MRA->>MRA: Log Usage Statistics
    
    MRA-->>Agents: Selected Musical Patterns
    
    Note over Agents: Multi-Agent Generation
    Agents->>Agents: Generate Musical Elements
    Agents-->>User: Final Music Output
    
    rect rgb(240, 248, 255)
        Note over MRA, VDB: RAG System Core
    end
    
    rect rgb(255, 248, 240)
        Note over Agents: Generation Layer
    end
```

### RAG Process Details

1. **Query Construction**: Convert musical requirements to searchable vectors
2. **Similarity Search**: Find patterns matching the musical style/genre
3. **Pattern Selection**: Score and rank retrieved patterns
4. **Fallback Handling**: Use database fallbacks for robust operation
5. **Usage Logging**: Track RAG queries and responses for transparency
6. **Pattern Distribution**: Provide selected patterns to generation agents

---

## System Performance Metrics

### RAG System Performance Distribution
```mermaid
pie title RAG System Performance
    "Pattern Retrieval" : 25
    "LLM Processing" : 30
    "Musical Generation" : 35
    "Audio Synthesis" : 10
```

### Generation Agent Workload Distribution
```mermaid
pie title Agent Workload
    "Melody Agent" : 30
    "Chord Agent" : 25
    "Instrument Agent" : 20
    "Drum Agent" : 15
    "Vocal Agent" : 10
```

### System Resource Usage
```mermaid
pie title Resource Usage
    "CPU Processing" : 40
    "Memory (RAM)" : 25
    "Disk I/O" : 20
    "Network (API)" : 15
```

---

## Technology Stack Integration

This diagram shows how all the technologies work together in the OrchestrAIte system.

### Application & Framework Layers
```mermaid
graph TD
    subgraph "Frontend Layer"
        CLI[Command Line Interface]
        API[REST API - Future]
    end
    
    subgraph "Framework Layer"
        LG[LangGraph Multi-Agent Framework]
        GEM[Gemini LLM Integration]
    end
    
    CLI --> LG
    API --> LG
    LG --> GEM
    
    style CLI fill:#e8f5e8
    style LG fill:#fff3e0
```

### AI/ML & Music Processing
```mermaid
graph TD
    subgraph "AI/ML Layer"
        RAG[RAG System]
        CHR[(ChromaDB)]
        EMB[Embedding Generation]
    end
    
    subgraph "Music Processing Layer"
        PM[Pretty MIDI]
        FS[FluidSynth]
        AU[Audio Utils]
    end
    
    RAG --> CHR
    RAG --> EMB
    PM --> FS
    FS --> AU
    
    style RAG fill:#f3e5f5
    style PM fill:#fff3e0
```

### Storage & File System
```mermaid
graph TD
    subgraph "Storage Layer"
        VDB[(Vector Database)]
        FS_FILES[File System]
        MIDI[MIDI Files]
        OUT[Output Files]
    end
    
    VDB --> FS_FILES
    MIDI --> FS_FILES
    OUT --> FS_FILES
    
    style VDB fill:#ffebee
    style FS_FILES fill:#f0f0f0
```

This architecture document provides a comprehensive view of how OrchestrAIte processes musical data and generates new compositions using a sophisticated multi-agent RAG system.
