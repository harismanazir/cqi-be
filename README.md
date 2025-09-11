# 🤖 LangGraph Multi-Agent Code Quality Intelligence (CQI)

An advanced AI-powered code analysis system built on **LangGraph** that orchestrates specialized AI agents for comprehensive code quality assessment. Each agent is an expert in a specific domain, working together through intelligent workflow management.

## ✨ **Revolutionary Features**

### 🧠 **LangGraph-Powered Workflow Orchestration**
- **Intelligent Agent Coordination**: Smart dependency management and execution sequencing
- **State-Aware Analysis**: Agents share insights and learn from each other's findings
- **Conditional Routing**: Dynamic workflow paths based on code characteristics
- **Error Recovery**: Built-in retry mechanisms and graceful failure handling
- **Progress Tracking**: Real-time workflow progress and observability

### 🎯 **4 Specialized AI Agents**

| 🤖 Agent | 🔍 Specialization | 🎯 Focus Areas | 🔗 Dependencies |
|-----------|-------------------|----------------|------------------|
| 🛡️ **Security** | Vulnerability Detection | SQL injection, XSS, hardcoded secrets | *None (Priority 1)* |
| 🔧 **Complexity** | Code Structure Analysis | Cyclomatic complexity, SOLID principles | *None (Priority 2)* |
| ⚡ **Performance** | Optimization Analysis | Algorithm efficiency, bottlenecks | *→ Complexity* |
| 📚 **Documentation** | Code Documentation | Missing docstrings, API docs | *→ Complexity* |
| 

### 🚀 **Smart Analysis Engine**
- **Automatic RAG Activation**: Seamlessly handles large codebases (>8K tokens) with retrieval-augmented generation
- **Language-Aware Chunking**: Intelligent code splitting that respects function/class boundaries
- **Cross-Agent Communication**: Agents share findings to enhance analysis quality
- **Vector Similarity Search**: FAISS-powered semantic code understanding
- **Multi-Language Support**: Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more

## 🏗️ **LangGraph Workflow Architecture**

```mermaid
graph TD
    A[Initialize Analysis] --> B[Discover Files]
    B --> C[Setup RAG System]
    C --> D[Route Agents]
    D --> E1[Security Agent]
    D --> E2[Complexity Agent]
    E1 --> F1[Performance Agent]
    E2 --> F1
    E1 --> F2[Documentation Agent]
    E2--> F2
    F1 --> G[Aggregate Results]
    F2 --> G
    G --> H[Finalize Analysis]
```

**🧠 Intelligent Features:**
- **Agent Dependencies**: Performance, Documentation waits for Security + Complexity insights
- **Dynamic Routing**: Workflow adapts based on findings
- **State Sharing**: Agents build on each other's analysis
- **Parallel Execution**: Independent agents run concurrently for speed

## 🚀 Installation & Setup

### **1. Clone & Install**
```bash
git clone <repository-url>
cd cqi
uv venv
source .venv/Scripts/activate  # Windows
source .venv/bin/activate      # Linux/Mac
```

### **2. Install LangGraph Dependencies**
```bash
uv pip install -r requirements.txt
```

### **3. Configure Groq API**
1. Get your **FREE** API key from [Groq Console](https://console.groq.com/keys)
2. Copy  to `.env`
3. Add your API key:
```bash
GROQ_API_KEY=your_actual_groq_api_key_here
```

## 🎮 Usage Examples

### **🔍 Full Project Analysis**
```bash
python main.py analyze /path/to/your/project --detailed
```

### **⚡ Targeted Analysis**
```bash
# Security-focused analysis
python main.py analyze /path/to/project --agents security,performance

# Single file deep-dive
python main.py analyze /path/to/file.py --detailed
```

### **🤖 Interactive LangGraph Mode**
```bash
python main.py interactive /path/to/project
```

**Interactive Commands:**
```
LangGraph CQI> summary              # Show analysis summary
LangGraph CQI> workflow             # Show LangGraph execution details
LangGraph CQI> agents               # Show agent performance
LangGraph CQI> issues critical      # Show critical issues only
LangGraph CQI> issues security      # Show security agent findings
```

### **📊 Export & Integration**
```bash
# JSON export for CI/CD
python main.py analyze /path/to/project -o results.json

# View available agents
python main.py agents

# Show workflow architecture
python main.py workflow
```

## 📊 **Example Output**

```bash
🤖 LangGraph Multi-Agent Code Quality Intelligence
AI-powered code analysis with intelligent workflow orchestration
Powered by LangGraph + Groq API

┌─────────────────┬─────────────────┐
│ 🎯 Analysis Summary              │
├─────────────────┼─────────────────┤
│ Files Processed │ 127             │
│ Total Lines     │ 45,230          │
│ Processing Time │ 18.7s           │
│ Workflow Status │ Completed       │
│ Total Issues    │ 89              │
│ Total Tokens    │ 156,430         │
└─────────────────┴─────────────────┘

┌─────────────┬───────┬────────────┐
│ ⚠️ Issues by Severity            │
├─────────────┼───────┼────────────┤
│ Critical    │ 2     │ 2.2%       │
│ High        │ 8     │ 9.0%       │
│ Medium      │ 31    │ 34.8%      │
│ Low         │ 48    │ 53.9%      │
└─────────────┴───────┴────────────┘

┌──────────────┬────────┬────────┬─────────┬────────────┐
│ 🤖 Agent Performance                                  │
├──────────────┼────────┼────────┼─────────┼────────────┤
│ Security     │ 12     │ 3.2s   │ 8,430   │ 0.94       │
│ Complexity   │ 28     │ 2.8s   │ 12,100  │ 0.91       │
│ Performance  │ 15     │ 4.1s   │ 15,200  │ 0.87       │
│ Testing      │ 21     │ 3.9s   │ 11,800  │ 0.89       │
│ Documentation│ 13     │ 2.4s   │ 7,900   │ 0.92       │
│ Duplication  │ 0      │ 1.8s   │ 6,200   │ 0.88       │
└──────────────┴────────┴────────┴─────────┴────────────┘
```

## ⚙️ **Advanced Configuration**

### **LangGraph Settings** (`.env`)
```bash
# Model Selection
DEFAULT_MODEL=mixtral-8x7b-32768    # Primary analysis model
FAST_MODEL=llama3-8b-8192          # Quick tasks

# RAG Configuration
MAX_TOKENS_BEFORE_RAG=8000         # When to activate RAG
CHUNK_SIZE=2000                    # Code chunk size
CHUNK_OVERLAP=200                  # Overlap for context

# Workflow Control
MAX_CONCURRENT_AGENTS=3            # Parallel agent limit
AGENT_TIMEOUT=30                   # Per-agent timeout
```

### **Agent Dependency Customization**
Modify `workflow/state.py` to customize agent execution order:

```python
AGENT_DEPENDENCIES = {
    'performance': ['complexity'],           # Performance needs complexity insights
    'testing': ['security', 'complexity'],  # Testing benefits from both
    'documentation': ['complexity'],        # Documentation uses complexity analysis
    'duplication': ['complexity']           # Duplication detection needs structure info
}
```

## 🔬 **Architecture Deep Dive**

```
cqi/
├── workflow/                 # LangGraph orchestration
│   ├── state.py             # Workflow state management
│   └── graph.py             # LangGraph workflow definition
├── agents/                  # LangGraph-integrated agents
│   ├── langgraph_base.py    # Base LangGraph agent class
│   ├── lg_security.py       # Security vulnerability agent
│   ├── lg_performance.py    # Performance optimization agent
│   ├── lg_complexity.py     # Code complexity agent
│   ├── lg_documentation.py  # Documentation quality agent
│   ├── lg_testing.py        # Test coverage agent
│   └── lg_duplication.py    # Code duplication agent
├── rag/                     # RAG system for large codebases
│   ├── chunker.py           # Intelligent code chunking
│   └── embeddings.py        # Vector search & retrieval
├── core/
│   └── langgraph_manager.py # LangGraph workflow manager
├── config/
│   └── settings.py          # Configuration management
└── cli.py                  # Rich CLI interface
```

## 🆚 **Why LangGraph vs Traditional Tools?**

| Feature | Traditional Static Analysis | Multi-Agent CQI |
|---------|---------------------------|-----------------|
| **Understanding** | Pattern matching only | Semantic code comprehension |
| **Context** | Single file analysis | Cross-file relationship awareness |
| **Workflow** | Fixed sequential analysis | Intelligent agent orchestration |
| **Adaptability** | Rule-based detection | AI-powered contextual analysis |
| **Collaboration** | Independent tools | Agents share insights and findings |
| **Scalability** | Manual tool coordination | Automatic RAG for large codebases |
| **Evolution** | Manual rule updates | Learning from agent interactions |

## 🚀 **Performance & Scalability**

### **Smart Resource Management**
- **RAG Activation**: Automatically switches to RAG for files >8K tokens
- **Concurrent Execution**: Parallel agent processing where dependencies allow
- **Token Optimization**: Efficient chunking and context management
- **Memory Efficiency**: Streaming analysis for large codebases

### **Benchmark Results**
- **Small Projects** (<1K lines): ~5-15 seconds
- **Medium Projects** (1K-10K lines): ~15-60 seconds  
- **Large Projects** (10K+ lines): ~1-5 minutes with RAG
- **Token Efficiency**: 60-80% reduction vs naive approaches

## 🤝 **Contributing to LangGraph CQI**

We welcome contributions! Key areas:

1. **New Agents**: Add specialized analysis agents
2. **Workflow Enhancements**: Improve LangGraph orchestration
3. **Language Support**: Add support for new programming languages
4. **RAG Improvements**: Better chunking and retrieval strategies

## 📄 **License**

MIT License - Build amazing things with LangGraph CQI!

## 🙏 **Powered By**

- **🦜 LangGraph**: Advanced AI workflow orchestration
- **⚡ Groq**: Lightning-fast free AI inference
- **🧠 LangChain**: AI application framework
- **🔍 FAISS**: High-performance vector similarity search
- **🎨 Rich**: Beautiful terminal interfaces

---

## 🚀 **Ready to Experience Next-Gen Code Analysis?**

1. **Get your free Groq API key**: [console.groq.com](https://console.groq.com)
2. **Install dependencies**: `uv pip install -r requirements.txt` 
3. **Configure your API key** in `.env`
4. **Run your first analysis**: `python main.py analyze /path/to/your/project`

**Transform your code quality workflow with AI-powered LangGraph orchestration!** 

[![LangGraph](https://img.shields.io/badge/Powered%20by-LangGraph-blue)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Inference-Groq%20API-orange)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)