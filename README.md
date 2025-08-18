# 🎮 Game RAG Assistant

An intelligent Q&A system specifically designed for game documentation, lore, and mechanics. Built as part of the NSK AI RAG Bootcamp Phase 1 project.

## 🌟 Features

- **Multi-format Document Support**: Text, PDF, and website content
- **Intelligent Retrieval**: Hybrid dense + sparse retrieval with domain filtering
- **Persona-based Responses**: Multiple assistant personalities (Guide, Sage, Expert, Casual)
- **Difficulty Levels**: Adjustable response detail from brief to comprehensive
- **Sample Data**: Pre-loaded example documents for immediate testing
- **User-Friendly Interface**: Clean Streamlit UI with guided onboarding
- **Deployment Ready**: Optimized for cloud deployment

## 🚀 Quick Start

### Option 1: Try Online (Recommended)
[Live Demo - Coming Soon]

### Option 2: Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/your-username/game-rag-assistant.git
cd game-rag-assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run gameRAG.py
```

4. **Open your browser** to `http://localhost:8501`

## 📋 Requirements

Create a `requirements.txt` file with:

```
streamlit>=1.28.0
langchain>=0.0.350
langchain-community>=0.0.38
langchain-huggingface>=0.0.3
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
chromadb>=0.4.15
pandas>=2.0.0
PyPDF2>=3.0.1
python-multipart>=0.0.6
```

## 🎯 Tech Stack

- **Framework**: LangChain + Streamlit
- **Embeddings**: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS (with Chroma fallback)
- **Retrieval**: Hybrid (Dense + BM25 sparse retrieval)
- **Optional LLM**: OpenAI GPT-3.5-turbo
- **Document Processing**: LangChain loaders for PDF, text, web content

## 📖 Usage Guide

### 1. Initialize the System
- Click "🚀 Initialize System" to set up the RAG pipeline
- Optionally configure OpenAI API key for enhanced answers

### 2. Add Game Documentation
Choose from three methods:

**📝 Text Documents**
- Paste your game documentation directly
- Perfect for mechanics, lore, and character descriptions

**📄 PDF Files** 
- Upload game design documents, manuals, or guides
- Automatically extracts and processes all pages

**🌐 Websites**
- Load content from game wikis or documentation sites
- Great for community-maintained content

### 3. Ask Questions
- Use natural language queries about your game content
- Adjust persona and detail level in the sidebar
- Filter by content domain (lore, mechanics, code, assets)

## 💡 Example Queries

After loading sample data or your own documents, try:

- "How does the combat system work?"
- "Tell me about the fantasy world setting"
- "What are the main character classes?"
- "Explain the magic system"
- "How do I unlock new areas?"

## 🎮 Game-Specific Features

### Content Domains
- **Lore**: World-building, characters, story
- **Mechanics**: Game rules, systems, progression  
- **Code**: Technical documentation, APIs
- **Assets**: Art guides, audio, resources
- **General**: Mixed or uncategorized content

### Assistant Personas
- **Helpful Guide**: Clear, professional explanations
- **Wise Sage**: Mystical, thoughtful responses
- **Tech Expert**: Developer-focused technical details
- **Casual Gamer**: Friendly, gaming-focused language

### Response Levels
- **Very Brief**: One sentence answers
- **Brief**: Short paragraphs
- **Balanced**: Moderate detail (default)
- **Detailed**: Comprehensive explanations
- **Very Detailed**: In-depth analysis

## 🔧 Configuration

### Environment Variables
```bash
# Optional: For enhanced answers
OPENAI_API_KEY=your_openai_api_key_here
```

### Customization
Modify these constants in the code to customize for your game:

```python
GAME_DOMAINS = ["lore", "mechanics", "code", "assets", "general"]
PERSONAS = {
    "Your Custom Persona": "Your persona description..."
}
```

## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Push code to GitHub repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from your repository
4. Set environment variables in Streamlit Cloud settings


## 📊 Project Structure

```
game-rag-assistant/
├── game_rag_enhanced.py    # Main application
├── requirements.txt        # Python dependencies  
├── README.md              # This file
├── Procfile              # Heroku deployment
├── Dockerfile            # Container deployment
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## 🔍 How It Works

1. **Document Ingestion**: Loads and processes various document formats
2. **Text Chunking**: Splits documents into semantic chunks with overlap
3. **Embedding**: Converts chunks to vector representations
4. **Indexing**: Stores in FAISS vector database with BM25 sparse index
5. **Retrieval**: Hybrid search combining dense and sparse methods
6. **Generation**: Uses LLM or extractive summarization for answers
7. **Context**: Provides source citations and metadata


## 🐛 Known Limitations

- PDF processing may struggle with complex layouts
- Website loading depends on site structure and accessibility
- Large documents may require chunking parameter adjustment
- OpenAI integration requires valid API key and credits

## 🤝 Contributing

This project was created for the NSK AI RAG Bootcamp. Feel free to fork and enhance for your own games!

## 📄 License

MIT License - Feel free to use for your own game projects.

## 🙏 Acknowledgments

- NSK AI Community for the excellent bootcamp content
- LangChain team for the comprehensive RAG framework
- Stream