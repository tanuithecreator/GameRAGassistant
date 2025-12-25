import os
import io
import tempfile
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from collections import Counter
import logging

import streamlit as st
import pandas as pd
import numpy as np

# LangChain core & community
# Handle text_splitter import - moved to separate package in newer versions
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback for older LangChain versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma, FAISS as LCFAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.retrievers import BM25Retriever

# Handle both old and new LangChain import paths
try:
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    # Fallback for older LangChain versions
    from langchain.schema import Document
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser

from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory

# Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Optional OpenAI (graceful fallback)
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Configuration & Constants
# ------------------------------
GAME_DOMAINS = ["lore", "mechanics", "code", "assets", "general"]
PERSONAS = {
    "Helpful Guide": "You are a knowledgeable and friendly game guide who provides clear, helpful answers.",
    "Wise Sage": "You speak with ancient wisdom, offering guidance in a mystical, thoughtful manner.",
    "Tech Expert": "You are a technical expert who provides precise, developer-focused information.",
    "Casual Gamer": "You speak like an experienced gamer friend, using casual language and gaming terms."
}

DIFFICULTY_LEVELS = {
    0: "Very Brief - One sentence answers",
    1: "Brief - Short paragraph answers", 
    2: "Balanced - Moderate detail",
    3: "Detailed - Comprehensive explanations",
    4: "Very Detailed - In-depth analysis"
}

# Vector store persistence directory
VECTOR_STORE_DIR = Path(".vectorstore")

SAMPLE_DOCUMENTS = {
    "RPG Combat System": {
        "content": """# Combat System Documentation

## Basic Combat Mechanics
Combat in our RPG follows a turn-based system with real-time elements. Players can attack, defend, or use special abilities.

### Damage Calculation
- Base damage = Weapon damage + Strength modifier
- Critical hits occur on rolls of 18-20
- Armor reduces damage by a flat amount

### Status Effects
- Poison: Deals 5 damage per turn for 3 turns
- Stun: Character loses next turn
- Blessing: +2 to all rolls for 5 turns

### Experience and Leveling
Characters gain XP from combat victories. Level up occurs every 1000 XP, granting +10 HP and +1 to all stats.""",
        "metadata": {"domain": "mechanics", "game_element": "combat", "source": "sample_docs"}
    },
    
    "Fantasy World Lore": {
        "content": """# The Realm of Aethermoor

## History
Long ago, the realm of Aethermoor was united under the Crystal Throne. The ancient kings wielded powerful artifacts known as the Elemental Stones, which controlled the very forces of nature.

## The Great Sundering
A catastrophic event 500 years ago shattered the realm into five kingdoms:
- **Northwind**: Frozen lands ruled by ice magic
- **Goldenvale**: Agricultural heartland with earth magic
- **Stormwatch**: Coastal kingdom commanding wind and water
- **Emberpeak**: Mountain realm of fire and forge
- **Shadowmere**: Mysterious lands where dark magic thrives

## Current Politics
The five kingdoms maintain an uneasy peace, but ancient prophecies speak of a chosen one who will reunite the realm.""",
        "metadata": {"domain": "lore", "game_element": "worldbuilding", "source": "sample_docs"}
    }
}

# ------------------------------
# Utility Functions
# ------------------------------
def expand_query(query: str) -> str:
    """Expand query with synonyms and related terms for better retrieval"""
    expansions = {
        "combat": ["battle", "fighting", "attack", "defense"],
        "magic": ["spell", "enchantment", "sorcery"],
        "character": ["player", "hero", "protagonist"],
        "level": ["stage", "area", "zone"],
        "item": ["equipment", "gear", "artifact"],
        "quest": ["mission", "objective", "task"],
        "lore": ["story", "history", "background", "world"],
        "mechanics": ["rules", "systems", "gameplay"]
    }
    
    query_lower = query.lower()
    expanded_terms = [query]
    
    for key, synonyms in expansions.items():
        if key in query_lower:
            expanded_terms.extend(synonyms[:2])  # Add top 2 synonyms
    
    return " ".join(set(expanded_terms))  # Remove duplicates

def calculate_sentence_scores(query: str, sentences: List[str]) -> List[Tuple[float, str]]:
    """Calculate relevance scores for sentences based on query"""
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    scored = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue
        
        sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        
        # Calculate overlap
        overlap = len(query_words.intersection(sentence_words))
        if len(query_words) > 0:
            score = overlap / len(query_words)
        else:
            score = 0
        
        # Boost score for question words in query
        question_words = {"what", "how", "why", "when", "where", "who"}
        if any(qw in sentence.lower() for qw in question_words if qw in query.lower()):
            score *= 1.2
        
        if score > 0:
            scored.append((score, sentence))
    
    return sorted(scored, reverse=True, key=lambda x: x[0])

def rerank_documents(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    """Rerank documents using query-document similarity"""
    if not docs:
        return []
    
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    scored_docs = []
    
    for doc in docs:
        content = doc.page_content.lower()
        content_words = set(re.findall(r'\b\w+\b', content))
        
        # Calculate Jaccard similarity
        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))
        similarity = intersection / union if union > 0 else 0
        
        # Boost if query terms appear early in document
        content_first_100 = content[:100]
        early_matches = sum(1 for word in query_words if word in content_first_100)
        similarity += early_matches * 0.1
        
        scored_docs.append((similarity, doc))
    
    # Sort by score and return top_k
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored_docs[:top_k]]

# ------------------------------
# Core RAG System
# ------------------------------
class GameRAG:
    def __init__(self, use_openai: bool = False, openai_api_key: Optional[str] = None,
                 use_conversation_memory: bool = True, memory_window: int = 5):
        self.use_openai = use_openai and HAS_OPENAI
        self.openai_api_key = openai_api_key
        self.use_conversation_memory = use_conversation_memory
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Storage
        self.vectorstore = None
        self.retriever_dense = None
        self.retriever_sparse = None
        self.documents = []
        
        # Conversation memory
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            return_messages=True
        ) if use_conversation_memory else None
        
        # Vector store persistence
        VECTOR_STORE_DIR.mkdir(exist_ok=True)
        
        self._setup_models()
    
    def _setup_models(self):
        """Initialize embeddings and LLM"""
        try:
            if self.use_openai and self.openai_api_key:
                self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                    openai_api_key=self.openai_api_key
                )
                logger.info("Using OpenAI models")
            else:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                logger.info("Using HuggingFace embeddings (no generative LLM)")
        except Exception as e:
            logger.error(f"Model setup error: {e}")
            # Fallback to HuggingFace
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            except Exception as e2:
                logger.error(f"Failed to initialize embeddings: {e2}")
                raise
            self.use_openai = False
    
    def save_vectorstore(self, name: str = "default"):
        """Save vectorstore to disk for persistence"""
        if self.vectorstore is None:
            return False
        
        try:
            save_path = VECTOR_STORE_DIR / name
            if isinstance(self.vectorstore, LCFAISS):
                self.vectorstore.save_local(str(save_path))
                logger.info(f"Saved FAISS vectorstore to {save_path}")
            elif isinstance(self.vectorstore, Chroma):
                # Chroma persists automatically, but we can note it
                logger.info(f"Chroma vectorstore persists at {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving vectorstore: {e}")
            return False
    
    def load_vectorstore(self, name: str = "default") -> bool:
        """Load vectorstore from disk"""
        try:
            load_path = VECTOR_STORE_DIR / name
            if not load_path.exists():
                return False
            
            if self.embeddings is None:
                logger.error("Embeddings not initialized")
                return False
            
            # Try loading FAISS
            try:
                self.vectorstore = LCFAISS.load_local(
                    str(load_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.retriever_dense = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                logger.info(f"Loaded FAISS vectorstore from {load_path}")
                return True
            except:
                # If FAISS fails, might be Chroma or corrupted
                logger.warning(f"Could not load vectorstore from {load_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading vectorstore: {e}")
            return False
    
    def load_sample_data(self):
        """Load sample documents for immediate testing"""
        try:
            docs = []
            for title, data in SAMPLE_DOCUMENTS.items():
                doc = Document(
                    page_content=data["content"],
                    metadata={**data["metadata"], "title": title}
                )
                docs.append(doc)
            
            if docs:
                success = self.index_documents(docs)
                if success:
                    return len(docs)
                else:
                    return 0
            return 0
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            return 0
    
    def load_text_document(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """Load a text document with metadata"""
        if not text.strip():
            return []
        
        # Clean metadata
        clean_meta = {k: v for k, v in metadata.items() if v}
        doc = Document(page_content=text, metadata=clean_meta)
        return [doc]
    
    def load_pdf_document(self, file_bytes: bytes, metadata: Dict[str, Any]) -> List[Document]:
        """Load PDF document from bytes"""
        docs = []
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # Add metadata to all docs
            for doc in docs:
                doc.metadata.update(metadata)
                
        except Exception as e:
            logger.error(f"PDF loading error: {e}")
            raise
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        return docs
    
    def load_website(self, url: str, metadata: Dict[str, Any]) -> List[Document]:
        """Load content from a website"""
        try:
            loader = WebBaseLoader([url])
            docs = loader.load()
            for doc in docs:
                doc.metadata.update(metadata)
                doc.metadata["source"] = url
            return docs
        except Exception as e:
            logger.error(f"Website loading error: {e}")
            return []
    
    def index_documents(self, docs: List[Document], persist: bool = True):
        """Index documents for retrieval"""
        if not docs:
            raise ValueError("No documents to index")
        
        # Store original documents
        self.documents.extend(docs)
        
        # Split documents
        try:
            splits = self.text_splitter.split_documents(docs)
            if not splits:
                raise ValueError("No content after splitting")
            
            logger.info(f"Processing {len(splits)} document chunks...")
            
            # Ensure embeddings are properly initialized
            if self.embeddings is None:
                raise ValueError("Embeddings not initialized")
            
            # Create vector store with better error handling
            try:
                if self.vectorstore is None:
                    self.vectorstore = LCFAISS.from_documents(splits, self.embeddings)
                    logger.info("Created new FAISS vector store")
                else:
                    # Add to existing vectorstore
                    new_vectorstore = LCFAISS.from_documents(splits, self.embeddings)
                    self.vectorstore.merge_from(new_vectorstore)
                    logger.info("Merged documents into existing vector store")
            except Exception as faiss_error:
                logger.warning(f"FAISS failed: {faiss_error}, trying Chroma...")
                # Fallback to Chroma
                if self.vectorstore is None:
                    self.vectorstore = Chroma.from_documents(splits, self.embeddings)
                    logger.info("Created new Chroma vector store")
                else:
                    # For Chroma, we need to recreate with all documents
                    all_splits = self.text_splitter.split_documents(self.documents)
                    self.vectorstore = Chroma.from_documents(all_splits, self.embeddings)
                    logger.info("Recreated Chroma vector store with all documents")
            
            # Setup retrievers
            self.retriever_dense = self.vectorstore.as_retriever(
                search_kwargs={"k": 8}  # Retrieve more for reranking
            )
            
            # Rebuild sparse retriever with all splits
            all_splits = self.text_splitter.split_documents(self.documents)
            self.retriever_sparse = BM25Retriever.from_documents(all_splits)
            self.retriever_sparse.k = 8
            
            logger.info(f"Successfully indexed {len(splits)} chunks from {len(docs)} documents")
            
            # Persist vectorstore
            if persist:
                self.save_vectorstore()
            
            return True
            
        except Exception as e:
            logger.error(f"Indexing error: {e}")
            raise
    
    def retrieve_context(self, query: str, domain_filter: Optional[str] = None,
                        use_query_expansion: bool = True, use_reranking: bool = True,
                        top_k: int = 5) -> List[Document]:
        """Retrieve relevant context with query expansion and reranking"""
        if not self.retriever_dense:
            return []
        
        try:
            # Expand query if enabled
            search_query = expand_query(query) if use_query_expansion else query
            
            # Get dense retrieval results
            dense_docs = self.retriever_dense.get_relevant_documents(search_query)
            
            # Apply domain filter if specified
            if domain_filter:
                dense_docs = [
                    doc for doc in dense_docs 
                    if doc.metadata.get("domain", "").lower() == domain_filter.lower()
                ]
            
            # Get sparse retrieval results
            all_docs = dense_docs
            if self.retriever_sparse:
                sparse_docs = self.retriever_sparse.get_relevant_documents(search_query)
                if domain_filter:
                    sparse_docs = [
                        doc for doc in sparse_docs 
                        if doc.metadata.get("domain", "").lower() == domain_filter.lower()
                    ]
                
                # Combine and deduplicate
                all_docs = dense_docs + sparse_docs
                seen_content = set()
                unique_docs = []
                for doc in all_docs:
                    content_hash = hash(doc.page_content[:200])  # Use first 200 chars as hash
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        unique_docs.append(doc)
                all_docs = unique_docs
            
            # Rerank if enabled
            if use_reranking and all_docs:
                all_docs = rerank_documents(query, all_docs, top_k=top_k)
            
            return all_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Document], 
                       persona: str, difficulty: int,
                       conversation_history: Optional[List[Tuple[str, str]]] = None) -> str:
        """Generate an answer using LLM or improved extractive fallback"""
        
        if not context_docs:
            return "I couldn't find any relevant information in the indexed documents. Try asking about different topics or add more documents to the knowledge base."
        
        # Format context
        context_text = "\n\n".join([
            f"Source: {doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))}\n{doc.page_content}"
            for doc in context_docs
        ])
        
        if self.llm is None:
            # IMPROVED EXTRACTIVE MODE with sentence ranking
            return self._extractive_answer(query, context_docs, persona, difficulty)
        
        # Generate with LLM
        try:
            persona_instruction = PERSONAS.get(persona, PERSONAS["Helpful Guide"])
            difficulty_instruction = DIFFICULTY_LEVELS.get(difficulty, "Provide a balanced answer")
            
            # Build conversation history string if available
            conversation_context = ""
            if conversation_history and len(conversation_history) > 0:
                conversation_context = "\n\nPrevious conversation:\n"
                for user_msg, assistant_msg in conversation_history[-3:]:  # Last 3 exchanges
                    conversation_context += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
            
            template = """You are a game assistant. {persona_instruction}

Answer the user's question based ONLY on the provided context. If the context doesn't contain relevant information, say so clearly.

Answer style: {difficulty_instruction}
{conversation_context}
Context from documents:
{context}

Question: {question}

Answer:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            chain = (
                {
                    "context": lambda x: context_text,
                    "question": RunnablePassthrough(),
                    "persona_instruction": lambda x: persona_instruction,
                    "difficulty_instruction": lambda x: difficulty_instruction,
                    "conversation_context": lambda x: conversation_context
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            answer = chain.invoke(query)
            
            # Add sources
            sources = []
            for i, doc in enumerate(context_docs, 1):
                source_name = doc.metadata.get('title', doc.metadata.get('source', f'Document {i}'))
                domain = doc.metadata.get('domain', 'unknown')
                sources.append(f"{i}. **{source_name}** ({domain})")
            
            if sources:
                answer += f"\n\n**Sources:**\n" + "\n".join(sources)
            
            return answer
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating answer: {str(e)}. Please check your API key and try again."
    
    def _extractive_answer(self, query: str, context_docs: List[Document],
                          persona: str, difficulty: int) -> str:
        """Improved extractive answer generation using sentence ranking and selection"""
        # Extract all sentences from context documents
        all_sentences = []
        doc_sources = {}
        
        for doc in context_docs:
            # Split by sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)
            source_name = doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Only meaningful sentences
                    all_sentences.append(sentence)
                    doc_sources[sentence] = source_name
        
        # Score and rank sentences
        scored_sentences = calculate_sentence_scores(query, all_sentences)
        
        if not scored_sentences:
            # Fallback: return first meaningful chunks
            fallback = "\n\n".join([doc.page_content[:300] for doc in context_docs[:3]])
            return f"**Based on the available documents:**\n\n{fallback}..."
        
        # Select sentences based on difficulty
        difficulty_map = {0: 2, 1: 4, 2: 6, 3: 10, 4: 15}
        num_sentences = difficulty_map.get(difficulty, 6)
        selected = scored_sentences[:num_sentences]
        
        # Build answer with formatting
        answer_parts = []
        seen_sources = set()
        
        for score, sentence in selected:
            # Clean up sentence
            sentence = sentence.strip()
            if sentence.endswith('.'):
                answer_parts.append(sentence)
            else:
                answer_parts.append(sentence + ".")
            
            # Track sources
            if sentence in doc_sources:
                seen_sources.add(doc_sources[sentence])
        
        # Format based on difficulty
        if difficulty <= 1:
            answer = ". ".join([s for s, _ in selected[:3]])
            if not answer.endswith('.'):
                answer += "."
        else:
            answer = "\n\n".join([f"- {part}" for part in answer_parts])
        
        # Add header
        answer = "**Based on the available documents:**\n\n" + answer
        
        # Add sources
        if seen_sources:
            answer += f"\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sorted(seen_sources)])
        
        answer += "\n\n*💡 Tip: Enable OpenAI API for generated, conversational answers instead of extracted text.*"
        
        return answer

# ------------------------------
# Streamlit UI
# ------------------------------
def init_session_state():
    """Initialize session state variables"""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = 0
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []

def show_welcome():
    """Show welcome message and instructions"""
    st.markdown("""
    ## Hi I'm XLR8, your Gaming Assistant! 🎮

    I'll help you create an intelligent Q&A system for your game documentation, lore, and mechanics.

    ### Quick Start:
    1. **Initialize** the system below
    2. **Load sample data** or upload your own documents  
    3. **Ask questions** about your game content
    
    ### Supported Document Types:
    - Text documents and PDFs
    - Websites and game wikis
    - Lore, mechanics, and technical documentation
    """)

def setup_sidebar():
    """Setup sidebar configuration"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        use_openai = st.checkbox("Use OpenAI (better answers)", value=False)
        openai_key = None
        
        if use_openai:
            openai_key = st.text_input(
                "OpenAI API Key", 
                type="password", 
                placeholder="sk-...",
                help="Get your API key from https://platform.openai.com/"
            )
            if not openai_key:
                st.warning("Provide API key for OpenAI features")
        
        st.divider()
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            use_query_expansion = st.checkbox("Query Expansion", value=True,
                                             help="Expand queries with synonyms for better retrieval")
            use_reranking = st.checkbox("Reranking", value=True,
                                       help="Rerank retrieved documents by relevance")
            use_conversation_memory = st.checkbox("Conversation Memory", value=True,
                                                 help="Use conversation context in answers")
            top_k = st.slider("Documents to retrieve", 3, 10, 5,
                            help="Number of document chunks to retrieve")
        
        st.divider()
        
        # Answer configuration
        st.subheader("Answer Settings")
        persona = st.selectbox("Assistant Persona", list(PERSONAS.keys()))
        difficulty = st.select_slider(
            "Detail Level", 
            options=list(DIFFICULTY_LEVELS.keys()),
            value=2,
            format_func=lambda x: DIFFICULTY_LEVELS[x]
        )
        domain_filter = st.selectbox(
            "Filter by Domain", 
            ["All"] + GAME_DOMAINS,
            help="Limit answers to specific content type"
        )
        
        return {
            "use_openai": use_openai,
            "openai_key": openai_key,
            "persona": persona,
            "difficulty": difficulty,
            "domain_filter": domain_filter,
            "use_query_expansion": use_query_expansion,
            "use_reranking": use_reranking,
            "use_conversation_memory": use_conversation_memory,
            "top_k": top_k
        }

def main():
    st.set_page_config(
        page_title="Gaming Assistant",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    config = setup_sidebar()
    
    # Main content area
    if st.session_state.rag_system is None:
        show_welcome()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Initialize System", type="primary"):
                with st.spinner("Setting up RAG system..."):
                    try:
                        st.session_state.rag_system = GameRAG(
                            use_openai=config["use_openai"],
                            openai_api_key=config["openai_key"],
                            use_conversation_memory=config["use_conversation_memory"]
                        )
                        # Try to load existing vectorstore
                        if st.session_state.rag_system.load_vectorstore():
                            st.info("📚 Loaded existing knowledge base")
                        st.success("✅ System initialized successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Initialization failed: {str(e)}")
        
        with col2:
            st.info("💡 Initialize the system to start adding documents and asking questions")
    
    else:
        # System is initialized
        rag = st.session_state.rag_system
        
        # Document management
        with st.expander("📚 Document Management", expanded=st.session_state.documents_loaded == 0):
            tab1, tab2, tab3 = st.tabs(["📝 Text", "📄 PDF", "🌐 Website"])
            
            with tab1:
                st.subheader("Add Text Document")
                title = st.text_input("Document Title", placeholder="e.g., Combat System Guide", key="text_title")
                domain = st.selectbox("Domain", GAME_DOMAINS, key="text_domain")
                text_content = st.text_area(
                    "Document Content", 
                    height=200,
                    placeholder="Paste your game documentation, lore, or mechanics here...",
                    key="text_content"
                )
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("➕ Add Text", disabled=not title or not text_content, key="add_text_btn"):
                        try:
                            metadata = {"domain": domain, "title": title, "source": "user_input"}
                            docs = rag.load_text_document(text_content, metadata)
                            if docs:
                                rag.index_documents(docs)
                                st.session_state.documents_loaded += len(docs)
                                st.success(f"✅ Added '{title}' to knowledge base")
                                st.rerun()
                            else:
                                st.error("❌ Failed to create document")
                        except Exception as e:
                            st.error(f"❌ Error adding document: {str(e)}")
                with col2:
                    if st.button("📚 Load Sample Data", key="load_sample_btn"):
                        try:
                            count = rag.load_sample_data()
                            if count > 0:
                                st.session_state.documents_loaded += count
                                st.success(f"✅ Loaded {count} sample documents")
                                st.rerun()
                            else:
                                st.error("❌ Failed to load sample data")
                        except Exception as e:
                            st.error(f"❌ Error loading sample data: {str(e)}")
            
            with tab2:
                st.subheader("Upload PDF")
                uploaded_file = st.file_uploader("Choose PDF file", type="pdf", key="pdf_uploader")
                if uploaded_file:
                    pdf_title = st.text_input("PDF Title", value=uploaded_file.name.replace(".pdf", ""), key="pdf_title")
                    pdf_domain = st.selectbox("Domain", GAME_DOMAINS, key="pdf_domain")
                    
                    if st.button("📄 Process PDF", key="process_pdf_btn"):
                        with st.spinner("Processing PDF..."):
                            try:
                                metadata = {"domain": pdf_domain, "title": pdf_title, "source": "pdf_upload"}
                                docs = rag.load_pdf_document(uploaded_file.read(), metadata)
                                if docs:
                                    rag.index_documents(docs)
                                    st.session_state.documents_loaded += len(docs)
                                    st.success(f"✅ Processed PDF: {len(docs)} pages added")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to process PDF")
                            except Exception as e:
                                st.error(f"❌ Error processing PDF: {str(e)}")
            
            with tab3:
                st.subheader("Load from Website")
                url = st.text_input("Website URL", placeholder="https://your-game-wiki.com", key="web_url")
                if url:
                    web_title = st.text_input("Website Title", value="Web Content", key="web_title")
                    web_domain = st.selectbox("Domain", GAME_DOMAINS, key="web_domain")
                    
                    if st.button("🌐 Load Website", key="load_web_btn"):
                        with st.spinner("Loading website content..."):
                            try:
                                metadata = {"domain": web_domain, "title": web_title, "source": "website"}
                                docs = rag.load_website(url, metadata)
                                if docs:
                                    rag.index_documents(docs)
                                    st.session_state.documents_loaded += len(docs)
                                    st.success(f"✅ Loaded website content")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to load website")
                            except Exception as e:
                                st.error(f"❌ Error loading website: {str(e)}")
        
        # Status info and controls
        if st.session_state.documents_loaded > 0:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"📊 Knowledge base contains {st.session_state.documents_loaded} documents")
            with col2:
                if st.button("💾 Save Vector Store", help="Save the current vector store to disk"):
                    if rag.save_vectorstore():
                        st.success("✅ Saved!")
                    else:
                        st.error("❌ Save failed")
                if st.button("🗑️ Clear Chat", help="Clear chat history"):
                    st.session_state.chat_history = []
                    st.session_state.conversation_context = []
                    st.rerun()
        
        # Chat interface - FIXED RENDERING
        st.subheader("💬 Ask Questions")
        
        # Display chat history FIRST (this is the key fix)
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)
        
        # Chat input
        if prompt := st.chat_input("Ask about your game content..."):
            if st.session_state.documents_loaded == 0:
                st.warning("⚠️ Please add some documents first!")
            else:
                # Add user message to history IMMEDIATELY
                st.session_state.chat_history.append(("user", prompt))
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        domain_filt = None if config["domain_filter"] == "All" else config["domain_filter"]
                        
                        # Get conversation history for context
                        conv_history = None
                        if config["use_conversation_memory"]:
                            # Extract last few exchanges from chat history
                            conv_history = []
                            for i in range(0, len(st.session_state.chat_history) - 1, 2):
                                if i + 1 < len(st.session_state.chat_history):
                                    conv_history.append((
                                        st.session_state.chat_history[i][1],
                                        st.session_state.chat_history[i + 1][1]
                                    ))
                        
                        context_docs = rag.retrieve_context(
                            prompt,
                            domain_filt,
                            use_query_expansion=config["use_query_expansion"],
                            use_reranking=config["use_reranking"],
                            top_k=config["top_k"]
                        )
                        
                        answer = rag.generate_answer(
                            prompt,
                            context_docs,
                            config["persona"],
                            config["difficulty"],
                            conversation_history=conv_history
                        )
                        st.markdown(answer)
                
                # Add assistant response to history
                st.session_state.chat_history.append(("assistant", answer))
                st.rerun()
        
        # Example questions
        if st.session_state.documents_loaded > 0:
            st.subheader("💡 Try These Questions")
            example_cols = st.columns(3)
            examples = [
                "How does combat work?",
                "Tell me about the game world",
                "What are the main character abilities?"
            ]
            
            for col, example in zip(example_cols, examples):
                if col.button(f"💭 {example}", key=f"example_{hash(example)}"):
                    # Add user message
                    st.session_state.chat_history.append(("user", example))
                    
                    # Generate and add assistant response
                    domain_filt = None if config["domain_filter"] == "All" else config["domain_filter"]
                    
                    conv_history = None
                    if config["use_conversation_memory"]:
                        conv_history = []
                        for i in range(0, len(st.session_state.chat_history) - 1, 2):
                            if i + 1 < len(st.session_state.chat_history):
                                conv_history.append((
                                    st.session_state.chat_history[i][1],
                                    st.session_state.chat_history[i + 1][1]
                                ))
                    
                    context_docs = rag.retrieve_context(
                        example,
                        domain_filt,
                        use_query_expansion=config["use_query_expansion"],
                        use_reranking=config["use_reranking"],
                        top_k=config["top_k"]
                    )
                    answer = rag.generate_answer(
                        example,
                        context_docs,
                        config["persona"],
                        config["difficulty"],
                        conversation_history=conv_history
                    )
                    st.session_state.chat_history.append(("assistant", answer))
                    st.rerun()

if __name__ == "__main__":
    main()
