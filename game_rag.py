import os
import io
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

import streamlit as st
import pandas as pd

# LangChain core & community
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS as LCFAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

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
# Core RAG System
# ------------------------------
class GameRAG:
    def __init__(self, use_openai: bool = False, openai_api_key: Optional[str] = None):
        self.use_openai = use_openai and HAS_OPENAI
        self.openai_api_key = openai_api_key
        
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
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.use_openai = False
    
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
        finally:
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
    
    def index_documents(self, docs: List[Document]):
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
                search_kwargs={"k": 5}
            )
            self.retriever_sparse = BM25Retriever.from_documents(splits)
            self.retriever_sparse.k = 5
            
            logger.info(f"Successfully indexed {len(splits)} chunks from {len(docs)} documents")
            
        except Exception as e:
            logger.error(f"Indexing error: {e}")
            # Don't raise the error, just log it and continue
            st.error(f"Indexing failed: {str(e)}")
            return False
        
        return True
    
    def retrieve_context(self, query: str, domain_filter: Optional[str] = None) -> List[Document]:
        """Retrieve relevant context for a query"""
        if not self.retriever_dense:
            return []
        
        try:
            # Get dense retrieval results
            dense_docs = self.retriever_dense.get_relevant_documents(query)
            
            # Apply domain filter if specified
            if domain_filter:
                dense_docs = [
                    doc for doc in dense_docs 
                    if doc.metadata.get("domain", "").lower() == domain_filter.lower()
                ]
            
            # Get sparse retrieval results
            if self.retriever_sparse:
                sparse_docs = self.retriever_sparse.get_relevant_documents(query)
                if domain_filter:
                    sparse_docs = [
                        doc for doc in sparse_docs 
                        if doc.metadata.get("domain", "").lower() == domain_filter.lower()
                    ]
                
                # Simple fusion: combine and deduplicate
                all_docs = dense_docs + sparse_docs
                seen_content = set()
                unique_docs = []
                for doc in all_docs:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        unique_docs.append(doc)
                
                return unique_docs[:6]
            
            return dense_docs[:5]
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Document], 
                       persona: str, difficulty: int) -> str:
        """Generate an answer using the LLM or fallback to extractive summary"""
        
        if not context_docs:
            return "I couldn't find any relevant information in the indexed documents. Try asking about different topics or add more documents to the knowledge base."
        
        # Format context
        context_text = "\n\n".join([
            f"Source: {doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))}\n{doc.page_content}"
            for doc in context_docs
        ])
        
        if self.llm is None:
            # Extractive fallback
            summary = context_text[:1500] + "..." if len(context_text) > 1500 else context_text
            return f"**Based on the available documents:**\n\n{summary}\n\n*Note: Using extractive mode. For generated answers, configure OpenAI API key.*"
        
        # Generate with LLM
        try:
            persona_instruction = PERSONAS.get(persona, PERSONAS["Helpful Guide"])
            difficulty_instruction = DIFFICULTY_LEVELS.get(difficulty, "Provide a balanced answer")
            
            template = """You are a game assistant. {persona_instruction}

Answer the user's question based ONLY on the provided context. If the context doesn't contain relevant information, say so clearly.

Answer style: {difficulty_instruction}

Context:
{context}

Question: {question}

Answer:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            chain = (
                {
                    "context": lambda x: context_text,
                    "question": RunnablePassthrough(),
                    "persona_instruction": lambda x: persona_instruction,
                    "difficulty_instruction": lambda x: difficulty_instruction
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
            return f"Error generating answer: {str(e)}"

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
        
        return use_openai, openai_key, persona, difficulty, domain_filter

def main():
    st.set_page_config(
        page_title="Gaming Assistant",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    ''''
    st.title("🎮 Gaming Assistant")
    st.markdown("*Intelligent Q&A for your game documentation*")
    ''''
    
    init_session_state()
    use_openai, openai_key, persona, difficulty, domain_filter = setup_sidebar()
    
    # Main content area
    if st.session_state.rag_system is None:
        show_welcome()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Initialize System", type="primary"):
                with st.spinner("Setting up RAG system..."):
                    try:
                        st.session_state.rag_system = GameRAG(
                            use_openai=use_openai, 
                            openai_api_key=openai_key
                        )
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
                title = st.text_input("Document Title", placeholder="e.g., Combat System Guide")
                domain = st.selectbox("Domain", GAME_DOMAINS, key="text_domain")
                text_content = st.text_area(
                    "Document Content", 
                    height=200,
                    placeholder="Paste your game documentation, lore, or mechanics here..."
                )
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("➕ Add Text", disabled=not title or not text_content):
                        try:
                            metadata = {"domain": domain, "title": title, "source": "user_input"}
                            docs = rag.load_text_document(text_content, metadata)
                            if docs:
                                success = rag.index_documents(docs)
                                if success:
                                    st.session_state.documents_loaded += len(docs)
                                    st.success(f"✅ Added '{title}' to knowledge base")
                                else:
                                    st.error("❌ Failed to index document")
                        except Exception as e:
                            st.error(f"❌ Error adding document: {str(e)}")
                with col2:
                    if st.button("📚 Load Sample Data"):
                        try:
                            count = rag.load_sample_data()
                            if count > 0:
                                st.session_state.documents_loaded += count
                                st.success(f"✅ Loaded {count} sample documents")
                            else:
                                st.error("❌ Failed to load sample data")
                        except Exception as e:
                            st.error(f"❌ Error loading sample data: {str(e)}")
            
            with tab2:
                st.subheader("Upload PDF")
                uploaded_file = st.file_uploader("Choose PDF file", type="pdf")
                if uploaded_file:
                    pdf_title = st.text_input("PDF Title", value=uploaded_file.name.replace(".pdf", ""))
                    pdf_domain = st.selectbox("Domain", GAME_DOMAINS, key="pdf_domain")
                    
                    if st.button("📄 Process PDF"):
                        with st.spinner("Processing PDF..."):
                            try:
                                metadata = {"domain": pdf_domain, "title": pdf_title, "source": "pdf_upload"}
                                docs = rag.load_pdf_document(uploaded_file.read(), metadata)
                                if docs:
                                    success = rag.index_documents(docs)
                                    if success:
                                        st.session_state.documents_loaded += len(docs)
                                        st.success(f"✅ Processed PDF: {len(docs)} pages added")
                                    else:
                                        st.error("❌ Failed to index PDF content")
                                else:
                                    st.error("❌ Failed to process PDF")
                            except Exception as e:
                                st.error(f"❌ Error processing PDF: {str(e)}")
            
            with tab3:
                st.subheader("Load from Website")
                url = st.text_input("Website URL", placeholder="https://your-game-wiki.com")
                if url:
                    web_title = st.text_input("Website Title", value="Web Content")
                    web_domain = st.selectbox("Domain", GAME_DOMAINS, key="web_domain")
                    
                    if st.button("🌐 Load Website"):
                        with st.spinner("Loading website content..."):
                            try:
                                metadata = {"domain": web_domain, "title": web_title, "source": "website"}
                                docs = rag.load_website(url, metadata)
                                if docs:
                                    success = rag.index_documents(docs)
                                    if success:
                                        st.session_state.documents_loaded += len(docs)
                                        st.success(f"✅ Loaded website content")
                                    else:
                                        st.error("❌ Failed to index website content")
                                else:
                                    st.error("❌ Failed to load website")
                            except Exception as e:
                                st.error(f"❌ Error loading website: {str(e)}")
        
        # Status info
        if st.session_state.documents_loaded > 0:
            st.info(f"📊 Knowledge base contains {st.session_state.documents_loaded} documents")
        
        # Chat interface
        st.subheader("💬 Ask Questions")
        
        # Display chat history
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)
        
        # Chat input
        if prompt := st.chat_input("Ask about your game content..."):
            if st.session_state.documents_loaded == 0:
                st.warning("⚠️ Please add some documents first!")
            else:
                # Add user message
                st.session_state.chat_history.append(("user", prompt))
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        domain_filt = None if domain_filter == "All" else domain_filter
                        context_docs = rag.retrieve_context(prompt, domain_filt)
                        answer = rag.generate_answer(prompt, context_docs, persona, difficulty)
                        st.markdown(answer)
                        st.session_state.chat_history.append(("assistant", answer))
        
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
                if col.button(f"💭 {example}"):
                    # Add user message
                    st.session_state.chat_history.append(("user", example))
                    
                    # Generate and add assistant response
                    domain_filt = None if domain_filter == "All" else domain_filter
                    context_docs = rag.retrieve_context(example, domain_filt)
                    answer = rag.generate_answer(example, context_docs, persona, difficulty)
                    st.session_state.chat_history.append(("assistant", answer))
                    
                    st.rerun()

if __name__ == "__main__":
    main()