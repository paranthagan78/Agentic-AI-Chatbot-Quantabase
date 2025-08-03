# Replace the import section with these updated imports:
from dotenv import load_dotenv
load_dotenv()

import logging
import hashlib
import json
import time
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import glob

# Updated LangChain imports
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.schema.retriever import BaseRetriever
from langchain_community.llms import CTransformers

# Other imports
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
import redis
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec

# Backup local LLM
try:
    from ctransformers import AutoModelForCausalLM
    CTRANSFORMERS_AVAILABLE = True
    print("✅ ctransformers available")
except ImportError:
    CTRANSFORMERS_AVAILABLE = False
    print("⚠️ ctransformers not available - install with: pip install ctransformers")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Enhanced search result with confidence scoring"""
    content: str
    metadata: Dict
    semantic_score: float
    bm25_score: float
    hybrid_score: float
    confidence: float
    relevance_score: float = 0.0

class GeminiFlashLLM(LLM):
    """Custom Gemini Flash LLM wrapper for LangChain"""
    
    def __init__(self, api_key: str):
        super().__init__()
        # Configure the API key
        genai.configure(api_key=api_key)
        # Initialize the model - store as private attribute
        self._model = genai.GenerativeModel('gemini-1.5-flash')
        
    @property
    def _llm_type(self) -> str:
        return "gemini-flash"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            response = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=1024,
                    candidate_count=1
                )
            )
            
            return response.text if response.text else "I apologize, but I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"I encountered an issue generating a response. Please try again."

class AgenticChunker:
    """LLM-powered semantic chunking with fallback to traditional chunking"""
    
    def __init__(self, gemini_llm):
        self.llm = gemini_llm
        self.max_context_length = 1500  # Conservative limit for input text
        self.chunk_prompt = """
You are an expert document analyzer. Split this text into 2-3 meaningful chunks of 250-350 words each.

Text: {text}

Return chunks separated by "---CHUNK---".
"""
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Chunk a single document using LLM-based semantic splitting with fallback"""
        try:
            text = document.page_content
            
            # Check text length and use fallback for long texts
            if len(text) < 300:  # Small documents don't need chunking
                return [document]
            elif len(text) > self.max_context_length:  # Too long for LLM
                print(f"⚠️ Document too long ({len(text)} chars), using fallback chunking")
                return self._fallback_chunk(document)
            
            # Check if LLM is available and working
            if self.llm is None or not hasattr(self.llm, '_call'):
                return self._fallback_chunk(document)
            
            # Try LLM-based chunking with error handling
            try:
                prompt = self.chunk_prompt.format(text=text)
                response = self.llm._call(prompt)
                
                # Parse chunks from response
                chunks = [chunk.strip() for chunk in response.split("---CHUNK---") if chunk.strip()]
                
                # Validate chunks
                if not chunks or len(chunks) == 0:
                    print("⚠️ LLM returned no valid chunks, using fallback")
                    return self._fallback_chunk(document)
                
                # Create Document objects for each chunk
                chunked_docs = []
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:  # Skip very small chunks
                        continue
                        
                    chunk_metadata = document.metadata.copy()
                    chunk_metadata.update({
                        'chunk_id': i,
                        'total_chunks': len(chunks),
                        'chunk_method': 'agentic',
                        'original_length': len(text),
                        'chunk_length': len(chunk)
                    })
                    
                    chunked_docs.append(Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    ))
                
                return chunked_docs if chunked_docs else [document]
                
            except Exception as llm_error:
                print(f"⚠️ LLM chunking failed ({llm_error}), using fallback")
                return self._fallback_chunk(document)
                
        except Exception as e:
            print(f"⚠️ Agentic chunking error: {e}")
            return self._fallback_chunk(document)
    
    def _fallback_chunk(self, document: Document) -> List[Document]:
        """Fallback to traditional chunking if LLM chunking fails"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " "]
        )
        chunks = splitter.split_documents([document])
        
        # Update metadata to indicate fallback method
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'total_chunks': len(chunks),
                'chunk_method': 'fallback_traditional',
                'original_length': len(document.page_content),
                'chunk_length': len(chunk.page_content)
            })
        
        return chunks
    
class RedisSessionManager:
    """Redis-based session management for RAG chatbot"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0, session_ttl: int = 3600):
        self.session_ttl = session_ttl
        self.redis_client = None
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            logger.info("📝 Continuing without Redis (memory will be in-process only)")
            self.redis_client = None
    
    def get_session_memory(self, session_id: str) -> RedisChatMessageHistory:
        """Get Redis-backed chat history for a session"""
        if not self.redis_client:
            return None
        
        return RedisChatMessageHistory(
            session_id=session_id,
            url=f"redis://{self.redis_client.connection_pool.connection_kwargs['host']}:{self.redis_client.connection_pool.connection_kwargs['port']}/{self.redis_client.connection_pool.connection_kwargs['db']}",
            ttl=self.session_ttl
        )
    
    def cache_query_result(self, query: str, result: str, session_id: str):
        """Cache query-result pairs for faster responses"""
        if not self.redis_client:
            return
        
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()
        cache_key = f"cache:{session_id}:{query_hash}"
        
        self.redis_client.setex(
            cache_key,
            self.session_ttl,
            json.dumps({
                'query': query,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
        )
    
    def get_cached_result(self, query: str, session_id: str) -> Optional[str]:
        """Retrieve cached result for similar queries"""
        if not self.redis_client:
            return None
        
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()
        cache_key = f"cache:{session_id}:{query_hash}"
        
        cached = self.redis_client.get(cache_key)
        if cached:
            try:
                data = json.loads(cached)
                return data['result']
            except:
                pass
        return None

class EnhancedEmbeddingModel:
    """High-performance embedding model for semantic search"""
    
    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        self.model_name = model_name
        print(f"🔧 Loading embedding model: {model_name}")
        
        # Load model with optimized settings
        self.model = SentenceTransformer(
            model_name,
            device='cpu',  # Use 'cuda' if GPU available
            cache_folder=r'C:\Users\paran\.cache\huggingface\hub'
        )
        
        # Get embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Embedding model loaded - Dimension: {self.dimension}")
        
        if not (700 <= self.dimension <= 800):
            print(f"⚠️ Warning: Embedding dimension {self.dimension} is outside recommended range (700-800)")
    
    def embed_texts(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """Embed multiple texts with optimization"""
        # Add e5 prefix for better performance
        if self.model_name.startswith("intfloat/e5"):
            texts = [f"passage: {text}" for text in texts]
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            batch_size=32,
            convert_to_tensor=False
        )
        
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query"""
        # Add e5 prefix for queries
        if self.model_name.startswith("intfloat/e5"):
            query = f"query: {query}"
        
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embedding[0].tolist()

class HybridVectorStore:
    """Hybrid vector store supporting both ChromaDB and Pinecone"""
    
    def __init__(self, embedding_model: EnhancedEmbeddingModel, 
                 use_pinecone: bool = True, pinecone_api_key: str = None,
                 pinecone_environment: str = None):
        self.embedding_model = embedding_model
        self.use_pinecone = use_pinecone and pinecone_api_key
        self.vectorstore = None
        self.bm25_vectorizer = None
        self.document_corpus = []
        self.document_metadata = []
        
        if self.use_pinecone:
            self._setup_pinecone(pinecone_api_key, pinecone_environment)
        else:
            self._setup_chroma()
    
    def _setup_pinecone(self, api_key: str, environment: str):
        """Setup Pinecone vector store"""
        try:
            pc = Pinecone(api_key=api_key)
            
            index_name = "quantabase-retail-rag"
            
            # Create index if it doesn't exist
            existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
            
            if index_name not in existing_indexes:
                pc.create_index(
                    name=index_name,
                    dimension=self.embedding_model.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"✅ Created Pinecone index: {index_name}")
            
            # Connect to index
            index = pc.Index(index_name)
            
            # For now, use ChromaDB as Pinecone integration has issues
            logger.info("🔄 Using ChromaDB instead of Pinecone for stability")
            self._setup_chroma()
            
        except Exception as e:
            logger.error(f"❌ Pinecone setup failed: {e}")
            logger.info("🔄 Falling back to ChromaDB")
            self._setup_chroma()
    
    def _setup_chroma(self):
        """Setup ChromaDB vector store"""
        try:
            client = chromadb.PersistentClient(
                path="./chroma_db_enhanced",
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.vectorstore = Chroma(
                client=client,
                collection_name="retail_documents_enhanced",
                embedding_function=self._get_langchain_embeddings(),
                persist_directory="./chroma_db_enhanced"
            )
            
            logger.info("✅ ChromaDB vector store ready")
            
        except Exception as e:
            logger.error(f"❌ ChromaDB setup failed: {e}")
            raise
    
    def _get_langchain_embeddings(self):
        """Get LangChain compatible embeddings"""
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def add_documents(self, documents: List[Document]):
        """Add documents to vector store"""
        try:
            # Store for BM25 indexing
            self.document_corpus = [doc.page_content for doc in documents]
            self.document_metadata = [doc.metadata for doc in documents]
            
            # Add to vector store
            self.vectorstore.add_documents(documents)
            
            # Build BM25 index
            self._build_bm25_index()
            
            logger.info(f"✅ Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            raise
    
    def _build_bm25_index(self):
        """Build BM25 index for keyword search"""
        try:
            self.bm25_vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=5000,
                lowercase=True,
                analyzer='word'
            )
            
            self.bm25_matrix = self.bm25_vectorizer.fit_transform(self.document_corpus)
            logger.info("✅ BM25 index built successfully")
            
        except Exception as e:
            logger.error(f"⚠️ BM25 index building failed: {e}")
            self.bm25_vectorizer = None
    
    def hybrid_search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Perform hybrid search with semantic + BM25"""
        try:
            # Semantic search
            semantic_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
            
            # BM25 search
            bm25_results = []
            if self.bm25_vectorizer is not None:
                query_vector = self.bm25_vectorizer.transform([query])
                bm25_scores = cosine_similarity(query_vector, self.bm25_matrix).flatten()
                
                top_indices = bm25_scores.argsort()[-k*2:][::-1]
                for idx in top_indices:
                    if bm25_scores[idx] > 0:
                        bm25_results.append({
                            'content': self.document_corpus[idx],
                            'metadata': self.document_metadata[idx],
                            'score': bm25_scores[idx]
                        })
            
            # Combine results
            return self._combine_results(semantic_results, bm25_results, k)
            
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            return []
    
    def _combine_results(self, semantic_results, bm25_results, k: int) -> List[SearchResult]:
        """Combine and rank search results"""
        combined = {}
        
        # Process semantic results
        if semantic_results:
            max_dist = max(score for _, score in semantic_results)
            min_dist = min(score for _, score in semantic_results)
            dist_range = max_dist - min_dist if max_dist != min_dist else 1
            
            for doc, distance in semantic_results:
                content = doc.page_content
                key = hash(content)
                
                # Convert distance to similarity (0-1, higher is better)
                similarity = 1.0 - ((distance - min_dist) / dist_range)
                
                combined[key] = SearchResult(
                    content=content,
                    metadata=doc.metadata,
                    semantic_score=similarity,
                    bm25_score=0.0,
                    hybrid_score=0.0,
                    confidence=0.0
                )
        
        # Process BM25 results
        if bm25_results:
            max_bm25 = max(r['score'] for r in bm25_results)
            min_bm25 = min(r['score'] for r in bm25_results)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1
            
            for result in bm25_results:
                content = result['content']
                key = hash(content)
                
                normalized_bm25 = (result['score'] - min_bm25) / bm25_range
                
                if key in combined:
                    combined[key].bm25_score = max(combined[key].bm25_score, normalized_bm25)
                else:
                    combined[key] = SearchResult(
                        content=content,
                        metadata=result['metadata'],
                        semantic_score=0.0,
                        bm25_score=normalized_bm25,
                        hybrid_score=0.0,
                        confidence=0.0
                    )
        
        # Calculate hybrid scores
        for result in combined.values():
            # Weighted combination (70% semantic, 30% BM25)
            result.hybrid_score = 0.7 * result.semantic_score + 0.3 * result.bm25_score
            
            # Confidence calculation
            result.confidence = min(
                (result.semantic_score * 0.6 + result.bm25_score * 0.4),
                1.0
            )
            
            # Boost if both methods found it
            if result.semantic_score > 0 and result.bm25_score > 0:
                result.confidence = min(result.confidence * 1.2, 1.0)
        
        # Sort by hybrid score and return top k
        sorted_results = sorted(combined.values(), key=lambda x: x.hybrid_score, reverse=True)
        return sorted_results[:k]
    
from langchain.schema.retriever import BaseRetriever

class HybridRetriever(BaseRetriever):
    """Fixed Hybrid Retriever for vector store"""
    
    def __init__(self, vector_store):
        super().__init__()
        self._vector_store = vector_store
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        try:
            search_results = self._vector_store.hybrid_search(query, k=5)
            return [
                Document(
                    page_content=result.content,
                    metadata=result.metadata
                )
                for result in search_results
            ]
        except Exception as e:
            logger.error(f"Retriever error: {e}")
            # Fallback to simple similarity search
            try:
                docs = self._vector_store.vectorstore.similarity_search(query, k=5)
                return docs
            except:
                return []

class EnhancedRAGChatbot:
    """Next-generation RAG chatbot with Gemini Flash, Redis, and hybrid search"""
    
    def __init__(self, dataset_path: str, gemini_api_key: str, use_pinecone: bool = True, pinecone_api_key: str = None, pinecone_environment: str = None, redis_host: str = "localhost", redis_port: int = 6379):
        self.dataset_path = dataset_path
        self.session_id = f"user_{int(time.time())}"
        
        # Initialize components
        print("🚀 Initializing Enhanced RAG Chatbot...")
        
        # Setup LLM
        self.setup_llm(gemini_api_key)
        
        # Setup embeddings
        self.embedding_model = EnhancedEmbeddingModel("intfloat/e5-base-v2")
        
        # Setup Redis session manager
        self.redis_manager = RedisSessionManager(redis_host, redis_port)
        
        # Setup vector store
        self.vector_store = HybridVectorStore(
            self.embedding_model,
            use_pinecone=use_pinecone,
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_environment
        )
        
        # Setup agentic chunker - use active_llm instead of gemini_llm
        self.agentic_chunker = AgenticChunker(self.active_llm)
        
        # Load and process documents
        self.load_and_process_documents()
        
        # Setup QA chain
        self.setup_qa_chain()
        
        print("✅ Enhanced RAG Chatbot initialized successfully!")

    def find_orca_model_path(self):
        """Find the correct path to the Orca Mini 3B GGUF model"""
        base_path = r"C:\Users\paran\.cache\huggingface\hub\models--zoltanctoth--orca_mini_3B-GGUF"
        
        # Common GGUF file patterns
        gguf_patterns = [
            "*.gguf",
            "**/*.gguf",
            "**/orca*.gguf",
            "**/orca-mini*.gguf"
        ]
        
        print(f"🔍 Searching for GGUF model in: {base_path}")
        
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Model directory not found: {base_path}")
        
        # Search for GGUF files
        for pattern in gguf_patterns:
            full_pattern = os.path.join(base_path, pattern)
            matches = glob.glob(full_pattern, recursive=True)
            if matches:
                model_path = matches[0]  # Take the first match
                print(f"✅ Found GGUF model: {model_path}")
                return model_path
        
        # If no GGUF found, list available files for debugging
        print("❌ No GGUF files found. Available files:")
        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"   📄 {file_path}")
        
        raise FileNotFoundError("No GGUF model file found in the specified directory")
    
    def setup_llm(self, gemini_api_key: str):
        """Setup LLM with Gemini Flash primary and local fallback"""
        self.gemini_llm = None
        self.local_llm = None
        self.active_llm = None
        
        # Try Gemini first
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                test_model = genai.GenerativeModel('gemini-1.5-flash')
                # Test with a simple query
                test_response = test_model.generate_content("Hello")
                if test_response.text:
                    self.gemini_llm = GeminiFlashLLM(gemini_api_key)
                    self.active_llm = self.gemini_llm
                    print("✅ Gemini LLM loaded successfully")
                    return
            except Exception as e:
                print(f"⚠️ Gemini LLM failed: {e}")
                self.gemini_llm = None
        
        # Setup local fallback LLM with better context handling
        if CTRANSFORMERS_AVAILABLE:
            print("🤖 Setting up optimized Orca Mini 3B LLM...")
            try:
                model_path = self.find_orca_model_path()
                
                # OPTIMIZED CONFIGURATION WITH BETTER CONTEXT LIMITS
                self.local_llm = CTransformers(
                    model=model_path,
                    model_type="llama",
                    config={
                        # REDUCED LIMITS TO PREVENT CONTEXT OVERFLOW
                        'max_new_tokens': 256,      # Further reduced for stability
                        'temperature': 0.1,
                        'context_length': 1024,     # Reduced to prevent overflow warnings
                        'top_p': 0.8,
                        'repetition_penalty': 1.05,
                        'threads': 8,
                        'batch_size': 1,
                        'stream': False,
                        'top_k': 40,
                        'gpu_layers': 0,
                        'mmap': True,
                        'mlock': False,
                        # ADDITIONAL SAFETY SETTINGS
                        'reset': True,              # Reset context when needed
                    }
                )
                self.active_llm = self.local_llm
                print("✅ Optimized Orca Mini 3B loaded successfully!")
                print("⚠️ Using reduced context window to prevent token overflow")
                
            except Exception as e:
                print(f"❌ Error loading local model: {e}")
                self.local_llm = None
        
        # Final check
        if self.active_llm is None:
            raise Exception("No LLM could be initialized. Please check your configuration.")

    def load_and_process_documents(self):
        """Load and process documents with agentic chunking and fallback"""
        print("📄 Loading and processing documents...")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        # Load documents
        loader = DirectoryLoader(
            self.dataset_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8', 'autodetect_encoding': True},
            show_progress=True
        )
        
        documents = loader.load()
        print(f"📚 Loaded {len(documents)} documents")
        
        if not documents:
            raise ValueError("No documents found in the dataset path")
        
        # Apply agentic chunking with better error handling
        print("🧠 Applying agentic chunking...")
        all_chunks = []
        agentic_success = 0
        fallback_used = 0
        
        for i, doc in enumerate(documents):
            try:
                print(f"Processing document {i+1}/{len(documents)}...")
                chunks = self.agentic_chunker.chunk_document(doc)
                all_chunks.extend(chunks)
                
                # Check which method was used
                if chunks and chunks[0].metadata.get('chunk_method') == 'agentic':
                    agentic_success += 1
                else:
                    fallback_used += 1
                    
            except Exception as e:
                print(f"⚠️ Error processing document {i+1}: {e}")
                # Use fallback for this document
                fallback_chunks = self.agentic_chunker._fallback_chunk(doc)
                all_chunks.extend(fallback_chunks)
                fallback_used += 1
        
        print(f"📝 Chunking complete: {len(all_chunks)} total chunks")
        print(f"✅ Agentic chunking successful: {agentic_success}/{len(documents)} documents")
        print(f"🔄 Fallback chunking used: {fallback_used}/{len(documents)} documents")
        
        # Add to vector store
        self.vector_store.add_documents(all_chunks)
    
    def setup_qa_chain(self):
        """Setup QA chain with Redis memory"""
        print("⛓️ Setting up QA chain with Redis memory...")
        
        # Setup Redis-backed memory
        redis_history = self.redis_manager.get_session_memory(self.session_id)
        
        if redis_history:
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                chat_memory=redis_history,
                return_messages=True,
                output_key="answer",
                k=8
            )
        else:
            # Fallback to regular memory
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                k=8
            )
        
        # Enhanced prompt template
        custom_prompt = PromptTemplate(
            template="""You are an expert retail industry analyst and consultant. Use the provided context to deliver comprehensive, accurate, and actionable insights.

    Context Information:
    {context}

    Conversation History:
    {chat_history}

    Current Question: {question}

    Instructions:
    1. Provide detailed, professional analysis based on the context
    2. Include specific data, trends, and actionable recommendations when available
    3. Structure your response clearly with key points
    4. Maintain a professional yet approachable tone
    5. If the context doesn't contain sufficient information, clearly state this
    6. Focus on practical, implementable insights
    7. Keep responses concise but comprehensive (aim for 150-300 words)

    Professional Response:""",
            input_variables=["context", "chat_history", "question"]
        )

        # Create hybrid retriever
        hybrid_retriever = HybridRetriever(self.vector_store)
        
        # Create QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.active_llm,
            retriever=hybrid_retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )

        print("✅ QA chain setup complete!")
    
    def chat(self, question: str) -> str:
        """Enhanced chat method with caching and error handling"""
        print(f"\n❓ User: {question}")
        
        # Check cache first
        cached_result = self.redis_manager.get_cached_result(question, self.session_id)
        if cached_result:
            print("⚡ Using cached response")
            print(f"🤖 Bot: {cached_result}")
            return cached_result
        
        try:
            start_time = time.time()
            
            # Get response from QA chain
            result = self.qa_chain({
                "question": question,
                "chat_history": []  # Redis memory handles this
            })
            
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            # Clean and format response
            cleaned_answer = self._clean_response(answer)
            
            # Cache the result
            self.redis_manager.cache_query_result(question, cleaned_answer, self.session_id)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            print(f"🤖 Bot: {cleaned_answer}")
            print(f"⏱️ Response time: {response_time:.2f}s")
            
            if source_docs:
                print(f"📚 Sources: {len(source_docs)} documents referenced")
            
            return cleaned_answer
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            
            if self.active_llm == self.gemini_llm and self.local_llm:
                try:
                    print("🔄 Switching to local LLM due to Gemini error...")
                    self.active_llm = self.local_llm
                    # Recreate QA chain with local LLM
                    self.setup_qa_chain()
                    # Retry the query
                    result = self.qa_chain({
                        "question": question,
                        "chat_history": []
                    })
                    return self._clean_response(result["answer"])
                except Exception as fallback_error:
                    logger.error(f"Local LLM fallback failed: {fallback_error}")
            
            return "I apologize, but I'm having trouble processing your request. Please try again later."
               
    def _clean_response(self, response: str) -> str:
        """Clean and format the response"""
        if not response:
            return "I don't have sufficient information to answer your question."
        
        response = response.strip()
        
        # Remove common prefixes
        prefixes = ["answer:", "response:", "bot:", "assistant:", "professional response:"]
        response_lower = response.lower()
        for prefix in prefixes:
            if response_lower.startswith(prefix):
                response = response[len(prefix):].strip()
                break
        
        # Ensure reasonable length
        if len(response) > 2000:
            sentences = response.split('. ')
            trimmed = ""
            for sentence in sentences:
                if len(trimmed + sentence) <= 2000:
                    trimmed += sentence + ". "
                else:
                    break
            response = trimmed.strip()
        
        return response if response else "I don't have enough information to provide a comprehensive answer."
    
    # Missing code completion for Enhanced RAG Chatbot

    def start_interactive_chat(self):
        """Start interactive chat session"""
        print("\n" + "="*80)
        print("🚀 Enhanced RAG Chatbot - Gemini Flash Powered")
        print("💼 Retail Industry Expert Assistant")
        print("🔍 Hybrid Search: Semantic + Keyword Matching")
        print("⚡ Redis Session Memory + Response Caching")
        print("🧠 Agentic Chunking for Better Context")
        print("="*80)
        
        print(f"\n🔑 Session ID: {self.session_id}")
        print("📝 Commands: 'quit', 'exit', 'q' to exit | 'clear' to clear session")
        print("🎯 Ask me anything about retail operations, strategies, or trends!")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thank you for using the Enhanced RAG Chatbot!")
                    print(f"🔑 Session {self.session_id} ended")
                    break
                
                elif user_input.lower() == 'clear':
                    # Clear session memory
                    if self.redis_manager.redis_client:
                        pattern = f"*{self.session_id}*"
                        keys = self.redis_manager.redis_client.keys(pattern)
                        if keys:
                            self.redis_manager.redis_client.delete(*keys)
                            print("🧹 Session memory cleared")
                        else:
                            print("🧹 No session data to clear")
                    continue
                
                elif user_input.lower() == 'help':
                    print("\n📖 Available Commands:")
                    print("• Type any question about retail business")
                    print("• 'clear' - Clear session memory")
                    print("• 'stats' - Show session statistics")
                    print("• 'quit/exit/q' - Exit the chatbot")
                    continue
                
                elif user_input.lower() == 'stats':
                    # Show session statistics
                    if self.redis_manager.redis_client:
                        pattern = f"*{self.session_id}*"
                        keys = self.redis_manager.redis_client.keys(pattern)
                        cached_queries = len([k for k in keys if k.startswith(f"cache:{self.session_id}")])
                        print(f"📊 Session Stats: {cached_queries} cached responses")
                    continue
                
                if not user_input:
                    print("💬 Please enter a question about retail business.")
                    continue
                
                # Process the question
                response = self.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Session ended by user.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again or type 'quit' to exit.")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        stats = {
            'session_id': self.session_id,
            'cached_responses': 0,
            'total_documents': len(self.vector_store.document_corpus) if hasattr(self.vector_store, 'document_corpus') else 0,
            'embedding_model': self.embedding_model.model_name,
            'vector_store_type': 'Pinecone' if self.vector_store.use_pinecone else 'ChromaDB',
            'redis_connected': self.redis_manager.redis_client is not None
        }
        
        if self.redis_manager.redis_client:
            try:
                pattern = f"cache:{self.session_id}:*"
                cached_keys = self.redis_manager.redis_client.keys(pattern)
                stats['cached_responses'] = len(cached_keys)
            except:
                pass
        
        return stats

    def reset_session(self):
        """Reset current session and create new one"""
        old_session = self.session_id
        self.session_id = f"user_{int(time.time())}"
        
        # Clear old session data
        if self.redis_manager.redis_client:
            try:
                pattern = f"*{old_session}*"
                keys = self.redis_manager.redis_client.keys(pattern)
                if keys:
                    self.redis_manager.redis_client.delete(*keys)
            except:
                pass
        
        # Reinitialize QA chain with new session
        self.setup_qa_chain()
        
        return self.session_id

    def batch_process_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries in batch for testing/evaluation"""
        results = []
        
        for i, query in enumerate(queries):
            print(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            start_time = time.time()
            try:
                response = self.chat(query)
                processing_time = time.time() - start_time
                
                results.append({
                    'query': query,
                    'response': response,
                    'processing_time': processing_time,
                    'success': True,
                    'error': None
                })
            except Exception as e:
                processing_time = time.time() - start_time
                results.append({
                    'query': query,
                    'response': None,
                    'processing_time': processing_time,
                    'success': False,
                    'error': str(e)
                })
        
        return results

# Configuration and Environment Setup
def load_config():
    """Load configuration from environment variables or config file"""
    config = {
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
        'PINECONE_ENVIRONMENT': os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp-free'),
        'REDIS_HOST': os.getenv('REDIS_HOST', 'localhost'),
        'REDIS_PORT': int(os.getenv('REDIS_PORT', 6379)),
        'REDIS_DB': int(os.getenv('REDIS_DB', 0)),
        'USE_PINECONE': os.getenv('USE_PINECONE', 'true').lower() == 'true',
        'DATASET_PATH': os.getenv('DATASET_PATH', r'C:\Users\paran\OneDrive\Desktop\Quantabase\working\dataset'),
        'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL', 'intfloat/e5-base-v2'),
        'SESSION_TTL': int(os.getenv('SESSION_TTL', 3600))  # 1 hour
    }
    
    print(config)

    return config

def check_requirements():
    """Check if all required components are available"""
    requirements_check = {
        'gemini_api': False,
        'pinecone_api': False,
        'redis': False,
        'dataset': False,
        'embedding_model': False
    }
    
    config = load_config()
    
    # Check Gemini API
    if config['GEMINI_API_KEY']:
        try:
            genai.configure(api_key=config['GEMINI_API_KEY'])
            model = genai.GenerativeModel('gemini-1.5-flash')
            test_response = model.generate_content("Hello")
            if test_response.text:
                requirements_check['gemini_api'] = True
                print("✅ Gemini API connection successful")
        except Exception as e:
            print(f"⚠️ Gemini API test failed: {e}")
            print("🔄 Will use local LLM if available")
    else:
        print("⚠️ GEMINI_API_KEY not found - will use local LLM")
    
    # Check Pinecone API
    if config['USE_PINECONE'] and config['PINECONE_API_KEY']:
        try:
            pc = Pinecone(api_key=config['PINECONE_API_KEY'])
            pc.list_indexes()
            requirements_check['pinecone_api'] = True
            print("✅ Pinecone API connection successful")
        except Exception as e:
            print(f"❌ Pinecone API test failed: {e}")
    
    # Check Redis
    try:
        redis_client = redis.Redis(
            host=config['REDIS_HOST'],
            port=config['REDIS_PORT'],
            db=config['REDIS_DB'],
            socket_connect_timeout=5
        )
        redis_client.ping()
        requirements_check['redis'] = True
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")
        print("📝 Continuing without Redis (memory will be in-process only)")
    
    # Check dataset
    if os.path.exists(config['DATASET_PATH']):
        txt_files = []
        for root, dirs, files in os.walk(config['DATASET_PATH']):
            txt_files.extend([f for f in files if f.endswith('.txt')])
        
        if txt_files:
            requirements_check['dataset'] = True
            print(f"✅ Dataset found: {len(txt_files)} .txt files")
        else:
            print(f"❌ No .txt files found in {config['DATASET_PATH']}")
    else:
        print(f"❌ Dataset path not found: {config['DATASET_PATH']}")
    
    # Check embedding model
    try:
        test_model = SentenceTransformer(config['EMBEDDING_MODEL'])
        dim = test_model.get_sentence_embedding_dimension()
        requirements_check['embedding_model'] = True
        print(f"✅ Embedding model loaded: {config['EMBEDDING_MODEL']} (dim: {dim})")
        if not (700 <= dim <= 800):
            print(f"⚠️ Embedding dimension {dim} outside recommended range (700-800) - but will work fine")
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
    
    return requirements_check, config
    
def main():
    """Main function to run the Enhanced RAG Chatbot"""
    print("🚀 Starting Enhanced RAG Chatbot Setup...")
    print("=" * 80)
    
    # Check requirements
    requirements, config = check_requirements()
    
    # Check if we have at least one working LLM
    has_working_llm = requirements['gemini_api'] or CTRANSFORMERS_AVAILABLE
    
    if not has_working_llm:
        print("❌ No LLM available! Either fix Gemini API or install ctransformers:")
        print("   pip install ctransformers")
        return
    
    # Show which LLM will be used
    if requirements['gemini_api']:
        print("✅ Using Gemini Flash as primary LLM")
    elif CTRANSFORMERS_AVAILABLE:
        print("✅ Using Local LLM (ctransformers) - Gemini API unavailable")
    
    if not requirements['dataset']:
        print("❌ Dataset is required but not found. Please check your dataset path.")
        return
    
    if not requirements['embedding_model']:
        print("❌ Embedding model failed to load. Please check your internet connection.")
        return
    
    # Initialize chatbot (Redis is optional)
    try:
        print("\n🔧 Initializing Enhanced RAG Chatbot...")
        
        chatbot = EnhancedRAGChatbot(
            dataset_path=config['DATASET_PATH'],
            gemini_api_key=config['GEMINI_API_KEY'],
            use_pinecone=config['USE_PINECONE'],
            pinecone_api_key=config['PINECONE_API_KEY'],
            pinecone_environment=config['PINECONE_ENVIRONMENT'],
            redis_host=config['REDIS_HOST'],
            redis_port=config['REDIS_PORT']
        )
        
        print("\n📊 System Status:")
        stats = chatbot.get_session_stats()
        for key, value in stats.items():
            print(f"  • {key}: {value}")
        
        # Start interactive chat
        chatbot.start_interactive_chat()
        
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        print(f"❌ Initialization failed: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    # Add command line argument support
    if len(sys.argv) > 1:
        if sys.argv[1] == "check":
            check_requirements()
        else:
            print("Usage: python chatbot.py [test|check]")
    else:
        main()
