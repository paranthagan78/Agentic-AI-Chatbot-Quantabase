# Enhanced RAG Chatbot with Orca Mini 3B for Retail Sector using ctransformers
# Requirements: pip install langchain langchain-community chromadb sentence-transformers ctransformers

import os
import sys
import tempfile
import shutil
import time
import re
import math
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import json
import logging
from datetime import datetime
import pickle

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import warnings
from sentence_transformers import SentenceTransformer
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ChromaDB Embedding Function
class ChromaCompatibleEmbeddingFunction:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with embedding model"""
        print(f"🔧 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ Embedding model loaded successfully!")

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input, normalize_embeddings=True, show_progress_bar=False).tolist()
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text"""
        return self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents"""
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

@dataclass
class SearchResult:
    """Structure for hybrid search results"""
    content: str
    metadata: Dict
    semantic_score: float
    bm25_score: float
    hybrid_score: float
    confidence: float

class HybridRetriever:
    """Hybrid retriever combining semantic and BM25 search"""
    
    def __init__(self, vectorstore, semantic_weight=0.7, bm25_weight=0.3):
        self.vectorstore = vectorstore
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.bm25_vectorizer = None
        self.document_corpus = []
        self.document_metadata = []
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from vectorstore documents"""
        print("🔧 Building BM25 index for hybrid search...")
        
        try:
            # Get all documents from vectorstore
            collection = self.vectorstore._collection
            all_docs = collection.get()
            
            self.document_corpus = all_docs['documents']
            self.document_metadata = all_docs['metadatas']
            
            # Initialize BM25 vectorizer
            self.bm25_vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=5000,
                lowercase=True,
                analyzer='word'
            )
            
            # Fit BM25 on document corpus
            self.bm25_matrix = self.bm25_vectorizer.fit_transform(self.document_corpus)
            print(f"✅ BM25 index built with {len(self.document_corpus)} documents")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not build BM25 index: {e}")
            self.bm25_vectorizer = None
    
    def hybrid_search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Perform hybrid search combining semantic and BM25"""
        print(f"🔍 Performing hybrid search for: '{query[:50]}...'")
        
        # Semantic search
        semantic_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        print(f"🔍 Semantic search returned {len(semantic_results)} results")
        
        # BM25 search
        bm25_results = []
        if self.bm25_vectorizer is not None:
            try:
                query_vector = self.bm25_vectorizer.transform([query])
                bm25_scores = cosine_similarity(query_vector, self.bm25_matrix).flatten()
                
                # Get top BM25 results
                top_indices = bm25_scores.argsort()[-k*2:][::-1]
                for idx in top_indices:
                    if bm25_scores[idx] > 0:
                        bm25_results.append({
                            'content': self.document_corpus[idx],
                            'metadata': self.document_metadata[idx],
                            'score': bm25_scores[idx]
                        })
                        
                print(f"🔍 BM25 search returned {len(bm25_results)} results")
            except Exception as e:
                print(f"⚠️ BM25 search failed: {e}")
        
        # Combine and rank results
        hybrid_results = self._combine_results(semantic_results, bm25_results, query)
        print(f"✅ Hybrid search combined to {len(hybrid_results)} final results")
        
        return hybrid_results[:k]
    
    def _combine_results(self, semantic_results, bm25_results, query) -> List[SearchResult]:
        """Combine and rank semantic and BM25 results"""
        combined = {}
        
        # Normalize semantic scores
        if semantic_results:
            max_semantic_distance = max(score for _, score in semantic_results)
            min_semantic_distance = min(score for _, score in semantic_results)
            distance_range = max_semantic_distance - min_semantic_distance if max_semantic_distance != min_semantic_distance else 1
        
        # Process semantic results
        for doc, distance_score in semantic_results:
            content = doc.page_content
            key = hash(content)
            
            # Convert distance to similarity score (0-1, higher is better)
            if semantic_results:
                normalized_similarity = 1.0 - ((distance_score - min_semantic_distance) / distance_range)
            else:
                normalized_similarity = 0.5
            
            if key not in combined:
                combined[key] = SearchResult(
                    content=content,
                    metadata=doc.metadata,
                    semantic_score=normalized_similarity,
                    bm25_score=0.0,
                    hybrid_score=0.0,
                    confidence=0.0
                )
            combined[key].semantic_score = max(combined[key].semantic_score, normalized_similarity)
        
        # Normalize BM25 scores
        if bm25_results:
            max_bm25 = max(result['score'] for result in bm25_results)
            min_bm25 = min(result['score'] for result in bm25_results)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1
        
        # Process BM25 results
        for result in bm25_results:
            content = result['content']
            key = hash(content)
            
            # Normalize BM25 scores
            if bm25_results:
                normalized_bm25 = (result['score'] - min_bm25) / bm25_range
            else:
                normalized_bm25 = 0.0
            
            if key not in combined:
                combined[key] = SearchResult(
                    content=content,
                    metadata=result['metadata'],
                    semantic_score=0.0,
                    bm25_score=normalized_bm25,
                    hybrid_score=0.0,
                    confidence=0.0
                )
            else:
                combined[key].bm25_score = max(combined[key].bm25_score, normalized_bm25)
        
        # Calculate hybrid scores
        for result in combined.values():
            # Weighted combination
            result.hybrid_score = (
                self.semantic_weight * result.semantic_score +
                self.bm25_weight * result.bm25_score
            )
            
            # Confidence calculation
            result.confidence = min(
                (result.semantic_score * 0.6 + result.bm25_score * 0.4), 
                1.0
            )
            
            # Boost confidence if both methods found the result
            if result.semantic_score > 0 and result.bm25_score > 0:
                result.confidence = min(result.confidence * 1.2, 1.0)
        
        # Sort by hybrid score
        return sorted(combined.values(), key=lambda x: x.hybrid_score, reverse=True)
    
class EnhancedRAGChatbot:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.vectorstore = None
        self.qa_chain = None
        self.hybrid_retriever = None
        self.conversation_history = []
        
        # Check dependencies first
        if not self.check_dependencies():
            print("❌ Please install missing dependencies before continuing.")
            sys.exit(1)
        
        # Initialize components
        self.setup_embeddings()
        self.setup_llm()
        self.load_and_process_documents()
        self.setup_hybrid_retriever()
        self.setup_qa_chain()
    
    def setup_hybrid_retriever(self):
        """Setup hybrid retriever with semantic and BM25 search"""
        print("🔧 Setting up hybrid retriever...")
        self.hybrid_retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            semantic_weight=0.7,
            bm25_weight=0.3
        )
        print("✅ Hybrid retriever setup complete!")
    
    def setup_embeddings(self):
        """Initialize embeddings for retail domain"""
        print("🔧 Setting up embeddings...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'show_progress_bar': False,
                'batch_size': 64
            }
        )
        print("✅ Embeddings loaded successfully!")

    def check_dependencies(self):
        """Check if all required dependencies are available"""
        print("🔍 Checking dependencies...")
        
        missing = []
        
        try:
            import ctransformers
            print("✅ ctransformers: Available")
        except ImportError:
            missing.append("ctransformers")
            print("❌ ctransformers: Missing")
        
        try:
            import chromadb
            print("✅ chromadb: Available")
        except ImportError:
            missing.append("chromadb")
            print("❌ chromadb: Missing")
        
        try:
            import sentence_transformers
            print("✅ sentence-transformers: Available")
        except ImportError:
            missing.append("sentence-transformers")
            print("❌ sentence-transformers: Missing")
        
        if missing:
            print(f"\n⚠️ Missing dependencies: {', '.join(missing)}")
            print("💡 Install them with:")
            for dep in missing:
                print(f"   pip install {dep}")
            return False
        
        print("✅ All dependencies available!")
        return True
    
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
    
    def setup_llm(self):
        """Initialize Orca Mini 3B with optimized settings"""
        print("🤖 Setting up Orca Mini 3B LLM...")
        
        try:
            model_path = self.find_orca_model_path()
            
            self.llm = CTransformers(
                model=model_path,
                model_type="llama",
                config={
                    'max_new_tokens': 384,
                    'temperature': 0.05,
                    'context_length': 1536,
                    'top_p': 0.7,
                    'repetition_penalty': 1.02,
                    'threads': os.cpu_count() - 1,
                    'batch_size': 1,
                    'stream': False,
                    'top_k': 20,
                    'gpu_layers': 0,
                    'mmap': True,
                    'mlock': True,
                }
            )
            print("✅ Orca Mini 3B loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
    def load_and_process_documents(self):
        """Load and process documents"""
        print("📄 Loading retail sector documents...")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        # Load text files
        loader = DirectoryLoader(
            self.dataset_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8', 'autodetect_encoding': True},
            show_progress=True
        )
        
        documents = loader.load()
        print(f"📚 Loaded {len(documents)} retail documents")
        
        if len(documents) == 0:
            print("⚠️ No .txt files found. Checking for other file types...")
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    print(f"   📄 {os.path.join(root, file)}")
            raise ValueError("No documents found in the dataset path. Please add .txt files to the dataset folder.")
        
        # Apply chunking strategy
        texts = self.chunk_documents(documents)
        print(f"📝 Document chunking complete: {len(texts)} chunks")
        
        # Create ChromaDB vector store
        self.create_vectorstore(texts)
    
    def create_vectorstore(self, texts):
        """Create ChromaDB vector store"""
        print("🗃️ Creating ChromaDB vector store...")
        
        embedding_function = ChromaCompatibleEmbeddingFunction("sentence-transformers/all-MiniLM-L6-v2")
        text_contents = [doc.page_content for doc in texts]
        metadatas = [doc.metadata for doc in texts]
        
        import chromadb
        from chromadb.config import Settings
        
        client = chromadb.PersistentClient(
            path="./chroma_db_fast",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                persist_directory="./chroma_db_fast"
            )
        )
        
        collection_name = "retail_documents_fast"
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print("✅ Using existing ChromaDB collection")
        except:
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"description": "Retail documents"}
            )
            
            batch_size = 100
            total_batches = (len(text_contents) + batch_size - 1) // batch_size
            
            for i in range(0, len(text_contents), batch_size):
                end_idx = min(i + batch_size, len(text_contents))
                batch_texts = text_contents[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                
                for j, metadata in enumerate(batch_metadatas):
                    metadata['chunk_index'] = i + j
                
                batch_ids = [f"doc_{j}" for j in range(i, end_idx)]
                
                collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                batch_num = i // batch_size + 1
                print(f"   ✅ Batch {batch_num}/{total_batches} processed")
            
            print("✅ ChromaDB collection created!")
        
        # Create Langchain wrapper
        self.vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_function
        )
        
        print("✅ ChromaDB vector store ready!")

    def setup_qa_chain(self):
        """Setup QA chain with memory"""
        print("⛓️ Setting up QA chain...")
        
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=8
        )
        
        # Enhanced prompt template
        custom_prompt = PromptTemplate(
            template="""You are a knowledgeable and approachable retail expert. Use the information provided to deliver clear, helpful, comprehensive, and reliable answers that are relevant, accurate, and directly address the user's question.

Context: {context}
History: {chat_history}
Question: {question}

Please provide a detailed, professional response that:
1. Directly addresses the user's question with accurate and reliable information.
2. Uses specific and relevant details from the given context.
3. Offers actionable insights when appropriate.
4. Maintains a professional yet friendly tone.
5. Is well-structured, easy to read, and presented in a clean, properly formatted style.
6. The response should be optimally long — not too short, not too lengthy — but clear and engaging.
7. Ensures the answer is useful, understandable, and clearly connected to the user's intent.
8. If no accurate or relevant answer can be found in the context, respond with: "Sorry, I couldn't find a relevant answer based on the current information."
9. Ensure the response can be generated and read comfortably within 20 seconds.
10. Avoid speculation or assumptions not supported by the context.

Answer:""",
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        
        print("✅ QA chain setup complete!")
    
    def chat(self, question: str):
        """Chat method"""
        print(f"\n❓ User: {question}")
        
        try:
            result = self.qa_chain({
                "question": question,
                "chat_history": self.conversation_history[-6:]
            })
            
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            cleaned_answer = self.clean_response(answer)
            
            # Update conversation history
            self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content=cleaned_answer))
            
            # Keep history manageable
            if len(self.conversation_history) > 12:
                self.conversation_history = self.conversation_history[-12:]
            
            print(f"🤖 Bot: {cleaned_answer}")
            
            # Optional source display
            if source_docs and hasattr(self, 'debug_mode') and self.debug_mode:
                self.display_sources(source_docs)
            
            return cleaned_answer
            
        except Exception as e:
            error_response = "I apologize, but I encountered an issue. Please try rephrasing your question."
            return error_response

    def display_sources(self, source_docs):
        """Display source documents"""
        print(f"\n📚 Sources ({len(source_docs)} documents):")
        seen_sources = set()
        for i, doc in enumerate(source_docs[:3]):
            source = doc.metadata.get('source', 'Unknown')
            filename = os.path.basename(source) if source != 'Unknown' else f'Doc_{i+1}'
            if filename not in seen_sources:
                seen_sources.add(filename)
                print(f"   📄 {filename}")
    
    def clean_response(self, response: str) -> str:
        """Clean response text"""
        if not response:
            return "I don't have sufficient information to answer your question."
        
        response = response.strip()
        
        # Remove common prefixes
        prefixes = ["answer:", "response:", "bot:", "assistant:"]
        response_lower = response.lower()
        for prefix in prefixes:
            if response_lower.startswith(prefix):
                response = response[len(prefix):].strip()
                break
        
        # Limit length
        if len(response) > 1500:
            sentences = response.split('. ')
            final_response = ""
            for sentence in sentences:
                if len(final_response + sentence) <= 1500:
                    final_response += sentence + ". "
                else:
                    break
            response = final_response.strip()
        
        return response if response else "I don't have enough information to provide a comprehensive answer."
    
    def start_interactive_chat(self):
        """Interactive chat session"""
        print("\n" + "="*100)
        print("🚀 Enhanced Retail Sector RAG Chatbot")
        print("💼 Professional retail industry analysis")
        print("🔍 Hybrid Search: Semantic + BM25 keyword matching")
        print("⚡ Powered by Orca Mini 3B")
        print("📝 Type 'quit', 'exit', or 'q' to exit")
        print("="*100)
        
        print("\n🎛️ Special Commands:")
        print("   • 'insights' - View conversation analysis")
        print("   • 'debug on/off' - Toggle debugging information")
        
        self.debug_mode = False
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thank you for using the Enhanced Retail Sector Chatbot!")
                    insights = self.get_conversation_insights()
                    print(f"📊 Session Summary: {insights['total_exchanges']} exchanges completed")
                    if insights['common_topics']:
                        print(f"🏷️ Main topics discussed: {', '.join(insights['common_topics'])}")
                    break
                
                elif user_input.lower() == 'insights':
                    insights = self.get_conversation_insights()
                    print(f"\n📈 Conversation Insights:")
                    print(f"   Total exchanges: {insights['total_exchanges']}")
                    print(f"   Common topics: {', '.join(insights['common_topics'])}")
                    continue
                
                elif user_input.lower().startswith('debug'):
                    if 'on' in user_input.lower():
                        self.debug_mode = True
                        print("🔧 Debug mode enabled")
                    elif 'off' in user_input.lower():
                        self.debug_mode = False
                        print("🔧 Debug mode disabled")
                    continue
                
                if not user_input:
                    print("💬 Please enter a question about retail operations, strategies, or industry insights.")
                    continue
                
                answer = self.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Thank you for using the chatbot!")
                break
            except Exception as e:
                print(f"⚠️ An unexpected error occurred: {e}")
                if self.debug_mode:
                    import traceback
                    traceback.print_exc()
                print("Please try rephrasing your question or restart the chatbot.")
    
    def chunk_documents(self, texts):
        """Chunk documents for processing"""
        print("🧠 Applying document chunking...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            length_function=len,
            separators=["\n\n", "\n", ". ", " "],
            add_start_index=False
        )
        
        chunks = text_splitter.split_documents(texts)
        print(f"📝 Document chunking complete: {len(chunks)} chunks created")
        return chunks
    
    def get_conversation_insights(self):
        """Analyze conversation patterns"""
        if not self.conversation_history:
            return {
                'total_exchanges': 0,
                'common_topics': [],
                'session_duration': 'Current session'
            }
        
        insights = {
            'total_exchanges': len(self.conversation_history) // 2,
            'common_topics': [],
            'session_duration': 'Current session'
        }
        
        # Analyze query patterns
        human_messages = [msg.content for msg in self.conversation_history if isinstance(msg, HumanMessage)]
        
        # Find common topics
        all_words = ' '.join(human_messages).lower().split()
        word_freq = Counter(all_words)
        
        # Filter for retail-relevant terms
        retail_terms = ['retail', 'customer', 'sales', 'inventory', 'store', 'marketing', 'profit']
        common_topics = [word for word, count in word_freq.most_common(5) 
                        if word in retail_terms or len(word) > 4]
        
        insights['common_topics'] = common_topics[:3]
        
        return insights

def main():
    """Main function"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Update this path to your actual dataset location
    DATASET_PATH = r"C:\Users\paran\OneDrive\Desktop\Quantabase\june\dataset"
    
    print("🔧 Setting up environment...")
    
    chatbot = None
    try:
        print("🚀 Initializing Enhanced Retail Sector RAG Chatbot...")
        print("🧠 Loading Orca Mini 3B...")
        print("🔍 Setting up hybrid search...")
        print("⏳ This may take a few minutes on first initialization...")
        
        chatbot = EnhancedRAGChatbot(DATASET_PATH)
        
        print("\n🌟 Features Active:")
        print("   ✅ Hybrid Search (Semantic + BM25)")
        print("   ✅ Conversation Memory")
        print("   ✅ Document Chunking")
        
        chatbot.start_interactive_chat()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n💡 Install missing dependencies:")
        print("   pip install scikit-learn ctransformers chromadb sentence-transformers")
        print("   pip install langchain langchain-community")
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        print("\n💡 Troubleshooting steps:")
        print("   1. Ensure your dataset folder exists and contains .txt files")
        print("   2. Check the Orca Mini 3B model path")
        print("   3. Download the model: huggingface-cli download zoltanctoth/orca_mini_3B-GGUF")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if chatbot:
            print("🧹 Cleaning up resources...")
            try:
                insights = chatbot.get_conversation_insights()
                print(f"📈 Session completed with {insights['total_exchanges']} exchanges")
            except:
                pass

if __name__ == "__main__":
    main()
