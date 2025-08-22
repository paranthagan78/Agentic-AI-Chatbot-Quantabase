# Suppress warnings and TensorFlow logs
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

import sys
import glob
from dataclasses import dataclass
from typing import Dict, List
from collections import Counter
import numpy as np

# Core imports with optimizations
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import CTransformers
import chromadb
from chromadb.config import Settings

# Langchain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage

@dataclass
class SearchResult:
    """Lightweight structure for search results"""
    content: str
    metadata: Dict
    score: float

class OptimizedEmbeddingFunction:
    """Optimized embedding function with caching"""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"🔧 Loading optimized embedding model...")
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 256  # Reduce for faster processing
        self._cache = {}
        print("✅ Embedding model loaded!")

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input, normalize_embeddings=True, show_progress_bar=False, batch_size=32).tolist()
    
    def embed_query(self, text: str) -> list[float]:
        # Simple caching for repeated queries
        if text in self._cache:
            return self._cache[text]
        
        embedding = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
        self._cache[text] = embedding
        return embedding

class FastRetriever:
    """Optimized retriever focusing on semantic search only for speed"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        print("✅ Fast retriever initialized")
    
    def search(self, query: str, k: int = 3) -> List[SearchResult]:
        """Fast semantic search with minimal processing"""
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            search_results = []
            for doc, score in results:
                search_results.append(SearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=1.0 - score  # Convert distance to similarity
                ))
            
            return search_results
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            return []

class OptimizedRAGChatbot:
    """Ultra-fast RAG Chatbot optimized for low latency"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.vectorstore = None
        self.qa_chain = None
        self.retriever = None
        self.conversation_history = []
        
        print("🚀 Initializing Optimized RAG Chatbot...")
        
        # Streamlined initialization
        self.setup_embeddings()
        self.setup_llm()
        self.load_and_process_documents()
        self.setup_retriever()
        self.setup_qa_chain()
        
        print("✅ Fast RAG Chatbot ready!")
    
    def setup_embeddings(self):
        """Setup lightweight embeddings"""
        print("🔧 Setting up fast embeddings...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'show_progress_bar': False,
                'batch_size': 32
            }
        )
        print("✅ Fast embeddings loaded!")
    
    def setup_llm(self):
        """Setup LLM with optimized parameters for speed"""
        print("🤖 Setting up fast LLM...")
        
        try:
            # Find model path
            base_path = r"C:\Users\paran\.cache\huggingface\hub\models--zoltanctoth--orca_mini_3B-GGUF"
            gguf_files = glob.glob(os.path.join(base_path, "**/*.gguf"), recursive=True)
            
            if not gguf_files:
                raise FileNotFoundError("No GGUF model found")
            
            model_path = gguf_files[0]  
            
            self.llm = CTransformers(
                model=model_path,
                model_type="llama",
                config={
                    'max_new_tokens': 150,  # Reduced for faster responses
                    'temperature': 0.1,
                    'context_length': 1024,  # Reduced context
                    'top_p': 0.9,
                    'repetition_penalty': 1.05,  # Reduced
                    'threads': max(1, os.cpu_count() - 1),
                    'batch_size': 1,
                    'stream': True,
                    'top_k': 20,  # Reduced for speed
                    'gpu_layers': 0,
                }
            )
            print("✅ Fast LLM loaded!")
            
        except Exception as e:
            print(f"❌ LLM setup error: {e}")
            raise
    
    def load_and_process_documents(self):
        """Optimized document processing"""
        print("📄 Loading documents...")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        # Load documents
        loader = DirectoryLoader(
            self.dataset_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8', 'autodetect_encoding': True}
        )
        
        documents = loader.load()
        print(f"📚 Loaded {len(documents)} documents")
        
        if not documents:
            raise ValueError("No documents found")
        
        # Fast chunking
        texts = self.chunk_documents(documents)
        print(f"📝 Created {len(texts)} chunks")
        
        # Create vector store
        self.create_vectorstore(texts)
    
    def chunk_documents(self, documents):
        """Optimized chunking for speed"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Smaller chunks for faster processing
            chunk_overlap=30,
            length_function=len,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        return text_splitter.split_documents(documents)
    
    def create_vectorstore(self, texts):
        """Create optimized vector store"""
        print("🗃️ Creating vector store...")
        
        embedding_function = OptimizedEmbeddingFunction()
        
        client = chromadb.PersistentClient(
            path="./chroma_faster_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        collection_name = "fast_retail_docs"
        
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print("✅ Using existing collection")
        except:
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            
            # Batch processing for speed
            batch_size = 100
            text_contents = [doc.page_content for doc in texts]
            metadatas = [doc.metadata for doc in texts]
            
            for i in range(0, len(text_contents), batch_size):
                end_idx = min(i + batch_size, len(text_contents))
                batch_texts = text_contents[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                batch_ids = [f"doc_{j}" for j in range(i, end_idx)]
                
                collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
            print("✅ Vector store created!")
        
        self.vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_function
        )
    
    def setup_retriever(self):
        """Setup fast retriever"""
        print("🔧 Setting up fast retriever...")
        self.retriever = FastRetriever(self.vectorstore)
    
    def setup_qa_chain(self):
        """Setup optimized QA chain"""
        print("⛓️ Setting up fast QA chain...")
        
        # Minimal memory for speed
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=2  # Keep only last 2 exchanges
        )
        
        # Optimized prompt for concise, accurate responses
        fast_prompt = PromptTemplate(
            template="""Based on the context provided, give a direct, accurate answer to the user's question. Be concise and specific.

Context: {context}

Question: {question}

Instructions:
- Answer only what is asked
- Use information from context when available
- Be precise and factual
- Keep response under 100 words
- No extra information unless directly relevant

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Fast retrieval chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}  # Fewer documents for speed
            ),
            memory=memory,
            return_source_documents=False,  # Skip for speed
            verbose=False,
            combine_docs_chain_kwargs={"prompt": fast_prompt}
        )
        
        print("✅ Fast QA chain ready!")
    
    def chat(self, question: str) -> str:
        """Optimized chat method for fast responses"""
        if not question.strip():
            return "Please ask a specific question."
        
        try:
            # Get fast response
            result = self.qa_chain.invoke({
                "question": question,
                "chat_history": self.conversation_history[-2:]  # Minimal history
            })
            
            answer = result.get("answer", "").strip()
            
            # Clean response quickly
            answer = self.quick_clean(answer)
            
            # Update minimal history
            self.conversation_history.extend([
                HumanMessage(content=question),
                AIMessage(content=answer)
            ])
            
            # Keep history small
            if len(self.conversation_history) > 8:
                self.conversation_history = self.conversation_history[-8:]
            
            return answer
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
            return "I apologize, but I couldn't process your request. Please try again."
    
    def quick_clean(self, response: str) -> str:
        """Fast response cleaning"""
        if not response:
            return "I don't have enough information to answer that question."
        
        # Remove common prefixes quickly
        prefixes = ["answer:", "response:", "assistant:"]
        response_lower = response.lower()
        
        for prefix in prefixes:
            if response_lower.startswith(prefix):
                response = response[len(prefix):].strip()
                break
        
        # Quick length check
        if len(response) > 500:
            sentences = response.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + '. ') <= 500:
                    truncated += sentence + '. '
                else:
                    break
            response = truncated.strip()
        
        return response if response else "Please provide more specific details for an accurate answer."
    
    def start_interactive_chat(self):
        """Start fast interactive chat"""
        print("\n" + "="*60)
        print("🚀 OPTIMIZED Fast Retail RAG Chatbot")
        print("⚡ Ultra-low latency responses")
        print("🎯 Precise, contextual answers")
        print("💬 Type 'quit' to exit")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for using Fast RAG Chatbot!")
                    break
                
                if not user_input:
                    print("💬 Please ask about retail services, delivery, or policies.")
                    continue
                
                # Fast response
                print(f"🤖 Assistant: {self.chat(user_input)}")
                
            except KeyboardInterrupt:
                print("\n👋 Chat ended. Goodbye!")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")

# Usage example
if __name__ == "__main__":
    try:
        # Initialize with your dataset path
        dataset_path = r"C:\Users\paran\OneDrive\Desktop\Quantabase\working\dataset"  # Update this path
        chatbot = OptimizedRAGChatbot(dataset_path)
        chatbot.start_interactive_chat()
    except Exception as e:
        print(f"❌ Startup error: {e}")
        print("Please ensure your dataset path is correct and contains .txt files.")
