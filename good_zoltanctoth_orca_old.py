# Enhanced RAG Chatbot with Orca Mini 3B for Retail Sector using ctransformers
# Requirements: pip install langchain langchain-community chromadb sentence-transformers ctransformers

import os
import sys
import tempfile
import shutil
from typing import List
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

# ChromaDB Embedding Function with Better Embeddings
class ChromaCompatibleEmbeddingFunction:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        """Initialize with high-quality embeddings model"""
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
    
class EnhancedRAGChatbot:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.vectorstore = None
        self.qa_chain = None
        
        # Check dependencies first
        if not self.check_dependencies():
            print("❌ Please install missing dependencies before continuing.")
            sys.exit(1)
        
        # Initialize components
        self.setup_embeddings()
        self.setup_llm()
        self.load_and_process_documents()
        self.setup_qa_chain()
    
    def setup_embeddings(self):
        """Initialize high-quality embeddings for retail domain"""
        print("🔧 Setting up high-quality embeddings...")
        
        # Using the best available embedding model for better semantic understanding
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",  # High-quality embeddings
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'show_progress_bar': False,
                'batch_size': 32
            }
        )
        print("✅ High-quality embeddings loaded successfully!")

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
        """Initialize Orca Mini 3B with optimized settings for faster response"""
        print("🤖 Setting up optimized Orca Mini 3B LLM...")
        
        try:
            model_path = self.find_orca_model_path()
            
            # OPTIMIZED CONFIGURATION FOR SPEED
            self.llm = CTransformers(
                model=model_path,
                model_type="llama",
                config={
                    # SPEED OPTIMIZATIONS
                    'max_new_tokens': 512,      # Reduced from 1024 for faster response
                    'temperature': 0.1,         # Lower temperature for more focused responses
                    'context_length': 2048,     # Reduced from 4096 to speed up processing
                    'top_p': 0.8,              # Slightly reduced for faster sampling
                    'repetition_penalty': 1.05, # Reduced penalty
                    'threads': 8,              # Increase if you have more CPU cores
                    'batch_size': 1,           # Reduced for faster individual responses
                    'stream': False,
                    # ADDITIONAL SPEED SETTINGS
                    'top_k': 40,               # Limit vocabulary consideration
                    'gpu_layers': 0,           # Keep at 0 for CPU-only
                    'mmap': True,              # Enable memory mapping for efficiency
                    'mlock': False,            # Disable memory locking
                }
            )
            print("✅ Optimized Orca Mini 3B loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
    def load_and_process_documents(self):
        """Load and process documents from the dataset folder"""
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
            # List available files for debugging
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    print(f"   📄 {os.path.join(root, file)}")
            raise ValueError("No documents found in the dataset path. Please add .txt files to the dataset folder.")
        
        # Enhanced text splitting for better context preservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # Optimal chunk size for mpnet embeddings
            chunk_overlap=75,  # Good overlap for continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            add_start_index=True
        )
        
        texts = text_splitter.split_documents(documents)
        print(f"📝 Split into {len(texts)} text chunks")
        
        # Create ChromaDB vector store
        self.create_vectorstore(texts)
    
    def create_vectorstore(self, texts):
        """Create ChromaDB vector store with high-quality embeddings"""
        print("🗃️ Creating ChromaDB vector store with high-quality embeddings...")
        
        # Use the best embedding model
        embedding_function = ChromaCompatibleEmbeddingFunction("sentence-transformers/all-mpnet-base-v2")
        text_contents = [doc.page_content for doc in texts]
        metadatas = [doc.metadata for doc in texts]
        
        import chromadb
        from chromadb.config import Settings
        
        # Create persistent client with better settings
        client = chromadb.PersistentClient(
            path="./chroma_db_orca_enhanced",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        collection_name = "retail_documents_orca_enhanced"
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
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            # Add documents in batches for better performance
            batch_size = 100
            for i in range(0, len(text_contents), batch_size):
                end_idx = min(i + batch_size, len(text_contents))
                batch_texts = text_contents[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                batch_ids = [f"retail_doc_{j}" for j in range(i, end_idx)]
                
                collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                print(f"   ✅ Added batch {i//batch_size + 1}/{(len(text_contents) + batch_size - 1)//batch_size}")
            
            print("✅ ChromaDB collection created and populated!")
        
        # Create Langchain wrapper
        self.vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_function
        )
        
        print("✅ High-quality ChromaDB vector store ready!")
    
    def setup_qa_chain(self):
        """Setup optimized QA chain for faster retrieval"""
        print("⛓️ Setting up optimized QA chain...")
        
        # OPTIMIZED RETRIEVER - fewer documents for speed
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",    # Changed from MMR for speed
            search_kwargs={
                "k": 3,                  # Reduced from 8 for faster processing
            }
        )
        
        # REDUCED MEMORY FOR SPEED
        memory = ConversationBufferWindowMemory(
            k=5,                         # Reduced from 20 for speed
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # SIMPLIFIED PROMPT FOR FASTER PROCESSING
        custom_prompt = PromptTemplate(
            template="""You are a retail expert. Use the context to answer the question concisely and accurately.

    Context: {context}

    Question: {question}

    Answer:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            max_tokens_limit=1500        # Reduced for faster processing
        )
        
        print("✅ Optimized QA chain setup complete!")
    
    def chat(self, question: str):
        """Process user question and return detailed response"""
        print(f"\n❓ User: {question}")
        
        enhanced_question = self.enhance_question(question)
        
        try:
            result = self.qa_chain({"question": enhanced_question})
            
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            answer = self.clean_response(answer, question)
            
            print(f"🤖 Bot: {answer}")
            
            # Enhanced source information
            if source_docs:
                self.display_sources(source_docs)
            
            return answer
            
        except Exception as e:
            print(f"⚠️ Error processing question: {e}")
            error_response = "I apologize, but I encountered an issue processing your question. Please try rephrasing it or asking about a different aspect of retail operations."
            return error_response

    def display_sources(self, source_docs):
        """Display source information in an enhanced format"""
        seen_sources = set()
        unique_sources = []
        relevance_scores = []

        for i, doc in enumerate(source_docs):
            source = doc.metadata.get('source', 'Unknown')
            filename = os.path.basename(source) if source != 'Unknown' else f'Document_{i+1}'
            
            if filename not in seen_sources:
                seen_sources.add(filename)
                unique_sources.append(filename)
                # You could add relevance scoring here if available
                relevance_scores.append(f"Chunk {i+1}")

        print(f"\n📚 Knowledge sources used: {len(unique_sources)} document(s)")
        for i, (filename, chunk_info) in enumerate(zip(unique_sources[:6], relevance_scores[:6])):
            print(f"   {i+1}. {filename} ({chunk_info})")
        
        if len(unique_sources) > 6:
            print(f"   ... and {len(unique_sources) - 6} more sources")

    def enhance_question(self, question: str) -> str:
        """Add retail context to questions when appropriate"""
        retail_keywords = [
            'retail', 'store', 'customer', 'sales', 'inventory', 'merchandise', 
            'shopping', 'commerce', 'e-commerce', 'omnichannel', 'supply chain',
            'consumer', 'market', 'brand', 'product', 'pricing', 'promotion'
        ]
        
        if not any(keyword in question.lower() for keyword in retail_keywords):
            return f"In the context of retail business operations: {question}"
        
        return question
    
    def clean_response(self, response: str, question: str = "") -> str:
        """Enhanced response cleaning for professional outputs"""
        if not response:
            return "I don't have sufficient information in my retail knowledge base to provide a comprehensive answer to your question."
        
        response = response.strip()
        
        # Remove unwanted prefixes and artifacts
        prefixes_to_remove = [
            "answer:", "response:", "bot:", "assistant:", "context:", "question:",
            "human:", "ai:", "professional retail analysis:", "detailed answer:",
            "analysis:", "retail consultant:"
        ]
        
        response_lower = response.lower()
        for prefix in prefixes_to_remove:
            if response_lower.startswith(prefix):
                response = response[len(prefix):].strip()
                response_lower = response.lower()
        
        # Clean up formatting while preserving structure
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 3:  # Filter out very short lines
                # Skip meta-information lines
                line_lower = line.lower()
                skip_patterns = [
                    'context:', 'source:', 'document:', 'file:', 'based on the',
                    'according to the document', 'the context shows', 'from the provided'
                ]
                
                if not any(pattern in line_lower for pattern in skip_patterns):
                    cleaned_lines.append(line)
        
        if cleaned_lines:
            cleaned_response = '\n'.join(cleaned_lines)
            
            # Allow longer responses for comprehensive retail analysis
            max_length = 2500
            if len(cleaned_response) > max_length:
                # Smart truncation at sentence boundaries
                sentences = cleaned_response.split('. ')
                final_response = ""
                for sentence in sentences:
                    potential_response = final_response + sentence + ". "
                    if len(potential_response) <= max_length:
                        final_response = potential_response
                    else:
                        break
                
                if final_response and len(final_response) > 100:
                    return final_response.strip()
                else:
                    return cleaned_response[:max_length-3] + "..."
            
            return cleaned_response
        
        return "I don't have enough relevant information in my retail knowledge base to provide a comprehensive answer to your specific question."
    
    def start_interactive_chat(self):
        """Start interactive chat session with enhanced interface"""
        print("\n" + "="*90)
        print("🚀 Enhanced Retail Sector RAG Chatbot with Orca Mini 3B")
        print("💼 Professional retail industry analysis and insights")
        print("🧠 High-quality embeddings (all-mpnet-base-v2) for better understanding")
        print("🎯 Comprehensive responses with source attribution")
        print("💬 Extended conversation memory for complex discussions")
        print("⚡ Powered by ctransformers for efficient GGUF model loading")
        print("📝 Type 'quit', 'exit', or 'q' to end the session")
        print("="*90)
        
        print("\n🔍 Example questions you can ask:")
        print("   • What are the latest trends in retail technology adoption?")
        print("   • How can I improve customer retention in my retail store?")
        print("   • What are the best practices for inventory management?")
        print("   • How do I implement an effective omnichannel strategy?")
        print("   • What metrics should I track for retail performance?")
        print("   • How can I optimize my supply chain operations?")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thank you for using the Enhanced Retail Sector Chatbot!")
                    print("🔧 Remember to keep your knowledge base updated for best results!")
                    break
                
                if not user_input:
                    print("💬 Please enter a question about retail operations, strategies, or industry insights.")
                    continue
                
                self.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Thank you for using the chatbot!")
                break
            except Exception as e:
                print(f"⚠️ An unexpected error occurred: {e}")
                print("Please try rephrasing your question or restart the chatbot.")
    
    # 4. HARDWARE CHECK AND OPTIMIZATION
    def check_system_performance():
        """Check system specs and provide optimization recommendations"""
        import psutil
        import platform
        
        print("🔍 System Performance Check:")
        print(f"CPU Cores: {psutil.cpu_count()}")
        print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        print(f"Platform: {platform.system()} {platform.release()}")
        
        # Recommendations based on system
        cores = psutil.cpu_count()
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        if cores < 4:
            print("⚠️ Low CPU cores detected. Consider using threads=2")
        elif cores >= 8:
            print(f"✅ Good CPU cores. You can use threads={min(cores-2, 8)}")
        
        if ram_gb < 8:
            print("⚠️ Low RAM. Consider using a smaller model or reducing context_length")
        else:
            print("✅ Sufficient RAM for current model")

    # 5. QUICK RESPONSE TESTING
    def test_model_response_time(self):
        """Test model response time with a simple question"""
        import time
        
        test_question = "What is retail?"
        print(f"🧪 Testing response time with: '{test_question}'")
        
        start_time = time.time()
        
        try:
            # Simple LLM call without retrieval
            response = self.llm(test_question)
            end_time = time.time()
            
            response_time = end_time - start_time
            print(f"⏱️ Model response time: {response_time:.2f} seconds")
            print(f"📝 Response: {response[:100]}...")
            
            if response_time > 30:
                print("⚠️ Response time is slow. Consider:")
                print("   - Using a smaller model")
                print("   - Reducing max_new_tokens")
                print("   - Increasing CPU threads")
            else:
                print("✅ Response time is acceptable")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")

def main():
    """Main function with comprehensive error handling"""
    # Suppress warnings early
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Update this path to your actual dataset location
    DATASET_PATH = r"C:\Users\paran\OneDrive\Desktop\Quantabase\june\dataset"
    
    print("🔧 Setting up environment...")
    
    chatbot = None
    try:
        print("🚀 Initializing Enhanced Retail Sector RAG Chatbot...")
        print("🧠 Loading Orca Mini 3B with ctransformers...")
        print("📊 Setting up advanced vector store and retrieval system...")
        print("⏳ This may take a few minutes on first initialization...")
        
        chatbot = EnhancedRAGChatbot(DATASET_PATH)
        chatbot.start_interactive_chat()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n💡 Install missing dependencies:")
        print("   pip install ctransformers chromadb sentence-transformers")
        print("   pip install langchain langchain-community")
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        print("\n💡 Troubleshooting steps:")
        print("   1. Ensure your dataset folder exists and contains .txt files")
        print("   2. Check the Orca Mini 3B model path:")
        print("      Expected: C:\\Users\\paran\\.cache\\huggingface\\hub\\models--zoltanctoth--orca_mini_3B-GGUF")
        print("   3. Download the model: huggingface-cli download zoltanctoth/orca_mini_3B-GGUF")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n💡 Try running with debugging:")
        print("   python -c \"import traceback; traceback.print_exc()\"")
    finally:
        if chatbot:
            print("🧹 Cleaning up resources...")

if __name__ == "__main__":
    main()
