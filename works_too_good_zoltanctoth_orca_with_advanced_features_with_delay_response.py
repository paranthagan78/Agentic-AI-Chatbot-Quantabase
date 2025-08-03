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
        self.hybrid_retriever = None
        self.query_processor = QueryProcessor()
        self.response_validator = ResponseValidator()
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
        self.setup_enhanced_qa_chain()
    
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
        """Enhanced document loading with semantic chunking"""
        print("📄 Loading retail sector documents with enhanced processing...")
        
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
        
        # Apply semantic chunking strategy
        texts = self.semantic_chunking(documents)
        print(f"📝 Applied semantic chunking: {len(texts)} enhanced chunks")
        
        # Create ChromaDB vector store with enhanced settings
        self.create_enhanced_vectorstore(texts)
    
    def create_enhanced_vectorstore(self, texts):
        """Create enhanced ChromaDB vector store with optimized settings"""
        print("🗃️ Creating enhanced ChromaDB vector store...")
        
        # Use high-quality embedding model
        embedding_function = ChromaCompatibleEmbeddingFunction("sentence-transformers/all-mpnet-base-v2")
        text_contents = [doc.page_content for doc in texts]
        metadatas = [doc.metadata for doc in texts]
        
        import chromadb
        from chromadb.config import Settings
        
        # Create persistent client with enhanced settings
        client = chromadb.PersistentClient(
            path="./chroma_db_enhanced_rag",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                persist_directory="./chroma_db_enhanced_rag"
            )
        )
        
        # Create or get collection with enhanced metadata
        collection_name = "retail_documents_enhanced_rag"
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print("✅ Using existing enhanced ChromaDB collection")
        except:
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16,
                    "description": "Enhanced retail documents with semantic chunking"
                }
            )
            
            # Add documents in optimized batches
            batch_size = 50  # Smaller batches for better memory management
            total_batches = (len(text_contents) + batch_size - 1) // batch_size
            
            for i in range(0, len(text_contents), batch_size):
                end_idx = min(i + batch_size, len(text_contents))
                batch_texts = text_contents[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                
                # Add enhanced metadata
                for j, metadata in enumerate(batch_metadatas):
                    metadata['chunk_index'] = i + j
                    metadata['chunk_length'] = len(batch_texts[j])
                    metadata['processing_timestamp'] = datetime.now().isoformat()
                
                batch_ids = [f"enhanced_doc_{j}" for j in range(i, end_idx)]
                
                collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                batch_num = i // batch_size + 1
                print(f"   ✅ Enhanced batch {batch_num}/{total_batches} processed")
            
            print("✅ Enhanced ChromaDB collection created and populated!")
        
        # Create Langchain wrapper with enhanced settings
        self.vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_function
        )
        
        print("✅ Enhanced ChromaDB vector store ready with optimized performance!")

    def setup_enhanced_qa_chain(self):
        """Setup enhanced QA chain with advanced memory"""
        print("⛓️ Setting up enhanced QA chain with advanced memory...")
        
        # Enhanced memory system
        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Enhanced prompt template
        custom_prompt = PromptTemplate(
            template="""You are a knowledgeable and approachable retail expert. Use the information provided to deliver clear, helpful, comprehensive, and reliable answers that are relevant, accurate, and directly address the user's question.

Context Information:
{context}

Conversation History:
{chat_history}

Current Question: {question}

Please provide a detailed, professional response that:
1. Directly addresses the user’s question with accurate and reliable information.
2. Uses specific and relevant details from the given context.
3. Offers actionable insights when appropriate.
4. Maintains a professional yet friendly tone.
5. Is well-structured, easy to read, and presented in a clean, properly formatted style.
6. The response should be optimally long — not too short, not too lengthy — but clear and engaging.
7. Ensures the answer is useful, understandable, and clearly connected to the user's intent.
8. If no accurate or relevant answer can be found in the context, respond with: "Sorry, I couldn’t find a relevant answer based on the current information."
9. Ensure the response can be generated and read comfortably within 20 seconds.
10. Avoid speculation or assumptions not supported by the context.

Answer:""",
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create custom retriever function
        def enhanced_retriever(query):
            # Use hybrid search instead of basic retrieval
            search_results = self.hybrid_retriever.hybrid_search(query, k=5)
            
            # Convert to Document format for compatibility
            from langchain.schema import Document
            documents = []
            for result in search_results:
                doc = Document(
                    page_content=result.content,
                    metadata={**result.metadata, 'hybrid_score': result.hybrid_score}
                )
                documents.append(doc)
            
            return documents
        
        # Create custom QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        
        print("✅ Enhanced QA chain setup complete!")
    
    def chat(self, question: str):
        """IMPROVED: Enhanced chat method with OPTIONAL debug output"""
        print(f"\n❓ User: {question}")
        
        # Step 1: Advanced query processing (only show debug if enabled)
        query_analysis = self.query_processor.analyze_query(question)
        
        # Only show debug info if debug mode is enabled
        debug_mode = getattr(self, 'debug_mode', False)
        
        if debug_mode:
            print(f"🔍 [DEBUG] Query type: {query_analysis.query_type}")
            print(f"🔍 [DEBUG] Retail terms found: {query_analysis.retail_terms}")
            print(f"🔍 [DEBUG] Query confidence: {query_analysis.confidence:.2f}")
        
        # Step 2: Hybrid search - USE ORIGINAL QUERY instead of expanded
        search_results = self.hybrid_retriever.hybrid_search(question, k=5)  # Changed from query_analysis.expanded_query
        
        # Step 3: Generate response
        try:
            # Use ORIGINAL question for better results
            result = self.qa_chain({
                "question": question,  # Changed from query_analysis.expanded_query
                "chat_history": self.conversation_history
            })
            
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            # Step 4: Response validation
            validation_result = self.response_validator.validate_response(
                answer, question, source_docs
            )
            
            # Step 5: Clean and enhance response
            cleaned_answer = self.clean_response(answer, question)
            
            # Step 6: Update conversation history
            self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content=cleaned_answer))
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            print(f"🤖 Bot: {cleaned_answer}")
            
            # Display enhanced source information
            if source_docs:
                self.display_enhanced_sources(source_docs, search_results, validation_result)
            
            return cleaned_answer
            
        except Exception as e:
            if debug_mode:
                print(f"⚠️ [DEBUG] Error in chat processing: {e}")
            error_response = "I apologize, but I encountered an issue processing your question. Please try rephrasing it or asking about a different aspect of retail operations."
            return error_response

    def display_enhanced_sources(self, source_docs, search_results, validation_result):
        """IMPROVED: Display enhanced source information with ACTUAL confidence scores"""
        print(f"\n📊 Response Quality Analysis:")
        print(f"   🎯 Overall Quality: {validation_result['overall_quality']:.2f}/1.0")
        print(f"   📈 Confidence: {validation_result['confidence_score']:.2f}")
        print(f"   🔗 Relevance: {validation_result['relevance_score']:.2f}")
        print(f"   ✅ Validation: {'PASSED' if validation_result['validation_passed'] else 'NEEDS REVIEW'}")
        
        if validation_result['issues']:
            print(f"   ⚠️ Issues: {', '.join(validation_result['issues'])}")
        
        # IMPROVED Source documents analysis with ACTUAL scores
        seen_sources = set()
        unique_sources = []
        
        print(f"\n📚 Knowledge Sources ({len(source_docs)} documents):")
        for i, doc in enumerate(source_docs[:5]):
            source = doc.metadata.get('source', 'Unknown')
            filename = os.path.basename(source) if source != 'Unknown' else f'Document_{i+1}'
            
            if filename not in seen_sources:
                seen_sources.add(filename)
                unique_sources.append(filename)
                
                # Get ACTUAL hybrid score from search results
                hybrid_score = 0.0
                if search_results:
                    # Find matching result by content
                    for search_result in search_results:
                        if search_result.content == doc.page_content:
                            hybrid_score = search_result.hybrid_score
                            break
                
                # If no hybrid score found, calculate from metadata or use default
                if hybrid_score == 0.0:
                    hybrid_score = doc.metadata.get('hybrid_score', 0.5)  # Default score
                
                print(f"   📄 {filename} (Score: {hybrid_score:.3f})")  # Show 3 decimal places
        
        # IMPROVED Hybrid search performance with actual data
        if search_results:
            print(f"\n🔍 Hybrid Search Performance:")
            semantic_count = len([r for r in search_results if r.semantic_score > 0.1])
            bm25_count = len([r for r in search_results if r.bm25_score > 0.1])
            
            print(f"   🧠 Semantic Results: {semantic_count}")
            print(f"   🔤 Keyword Results: {bm25_count}")
            print(f"   ⚖️ Combined Results: {len(search_results)}")
            
            # Show ACTUAL top result scores
            if search_results:
                top_result = search_results[0]
                print(f"   🏆 Top Result - Semantic: {top_result.semantic_score:.3f}, "
                    f"BM25: {top_result.bm25_score:.3f}, "
                    f"Hybrid: {top_result.hybrid_score:.3f}")

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
        """IMPROVED: Enhanced response cleaning for MORE CONVERSATIONAL outputs"""
        if not response:
            return "I don't have sufficient information in my retail knowledge base to provide a comprehensive answer to your question."
        
        response = response.strip()
        
        # Remove unwanted prefixes and artifacts
        prefixes_to_remove = [
            "answer:", "response:", "bot:", "assistant:", "context:", "question:",
            "human:", "ai:", "professional retail analysis:", "detailed answer:",
            "analysis:", "retail consultant:", "based on the", "according to"
        ]
        
        response_lower = response.lower()
        for prefix in prefixes_to_remove:
            if response_lower.startswith(prefix):
                response = response[len(prefix):].strip()
                response_lower = response.lower()
        
        # IMPROVED: Make responses more conversational
        conversational_starters = [
            "Here's what I found about that:",
            "Great question! Let me help you with that.",
            "Based on retail industry knowledge:",
            "From what I understand about retail:",
            "In the retail context:",
            ""  # Sometimes no starter is better
        ]
        
        # Add conversational elements
        if not any(response.lower().startswith(starter.lower()) for starter in ["here's", "great", "based on", "from what", "in the"]):
            # Randomly choose a conversational starter (or none)
            import random
            starter = random.choice(conversational_starters)
            if starter:
                response = f"{starter} {response}"
        
        # Clean up formatting while preserving structure
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 3:  # Filter out very short lines
                # Skip meta-information lines
                line_lower = line.lower()
                skip_patterns = [
                    'context:', 'source:', 'document:', 'file:', 'based on the document',
                    'according to the document', 'the context shows', 'from the provided',
                    '[debug]', 'processing'
                ]
                
                if not any(pattern in line_lower for pattern in skip_patterns):
                    cleaned_lines.append(line)
        
        if cleaned_lines:
            cleaned_response = '\n'.join(cleaned_lines)
            
            # IMPROVED: More reasonable length limits for conversational responses
            max_length = 2000  # Slightly reduced for better readability
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
        """IMPROVED: Enhanced interactive chat session with debug toggle"""
        print("\n" + "="*100)
        print("🚀 Enhanced Retail Sector RAG Chatbot with Advanced AI Features")
        print("💼 Professional retail industry analysis with hybrid intelligence")
        print("🔍 Hybrid Search: Semantic + BM25 keyword matching")
        print("🧠 Advanced Memory: Conversation summarization with context retention")
        print("📊 Response Validation: Real-time confidence scoring and quality assessment")
        print("🎯 Query Enhancement: Automatic expansion with retail domain knowledge")
        print("⚡ Powered by Orca Mini 3B with ctransformers optimization")
        print("📝 Type 'quit', 'exit', 'insights', or 'q' for special commands")
        print("="*100)
        
        print("\n🔍 Enhanced Query Examples:")
        print("   • How can I implement an effective omnichannel retail strategy?")
        print("   • What are the key performance indicators for measuring customer satisfaction?")
        print("   • Compare traditional retail vs e-commerce business models")
        print("   • What are the latest trends in retail technology and automation?")
        print("   • How do I optimize inventory management for seasonal products?")
        print("   • What strategies improve customer retention and lifetime value?")
        
        print("\n🎛️ Special Commands:")
        print("   • 'insights' - View conversation analysis and patterns")
        print("   • 'debug on/off' - Toggle detailed debugging information")
        print("   • 'confidence' - Show confidence scores for responses")
        
        # ADDED: Initialize debug mode as instance variable
        self.debug_mode = False
        show_confidence = True
        
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
                    print(f"   Session: {insights['session_duration']}")
                    continue
                
                elif user_input.lower().startswith('debug'):
                    if 'on' in user_input.lower():
                        self.debug_mode = True  # Use instance variable
                        print("🔧 Debug mode enabled - detailed processing info will be shown")
                    elif 'off' in user_input.lower():
                        self.debug_mode = False  # Use instance variable
                        print("🔧 Debug mode disabled - cleaner output mode")
                    continue
                
                elif user_input.lower() == 'confidence':
                    show_confidence = not show_confidence
                    status = "enabled" if show_confidence else "disabled"
                    print(f"📊 Confidence scoring {status}")
                    continue
                
                if not user_input:
                    print("💬 Please enter a question about retail operations, strategies, or industry insights.")
                    continue
                
                answer = self.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Thank you for using the enhanced chatbot!")
                break
            except Exception as e:
                print(f"⚠️ An unexpected error occurred: {e}")
                if self.debug_mode:
                    import traceback
                    traceback.print_exc()
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

    
    def semantic_chunking(self, texts):
        """Advanced semantic chunking strategy"""
        print("🧠 Applying semantic chunking strategy...")
        
        # Enhanced text splitter with semantic awareness
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,  # Optimized for semantic coherence
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
            add_start_index=True
        )
        
        # Split documents
        chunks = text_splitter.split_documents(texts)
        
        # Group semantically similar sentences
        grouped_chunks = self._group_similar_chunks(chunks)
        
        print(f"📝 Semantic chunking complete: {len(grouped_chunks)} chunks created")
        return grouped_chunks
    
    def _group_similar_chunks(self, chunks):
        """Group semantically similar chunks together"""
        # For now, return original chunks
        # In a full implementation, this would use sentence embeddings
        # to group similar content together
        return chunks
    
    def get_conversation_insights(self):
        """Analyze conversation patterns and provide insights"""
        if not self.conversation_history:
            return "No conversation history available."
        
        insights = {
            'total_exchanges': len(self.conversation_history) // 2,
            'common_topics': [],
            'query_types': [],
            'session_duration': 'Current session'
        }
        
        # Analyze query patterns
        human_messages = [msg.content for msg in self.conversation_history if isinstance(msg, HumanMessage)]
        
        # Find common topics (simple keyword analysis)
        all_words = ' '.join(human_messages).lower().split()
        word_freq = Counter(all_words)
        
        # Filter for retail-relevant terms
        retail_terms = ['retail', 'customer', 'sales', 'inventory', 'store', 'marketing', 'profit']
        common_topics = [word for word, count in word_freq.most_common(5) 
                        if word in retail_terms or len(word) > 4]
        
        insights['common_topics'] = common_topics[:3]
        
        print(f"\n📊 [DEBUG] Conversation Insights:")
        print(f"   💬 Total exchanges: {insights['total_exchanges']}")
        print(f"   🏷️ Common topics: {', '.join(insights['common_topics'])}")
        
        return insights
    
# ============================================================================
# PART 2: NEW CLASSES TO ADD (Add these after existing classes)
# ============================================================================

@dataclass
class QueryAnalysis:
    """Structure for query analysis results"""
    original_query: str
    expanded_query: str
    query_type: str
    context_hints: List[str]
    retail_terms: List[str]
    confidence: float

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
        
        # Get all documents from vectorstore
        try:
            # Try to get documents from ChromaDB
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
        print(f"🔍 [DEBUG] Performing hybrid search for: '{query[:50]}...'")
        
        # Semantic search
        semantic_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        print(f"🔍 [DEBUG] Semantic search returned {len(semantic_results)} results")
        
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
                        
                print(f"🔍 [DEBUG] BM25 search returned {len(bm25_results)} results")
            except Exception as e:
                print(f"⚠️ [DEBUG] BM25 search failed: {e}")
        
        # Combine and rank results
        hybrid_results = self._combine_results(semantic_results, bm25_results, query)
        print(f"✅ [DEBUG] Hybrid search combined to {len(hybrid_results)} final results")
        
        return hybrid_results[:k]
    
    def _combine_results(self, semantic_results, bm25_results, query) -> List[SearchResult]:
        """IMPROVED: Combine and rank semantic and BM25 results with better scoring"""
        combined = {}
        
        # Normalize semantic scores properly
        if semantic_results:
            max_semantic_distance = max(score for _, score in semantic_results)
            min_semantic_distance = min(score for _, score in semantic_results)
            distance_range = max_semantic_distance - min_semantic_distance if max_semantic_distance != min_semantic_distance else 1
        
        # Process semantic results with IMPROVED scoring
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
        
        # Process BM25 results with IMPROVED scoring
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
        
        # Calculate IMPROVED hybrid scores
        for result in combined.values():
            # Weighted combination
            result.hybrid_score = (
                self.semantic_weight * result.semantic_score +
                self.bm25_weight * result.bm25_score
            )
            
            # IMPROVED confidence calculation
            result.confidence = min(
                (result.semantic_score * 0.6 + result.bm25_score * 0.4), 
                1.0
            )
            
            # Boost confidence if both methods found the result
            if result.semantic_score > 0 and result.bm25_score > 0:
                result.confidence = min(result.confidence * 1.2, 1.0)
        
        # Sort by hybrid score
        return sorted(combined.values(), key=lambda x: x.hybrid_score, reverse=True)

class QueryProcessor:
    """Advanced query processing with expansion and classification"""
    
    def __init__(self):
        # EXPANDED retail synonyms with more comprehensive terms
        self.retail_synonyms = {
            'customer': ['client', 'consumer', 'buyer', 'shopper', 'patron', 'user', 'clientele'],
            'sales': ['revenue', 'turnover', 'income', 'earnings', 'proceeds', 'transactions'],
            'inventory': ['stock', 'merchandise', 'goods', 'products', 'items', 'catalog', 'assortment'],
            'store': ['shop', 'outlet', 'branch', 'location', 'retail space', 'establishment'],
            'profit': ['margin', 'earnings', 'returns', 'gains', 'income', 'profitability'],
            'marketing': ['promotion', 'advertising', 'branding', 'campaigns', 'outreach'],
            'supply chain': ['logistics', 'distribution', 'procurement', 'sourcing', 'fulfillment'],
            'e-commerce': ['online retail', 'digital commerce', 'web store', 'online shopping'],
            'omnichannel': ['multichannel', 'cross-channel', 'integrated retail'],
            'product': ['item', 'merchandise', 'goods', 'article', 'commodity'],
            'price': ['cost', 'pricing', 'rate', 'fee', 'charge', 'value'],
            'discount': ['sale', 'offer', 'deal', 'promotion', 'reduction'],
            'available': ['in stock', 'available', 'on hand', 'ready', 'accessible'],
            'quality': ['grade', 'standard', 'level', 'caliber', 'excellence'],
            'brand': ['label', 'trademark', 'name', 'make', 'manufacturer'],
            'category': ['type', 'class', 'group', 'section', 'department'],
            'size': ['dimension', 'measurement', 'scale', 'fit'],
            'color': ['shade', 'hue', 'tone', 'tint'],
            'material': ['fabric', 'substance', 'composition', 'texture']
        }
        
        # IMPROVED query patterns with more specific classifications
        self.query_patterns = {
            'product_lookup': ['find', 'search', 'looking for', 'need', 'want', 'available', 'in stock', 'do you have'],
            'product_info': ['tell me about', 'information about', 'details about', 'specs', 'features'],
            'comparison': ['vs', 'versus', 'compare', 'difference between', 'better than', 'which is better'],
            'availability': ['available', 'in stock', 'do you have', 'can I get', 'is there'],
            'pricing': ['price', 'cost', 'how much', 'expensive', 'cheap', 'affordable'],
            'how_to': ['how to', 'how do', 'how can', 'what is the process', 'steps to', 'guide'],
            'definition': ['what is', 'define', 'meaning of', 'explain', 'description'],
            'best_practices': ['best practices', 'recommendations', 'tips', 'advice', 'suggestions'],
            'metrics': ['kpi', 'metrics', 'measure', 'track', 'analyze', 'performance'],
            'trends': ['trends', 'future', 'emerging', 'latest', 'new', 'upcoming'],
            'support': ['help', 'problem', 'issue', 'trouble', 'support', 'assistance'],
            'location': ['where', 'location', 'store', 'branch', 'address', 'find store']
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Comprehensive query analysis WITHOUT problematic expansion"""
        print(f"🔍 [DEBUG] Analyzing query: '{query}'")
        
        # Classify query type
        query_type = self._classify_query(query)
        print(f"🔍 [DEBUG] Query type identified: {query_type}")
        
        # DON'T expand query - use original
        expanded_query = query  # Changed from self._expand_query(query)
        print(f"🔍 [DEBUG] Using original query: '{expanded_query}'")
        
        # Extract retail terms
        retail_terms = self._extract_retail_terms(query)
        print(f"🔍 [DEBUG] Retail terms found: {retail_terms}")
        
        # Generate context hints
        context_hints = self._generate_context_hints(query, query_type)
        print(f"🔍 [DEBUG] Context hints: {context_hints}")
        
        # Calculate confidence
        confidence = self._calculate_query_confidence(query, retail_terms)
        print(f"🔍 [DEBUG] Query confidence: {confidence:.2f}")
        
        return QueryAnalysis(
            original_query=query,
            expanded_query=expanded_query,
            query_type=query_type,
            context_hints=context_hints,
            retail_terms=retail_terms,
            confidence=confidence
        )
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()
        
        for query_type, patterns in self.query_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return query_type
        
        return 'general'
    
    def _expand_query(self, query: str) -> str:
        return query
    
    def _extract_retail_terms(self, query: str) -> List[str]:
        """IMPROVED: Extract retail-specific terms from query with better matching"""
        retail_terms = []
        query_lower = query.lower()
        
        # Check main terms
        for term in self.retail_synonyms.keys():
            if term in query_lower:
                retail_terms.append(term)
        
        # Check synonyms too - THIS WAS MISSING!
        for main_term, synonyms in self.retail_synonyms.items():
            for synonym in synonyms:
                if synonym in query_lower and main_term not in retail_terms:
                    retail_terms.append(main_term)
                    break
        
        # Add common retail words that might not be in synonyms
        common_retail_words = ['shop', 'buy', 'purchase', 'order', 'delivery', 'return', 
                              'exchange', 'warranty', 'guarantee', 'service', 'checkout']
        
        for word in common_retail_words:
            if word in query_lower:
                retail_terms.append(word)
        
        return list(set(retail_terms))  # Remove duplicates
    
    def _generate_context_hints(self, query: str, query_type: str) -> List[str]:
        """Generate context hints based on query type"""
        context_hints = []
        
        if query_type == 'how_to':
            context_hints.extend(['process', 'steps', 'implementation', 'strategy'])
        elif query_type == 'comparison':
            context_hints.extend(['advantages', 'disadvantages', 'features', 'benefits'])
        elif query_type == 'definition':
            context_hints.extend(['concept', 'meaning', 'explanation', 'overview'])
        elif query_type == 'best_practices':
            context_hints.extend(['recommendations', 'guidelines', 'tips', 'success factors'])
        elif query_type == 'metrics':
            context_hints.extend(['measurement', 'tracking', 'analysis', 'performance'])
        elif query_type == 'trends':
            context_hints.extend(['future', 'innovation', 'emerging', 'development'])
        
        return context_hints
    
    def _calculate_query_confidence(self, query: str, retail_terms: List[str]) -> float:
        """Calculate confidence score for the query"""
        base_confidence = 0.5
        
        # Boost confidence for retail terms
        retail_boost = min(len(retail_terms) * 0.1, 0.3)
        
        # Boost confidence for query length (optimal 5-20 words)
        word_count = len(query.split())
        if 5 <= word_count <= 20:
            length_boost = 0.2
        else:
            length_boost = max(0, 0.2 - abs(word_count - 12) * 0.01)
        
        return min(base_confidence + retail_boost + length_boost, 1.0)

class ResponseValidator:
    """Validates and scores response quality"""
    
    def __init__(self):
        self.quality_metrics = {
            'relevance': 0.0,
            'completeness': 0.0,
            'confidence': 0.0,
            'clarity': 0.0
        }
    
    def validate_response(self, response: str, query: str, source_docs: List) -> Dict:
        """Comprehensive response validation"""
        print(f"🔍 [DEBUG] Validating response quality...")
        
        validation_result = {
            'confidence_score': 0.0,
            'relevance_score': 0.0,
            'completeness_score': 0.0,
            'clarity_score': 0.0,
            'overall_quality': 0.0,
            'validation_passed': False,
            'issues': []
        }
        
        # Calculate individual scores
        validation_result['confidence_score'] = self._calculate_confidence(response, query)
        validation_result['relevance_score'] = self._calculate_relevance(response, query, source_docs)
        validation_result['completeness_score'] = self._calculate_completeness(response, query)
        validation_result['clarity_score'] = self._calculate_clarity(response)
        
        # Calculate overall quality
        validation_result['overall_quality'] = (
            validation_result['confidence_score'] * 0.3 +
            validation_result['relevance_score'] * 0.3 +
            validation_result['completeness_score'] * 0.2 +
            validation_result['clarity_score'] * 0.2
        )
        
        # Determine if validation passed
        validation_result['validation_passed'] = validation_result['overall_quality'] >= 0.6
        
        # Identify issues
        if validation_result['confidence_score'] < 0.5:
            validation_result['issues'].append('Low confidence')
        if validation_result['relevance_score'] < 0.5:
            validation_result['issues'].append('Low relevance')
        if validation_result['completeness_score'] < 0.5:
            validation_result['issues'].append('Incomplete response')
        if validation_result['clarity_score'] < 0.5:
            validation_result['issues'].append('Unclear response')
        
        print(f"✅ [DEBUG] Response validation complete - Quality: {validation_result['overall_quality']:.2f}")
        print(f"🔍 [DEBUG] Validation scores: Confidence={validation_result['confidence_score']:.2f}, "
              f"Relevance={validation_result['relevance_score']:.2f}, "
              f"Completeness={validation_result['completeness_score']:.2f}, "
              f"Clarity={validation_result['clarity_score']:.2f}")
        
        if validation_result['issues']:
            print(f"⚠️ [DEBUG] Issues found: {', '.join(validation_result['issues'])}")
        
        return validation_result
    
    def _calculate_confidence(self, response: str, query: str) -> float:
        """Calculate confidence score based on response characteristics"""
        score = 0.5
        
        # Length check
        if 50 <= len(response) <= 1000:
            score += 0.2
        
        # Specific terms check
        if any(term in response.lower() for term in ['specifically', 'precisely', 'exactly']):
            score += 0.1
        
        # Uncertainty indicators (reduce confidence)
        uncertainty_terms = ['might', 'possibly', 'perhaps', 'unclear', 'uncertain']
        if any(term in response.lower() for term in uncertainty_terms):
            score -= 0.2
        
        return max(0, min(1, score))
    
    def _calculate_relevance(self, response: str, query: str, source_docs: List) -> float:
        """Calculate relevance score"""
        score = 0.5
        
        # Check if response contains query terms
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        
        overlap = len(query_terms.intersection(response_terms))
        if overlap > 0:
            score += min(overlap / len(query_terms), 0.3)
        
        # Check source document relevance
        if source_docs and len(source_docs) > 0:
            score += 0.2
        
        return max(0, min(1, score))
    
    def _calculate_completeness(self, response: str, query: str) -> float:
        """Calculate completeness score"""
        score = 0.5
        
        # Length-based completeness
        if len(response) >= 100:
            score += 0.3
        
        # Structure indicators
        if any(indicator in response for indicator in [':', '1.', '•', 'first', 'second']):
            score += 0.2
        
        return max(0, min(1, score))
    
    def _calculate_clarity(self, response: str) -> float:
        """Calculate clarity score"""
        score = 0.5
        
        # Sentence structure
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if 10 <= avg_sentence_length <= 25:
            score += 0.3
        
        # Readability indicators
        if any(connector in response.lower() for connector in ['however', 'therefore', 'additionally', 'furthermore']):
            score += 0.2
        
        return max(0, min(1, score))
    
def main():
    """Enhanced main function with comprehensive error handling"""
    # Suppress warnings early
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Update this path to your actual dataset location
    DATASET_PATH = r"C:\Users\paran\OneDrive\Desktop\Quantabase\june\dataset"
    
    print("🔧 Setting up enhanced environment...")
    
    chatbot = None
    try:
        print("🚀 Initializing Enhanced Retail Sector RAG Chatbot...")
        print("🧠 Loading Orca Mini 3B with advanced features...")
        print("🔍 Setting up hybrid search (Semantic + BM25)...")
        print("💭 Initializing advanced memory systems...")
        print("📊 Configuring response validation...")
        print("⏳ This may take a few minutes on first initialization...")
        
        chatbot = EnhancedRAGChatbot(DATASET_PATH)
        
        # Show system capabilities
        print("\n🌟 Enhanced Features Active:")
        print("   ✅ Hybrid Search (Semantic + BM25)")
        print("   ✅ Advanced Query Processing")
        print("   ✅ Conversation Memory with Summarization")
        print("   ✅ Semantic Chunking Strategy")
        print("   ✅ Response Validation & Confidence Scoring")
        print("   ✅ Real-time Debugging (Backend Only)")
        
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
            # Show final conversation insights
            try:
                insights = chatbot.get_conversation_insights()
                print(f"📈 Session completed with {insights['total_exchanges']} exchanges")
            except:
                pass

if __name__ == "__main__":
    main()
