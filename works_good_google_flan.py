# Enhanced RAG Chatbot with Better Language Model for Retail Sector
# Requirements: pip install langchain langchain-community chromadb sentence-transformers transformers torch

import os
import tempfile
import shutil
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline  # FIXED: Changed to AutoModelForSeq2SeqLM
import torch
import warnings
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ChromaDB Embedding Function - UPDATED VERSION
class ChromaCompatibleEmbeddingFunction:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input, normalize_embeddings=True, show_progress_bar=False).tolist()
    
    # Required for Langchain compatibility
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
        self.temp_dir = None
        
        # Initialize components
        self.setup_embeddings()
        self.setup_llm()
        self.load_and_process_documents()
        self.setup_qa_chain()
    
    def setup_embeddings(self):
        """Initialize better embeddings for retail domain"""
        print("🔧 Setting up embeddings...")
        # Using a better embedding model for better semantic understanding
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",  # Better model
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Enhanced embeddings loaded successfully!")
    
    def setup_llm(self):
        """Initialize Google Flan-T5 for better question answering"""
        print("🤖 Setting up enhanced LLM...")
        
        # Using Google Flan-T5-base - much better for Q&A tasks
        model_name = "google/flan-t5-base"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # FIXED: Changed from AutoModelForCausalLM to AutoModelForSeq2SeqLM for T5 models
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Create pipeline with better parameters for Q&A
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.3,  # Lower temperature for more focused answers
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        print("✅ Enhanced LLM (Flan-T5) loaded successfully!")
    
    def load_and_process_documents(self):
        """Load and process documents from the dataset folder"""
        print("📄 Loading retail sector documents...")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        loader = DirectoryLoader(
            self.dataset_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8', 'autodetect_encoding': True}
        )
        
        documents = loader.load()
        print(f"📚 Loaded {len(documents)} retail documents")
        
        if len(documents) == 0:
            raise ValueError("No documents found in the dataset path. Please add .txt files to the dataset folder.")
        
        # Enhanced text splitting for better context preservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,  # Larger chunks for better context
            chunk_overlap=75,  # More overlap for continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        texts = text_splitter.split_documents(documents)
        print(f"📝 Split into {len(texts)} text chunks")
        
        # Create ChromaDB vector store
        self.create_vectorstore(texts)
    
    def create_vectorstore(self, texts):
        """Create ChromaDB vector store with better embeddings"""
        print("🗃️ Creating enhanced ChromaDB vector store...")
        
        # Create custom embedding function for ChromaDB compatibility
        embedding_function = ChromaCompatibleEmbeddingFunction("sentence-transformers/all-mpnet-base-v2")

        # Extract text content from Document objects
        text_contents = [doc.page_content for doc in texts]
        metadatas = [doc.metadata for doc in texts]
        
        # Create collection manually with proper interface
        import chromadb
        from chromadb.config import Settings
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path="./chroma_db_enhanced")
        
        # Create or get collection
        collection = client.get_or_create_collection(
            name="retail_documents",
            embedding_function=embedding_function
        )
        
        # Add documents to collection
        collection.add(
            documents=text_contents,
            metadatas=metadatas,
            ids=[f"retail_doc_{i}" for i in range(len(text_contents))]
        )
        
        # Create Langchain wrapper
        self.vectorstore = Chroma(
            client=client,
            collection_name="retail_documents",
            embedding_function=embedding_function
        )
        
        print("✅ Enhanced ChromaDB vector store created successfully!")
    
    def setup_qa_chain(self):
        """Setup enhanced ConversationalRetrievalChain with custom prompt"""
        print("⛓️ Setting up enhanced QA chain...")
        
        # Better retriever with more relevant documents
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={"k": 4, "fetch_k": 8}  # Get more diverse results
        )
        
        # Enhanced memory for better context
        memory = ConversationBufferWindowMemory(
            k=5,  # Remember more conversation history
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Custom prompt template for retail-specific Q&A
        custom_prompt = PromptTemplate(
            template="""You are a knowledgeable assistant specializing in the retail sector. Use the following context from retail documents to answer the question accurately and comprehensively.

Context from retail documents:
{context}

Previous conversation:
{chat_history}

Human Question: {question}

Instructions:
- Provide accurate, detailed answers based on the retail context provided
- Focus specifically on retail industry insights, trends, and practices
- If the context doesn't contain enough information, say so clearly
- Keep answers informative but concise
- Use specific examples from the context when available

Answer:""",
            input_variables=["context", "chat_history", "question"]
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        
        print("✅ Enhanced QA chain setup complete!")
    
    def chat(self, question: str):
        """Process user question and return enhanced response"""
        print(f"\n❓ User: {question}")
        
        # Add retail context to question if needed
        enhanced_question = self.enhance_question(question)
        
        try:
            result = self.qa_chain({"question": enhanced_question})
            
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            # Enhanced response cleaning
            answer = self.clean_response(answer, question)
            
            print(f"🤖 Bot: {answer}")
            
            if source_docs:
                # Track unique source filenames
                seen_sources = set()
                unique_sources = []

                for doc in source_docs:
                    source = doc.metadata.get('source', 'Unknown')
                    filename = os.path.basename(source) if source != 'Unknown' else 'Memory'
                    if filename not in seen_sources:
                        seen_sources.add(filename)
                        unique_sources.append(filename)

                print(f"\n📚 Sources used: {len(unique_sources)} retail document(s)")
                for i, filename in enumerate(unique_sources[:3]):
                    print(f"   {i+1}. {filename}")
            
            return answer
            
        except Exception as e:
            print(f"⚠️ Error processing question: {e}")
            return "I apologize, but I encountered an issue processing your question. Please try rephrasing it."

    def enhance_question(self, question: str) -> str:
        """Add retail context to questions when appropriate"""
        retail_keywords = ['retail', 'store', 'customer', 'sales', 'inventory', 'merchandise', 'shopping']
        
        # If question doesn't contain retail context, add it
        if not any(keyword in question.lower() for keyword in retail_keywords):
            return f"In the context of retail sector: {question}"
        
        return question
    
    def clean_response(self, response: str, question: str = "") -> str:
        """Enhanced response cleaning for better readability"""
        if not response:
            return "I don't have enough information in my retail knowledge base to answer that question accurately."
        
        response = response.strip()
        
        # Remove unwanted prefixes and artifacts
        prefixes_to_remove = [
            "answer:", "response:", "bot:", "assistant:", "context:", "question:",
            "human:", "ai:", "based on the context", "according to the documents"
        ]
        
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Remove repetitive phrases
        repetitive_phrases = [
            "based on the retail documents",
            "according to the context provided",
            "from the retail sector information"
        ]
        
        for phrase in repetitive_phrases:
            response = response.replace(phrase, "").strip()
        
        # Clean up formatting
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 3:
                # Remove lines that are just metadata
                if not any(meta in line.lower() for meta in ['context:', 'source:', 'document:', 'file:']):
                    cleaned_lines.append(line)
        
        if cleaned_lines:
            cleaned_response = ' '.join(cleaned_lines)
            
            # Ensure response isn't too long
            if len(cleaned_response) > 500:
                # Find last complete sentence within limit
                sentences = cleaned_response.split('. ')
                final_response = ""
                for sentence in sentences:
                    if len(final_response + sentence + '. ') <= 500:
                        final_response += sentence + ". "
                    else:
                        break
                
                if final_response:
                    return final_response.strip()
                else:
                    return cleaned_response[:497] + "..."
            
            return cleaned_response
        
        return "I don't have enough information in my retail knowledge base to provide a comprehensive answer to your question."
    
    def start_interactive_chat(self):
        """Start enhanced interactive chat session"""
        print("\n" + "="*70)
        print("🚀 Enhanced Retail Sector RAG Chatbot is ready!")
        print("💼 Specialized in retail industry questions and insights")
        print("💡 Ask questions about retail trends, strategies, operations, etc.")
        print("📝 Type 'quit' to exit")
        print("="*70)
        
        # Welcome message with sample questions
        print("\n🔍 Sample questions you can ask:")
        print("   • What are the latest retail trends?")
        print("   • How can I improve customer experience?")
        print("   • What are effective inventory management strategies?")
        print("   • How does omnichannel retailing work?")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("👋 Thank you for using the Retail Sector Chatbot!")
                    break
                
                if not user_input:
                    print("💬 Please enter a question about the retail sector.")
                    continue
                
                self.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Thank you for using the Retail Sector Chatbot!")
                break
            except Exception as e:
                print(f"⚠️ An error occurred: {e}")
                print("Please try asking your question differently.")
    
    def cleanup(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                print(f"⚠️ Could not clean up {self.temp_dir}: {e}")

def main():
    """Main function with enhanced error handling"""
    DATASET_PATH = r"C:\Users\paran\OneDrive\Desktop\Quantabase\june\dataset"
    
    chatbot = None
    try:
        print("🚀 Initializing Enhanced Retail Sector RAG Chatbot...")
        print("📦 Loading advanced models and setting up vector store...")
        print("⏳ This may take a few minutes on first run (downloading models)...")
        print("🎯 Optimized for retail sector question-answering...")
        
        chatbot = EnhancedRAGChatbot(DATASET_PATH)
        chatbot.start_interactive_chat()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except FileNotFoundError as e:
        print(f"❌ Dataset folder not found: {e}")
        print("💡 Please ensure your dataset folder exists and contains .txt files")
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   • Make sure your dataset folder contains .txt files")
        print("   • Check if you have sufficient RAM (at least 4GB free)")
        print("   • Ensure all required packages are installed")
    finally:
        # Cleanup
        if chatbot:
            chatbot.cleanup()

if __name__ == "__main__":
    main()
