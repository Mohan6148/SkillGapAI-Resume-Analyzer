# chatbot.py

"""
Standalone AI Chatbot Module for Skill Gap Analyzer
Features:
- Vector database with FAISS
- RAG (Retrieval-Augmented Generation)
- OpenAI integration
- Semantic search capabilities
"""

import streamlit as st
import numpy as np
import json
import importlib
from typing import List, Dict, Any, Optional
from datetime import datetime

# Try to import required libraries
try:
    faiss = importlib.import_module("faiss")
    from sentence_transformers import SentenceTransformer
    openai = importlib.import_module("openai")
    DEPENDENCIES_AVAILABLE = True
    MISSING_DEPS = ""
except ImportError as e:
    faiss = None
    SentenceTransformer = None
    openai = None
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS = str(e)

class VectorDatabase:
    """FAISS-based vector database for semantic search"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the vector database
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.metadata = []
        self.dimension = None
        self.embeddings = None
        self.is_built = False
        
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load the sentence transformer model"""
        try:
            if not DEPENDENCIES_AVAILABLE:
                st.error(f"❌ Missing dependencies: {MISSING_DEPS}")
                return False
                
            self.embedding_model = SentenceTransformer(self.model_name)
            # Get the dimension of embeddings
            test_embedding = self.embedding_model.encode(["test"])
            self.dimension = test_embedding.shape[1]
            return True
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            return False
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None) -> bool:
        """
        Add documents to the vector database
        
        Args:
            documents: List of text documents
            metadata: List of metadata dictionaries for each document
            
        Returns:
            bool: Success status
        """
        if not self.embedding_model:
            st.error("❌ Embedding model not loaded")
            return False
        
        if metadata is None:
            metadata = [{"source": "unknown", "type": "text"}] * len(documents)
        
        try:
            # Generate embeddings for new documents
            new_embeddings = self.embedding_model.encode(documents)
            
            # Add to documents list
            self.documents.extend(documents)
            self.metadata.extend(metadata)
            
            new_embeddings = new_embeddings.astype('float32')

            # Create or update FAISS index when available.
            if faiss is not None:
                if self.index is None:
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self.index.add(new_embeddings)
                else:
                    self.index.add(new_embeddings)
            else:
                # Fallback to in-memory embeddings search if FAISS is unavailable.
                if self.embeddings is None:
                    self.embeddings = new_embeddings
                else:
                    self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            self.is_built = True
            return True
        except Exception as e:
            st.error(f"❌ Failed to add documents: {e}")
            return False
    
    def search(self, query: str, k: int = 5, threshold: float = 0.7) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of search results with documents, metadata, and similarity scores
        """
        if not self.is_built:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            query_embedding = query_embedding.astype('float32')

            if faiss is not None and self.index is not None:
                distances, indices = self.index.search(query_embedding, k)
            elif self.embeddings is not None:
                # Euclidean distance fallback search.
                all_distances = np.linalg.norm(self.embeddings - query_embedding[0], axis=1)
                top_indices = np.argsort(all_distances)[:k]
                distances = np.array([all_distances[top_indices]])
                indices = np.array([top_indices])
            else:
                return []
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    similarity = 1 - dist  # Convert distance to similarity
                    if similarity >= threshold:
                        results.append({
                            "document": self.documents[idx],
                            "metadata": self.metadata[idx],
                            "similarity": similarity,
                            "index": idx,
                            "distance": dist
                        })
            
            return results
        except Exception as e:
            st.error(f"❌ Search failed: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            "total_documents": len(self.documents),
            "is_built": self.is_built,
            "model_name": self.model_name,
            "dimension": self.dimension
        }

class SkillGapChatbot:
    """RAG-based chatbot for skill gap analysis"""
    
    def __init__(self):
        """Initialize the chatbot"""
        self.vector_db = VectorDatabase()
        self.openai_api_key = None
        self.conversation_history = []
        self.is_initialized = False
        
    def initialize_openai(self, api_key: str) -> bool:
        """
        Initialize OpenAI API
        
        Args:
            api_key: OpenAI API key
            
        Returns:
            bool: Success status
        """
        try:
            if not DEPENDENCIES_AVAILABLE:
                st.error("❌ OpenAI library not installed")
                return False
                
            openai.api_key = api_key
            self.openai_api_key = api_key
            self.is_initialized = True
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize OpenAI: {e}")
            return False
    
    def build_knowledge_base(self, session_state: Dict) -> bool:
        """
        Build knowledge base from session state
        
        Args:
            session_state: Streamlit session state dictionary
            
        Returns:
            bool: Success status
        """
        documents = []
        metadata = []
        
        # 1. Resume Information
        if session_state.get('cleaned_resume'):
            documents.append(f"Resume Content: {session_state['cleaned_resume']}")
            metadata.append({"source": "resume", "type": "content", "priority": "high"})
        
        # 2. Job Description
        if session_state.get('cleaned_jd'):
            documents.append(f"Job Description: {session_state['cleaned_jd']}")
            metadata.append({"source": "jd", "type": "content", "priority": "high"})
        
        # 3. Skills Information
        if session_state.get('resume_skills'):
            documents.append(f"Your Skills: {', '.join(session_state['resume_skills'])}")
            metadata.append({"source": "resume", "type": "skills", "priority": "high"})
        
        if session_state.get('jd_skills'):
            documents.append(f"Required Skills: {', '.join(session_state['jd_skills'])}")
            metadata.append({"source": "jd", "type": "skills", "priority": "high"})
        
        # 4. Gap Analysis Results
        if session_state.get('analysis_result'):
            result = session_state['analysis_result']
            
            # Extract skills based on result type
            if isinstance(result, dict):
                matched = result.get('matched_skills', [])
                missing = result.get('missing_skills', [])
                partial = result.get('partial_matches', [])
            else:
                matched = [m.jd_skill for m in result.matched_skills] if hasattr(result, 'matched_skills') else []
                missing = [m.jd_skill for m in result.missing_skills] if hasattr(result, 'missing_skills') else []
                partial = [m.jd_skill for m in result.partial_matches] if hasattr(result, 'partial_matches') else []
            
            # Add each skill as separate document for better retrieval
            for skill in matched:
                documents.append(f"✅ Matched Skill: {skill} - You have this skill and it matches the job requirements perfectly")
                metadata.append({"source": "analysis", "type": "matched_skill", "skill": skill, "priority": "medium"})
            
            for skill in missing:
                documents.append(f"❌ Missing Skill: {skill} - This skill is required but not found in your resume")
                metadata.append({"source": "analysis", "type": "missing_skill", "skill": skill, "priority": "high"})
            
            for skill in partial:
                documents.append(f"⚠️ Partial Match: {skill} - You have some experience with this skill but may need to improve")
                metadata.append({"source": "analysis", "type": "partial_skill", "skill": skill, "priority": "medium"})
        
        # 5. ATS Analysis
        if session_state.get('ats_analysis'):
            ats = session_state['ats_analysis']
            score = session_state.get('ats_score', 0) * 100
            
            documents.append(f"📊 Overall ATS Score: {score:.1f}% - This is how well your resume will be parsed by automated systems")
            metadata.append({"source": "ats", "type": "score", "priority": "high"})
            
            # Add each ATS factor
            factor_scores = ats.get('factor_scores', {})
            for factor, data in factor_scores.items():
                factor_name = factor.replace('_', ' ').title()
                documents.append(f"📈 ATS {factor_name}: {data['score']*100:.1f}% - Status: {data.get('category', 'unknown')}")
                metadata.append({"source": "ats", "type": "factor", "factor": factor, "priority": "medium"})
            
            # Add recommendations
            if ats.get('missing_keywords'):
                documents.append(f"🔑 Missing Keywords: {', '.join(ats['missing_keywords'])} - Add these to improve your ATS score")
                metadata.append({"source": "ats", "type": "missing_keywords", "priority": "high"})
            
            if ats.get('formatting_issues'):
                for issue in ats['formatting_issues']:
                    documents.append(f"⚠️ Formatting Issue: {issue} - This affects how ATS systems parse your resume")
                    metadata.append({"source": "ats", "type": "formatting_issue", "priority": "medium"})
        
        # 6. Learning Path
        if session_state.get('learning_path'):
            for item in session_state['learning_path']:
                documents.append(f"🎓 Learning Path for {item['skill']}: Priority - {item['priority']}, Estimated Time - {item['estimated_time']}")
                metadata.append({"source": "learning_path", "type": "skill_plan", "skill": item['skill'], "priority": "high"})
                
                # Add resources
                if 'resources' in item:
                    for resource in item['resources']:
                        documents.append(f"📚 Resource for {item['skill']}: {resource}")
                        metadata.append({"source": "learning_path", "type": "resource", "skill": item['skill'], "priority": "low"})
        
        # Build the vector database
        if documents:
            success = self.vector_db.add_documents(documents, metadata)
            if success:
                return True
        
        return False
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant context for the query
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with metadata
        """
        results = self.vector_db.search(query, k=k, threshold=0.6)
        
        # Sort by priority and similarity
        priority_order = {"high": 3, "medium": 2, "low": 1}
        results.sort(key=lambda x: (
            priority_order.get(x["metadata"].get("priority", "low"), 1), 
            x["similarity"]
        ), reverse=True)
        
        return results[:3]  # Return top 3 most relevant
    
    def generate_response(self, query: str, session_state: Dict) -> str:
        """
        Generate response using RAG with vector database
        
        Args:
            query: User query
            session_state: Streamlit session state
            
        Returns:
            Generated response
        """
        if not self.is_initialized:
            return "Please set your OpenAI API key to use the AI assistant."
        
        # Retrieve relevant context
        context_results = self.retrieve_context(query)
        
        if not context_results:
            return "I couldn't find relevant information in your analysis. Could you rephrase your question or make sure you've completed the analysis first?"
        
        # Build context string
        context_parts = []
        for result in context_results:
            context_parts.append(f"[{result['metadata']['source'].upper()}] {result['document']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate response
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert career advisor and AI assistant for the Skill Gap Analyzer application. 
                        
                        Your role is to help users understand their skill analysis results and provide actionable advice.
                        
                        Guidelines:
                        1. Always base your answers on the provided context
                        2. Explain technical concepts in simple, easy-to-understand language
                        3. Provide specific, actionable advice whenever possible
                        4. Be encouraging and supportive
                        5. If you don't know something, admit it honestly
                        6. Keep responses concise but informative (max 3-4 paragraphs)
                        7. Use emojis to make responses more engaging"""
                    },
                    {
                        "role": "user",
                        "content": f"""Context from user's analysis:
{context}

User Question: {query}

Please provide a helpful answer based on this context."""
                    }
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"


def _inject_chat_styles() -> None:
    """Inject custom CSS for a richer and cleaner chatbot sidebar UI."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Space+Grotesk:wght@600;700&display=swap');

        [data-testid="stSidebar"] {
            background:
                radial-gradient(120% 80% at 10% 5%, rgba(15, 157, 88, 0.16), transparent 55%),
                radial-gradient(100% 70% at 90% 10%, rgba(66, 133, 244, 0.15), transparent 45%),
                linear-gradient(180deg, #0f172a 0%, #101826 100%);
        }

        [data-testid="stSidebar"] * {
            font-family: 'Manrope', sans-serif;
        }

        .chat-title {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            letter-spacing: 0.4px;
            color: #e2f4e8;
            font-size: 1.05rem;
            margin-bottom: 0.35rem;
        }

        .chat-subtitle {
            color: #b7c9de;
            font-size: 0.84rem;
            margin-bottom: 0.85rem;
        }

        .chat-panel {
            border: 1px solid rgba(196, 214, 255, 0.16);
            background: linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.02));
            border-radius: 14px;
            padding: 0.65rem 0.75rem;
            margin: 0.45rem 0 0.75rem 0;
            box-shadow: 0 10px 20px rgba(0,0,0,0.18);
        }

        .chat-badge {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.2rem 0.5rem;
            border-radius: 999px;
            background: rgba(15, 157, 88, 0.18);
            color: #d4ffe4;
            margin-top: 0.35rem;
        }

        .quick-actions-title {
            font-size: 0.86rem;
            font-weight: 700;
            color: #d5e6ff;
            margin-bottom: 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Global chatbot instance
chatbot = SkillGapChatbot()

def render_chatbox():
    """Render the chatbox in the sidebar"""
    _inject_chat_styles()

    with st.sidebar:
        st.markdown("---")
        st.markdown('<div class="chat-title">🤖 AI Skill Assistant</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="chat-subtitle">Ask focused questions about ATS score, missing skills, and your learning plan.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
        
        # Check dependencies
        if not DEPENDENCIES_AVAILABLE:
            st.error("❌ Required libraries not installed")
            st.code("pip install faiss-cpu sentence-transformers openai")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # API Key Input
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password", 
            help="Enter your OpenAI API key to enable AI responses"
        )
        
        if api_key and not chatbot.is_initialized:
            with st.spinner("🔑 Initializing..."):
                if chatbot.initialize_openai(api_key):
                    st.success("✅ API key configured!")
                    # Try to build knowledge base if analysis is complete
                    if st.session_state.get('processing_complete'):
                        with st.spinner("🧠 Building knowledge base..."):
                            if chatbot.build_knowledge_base(st.session_state):
                                st.success("✅ Knowledge base ready!")
                else:
                    st.error("❌ Failed to initialize API key")

        if chatbot.is_initialized:
            st.markdown('<span class="chat-badge">Ready for AI responses</span>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        
        # Initialize chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "👋 Hello! I'm your AI skill assistant. I can help you understand your analysis results, explain skill gaps, and provide personalized advice. Ask me anything! 💬"}
            ]
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your skill analysis..."):
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                if chatbot.is_initialized:
                    # Build knowledge base if needed
                    if not chatbot.vector_db.is_built and st.session_state.get('processing_complete'):
                        with st.spinner("🧠 Building knowledge base..."):
                            chatbot.build_knowledge_base(st.session_state)
                    
                    # Generate response
                    with st.spinner("🤔 Thinking..."):
                        response = chatbot.generate_response(prompt, st.session_state)
                else:
                    response = "Please enter your OpenAI API key above to start chatting."
                
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        # Show database stats
        if chatbot.vector_db.is_built:
            stats = chatbot.vector_db.get_stats()
            st.info(f"📊 Knowledge Base: {stats['total_documents']} documents indexed")
        
        # Quick actions
        st.markdown("---")
        st.markdown('<div class="quick-actions-title">🚀 Quick Actions</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Explain ATS Score", help="Get explanation of your ATS score"):
                if chatbot.is_initialized:
                    st.session_state.chat_messages.append({
                        "role": "user", 
                        "content": "Can you explain my ATS score and how to improve it?"
                    })
                    st.rerun()
        
        with col2:
            if st.button("🎯 Top Skill Gaps", help="See your most critical skill gaps"):
                if chatbot.is_initialized:
                    st.session_state.chat_messages.append({
                        "role": "user", 
                        "content": "What are my most critical skill gaps that I should focus on first?"
                    })
                    st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", help="Clear chat history"):
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "👋 Chat cleared! How can I help you today?"}
            ]
            st.rerun()

def get_chatbot():
    """Get the global chatbot instance"""
    return chatbot

# Export functions for easy import
__all__ = ['render_chatbox', 'get_chatbot', 'SkillGapChatbot', 'VectorDatabase']