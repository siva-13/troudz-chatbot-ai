"""
Main RAG Application - FIXED VERSION
Integrates all components with all required methods
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import json

# Import our custom modules
from document_processor import DocumentProcessor
from vector_store import VectorStore  
from dynamodb_manager import DynamoDBManager
from conversation_engine import ConversationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGApplication:
    def __init__(self, config: Dict[str, Any]):
        """Initialize RAG Application with all components"""
        self.config = config

        # Initialize components
        self.document_processor = DocumentProcessor(
            openai_api_key=config['openai_api_key'],
            embedding_model=config.get('embedding_model', 'text-embedding-3-small')
        )

        self.vector_store = VectorStore(
            dimension=config.get('embedding_dimension', 1536),
            index_type=config.get('index_type', 'cosine')
        )

        self.db_manager = DynamoDBManager(
            region_name=config.get('aws_region', 'us-east-1'),
            profile_name=config.get('aws_profile')
        )

        self.conversation_engine = ConversationEngine(
            openai_api_key=config['openai_api_key'],
            model=config.get('chat_model', 'gpt-3.5-turbo')
        )

        self.is_initialized = False
        self.vector_store_path = config.get('vector_store_path', './data/vector_index')

        logger.info("RAG Application initialized")

    async def initialize(self):
        """Initialize all components and load existing data"""
        try:
            try:
                self.db_manager.connect_to_tables()
            except:
                logger.info("Tables don't exist, creating new ones...")
                self.db_manager.create_tables()

            if os.path.exists(f"{self.vector_store_path}.faiss"):
                logger.info("Loading existing vector store...")
                self.vector_store.load_index(self.vector_store_path)
            else:
                logger.info("No existing vector store found, starting fresh")

            self.is_initialized = True
            logger.info("RAG Application fully initialized")

        except Exception as e:
            logger.error(f"Error initializing application: {e}")
            raise

    def create_user(self, name: str, phone_number: str, email: str, 
                   additional_info: str = "") -> Dict[str, Any]:
        """Create a new user in the system"""
        if not self.is_initialized:
            raise Exception("Application not initialized. Call initialize() first.")

        # Generate user ID
        user_id = f"user_{datetime.now().strftime('%Y%m%d')}_{hash(email) % 10000:04d}"

        return self.db_manager.create_user(
            user_id=user_id,
            name=name,
            phone_number=phone_number,
            email=email,
            additional_info=additional_info
        )

    def start_conversation(self, user_id: str) -> Dict[str, Any]:
        """Start a new conversation session for a user"""
        if not self.is_initialized:
            raise Exception("Application not initialized. Call initialize() first.")

        # Verify user exists
        user = self.db_manager.get_user(user_id)
        if not user:
            raise Exception(f"User {user_id} not found")

        # Create new session
        session_id = self.conversation_engine.create_session(user_id)

        return {
            'session_id': session_id,
            'user_id': user_id,
            'user_name': user.get('Name', 'Unknown'),
            'created_at': datetime.now().isoformat(),
            'turns_remaining': self.conversation_engine.max_turns_per_session
        }

    def ingest_document(self, file_path: str, document_id: str = None) -> Dict[str, Any]:
        """Ingest a Word document into the RAG system"""
        if not self.is_initialized:
            raise Exception("Application not initialized. Call initialize() first.")

        try:
            logger.info(f"Starting document ingestion: {file_path}")

            processed_doc = self.document_processor.process_document(file_path, document_id)

            embeddings = [chunk['embedding'] for chunk in processed_doc['chunks']]
            chunk_data = processed_doc['chunks']

            self.vector_store.add_vectors(embeddings, chunk_data)
            self.vector_store.save_index(self.vector_store_path)

            result = {
                'document_id': processed_doc['document_id'],
                'filename': processed_doc['filename'],
                'chunks_created': len(processed_doc['chunks']),
                'total_tokens': processed_doc['total_tokens'],
                'status': 'success',
                'ingested_at': datetime.now().isoformat()
            }

            logger.info(f"Successfully ingested document: {processed_doc['document_id']}")
            return result

        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'ingested_at': datetime.now().isoformat()
            }

    def process_message(self, user_message: str, session_id: str, user_id: str,
                       similarity_threshold: float = 0.7, max_chunks: int = 3) -> Dict[str, Any]:
        """Process a user message with full RAG pipeline"""
        if not self.is_initialized:
            raise Exception("Application not initialized. Call initialize() first.")

        try:
            logger.info(f"Processing message in session {session_id}")

            # Get conversation context from DynamoDB
            conversation_context = self.db_manager.get_conversation_context(session_id)

            # Enhance query for vector retrieval
            enhanced_query = self.conversation_engine.extract_query_for_retrieval(
                user_message, conversation_context
            )

            # Generate query embedding for retrieval
            query_embeddings = self.document_processor.generate_embeddings([enhanced_query])
            query_embedding = query_embeddings[0]

            # Retrieve relevant chunks from vector store
            retrieved_chunks = self.vector_store.search(
                query_embedding=query_embedding,
                k=max_chunks * 2  # Get more initially for filtering
            )

            # Filter by similarity threshold
            filtered_chunks = [
                chunk for chunk in retrieved_chunks 
                if chunk.get('similarity_score', 0) >= similarity_threshold
            ][:max_chunks]

            # Process message with conversation engine
            response_data = self.conversation_engine.process_user_message(
                user_message=user_message,
                session_id=session_id,
                user_id=user_id,
                conversation_context=conversation_context,
                retrieved_chunks=filtered_chunks
            )

            # Store user message in DynamoDB
            self.db_manager.create_message(
                session_id=session_id,
                user_id=user_id,
                role='User',
                message_content=user_message
            )

            # Store assistant response in DynamoDB
            self.db_manager.create_message(
                session_id=session_id,
                user_id=user_id,
                role='Assistant',
                message_content=response_data['response']
            )

            # Handle session rollover if needed
            if response_data.get('needs_new_session') and response_data.get('summary'):
                # Store summary
                self.db_manager.create_conversation_summary(
                    session_id=session_id,
                    user_id=user_id,
                    summary_content=response_data['summary']
                )

                # Create continuation message in new session
                new_session_id = response_data['new_session_id']
                continuation_message = self.conversation_engine.create_session_continuation_message(
                    old_session_id=session_id,
                    new_session_id=new_session_id,
                    summary=response_data['summary']
                )

                # Store continuation message
                self.db_manager.create_message(
                    session_id=new_session_id,
                    user_id=user_id,
                    role='Assistant',
                    message_content=continuation_message
                )

            # Prepare final response
            final_response = {
                'response': response_data['response'],
                'session_id': response_data.get('new_session_id', session_id),
                'retrieved_chunks': len(filtered_chunks),
                'similarity_scores': [chunk.get('similarity_score', 0) for chunk in filtered_chunks],
                'sources': list(set([chunk.get('document_id', 'Unknown') for chunk in filtered_chunks])),
                'session_continued': response_data.get('needs_new_session', False),
                'turn_count': response_data.get('turn_count', 0),
                'timestamp': response_data.get('timestamp')
            }

            logger.info(f"Successfully processed message with {len(filtered_chunks)} relevant chunks")
            return final_response

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                'response': "I'm sorry, I encountered an error processing your message. Please try again.",
                'session_id': session_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_user_conversation_history(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        if not self.is_initialized:
            raise Exception("Application not initialized. Call initialize() first.")

        session_ids = self.db_manager.get_user_sessions(user_id, limit)

        history = []
        for session_id in session_ids:
            context = self.db_manager.get_conversation_context(session_id)
            stats = self.conversation_engine.get_conversation_stats(context)

            # Get first and last message timestamps
            messages = context.get('recent_messages', [])
            if messages:
                first_message = min(messages, key=lambda x: x.get('Timestamp', ''))
                last_message = max(messages, key=lambda x: x.get('Timestamp', ''))

                history.append({
                    'session_id': session_id,
                    'message_count': stats['total_messages'],
                    'started_at': first_message.get('Timestamp'),
                    'last_activity': last_message.get('Timestamp'),
                    'has_summary': stats['has_summary'],
                    'preview': messages[0].get('MessageContent', '')[:100] + '...' if messages else ''
                })

        return history

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and statistics"""
        try:
            vector_stats = self.vector_store.get_stats()
            db_stats = self.db_manager.get_database_stats()

            return {
                'initialized': self.is_initialized,
                'vector_store': vector_stats,
                'database': db_stats,
                'configuration': {
                    'embedding_model': self.config.get('embedding_model', 'text-embedding-3-small'),
                    'chat_model': self.config.get('chat_model', 'gpt-3.5-turbo'),
                    'max_turns_per_session': self.conversation_engine.max_turns_per_session,
                    'vector_store_type': self.vector_store.index_type
                },
                'status_checked_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'initialized': self.is_initialized,
                'error': str(e),
                'status_checked_at': datetime.now().isoformat()
            }

    def cleanup(self):
        """Clean up resources and save state"""
        try:
            if self.vector_store and self.vector_store.index.ntotal > 0:
                self.vector_store.save_index(self.vector_store_path)
                logger.info("Vector store saved")

            logger.info("Application cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def create_sample_config():
    """Create sample configuration dictionary"""
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here'),
        'embedding_model': 'text-embedding-3-small',
        'chat_model': 'gpt-3.5-turbo',
        'embedding_dimension': 1536,
        'index_type': 'cosine',
        'aws_region': 'us-east-1',
        'aws_profile': None,
        'vector_store_path': './data/vector_index'
    }

if __name__ == "__main__":
    print("RAG Application - Complete Implementation with AWS DynamoDB")
