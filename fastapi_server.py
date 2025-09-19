"""
FastAPI REST API Server for RAG Application
Provides RESTful endpoints for all RAG functionality
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import asyncio
import os
import tempfile
import logging
from datetime import datetime
import json

from main_app import RAGApplication, create_sample_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class UserCreateRequest(BaseModel):
    name: str = Field(..., description="User's full name")
    phone_number: str = Field(..., description="User's phone number")
    email: str = Field(..., description="User's email address")
    additional_info: str = Field("", description="Additional user information")

class SessionCreateRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")

class MessageRequest(BaseModel):
    user_message: str = Field(..., description="User's message")
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    similarity_threshold: float = Field(0.7, description="Minimum similarity for chunk retrieval")
    max_chunks: int = Field(3, description="Maximum chunks to retrieve")

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise RAG System API",
    description="RESTful API for Retrieval-Augmented Generation with document ingestion and conversation management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG application instance
rag_app: Optional[RAGApplication] = None

async def get_rag_app() -> RAGApplication:
    """Dependency to get initialized RAG application"""
    global rag_app
    if rag_app is None or not rag_app.is_initialized:
        raise HTTPException(status_code=503, detail="RAG application not initialized")
    return rag_app

@app.on_event("startup")
async def startup_event():
    """Initialize RAG application on startup"""
    global rag_app
    try:
        config = create_sample_config()

        config.update({
            'openai_api_key': os.getenv('OPENAI_API_KEY', config['openai_api_key']),
            'aws_region': os.getenv('AWS_REGION', config['aws_region']),
            'aws_profile': os.getenv('AWS_PROFILE', config['aws_profile']),
            'vector_store_path': os.getenv('VECTOR_STORE_PATH', config['vector_store_path'])
        })

        rag_app = RAGApplication(config)
        await rag_app.initialize()
        logger.info("RAG Application initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize RAG application: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global rag_app
    if rag_app:
        rag_app.cleanup()
        logger.info("RAG Application cleaned up")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "initialized": rag_app.is_initialized if rag_app else False
    }

# System status endpoint
@app.get("/system/status")
async def get_system_status(app: RAGApplication = Depends(get_rag_app)):
    """Get comprehensive system status"""
    return app.get_system_status()

# User management endpoints
@app.post("/users")
async def create_user(
    user_data: UserCreateRequest,
    app: RAGApplication = Depends(get_rag_app)
):
    """Create a new user"""
    try:
        user = app.create_user(
            name=user_data.name,
            phone_number=user_data.phone_number,
            email=user_data.email,
            additional_info=user_data.additional_info
        )
        return user
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users/{user_id}")
async def get_user(
    user_id: str,
    app: RAGApplication = Depends(get_rag_app)
):
    """Get user by ID"""
    user = app.db_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# Document management endpoints
@app.post("/documents/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: Optional[str] = None,
    app: RAGApplication = Depends(get_rag_app)
):
    """Ingest a Word document into the RAG system"""
    if not file.filename.endswith(('.docx', '.doc')):
        raise HTTPException(
            status_code=400, 
            detail="Only Word documents (.docx, .doc) are supported"
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        result = app.ingest_document(temp_file_path, document_id)

        background_tasks.add_task(os.unlink, temp_file_path)

        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['error'])

        return result

    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

# Conversation management endpoints
@app.post("/conversations/start")
async def start_conversation(
    session_data: SessionCreateRequest,
    app: RAGApplication = Depends(get_rag_app)
):
    """Start a new conversation session"""
    try:
        session = app.start_conversation(session_data.user_id)
        return session
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/conversations/message")
async def send_message(
    message_data: MessageRequest,
    app: RAGApplication = Depends(get_rag_app)
):
    """Send a message and get RAG response"""
    try:
        response = app.process_message(
            user_message=message_data.user_message,
            session_id=message_data.session_id,
            user_id=message_data.user_id,
            similarity_threshold=message_data.similarity_threshold,
            max_chunks=message_data.max_chunks
        )

        if 'error' in response:
            raise HTTPException(status_code=400, detail=response['error'])

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
