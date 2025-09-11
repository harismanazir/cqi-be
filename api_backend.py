
#!/usr/bin/env python3
"""
FastAPI Backend for LangGraph Code Quality Intelligence
Fixed version to ensure seamless frontend-backend connectivity
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import os
import tempfile
import uuid
from datetime import datetime
import zipfile
import json
import shutil
from extensions.github_integration import GitHubRepoAnalyzer
# Import your LangGraph system (these must exist in your project)
try:
    from main import LangGraphCQI
    from interactive_qa import EnhancedQAAgent
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Make sure main.py and interactive_qa.py exist in your project")
    
    # Mock classes for development/testing
    class LangGraphCQI:
        def __init__(self, enable_rag=True, enable_cache=True, use_langgraph=True):
            pass
        
        async def analyze_file(self, file_path, selected_agents=None, detailed=True):
            # Return mock data matching your CLI output format
            return {
                'language': 'python',
                'lines_of_code': 42,
                'all_issues': [
                    {
                        'severity': 'HIGH',
                        'title': 'Hardcoded Credentials',
                        'agent': 'Security',
                        'line': 13,
                        'description': 'API key is hardcoded in the code',
                        'suggestion': 'Use environment variables or a secure secrets management system'
                    },
                    {
                        'severity': 'MEDIUM',
                        'title': 'Input Validation Issues',
                        'agent': 'Security',
                        'line': 16,
                        'description': 'The function does not validate the input query',
                        'suggestion': 'Add input validation to prevent potential attacks'
                    }
                ],
                'agent_stats': {
                    'security': {'issue_count': 2, 'processing_time': 0.69, 'confidence': 0.90},
                    'performance': {'issue_count': 1, 'processing_time': 0.73, 'confidence': 0.80},
                    'complexity': {'issue_count': 0, 'processing_time': 1.11, 'confidence': 0.85},
                    'documentation': {'issue_count': 3, 'processing_time': 1.31, 'confidence': 0.80}
                },
                'processing_time': 19.14,
                'total_tokens': 3456,
                'total_api_calls': 4,
                'completed_agents': ['security', 'performance', 'complexity', 'documentation']
            }
    
    class EnhancedQAAgent:
        def __init__(self, codebase_path=".", run_analysis=True):
            self.codebase_path = codebase_path
        
        async def initialize(self, enable_rag=True, run_analysis=True):
            pass
        
        async def ask_question(self, question):
            return type('obj', (object,), {
                'answer': 'Mock response',
                'confidence': 0.8,
                'source': 'mock',
                'processing_time': 0.5,
                'follow_up_suggestions': [],
                'related_files': []
            })()

# ---------------------------
# Pydantic models (Updated to match CLI output)
# ---------------------------

class IssueModel(BaseModel):
    severity: str
    title: str
    agent: str
    file: str
    line: int
    description: str
    fix: str

class AgentPerformance(BaseModel):
    agent: str
    issues: int
    time: float
    confidence: float
    status: str

class AnalysisResult(BaseModel):
    file: str
    language: str
    lines: int
    total_issues: int
    processing_time: float
    tokens_used: int
    api_calls: int
    completed_agents: List[str]

    # Severity breakdown
    high_issues: int
    medium_issues: int
    low_issues: int
    critical_issues: int = 0

    # Agent performance
    agent_performance: List[AgentPerformance]

    agent_breakdown: Dict[str, int] = {}

    # Detailed issues (top 20)
    detailed_issues: List[IssueModel]

    # Additional metadata
    timestamp: str
    job_id: str

class ChatMessage(BaseModel):
    session_id: str
    message: str

class UploadResponse(BaseModel):
    success: bool
    files: List[Dict[str, Any]]
    upload_dir: str
    total_files: int

class AnalyzeRequest(BaseModel):
    file_paths: List[str]
    detailed: bool = True
    rag: bool = True

class ChatStartRequest(BaseModel):
    upload_dir: str = ""
    github_repo: str = ""
    branch: str = ""

# ---------------------------
# FastAPI app + CORS
# ---------------------------

app = FastAPI(
    title="LangGraph Code Quality Intelligence API",
    description="Seamless integration with LangGraph multi-agent analysis",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Global in-memory storage
# ---------------------------

analysis_jobs: Dict[str, Dict] = {}
websocket_connections: Dict[str, WebSocket] = {}

# IMPORTANT: Store GitHub temp directories to prevent cleanup
github_temp_dirs: Dict[str, str] = {}

def generate_job_id() -> str:
    return str(uuid.uuid4())

def normalize_github_url(url: str) -> str:
    """Normalize GitHub URL for consistent comparison"""
    if not url:
        return ""
    # Remove common prefixes and suffixes
    url = url.strip().rstrip('/')
    url = url.replace('https://github.com/', '')
    url = url.replace('http://github.com/', '')
    url = url.replace('git@github.com:', '')
    url = url.replace('.git', '')
    return url.lower()

def convert_langgraph_output_to_api_format(raw_result: Dict, job_id: str, file_path: str) -> AnalysisResult:
    """Convert raw LangGraph output to API format that matches CLI output exactly"""
    print(f"[CONVERT] Converting LangGraph output for file: {file_path}")
    print(f"[CONVERT] Raw result keys: {raw_result.keys()}")

    file_name = os.path.basename(file_path)
    language = raw_result.get('language', 'Unknown')

    all_issues = raw_result.get('all_issues', [])
    print(f"[CONVERT] Found {len(all_issues)} issues")

    # Count issues by severity - FIXED: Separate critical from high
    critical_count = len([i for i in all_issues if i.get('severity', '').lower() == 'critical'])
    high_count = len([i for i in all_issues if i.get('severity', '').lower() == 'high'])
    medium_count = len([i for i in all_issues if i.get('severity', '').lower() == 'medium'])
    low_count = len([i for i in all_issues if i.get('severity', '').lower() == 'low'])

    # Convert agent stats
    agent_performance = []
    agent_stats = raw_result.get('agent_stats', {})
    print(f"[CONVERT] Agent stats: {agent_stats}")
    
    # FIXED: Calculate agent breakdown for frontend with correct keys
    agent_breakdown = {}
    for agent, stats in agent_stats.items():
        agent_performance.append(AgentPerformance(
            agent=agent.title(),
            issues=stats.get('issue_count', 0),
            time=stats.get('processing_time', 0.0),
            confidence=stats.get('confidence', 0.0),
            status="SUCCESS"
        ))
        # Add to agent breakdown with lowercase key for frontend
        agent_breakdown[agent.lower()] = stats.get('issue_count', 0)

    # Sort issues by severity for detailed_issues
    severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
    sorted_issues = sorted(
        all_issues,
        key=lambda x: severity_order.get(x.get('severity', '').lower(), 0),
        reverse=True
    )

    detailed_issues = []
    for i, issue in enumerate(sorted_issues[:20]):  # Top 20 issues
        detailed_issues.append(IssueModel(
            severity=issue.get('severity', 'unknown').upper(),
            title=issue.get('title', 'Unknown Issue'),
            agent=issue.get('agent', 'unknown').title(),
            file=file_name,
            line=issue.get('line_number', issue.get('line', 0)),
            description=issue.get('description', 'No description'),
            fix=issue.get('suggestion', issue.get('fix', 'No fix suggested'))
        ))

    result = AnalysisResult(
        file=file_name,
        language=language.title(),
        lines=raw_result.get('lines_of_code', 0),
        total_issues=len(all_issues),
        processing_time=raw_result.get('processing_time', 0.0),
        tokens_used=raw_result.get('total_tokens', 0),
        api_calls=raw_result.get('total_api_calls', 0),
        completed_agents=raw_result.get('completed_agents', []),

        # FIXED: Use separate counts for critical and high
        critical_issues=critical_count,
        high_issues=high_count,
        medium_issues=medium_count,
        low_issues=low_count,
        

        agent_performance=agent_performance,
        agent_breakdown=agent_breakdown,
        detailed_issues=detailed_issues,

        timestamp=datetime.now().isoformat(),
        job_id=job_id
    )
    
    print(f"[CONVERT] Converted result: {result.total_issues} issues, {len(result.detailed_issues)} detailed")
    print(f"[CONVERT] Agent breakdown: {agent_breakdown}")
    print(f"[CONVERT] Severity counts - Critical: {critical_count}, High: {high_count}, Medium: {medium_count}, Low: {low_count}")
    return result
    

async def broadcast_progress(job_id: str, progress: int, message: str):
    """Send progress updates via WebSocket"""
    print(f"[WEBSOCKET] Broadcasting progress for {job_id}: {progress}% - {message}")
    if job_id in websocket_connections:
        try:
            await websocket_connections[job_id].send_json({
                "type": "progress",
                "job_id": job_id,
                "progress": progress,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            print(f"[WEBSOCKET] Error broadcasting: {e}")
            websocket_connections.pop(job_id, None)

# ---------------------------
# API endpoints
# ---------------------------

@app.get("/")
async def root():
    return {
        "message": "LangGraph Code Quality Intelligence API", 
        "status": "online", 
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload files or folders for analysis"""
    print(f"[UPLOAD] Received {len(files)} files")

    try:
        uploaded_files = []
        upload_dir = tempfile.mkdtemp(prefix="cqi_upload_")
        print(f"[UPLOAD] Created temp directory: {upload_dir}")

        for file in files:
            print(f"[UPLOAD] Processing: {file.filename}")

            safe_filename = file.filename.replace(" ", "_").replace("..", "")
            file_path = os.path.join(upload_dir, safe_filename)

            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            # Handle ZIP files
            if file.filename.endswith('.zip'):
                extract_dir = os.path.join(upload_dir, "extracted")
                os.makedirs(extract_dir, exist_ok=True)

                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)

                    # Find code files in extracted content
                    code_extensions = ('.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go')
                    for root, dirs, filenames in os.walk(extract_dir):
                        for filename in filenames:
                            if filename.endswith(code_extensions):
                                extracted_path = os.path.join(root, filename)
                                uploaded_files.append({
                                    "name": filename,
                                    "path": extracted_path,
                                    "size": os.path.getsize(extracted_path),
                                    "type": "code"
                                })
                except Exception as e:
                    print(f"[UPLOAD] ZIP extraction failed: {e}")
                    uploaded_files.append({
                        "name": file.filename,
                        "path": file_path,
                        "size": len(content),
                        "type": "archive"
                    })
            else:
                uploaded_files.append({
                    "name": file.filename,
                    "path": file_path,
                    "size": len(content),
                    "type": "code"
                })

        print(f"[UPLOAD] Successfully processed {len(uploaded_files)} files")

        return UploadResponse(
            success=True,
            files=uploaded_files,
            upload_dir=upload_dir,
            total_files=len(uploaded_files)
        )

    except Exception as e:
        print(f"[UPLOAD] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/analyze/{job_id}")
async def start_analysis(job_id: str, request: AnalyzeRequest):
    """Start LangGraph analysis - exact same as CLI main.py analyze"""
    print(f"[ANALYZE] Starting analysis for job {job_id}")
    print(f"[ANALYZE] Files: {[os.path.basename(f) for f in request.file_paths]}")
    print(f"[ANALYZE] Options: detailed={request.detailed}, rag={request.rag}")

    try:
        # Initialize job tracking
        analysis_jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "progress": 10,
            "message": "Initializing LangGraph analysis...",
            "start_time": datetime.now(),
            "file_paths": request.file_paths
        }

        await broadcast_progress(job_id, 10, "Initializing LangGraph analysis...")

        # Initialize LangGraph system
        print("[ANALYZE] Initializing LangGraph multi-agent system...")
        analyzer = LangGraphCQI(enable_rag=request.rag, enable_cache=True, use_langgraph=True)

        await broadcast_progress(job_id, 30, "LangGraph multi-agent system ready...")

        results = []
        total_files = len(request.file_paths)

        for i, file_path in enumerate(request.file_paths):
            progress = 30 + int((i / total_files) * 60)
            filename = os.path.basename(file_path)

            analysis_jobs[job_id]["progress"] = progress
            analysis_jobs[job_id]["message"] = f"Analyzing {filename}..."

            await broadcast_progress(job_id, progress, f"Analyzing {filename}...")

            print(f"[ANALYZE] Processing: {filename}")

            # Run LangGraph analysis (same as CLI)
            try:
                result = await analyzer.analyze_file(
                    file_path,
                    selected_agents=None,
                    detailed=request.detailed
                )

                api_result = convert_langgraph_output_to_api_format(result, job_id, file_path)
                results.append(api_result)

                print(f"[ANALYZE] Completed: {filename} - {api_result.total_issues} issues found")
            
            except Exception as file_error:
                print(f"[ANALYZE] Error analyzing {filename}: {file_error}")
                # Continue with other files
                continue

        # Complete analysis
        analysis_jobs[job_id]["status"] = "completed"
        analysis_jobs[job_id]["progress"] = 100
        analysis_jobs[job_id]["message"] = "Analysis completed!"
        analysis_jobs[job_id]["results"] = results
        analysis_jobs[job_id]["completion_time"] = datetime.now()

        await broadcast_progress(job_id, 100, "Analysis completed!")

        print(f"[ANALYZE] Job {job_id} completed successfully with {len(results)} files analyzed")

        return {"success": True, "job_id": job_id, "results_count": len(results)}

    except Exception as e:
        print(f"[ANALYZE] Error: {e}")
        # Mark job as failed
        if job_id in analysis_jobs:
            analysis_jobs[job_id]["status"] = "failed"
            analysis_jobs[job_id]["message"] = f"Analysis failed: {str(e)}"
        else:
            analysis_jobs[job_id] = {
                "job_id": job_id,
                "status": "failed",
                "progress": 0,
                "message": f"Analysis failed: {str(e)}",
                "start_time": datetime.now()
            }

        await broadcast_progress(job_id, 0, f"Analysis failed: {str(e)}")

        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/status/{job_id}")
async def get_analysis_status(job_id: str):
    """Get analysis job status"""
    print(f"[STATUS] Checking status for job: {job_id}")
    
    if job_id not in analysis_jobs:
        print(f"[STATUS] Job {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")

    job = analysis_jobs[job_id]
    print(f"[STATUS] Job {job_id} status: {job['status']} ({job['progress']}%)")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "start_time": job["start_time"].isoformat(),
        "completion_time": job.get("completion_time").isoformat() if job.get("completion_time") else None
    }

@app.get("/api/results/{job_id}")
async def get_analysis_results(job_id: str):
    """Get analysis results - enhanced with GitHub metadata"""
    print(f"[RESULTS] Fetching results for job: {job_id}")
    
    if job_id not in analysis_jobs:
        print(f"[RESULTS] Job {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")

    job = analysis_jobs[job_id]

    if job["status"] != "completed":
        print(f"[RESULTS] Job {job_id} not completed, status: {job['status']}")
        raise HTTPException(status_code=400, detail=f"Analysis not completed. Status: {job['status']}")

    results = job.get("results", [])
    
    # Enhance with GitHub metadata if this was a GitHub analysis
    response_data = {
        "success": True,
        "job_id": job_id,
        "results": results,
        "total_files": len(results),
        "completion_time": job.get("completion_time").isoformat() if job.get("completion_time") else None
    }
    
    # Add GitHub-specific metadata
    if job.get("is_github", False):
        response_data["github_metadata"] = {
            "repo_url": job.get("repo_url"),
            "branch": job.get("branch"),
            "repo_stats": job.get("repo_stats", {}),
            "analysis_type": "github_repository",
            "temp_dir": job.get("temp_dir")  # IMPORTANT: Include temp_dir for chat
        }
    
    print(f"[RESULTS] Returning {len(results)} file results")
    if job.get("is_github"):
        print(f"[RESULTS] GitHub repository: {job.get('repo_url')}")
    
    return response_data

qa_agents: Dict[str, EnhancedQAAgent] = {}

# FIXED: Enhanced start_chat_session to handle GitHub repos properly
@app.post("/api/chat/start")
async def start_chat_session(request: ChatStartRequest):
    try:
        session_id = generate_job_id()
        
        print(f"[CHAT] Starting interactive session: {session_id}")
        print(f"[CHAT] Request details:")
        print(f"[CHAT] - upload_dir: {request.upload_dir}")
        print(f"[CHAT] - github_repo: {request.github_repo}")
        print(f"[CHAT] - branch: {request.branch}")
        
        # UNIFIED LOGIC: Both flows use upload_dir
        codebase_path = "."
        analysis_context = "current directory"
        
        if request.upload_dir and os.path.exists(request.upload_dir):
            # Works for BOTH file uploads AND GitHub repos now!
            codebase_path = request.upload_dir
            
            # Determine context type
            if request.github_repo:
                analysis_context = f"GitHub repository {request.github_repo} (branch: {request.branch})"
                print(f"[CHAT] Using GitHub repository directory: {codebase_path}")
            else:
                analysis_context = f"uploaded files in {request.upload_dir}"
                print(f"[CHAT] Using uploaded files directory: {codebase_path}")
            
            # Verify the directory has files
            try:
                files_in_dir = os.listdir(codebase_path)
                print(f"[CHAT] Directory contains {len(files_in_dir)} items: {files_in_dir[:5]}...")
            except Exception as e:
                print(f"[CHAT] Warning: Could not list directory contents: {e}")
                
        else:
            print(f"[CHAT] No upload directory provided or directory doesn't exist: {request.upload_dir}")
            
            # Only fail if we expected a directory
            if request.upload_dir or request.github_repo:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Analysis directory not found. Please analyze the {'repository' if request.github_repo else 'files'} first."
                )
        
        print(f"[CHAT] Final codebase path: {codebase_path}")
        
        # Initialize the Q&A agent with the correct path
        qa_agent = EnhancedQAAgent(codebase_path=codebase_path)
        await qa_agent.initialize(enable_rag=True, run_analysis=True)
        
        # Store the agent for this session
        qa_agents[session_id] = qa_agent
        
        return {
            "success": True,
            "session_id": session_id,
            "message": f"Interactive Q&A session started with {analysis_context}",
            "codebase_info": {
                "path": codebase_path,
                "status": "ready",
                "context": analysis_context,
                "github_repo": request.github_repo if request.github_repo else None,
                "branch": request.branch if request.branch else None
            }
        }
    except Exception as e:
        print(f"[CHAT] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to start chat: {str(e)}")

# Fix the send_chat_message endpoint
@app.post("/api/chat/message")
async def send_chat_message(request: ChatMessage):
    try:
        if request.session_id not in qa_agents:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        qa_agent = qa_agents[request.session_id]
        
        print(f"[CHAT] Processing message: {request.message[:50]}...")
        print(f"[CHAT] Using codebase path: {qa_agent.codebase_path}")
        
        # Use the actual Q&A system
        response = await qa_agent.ask_question(request.message)
        
        return {
            "success": True,
            "response": {
                "content": response.answer,
                "confidence": response.confidence,
                "source": response.source,
                "processing_time": response.processing_time,
                "follow_up_suggestions": response.follow_up_suggestions,
                "related_files": response.related_files or []
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"[CHAT] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat message failed: {str(e)}")

@app.websocket("/api/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket for real-time progress updates"""
    print(f"[WEBSOCKET] Client connecting for job: {job_id}")
    await websocket.accept()
    websocket_connections[job_id] = websocket

    try:
        while True:
            await asyncio.sleep(1)

            if job_id in analysis_jobs:
                job = analysis_jobs[job_id]
                await websocket.send_json({
                    "type": "heartbeat",
                    "job_id": job_id,
                    "status": job["status"],
                    "progress": job["progress"],
                    "message": job["message"]
                })

                if job["status"] in ["completed", "failed"]:
                    print(f"[WEBSOCKET] Job {job_id} finished, closing connection")
                    break

    except WebSocketDisconnect:
        print(f"[WEBSOCKET] Client disconnected for job: {job_id}")
        websocket_connections.pop(job_id, None)
    except Exception as e:
        print(f"[WEBSOCKET] Error: {e}")
        websocket_connections.pop(job_id, None)

# ---------------------------
# Development endpoint for testing
# ---------------------------

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint for development"""
    return {
        "message": "API is working!",
        "active_jobs": len(analysis_jobs),
        "active_sessions": len(qa_agents),
        "websocket_connections": len(websocket_connections),
        "timestamp": datetime.now().isoformat()
    }


class GitHubAnalyzeRequest(BaseModel):
    repo_url: str
    branch: str = "main"
    agents: List[str] = ["security", "performance", "complexity", "documentation"]
    detailed: bool = True

class GitHubValidationResponse(BaseModel):
    valid: bool
    owner: Optional[str] = None
    repo_name: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    branches: Optional[List[str]] = None
    error: Optional[str] = None

# ========== GITHUB INTEGRATION ENDPOINTS ==========

@app.post("/api/github/analyze")
async def analyze_github_repository(request: GitHubAnalyzeRequest):
    job_id = generate_job_id()
    temp_dir = None
    
    try:
        # Initialize GitHub analyzer
        github_analyzer = GitHubRepoAnalyzer()
        
        # Download repository
        temp_dir = await github_analyzer.download_repo(request.repo_url, request.branch)
        print(f"[GITHUB-API] Repository downloaded to: {temp_dir}")
        
        # IMPORTANT: Store temp directory globally like file uploads
        github_temp_dirs[job_id] = temp_dir
        
        # Get repository statistics
        repo_stats = github_analyzer.get_repository_stats(temp_dir)
        
        # Use existing file discovery logic
        from main import LangGraphCQI
        cqi = LangGraphCQI(enable_rag=True, enable_cache=True)
        code_files = cqi._discover_files(temp_dir)
        
        # Initialize job tracking - SAME AS FILE UPLOAD
        analysis_jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "progress": 20,
            "message": f"Downloaded {request.repo_url}, analyzing {len(code_files)} files...",
            "start_time": datetime.now(),
            "file_paths": code_files,
            "repo_url": request.repo_url,
            "branch": request.branch,
            "temp_dir": temp_dir,  # Store like upload_dir
            "repo_stats": repo_stats,
            "is_github": True,
            "github_metadata": {
                "repo_url": request.repo_url,
                "branch": request.branch,
                "stats": repo_stats,
                "temp_dir": temp_dir
            }
        }
        
        await broadcast_progress(job_id, 20, f"Repository downloaded, analyzing {len(code_files)} files...")
        
        # Use existing analysis logic
        analysis_request = AnalyzeRequest(
            file_paths=code_files,
            detailed=request.detailed,
            rag=True
        )
        
        # Start analysis using existing pipeline
        result = await start_analysis(job_id, analysis_request)
        
        print(f"[GITHUB-API] Analysis completed for {request.repo_url}")
        print(f"[GITHUB-API] Temp directory preserved: {temp_dir}")
        
        # RETURN temp_dir in response (like file upload returns upload_dir)
        return {
            "success": True,
            "job_id": job_id,
            "repo_url": request.repo_url,
            "branch": request.branch,
            "files_analyzed": len(code_files),
            "repo_stats": repo_stats,
            "upload_dir": temp_dir,  # Use same key as file upload!
            "temp_dir": temp_dir,    # Also include for backward compatibility
            "total_files": len(code_files)
        }
        
    except Exception as e:
        print(f"[GITHUB-API] Error: {str(e)}")
        
        # Clean up temp directory ONLY on error
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"[GITHUB-API] Cleaned up temp directory on error: {temp_dir}")
            except:
                pass
        
        # Mark job as failed
        analysis_jobs[job_id] = {
            "job_id": job_id,
            "status": "failed",
            "progress": 0,
            "message": f"GitHub analysis failed: {str(e)}",
            "start_time": datetime.now(),
            "is_github": True
        }
        
        await broadcast_progress(job_id, 0, f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GitHub analysis failed: {str(e)}")

    # REMOVE the finally block completely - no cleanup!

class GitHubValidateRequest(BaseModel):
    repo_url: str

@app.post("/api/github/validate")
async def validate_github_repository(request: GitHubValidateRequest):
    repo_url = request.repo_url
    """Validate GitHub repository URL and get metadata"""
    
    print(f"[GITHUB-API] Validating repository: {repo_url}")
    
    try:
        github_analyzer = GitHubRepoAnalyzer()
        validation_result = await github_analyzer.validate_repository(repo_url)
        
        print(f"[GITHUB-API] Validation result: {validation_result.get('valid', False)}")
        
        return validation_result
        
    except Exception as e:
        print(f"[GITHUB-API] Validation error: {str(e)}")
        return {
            "valid": False,
            "error": f"Validation failed: {str(e)}"
        }

@app.get("/api/github/branches/{owner}/{repo}")
async def get_repository_branches(owner: str, repo: str):
    """Get available branches for a repository"""
    
    print(f"[GITHUB-API] Getting branches for: {owner}/{repo}")
    
    try:
        github_analyzer = GitHubRepoAnalyzer()
        branches = await github_analyzer.get_repository_branches(owner, repo)
        
        print(f"[GITHUB-API] Found {len(branches)} branches")
        
        return {
            "success": True,
            "branches": branches,
            "default_branch": branches[0] if branches else "main"
        }
        
    except Exception as e:
        print(f"[GITHUB-API] Error getting branches: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "branches": ["main", "master"]  # Fallback
        }

# IMPORTANT: Cleanup endpoint for managing temp directories
@app.delete("/api/github/cleanup/{job_id}")
async def cleanup_github_temp_dir(job_id: str):
    """Clean up GitHub temporary directory"""
    try:
        if job_id in github_temp_dirs:
            temp_dir = github_temp_dirs[job_id]
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"[CLEANUP] Removed temp directory: {temp_dir}")
            del github_temp_dirs[job_id]
            return {"success": True, "message": f"Cleaned up temp directory for job {job_id}"}
        else:
            return {"success": False, "message": f"No temp directory found for job {job_id}"}
    except Exception as e:
        print(f"[CLEANUP] Error: {e}")
        return {"success": False, "error": str(e)}

# OPTIONAL: Add a debug endpoint to check GitHub job status
@app.get("/api/debug/github-jobs")
async def debug_github_jobs():
    """Debug endpoint to check GitHub job tracking"""
    github_jobs = {}
    
    for job_id, job in analysis_jobs.items():
        if job.get("is_github"):
            github_jobs[job_id] = {
                "repo_url": job.get("repo_url"),
                "normalized_repo_url": job.get("normalized_repo_url"),
                "branch": job.get("branch"),
                "temp_dir": job.get("temp_dir"),
                "temp_dir_exists": os.path.exists(job.get("temp_dir", "")),
                "status": job.get("status"),
                "start_time": job.get("start_time").isoformat() if job.get("start_time") else None
            }
    
    return {
        "total_github_jobs": len(github_jobs),
        "github_temp_dirs": github_temp_dirs,
        "jobs": github_jobs
    }

# ---------------------------
# Run server (development)
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting LangGraph Code Quality Intelligence API...")
    print("📊 Frontend will receive exact same output as CLI main.py analyze")
    print("🔗 CORS enabled for localhost:3000, localhost:5173, localhost:5174")
    
    # Use PORT environment variable for deployment (DigitalOcean, Render, Heroku, etc.)
    port = int(os.environ.get("PORT", 8000))
    print(f"💡 Access test endpoint at: http://localhost:{port}/api/test")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)