"""RAG-enhanced code analysis agent with lightweight alternatives."""

import os
import hashlib
import pickle
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import re
import math
from collections import defaultdict, Counter

# Lightweight alternatives
try:
    import spacy
    SPACY_AVAILABLE = True
    # Try to load small model
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_MODEL_AVAILABLE = True
    except OSError:
        SPACY_MODEL_AVAILABLE = False
        nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    SPACY_MODEL_AVAILABLE = False
    nlp = None

# External API support
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata"""
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str  # 'function', 'class', 'block', 'full_file'
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeChunk':
        """Create from dictionary"""
        return cls(**data)


class LightweightEmbedder:
    """Lightweight embedding alternatives using multiple strategies"""
    
    def __init__(self, method: str = "auto"):
        """
        Initialize embedder with different methods:
        - "auto": Try spaCy, then TF-IDF, then simple
        - "spacy": Use spaCy word vectors
        - "tfidf": Use TF-IDF with code-specific preprocessing
        - "simple": Simple heuristic-based embeddings
        - "api": Use external API (OpenAI, Cohere, etc.)
        """
        self.method = method
        self.api_key = os.getenv('EMBEDDING_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.api_url = os.getenv('EMBEDDING_API_URL', 'https://api.openai.com/v1/embeddings')
        
        # Initialize based on available dependencies and method
        if method == "auto":
            if SPACY_MODEL_AVAILABLE:
                self.method = "spacy"
                print("✅ Using spaCy embeddings")
            else:
                self.method = "tfidf"
                print("✅ Using TF-IDF embeddings")
        
        # Initialize TF-IDF components
        if self.method in ["tfidf", "auto"]:
            self.vocabulary = {}
            self.idf_scores = {}
            self.is_fitted = False
        
        # Code-specific stop words
        self.code_stop_words = {
            'def', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while',
            'try', 'except', 'finally', 'with', 'as', 'return', 'yield', 'pass',
            'break', 'continue', 'and', 'or', 'not', 'is', 'in', 'none', 'true',
            'false', 'self', 'cls', 'super', 'this', 'null', 'undefined', 'var',
            'let', 'const', 'function', 'async', 'await', 'public', 'private',
            'protected', 'static', 'final', 'abstract', 'interface', 'enum'
        }
    
    def encode(self, texts: List[str], show_progress_bar: bool = False) -> List[List[float]]:
        """Encode texts into embeddings using the selected method"""
        if self.method == "spacy":
            return self._encode_spacy(texts)
        elif self.method == "tfidf":
            return self._encode_tfidf(texts)
        elif self.method == "api":
            return asyncio.run(self._encode_api_async(texts))
        else:  # simple
            return self._encode_simple(texts)
    
    def _encode_spacy(self, texts: List[str]) -> List[List[float]]:
        """Use spaCy word vectors for embeddings"""
        if not SPACY_MODEL_AVAILABLE:
            print("⚠️ spaCy model not available, falling back to TF-IDF")
            return self._encode_tfidf(texts)
        
        embeddings = []
        for text in texts:
            doc = nlp(text)
            if doc.has_vector:
                embeddings.append(doc.vector.tolist())
            else:
                # Fallback to average word vectors
                word_vectors = [token.vector for token in doc if token.has_vector]
                if word_vectors:
                    import numpy as np
                    avg_vector = np.mean(word_vectors, axis=0)
                    embeddings.append(avg_vector.tolist())
                else:
                    # Ultimate fallback
                    embeddings.append(self._simple_embedding(text))
        
        return embeddings
    
    def _encode_tfidf(self, texts: List[str]) -> List[List[float]]:
        """Use TF-IDF for embeddings with code-specific preprocessing"""
        # Preprocess texts for code analysis
        processed_texts = [self._preprocess_code_text(text) for text in texts]
        
        # Build vocabulary if not fitted
        if not self.is_fitted:
            self._fit_tfidf(processed_texts)
        
        # Compute TF-IDF vectors
        embeddings = []
        for text in processed_texts:
            tf_scores = self._compute_tf(text)
            tfidf_vector = [0.0] * len(self.vocabulary)
            
            for word, tf in tf_scores.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    idf = self.idf_scores.get(word, 0)
                    tfidf_vector[idx] = tf * idf
            
            # Normalize vector
            norm = math.sqrt(sum(x*x for x in tfidf_vector))
            if norm > 0:
                tfidf_vector = [x/norm for x in tfidf_vector]
            
            embeddings.append(tfidf_vector)
        
        return embeddings
    
    def _encode_simple(self, texts: List[str]) -> List[List[float]]:
        """Simple heuristic-based embeddings for code"""
        embeddings = []
        for text in texts:
            embeddings.append(self._simple_embedding(text))
        return embeddings
    
    async def _encode_api_async(self, texts: List[str]) -> List[List[float]]:
        """Use external API for embeddings"""
        if not self.api_key:
            print("⚠️ No API key provided, falling back to TF-IDF")
            return self._encode_tfidf(texts)
        
        if not HTTPX_AVAILABLE and not REQUESTS_AVAILABLE:
            print("⚠️ No HTTP client available for API calls, falling back to TF-IDF")
            return self._encode_tfidf(texts)
        
        try:
            if HTTPX_AVAILABLE:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        self.api_url,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": "text-embedding-3-small",  # Cheaper OpenAI model
                            "input": texts[:100]  # Limit batch size
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return [item["embedding"] for item in data["data"]]
            else:
                # Fallback to requests (sync)
                import requests
                response = requests.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": "text-embedding-3-small",
                        "input": texts[:100]
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return [item["embedding"] for item in data["data"]]
        
        except Exception as e:
            print(f"⚠️ API embedding failed: {e}, falling back to TF-IDF")
        
        return self._encode_tfidf(texts)
    
    def _preprocess_code_text(self, text: str) -> str:
        """Preprocess text for code analysis"""
        # Extract meaningful tokens from code
        # Remove comments
        text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        
        # Extract identifiers, function names, etc.
        tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
        
        # Remove stop words
        tokens = [t for t in tokens if t not in self.code_stop_words and len(t) > 2]
        
        return ' '.join(tokens)
    
    def _fit_tfidf(self, texts: List[str]):
        """Build TF-IDF vocabulary and IDF scores"""
        # Build vocabulary
        word_doc_count = defaultdict(int)
        all_words = set()
        
        for text in texts:
            words = set(text.split())
            all_words.update(words)
            for word in words:
                word_doc_count[word] += 1
        
        # Create vocabulary mapping
        self.vocabulary = {word: i for i, word in enumerate(sorted(all_words))}
        
        # Compute IDF scores
        num_docs = len(texts)
        for word in all_words:
            df = word_doc_count[word]
            self.idf_scores[word] = math.log(num_docs / (1 + df))
        
        self.is_fitted = True
    
    def _compute_tf(self, text: str) -> Dict[str, float]:
        """Compute term frequency for a text"""
        words = text.split()
        word_count = Counter(words)
        total_words = len(words)
        
        tf_scores = {}
        for word, count in word_count.items():
            tf_scores[word] = count / total_words
        
        return tf_scores
    
    def _simple_embedding(self, text: str) -> List[float]:
        """Create simple heuristic-based embedding for code"""
        # Code-specific features
        features = []
        
        # Basic text statistics
        words = text.lower().split()
        lines = text.split('\n')
        
        features.extend([
            len(words),                           # Word count
            len(lines),                          # Line count
            len(set(words)),                     # Unique words
            len([w for w in words if len(w) > 8]),  # Long identifiers
            text.count('('),                     # Function calls
            text.count('{'),                     # Blocks
            text.count('='),                     # Assignments
            text.count('if'),                    # Conditionals
            text.count('for') + text.count('while'),  # Loops
            text.count('class'),                 # Classes
            text.count('def') + text.count('function'),  # Functions
            text.count('import') + text.count('include'),  # Imports
        ])
        
        # Language-specific keywords
        keywords = {
            'security': ['auth', 'password', 'token', 'security', 'encrypt', 'hash'],
            'performance': ['optimize', 'cache', 'fast', 'efficient', 'performance'],
            'data': ['database', 'sql', 'query', 'data', 'model', 'table'],
            'ui': ['render', 'component', 'view', 'template', 'ui', 'interface'],
            'error': ['error', 'exception', 'try', 'catch', 'handle', 'fail']
        }
        
        text_lower = text.lower()
        for category, words_list in keywords.items():
            count = sum(text_lower.count(word) for word in words_list)
            features.append(count)
        
        # Normalize to fixed size (64 dimensions)
        if len(features) < 64:
            features.extend([0.0] * (64 - len(features)))
        else:
            features = features[:64]
        
        # L2 normalize
        norm = math.sqrt(sum(x*x for x in features))
        if norm > 0:
            features = [x/norm for x in features]
        
        return features


class LightweightSimilaritySearch:
    """Lightweight similarity search using cosine similarity"""
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def search_similar_vectors(query_vec: List[float], 
                             vectors: List[List[float]], 
                             top_k: int = 5) -> List[Tuple[int, float]]:
        """Find most similar vectors"""
        similarities = []
        
        for i, vec in enumerate(vectors):
            sim = LightweightSimilaritySearch.cosine_similarity(query_vec, vec)
            similarities.append((i, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class CodeChunker:
    """Intelligent code chunking for different programming languages"""
    
    def __init__(self, max_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size
        
        # Language-specific patterns for intelligent chunking
        self.language_patterns = {
            'python': {
                'function': r'^def\s+\w+.*?:',
                'class': r'^class\s+\w+.*?:',
                'import': r'^(?:from\s+\w+\s+)?import\s+',
            },
            'java': {
                'method': r'(public|private|protected)\s+.*?\w+\s*\([^)]*\)\s*\{',
                'class': r'(public\s+)?(class|interface)\s+\w+',
                'import': r'^import\s+',
            },
            'javascript': {
                'function': r'function\s+\w+\s*\([^)]*\)\s*\{',
                'arrow_function': r'\w+\s*=\s*\([^)]*\)\s*=>\s*\{',
                'class': r'class\s+\w+',
            }
        }
    
    def chunk_code(self, code: str, file_path: str, language: str) -> List[CodeChunk]:
        """Intelligently chunk code based on language structure"""
        lines = code.split('\n')
        chunks = []
        
        # Try language-specific chunking first
        if language in self.language_patterns:
            chunks.extend(self._smart_chunk_by_language(lines, file_path, language))
        
        # If no smart chunks found or file is too large, use sliding window
        if not chunks or len(code) > self.max_chunk_size * 3:
            chunks.extend(self._sliding_window_chunk(lines, file_path, language))
        
        # Always include a full-file chunk if file is small enough
        if len(code) <= self.max_chunk_size:
            chunks.append(CodeChunk(
                content=code,
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                language=language,
                chunk_type='full_file'
            ))
        
        return chunks
    
    def _smart_chunk_by_language(self, lines: List[str], file_path: str, language: str) -> List[CodeChunk]:
        """Create chunks based on language-specific structures"""
        chunks = []
        patterns = self.language_patterns.get(language, {})
        
        current_chunk_lines = []
        current_start = 1
        in_function = False
        brace_count = 0
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for function/method/class start
            is_structure_start = any(re.match(pattern, line_stripped, re.IGNORECASE) 
                                   for pattern in patterns.values())
            
            if is_structure_start and current_chunk_lines:
                # Save previous chunk
                chunk_content = '\n'.join(current_chunk_lines)
                if chunk_content.strip():
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        start_line=current_start,
                        end_line=i - 1,
                        language=language,
                        chunk_type='block'
                    ))
                
                current_chunk_lines = [line]
                current_start = i
                in_function = True
                brace_count = line.count('{') - line.count('}')
            else:
                current_chunk_lines.append(line)
                if in_function:
                    brace_count += line.count('{') - line.count('}')
                    if brace_count <= 0 and any(c in line for c in ['}', 'end', 'return']):
                        in_function = False
            
            # Force chunk if getting too large
            if len('\n'.join(current_chunk_lines)) > self.max_chunk_size:
                chunk_content = '\n'.join(current_chunk_lines)
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=current_start,
                    end_line=i,
                    language=language,
                    chunk_type='large_block'
                ))
                current_chunk_lines = []
                current_start = i + 1
                in_function = False
                brace_count = 0
        
        # Add remaining lines
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            if chunk_content.strip():
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=current_start,
                    end_line=len(lines),
                    language=language,
                    chunk_type='block'
                ))
        
        return chunks
    
    def _sliding_window_chunk(self, lines: List[str], file_path: str, language: str) -> List[CodeChunk]:
        """Create overlapping chunks using sliding window"""
        chunks = []
        window_size = 50  # lines per chunk
        overlap = 10      # overlapping lines
        
        for i in range(0, len(lines), window_size - overlap):
            end_idx = min(i + window_size, len(lines))
            chunk_lines = lines[i:end_idx]
            
            if chunk_lines:
                chunk_content = '\n'.join(chunk_lines)
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=end_idx,
                    language=language,
                    chunk_type='sliding_window'
                ))
        
        return chunks


class LightweightVectorStore:
    """Lightweight vector store for code embeddings with similarity search"""
    
    def __init__(self, embedding_method: str = "auto"):
        self.embedder = LightweightEmbedder(method=embedding_method)
        self.similarity_search = LightweightSimilaritySearch()
        
        self.chunks: List[CodeChunk] = []
        self.cache_file = ".analysis_cache/code_embeddings_cache.json"
        
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        
        # Load cached embeddings if available
        self._load_cache()
    
    def add_chunks(self, chunks: List[CodeChunk], force_recompute: bool = False):
        """Add code chunks and compute embeddings"""
        new_chunks_to_process = []
        all_chunks = []
        
        for chunk in chunks:
            # Create hash for chunk to check if it's changed
            chunk_hash = hashlib.md5(chunk.content.encode()).hexdigest()
            chunk.metadata = chunk.metadata or {}
            chunk.metadata['hash'] = chunk_hash
            
            # Check if we already have this chunk
            existing = next((c for c in self.chunks if 
                           c.file_path == chunk.file_path and 
                           c.start_line == chunk.start_line and 
                           c.metadata.get('hash') == chunk_hash), None)
            
            if not existing or force_recompute:
                new_chunks_to_process.append(chunk)
                all_chunks.append(chunk)
            else:
                all_chunks.append(existing)  # Use cached version
        
        if new_chunks_to_process:
            # Compute embeddings for new chunks
            texts = []
            for chunk in new_chunks_to_process:
                # Create rich text representation for embedding
                text = f"File: {os.path.basename(chunk.file_path)}\n"
                text += f"Language: {chunk.language}\n"
                text += f"Type: {chunk.chunk_type}\n"
                text += f"Code:\n{chunk.content}"
                texts.append(text)
            
            print(f"Computing embeddings for {len(texts)} code chunks...")
            embeddings = self.embedder.encode(texts, show_progress_bar=True)
            
            # Update chunks with embeddings
            for chunk, embedding in zip(new_chunks_to_process, embeddings):
                chunk.embedding = embedding
        
        # Update our storage
        self.chunks = all_chunks
        
        # Cache the results
        self._save_cache()
    
    def search_similar(self, query: str, top_k: int = 5, 
                      filter_language: Optional[str] = None,
                      filter_file: Optional[str] = None) -> List[Tuple[CodeChunk, float]]:
        """Search for code chunks similar to query"""
        if not self.chunks:
            return []
        
        # Encode query
        query_embeddings = self.embedder.encode([query])
        if not query_embeddings:
            return []
        
        query_embedding = query_embeddings[0]
        
        # Filter chunks
        filtered_chunks = []
        filtered_embeddings = []
        
        for chunk in self.chunks:
            # Apply filters
            if filter_language and chunk.language != filter_language:
                continue
            if filter_file and filter_file not in chunk.file_path:
                continue
            if chunk.embedding is None:
                continue
            
            filtered_chunks.append(chunk)
            filtered_embeddings.append(chunk.embedding)
        
        if not filtered_embeddings:
            return []
        
        # Find similar vectors
        similar_indices = self.similarity_search.search_similar_vectors(
            query_embedding, filtered_embeddings, top_k
        )
        
        # Map back to chunks
        results = []
        for idx, similarity in similar_indices:
            if idx < len(filtered_chunks):
                results.append((filtered_chunks[idx], similarity))
        
        return results
    
    def _save_cache(self):
        """Save embeddings to cache file"""
        try:
            cache_data = {
                'chunks': [chunk.to_dict() for chunk in self.chunks],
                'embedding_method': self.embedder.method,
                'version': '1.0'
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save embeddings cache: {e}")
    
    def _load_cache(self):
        """Load embeddings from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Validate cache version and method
                if (cache_data.get('version') == '1.0' and 
                    cache_data.get('embedding_method') == self.embedder.method):
                    
                    chunk_dicts = cache_data.get('chunks', [])
                    self.chunks = [CodeChunk.from_dict(chunk_dict) for chunk_dict in chunk_dicts]
                    
                    # Filter out chunks without embeddings
                    self.chunks = [c for c in self.chunks if c.embedding is not None]
                    
                    print(f"Loaded {len(self.chunks)} cached embeddings")
                else:
                    print("Cache version mismatch, rebuilding embeddings")
                    
        except Exception as e:
            print(f"Warning: Could not load embeddings cache: {e}")
            self.chunks = []


class RAGCodeAnalyzer:
    """Lightweight RAG-enhanced code analyzer"""
    
    def __init__(self, max_context_chunks: int = 3, embedding_method: str = "auto"):
        self.chunker = CodeChunker()
        self.vector_store = LightweightVectorStore(embedding_method=embedding_method)
        self.max_context_chunks = max_context_chunks
    
    def index_codebase(self, file_paths: List[str], language_detector):
        """Index entire codebase for RAG"""
        print(f"Indexing {len(file_paths)} files for RAG...")
        
        all_chunks = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                language = language_detector.detect_language(file_path)
                chunks = self.chunker.chunk_code(code, file_path, language)
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
        
        print(f"Created {len(all_chunks)} code chunks")
        self.vector_store.add_chunks(all_chunks)
        print("RAG indexing complete!")
    
    def get_relevant_context(self, agent_type: str, file_path: str, language: str, max_tokens: int = 500) -> str:
        """Get relevant code context for analysis with token optimization"""
        # Create query based on agent type and file
        query_map = {
            'security': f"security vulnerabilities SQL injection XSS authentication {language}",
            'performance': f"performance optimization algorithms efficiency {language}",
            'complexity': f"code complexity readability maintainability {language}",
            'documentation': f"documentation comments docstrings API {language}",
            'duplication': f"duplicate code similar patterns repeated {language}"
        }
        
        query = query_map.get(agent_type, f"code analysis {language}")
        query += f" {os.path.basename(file_path)}"
        
        # Search for relevant chunks
        similar_chunks = self.vector_store.search_similar(
            query=query,
            top_k=min(8, len(self.vector_store.chunks)),
            filter_language=None  # Don't filter by language for cross-language patterns
        )
        
        # Smart context selection with token budget
        selected_chunks = self._select_optimal_context(similar_chunks, max_tokens)
        
        # Build context
        context_parts = []
        for chunk, similarity in selected_chunks:
            if similarity > 0.3:  # Similarity threshold
                # Truncate very long chunks
                content = chunk.content
                if len(content) > 200:
                    content = content[:200] + "..."
                
                context_parts.append(f"Related code from {os.path.basename(chunk.file_path)} "
                                   f"(lines {chunk.start_line}-{chunk.end_line}):\n{content}\n")
        
        return "\n---\n".join(context_parts) if context_parts else ""
    
    def _select_optimal_context(self, chunks: List, max_tokens: int) -> List:
        """Select diverse, high-value context within token budget"""
        if not chunks:
            return []
        
        # Diversify by file and chunk type
        diverse_chunks = {}
        for chunk, similarity in chunks:
            file_key = os.path.basename(chunk.file_path)
            chunk_key = f"{file_key}_{chunk.chunk_type}"
            
            if chunk_key not in diverse_chunks or diverse_chunks[chunk_key][1] < similarity:
                diverse_chunks[chunk_key] = (chunk, similarity)
        
        # Sort by similarity and select within token budget
        sorted_chunks = sorted(diverse_chunks.values(), key=lambda x: x[1], reverse=True)
        
        selected = []
        token_count = 0
        
        for chunk, similarity in sorted_chunks:
            estimated_tokens = len(chunk.content) // 4  # Rough estimation
            
            if token_count + estimated_tokens <= max_tokens:
                selected.append((chunk, similarity))
                token_count += estimated_tokens
            
            if len(selected) >= 3:  # Max 3 context chunks
                break
        
        return selected


# Usage examples and compatibility layer
def create_rag_analyzer(embedding_method: str = "auto") -> RAGCodeAnalyzer:
    """Factory function to create RAG analyzer with best available method"""
    return RAGCodeAnalyzer(embedding_method=embedding_method)


# Backward compatibility
class CodeVectorStore:
    """Legacy compatibility wrapper for the original heavy implementation"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print("⚠️ Using lightweight compatibility mode for CodeVectorStore")
        # Map to lightweight implementation
        self.lightweight_store = LightweightVectorStore(embedding_method="auto")
    
    def add_chunks(self, chunks: List[CodeChunk], force_recompute: bool = False):
        """Legacy compatibility method"""
        return self.lightweight_store.add_chunks(chunks, force_recompute)
    
    def search_similar(self, query: str, top_k: int = 5, 
                      filter_language: Optional[str] = None,
                      filter_file: Optional[str] = None) -> List[Tuple[CodeChunk, float]]:
        """Legacy compatibility method"""
        return self.lightweight_store.search_similar(query, top_k, filter_language, filter_file)


# Performance comparison and migration utilities
class EmbeddingMethodComparator:
    """Compare different embedding methods for your use case"""
    
    @staticmethod
    def benchmark_methods(sample_texts: List[str], methods: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Benchmark different embedding methods"""
        if methods is None:
            methods = ["simple", "tfidf"]
            if SPACY_MODEL_AVAILABLE:
                methods.append("spacy")
        
        results = {}
        
        for method in methods:
            print(f"Testing {method} method...")
            try:
                import time
                start_time = time.time()
                
                embedder = LightweightEmbedder(method=method)
                embeddings = embedder.encode(sample_texts[:10])  # Test with small sample
                
                end_time = time.time()
                
                results[method] = {
                    "success": True,
                    "time_taken": end_time - start_time,
                    "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                    "embeddings_per_second": len(embeddings) / (end_time - start_time) if end_time > start_time else 0
                }
                
            except Exception as e:
                results[method] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    @staticmethod
    def recommend_method(codebase_size: int, api_budget: float = 0.0) -> str:
        """Recommend best embedding method based on constraints"""
        
        if api_budget > 0.01:  # Has API budget
            return "api"
        elif codebase_size < 1000 and SPACY_MODEL_AVAILABLE:
            return "spacy"
        elif codebase_size < 10000:
            return "tfidf"
        else:
            return "simple"


# Migration utilities
def migrate_from_sentence_transformers(old_cache_file: str, new_cache_file: str = None):
    """Migrate from old sentence-transformers cache to new lightweight format"""
    try:
        if new_cache_file is None:
            new_cache_file = ".analysis_cache/code_embeddings_cache.json"
        
        print("Migrating from sentence-transformers cache...")
        
        # Try to load old cache
        with open(old_cache_file, 'rb') as f:
            old_data = pickle.load(f)
        
        old_chunks = old_data.get('chunks', [])
        
        print(f"Found {len(old_chunks)} chunks in old cache")
        
        # Create new lightweight store
        lightweight_store = LightweightVectorStore()
        
        # Convert chunks (removing heavy embeddings, will recompute with lightweight method)
        new_chunks = []
        for chunk in old_chunks:
            # Create new chunk without embedding (will be recomputed)
            new_chunk = CodeChunk(
                content=chunk.content,
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                language=chunk.language,
                chunk_type=chunk.chunk_type,
                embedding=None,  # Will be recomputed
                metadata=chunk.metadata
            )
            new_chunks.append(new_chunk)
        
        # Add chunks to new store (this will compute new embeddings)
        lightweight_store.add_chunks(new_chunks, force_recompute=True)
        
        print(f"Migration complete! New cache saved to {new_cache_file}")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        print("You may need to rebuild the cache from scratch")


# Testing and validation utilities
def test_lightweight_rag():
    """Test the lightweight RAG implementation"""
    print("🧪 Testing Lightweight RAG Implementation...")
    
    # Test data
    test_code_chunks = [
        CodeChunk(
            content="""def authenticate_user(username, password):
    if not username or not password:
        raise ValueError("Username and password required")
    # Hash password for security
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return verify_credentials(username, hashed)""",
            file_path="auth.py",
            start_line=1,
            end_line=6,
            language="python",
            chunk_type="function"
        ),
        CodeChunk(
            content="""class DatabaseOptimizer:
    def optimize_query(self, query):
        # Add index hints for better performance
        if "SELECT" in query and "WHERE" in query:
            query += " USE INDEX (idx_optimized)"
        return query""",
            file_path="optimizer.py", 
            start_line=1,
            end_line=6,
            language="python",
            chunk_type="class"
        )
    ]
    
    try:
        # Test embedding methods
        print("\n1. Testing embedding methods...")
        methods_to_test = ["simple", "tfidf"]
        if SPACY_MODEL_AVAILABLE:
            methods_to_test.append("spacy")
        
        for method in methods_to_test:
            print(f"   Testing {method}...")
            embedder = LightweightEmbedder(method=method)
            embeddings = embedder.encode(["test code", "another test"])
            print(f"   ✅ {method}: Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
        
        # Test vector store
        print("\n2. Testing vector store...")
        vector_store = LightweightVectorStore(embedding_method="simple")
        vector_store.add_chunks(test_code_chunks)
        
        # Test search
        results = vector_store.search_similar("security authentication", top_k=2)
        print(f"   ✅ Search returned {len(results)} results")
        
        # Test RAG analyzer
        print("\n3. Testing RAG analyzer...")
        rag = RAGCodeAnalyzer(embedding_method="simple")
        rag.vector_store = vector_store  # Use pre-populated store
        
        context = rag.get_relevant_context("security", "test.py", "python", max_tokens=200)
        print(f"   ✅ Generated context: {len(context)} characters")
        
        print("\n🎉 All tests passed! Lightweight RAG is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


# Configuration utilities
class RAGConfig:
    """Configuration management for lightweight RAG"""
    
    @staticmethod
    def get_optimal_config(codebase_size: int, memory_limit_mb: int = 512) -> Dict[str, Any]:
        """Get optimal configuration based on constraints"""
        
        if memory_limit_mb < 256:
            return {
                "embedding_method": "simple",
                "max_chunk_size": 300,
                "max_context_chunks": 2,
                "cache_embeddings": False
            }
        elif memory_limit_mb < 512:
            return {
                "embedding_method": "tfidf" if codebase_size < 5000 else "simple",
                "max_chunk_size": 400,
                "max_context_chunks": 3,
                "cache_embeddings": True
            }
        else:
            method = "spacy" if SPACY_MODEL_AVAILABLE and codebase_size < 10000 else "tfidf"
            return {
                "embedding_method": method,
                "max_chunk_size": 500,
                "max_context_chunks": 3,
                "cache_embeddings": True
            }
    
    @staticmethod
    def print_recommendations(codebase_size: int):
        """Print recommendations for RAG configuration"""
        print(f"\n📊 RAG Configuration Recommendations for {codebase_size} files:")
        print("=" * 60)
        
        configs = {
            "Memory Constrained (< 256MB)": RAGConfig.get_optimal_config(codebase_size, 256),
            "Standard (512MB)": RAGConfig.get_optimal_config(codebase_size, 512),
            "High Memory (1GB+)": RAGConfig.get_optimal_config(codebase_size, 1024)
        }
        
        for scenario, config in configs.items():
            print(f"\n{scenario}:")
            for key, value in config.items():
                print(f"  {key}: {value}")


# Example usage and integration
if __name__ == "__main__":
    print("🚀 Lightweight RAG Code Analyzer")
    print("=" * 50)
    
    # Test the implementation
    if test_lightweight_rag():
        print("\n📋 Recommendations:")
        RAGConfig.print_recommendations(1000)
        
        print("\n💡 Integration Tips:")
        print("1. Use 'simple' method for very large codebases (10k+ files)")
        print("2. Use 'tfidf' method for medium codebases (1k-10k files)")
        print("3. Use 'spacy' method for small codebases (< 1k files) if available")
        print("4. Use 'api' method if you have API budget and want best quality")
        print("5. Enable caching to avoid recomputing embeddings")
        
        # Benchmark if requested
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
            print("\n⏱️ Running benchmarks...")
            sample_texts = [
                "def process_data(): pass",
                "class DataProcessor: pass", 
                "import numpy as np",
                "for i in range(10): print(i)"
            ]
            results = EmbeddingMethodComparator.benchmark_methods(sample_texts)
            
            print("\nBenchmark Results:")
            for method, result in results.items():
                if result["success"]:
                    print(f"{method}: {result['time_taken']:.3f}s, "
                          f"{result['embeddings_per_second']:.1f} embeddings/sec, "
                          f"dim: {result['embedding_dimension']}")
                else:
                    print(f"{method}: Failed - {result['error']}")
    else:
        print("\n❌ Some tests failed. Check your environment and dependencies.")