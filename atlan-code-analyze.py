#!/usr/bin/env python3
"""
Atlan Code Analysis CLI Tool
A comprehensive code quality intelligence system with LangGraph multi-agent analysis

Usage:
    atlan-code-analyze --path="/path/to/code" [options]
    atlan-code-analyze --repo="https://github.com/user/repo" [options]
    atlan-code-analyze --qa --path="/path/to/code"
    atlan-code-analyze --qa --repo="https://github.com/user/repo"
"""

import os
import sys
import asyncio
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing modules
try:
    from main import LangGraphCQI
    from interactive_qa import InteractiveQASession
    from extensions.github_integration import GitHubRepoAnalyzer
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Make sure all required files are in the same directory")
    sys.exit(1)


class AtlanCodeAnalyzer:
    """Main CLI class for Atlan Code Analysis"""
    
    def __init__(self):
        self.temp_dirs = []  # Track temp directories for cleanup
        
    async def analyze_path(self, path: str, detailed: bool = False, agents: Optional[List[str]] = None,
                          enable_rag: bool = True, enable_cache: bool = True, max_files: Optional[int] = None):
        """Analyze local file or directory path"""
        
        if not os.path.exists(path):
            print(f"❌ Error: Path does not exist: {path}")
            return False
        
        print(f"🔍 Analyzing local path: {path}")
        print("=" * 60)
        
        # Initialize LangGraph CQI system
        cqi = LangGraphCQI(
            enable_rag=enable_rag,
            enable_cache=enable_cache,
            use_langgraph=True
        )
        
        try:
            if os.path.isfile(path):
                # Analyze single file
                result = await cqi.analyze_file(path, agents, detailed)
                self._print_analysis_summary(result, "file")
            else:
                # Analyze directory
                result = await cqi.analyze_directory(
                    path, agents, detailed, max_parallel=4, max_files=max_files
                )
                self._print_analysis_summary(result, "directory")
            
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            return False
    
    async def analyze_github_repo(self, repo_url: str, branch: str = "main", 
                                 detailed: bool = False, agents: Optional[List[str]] = None,
                                 enable_rag: bool = True, enable_cache: bool = True,
                                 max_files: Optional[int] = None):
        """Analyze GitHub repository"""
        
        print(f"🐙 Analyzing GitHub repository: {repo_url}")
        print(f"📝 Branch: {branch}")
        print("=" * 60)
        
        temp_dir = None
        
        try:
            # Initialize GitHub analyzer
            github_analyzer = GitHubRepoAnalyzer()
            
            # Validate repository first
            print("🔍 Validating repository...")
            validation = await github_analyzer.validate_repository(repo_url)
            
            if not validation.get('valid', False):
                print(f"❌ Repository validation failed: {validation.get('error', 'Unknown error')}")
                return False
            
            print(f"✅ Repository validated: {validation['owner']}/{validation['repo_name']}")
            if validation.get('description'):
                print(f"📄 Description: {validation['description']}")
            if validation.get('language'):
                print(f"💻 Primary language: {validation['language']}")
            
            # Download repository
            print("\n📥 Downloading repository...")
            temp_dir = await github_analyzer.download_repo(repo_url, branch)
            self.temp_dirs.append(temp_dir)
            
            # Get repository stats
            repo_stats = github_analyzer.get_repository_stats(temp_dir)
            print(f"📊 Repository stats: {repo_stats['total_files']} files, {repo_stats['total_lines']:,} lines")
            
            # Analyze the downloaded code
            print("\n🧠 Starting LangGraph analysis...")
            cqi = LangGraphCQI(
                enable_rag=enable_rag,
                enable_cache=enable_cache,
                use_langgraph=True
            )
            
            result = await cqi.analyze_directory(
                temp_dir, agents, detailed, max_parallel=4, max_files=max_files
            )
            
            # Enhance result with GitHub metadata
            result['github_metadata'] = {
                'repo_url': repo_url,
                'branch': branch,
                'validation': validation,
                'repo_stats': repo_stats
            }
            
            self._print_analysis_summary(result, "github_repo")
            return True
            
        except Exception as e:
            print(f"❌ GitHub analysis failed: {str(e)}")
            return False
        
        finally:
            # Cleanup temp directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"🧹 Cleaned up temporary directory")
                except Exception as cleanup_error:
                    print(f"⚠️ Warning: Could not clean up temp directory: {cleanup_error}")
    
    async def start_qa_session(self, path: Optional[str] = None, repo_url: Optional[str] = None, 
                              branch: str = "main", enable_rag: bool = True, run_analysis: bool = True):
        """Start interactive Q&A session"""
        
        codebase_path = None
        temp_dir = None
        
        try:
            if repo_url:
                # Handle GitHub repository
                print(f"🐙 Setting up Q&A for GitHub repository: {repo_url}")
                print("=" * 60)
                
                github_analyzer = GitHubRepoAnalyzer()
                
                # Validate and download
                validation = await github_analyzer.validate_repository(repo_url)
                if not validation.get('valid', False):
                    print(f"❌ Repository validation failed: {validation.get('error', 'Unknown error')}")
                    return False
                
                print(f"✅ Repository: {validation['owner']}/{validation['repo_name']}")
                
                print("📥 Downloading repository for Q&A...")
                temp_dir = await github_analyzer.download_repo(repo_url, branch)
                self.temp_dirs.append(temp_dir)
                codebase_path = temp_dir
                
                print(f"📁 Repository downloaded to: {temp_dir}")
                
            elif path:
                # Handle local path
                if not os.path.exists(path):
                    print(f"❌ Error: Path does not exist: {path}")
                    return False
                
                print(f"📁 Setting up Q&A for local path: {path}")
                print("=" * 60)
                codebase_path = path
            
            else:
                print("❌ Error: Either --path or --repo must be specified for Q&A")
                return False
            
            # Initialize Q&A session
            print("🧠 Initializing interactive Q&A system...")
            session = InteractiveQASession(
                codebase_path=codebase_path,
                enable_rag=enable_rag,
                run_analysis=run_analysis
            )
            
            await session.initialize()
            await session.start_session()
            
            return True
            
        except KeyboardInterrupt:
            print("\n👋 Q&A session ended by user")
            return True
        except Exception as e:
            print(f"❌ Q&A session failed: {str(e)}")
            return False
        
        finally:
            # Cleanup temp directory for GitHub repos
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"\n🧹 Cleaned up temporary directory")
                except Exception as cleanup_error:
                    print(f"⚠️ Warning: Could not clean up temp directory: {cleanup_error}")
    
    def _print_analysis_summary(self, result: dict, analysis_type: str):
        """Print analysis summary with enhanced formatting"""
        
        if 'error' in result:
            print(f"❌ Analysis Error: {result['error']}")
            return
        
        print("\n" + "="*70)
        print("📊 ANALYSIS COMPLETE")
        print("="*70)
        
        # Basic statistics
        if analysis_type == "file":
            print(f"📄 File: {os.path.basename(result.get('file_path', 'Unknown'))}")
            print(f"💻 Language: {result.get('language', 'Unknown').title()}")
            print(f"📏 Lines: {result.get('total_lines', 0):,}")
        elif analysis_type in ["directory", "github_repo"]:
            print(f"📁 Files Processed: {result.get('total_files', 0)}")
            print(f"📏 Total Lines: {result.get('total_lines', 0):,}")
            
            # Language breakdown
            languages = result.get('languages', {})
            if languages:
                print(f"💻 Languages Found: {len(languages)}")
                for lang, stats in languages.items():
                    print(f"   - {lang.title()}: {stats['files']} files, {stats['issues']} issues")
        
        # GitHub metadata
        if analysis_type == "github_repo" and result.get('github_metadata'):
            metadata = result['github_metadata']
            print(f"🐙 Repository: {metadata['repo_url']}")
            print(f"🌿 Branch: {metadata['branch']}")
            
            validation = metadata.get('validation', {})
            if validation.get('language'):
                print(f"🏷️ Primary Language: {validation['language']}")
            if validation.get('stars'):
                print(f"⭐ Stars: {validation['stars']:,}")
        
        # Issue summary
        total_issues = result.get('total_issues', 0)
        print(f"\n🚨 Issues Found: {total_issues}")
        
        severity_counts = result.get('issues_by_severity', {})
        if severity_counts:
            print("📈 Severity Breakdown:")
            for severity in ['critical', 'high', 'medium', 'low']:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
                    print(f"   {emoji[severity]} {severity.title()}: {count}")
        
        # Performance metrics
        print(f"\n⏱️ Processing Time: {result.get('processing_time', 0):.2f}s")
        print(f"🔢 LLM Tokens: {result.get('total_tokens', 0):,}")
        print(f"📞 API Calls: {result.get('total_llm_calls', 0)}")
        
        # Workflow info
        workflow_engine = result.get('workflow_engine', 'unknown')
        print(f"🧠 Engine: {workflow_engine.title()}")
        
        if result.get('completed_agents'):
            print(f"✅ Agents: {', '.join(result['completed_agents'])}")
        if result.get('failed_agents'):
            print(f"❌ Failed: {', '.join(result['failed_agents'])}")
        
        # Top issues (if detailed)
        all_issues = result.get('all_issues', [])
        if all_issues:
            print(f"\n🔍 TOP ISSUES (First 10):")
            print("-" * 70)
            
            # Sort by severity
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            sorted_issues = sorted(all_issues, 
                                 key=lambda x: severity_order.get(x.get('severity', 'low'), 3))
            
            for i, issue in enumerate(sorted_issues[:10], 1):
                severity_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
                severity = issue.get('severity', 'low')
                
                print(f"\n{i}. {severity_emoji.get(severity, '⚪')} [{severity.upper()}] {issue.get('title', 'Unknown Issue')}")
                print(f"   🤖 Agent: {issue.get('agent', 'unknown').title()}")
                
                file_path = issue.get('file_path', '')
                if file_path:
                    print(f"   📄 File: {os.path.basename(file_path)}")
                
                line_number = issue.get('line_number', issue.get('line', 0))
                if line_number and line_number > 0:
                    print(f"   📍 Line: {line_number}")
                
                print(f"   📝 {issue.get('description', 'No description')}")
                fix = issue.get('suggestion', issue.get('fix', 'No suggestion'))
                if fix and fix != 'No suggestion':
                    print(f"   💡 Fix: {fix}")
        
        print("\n" + "="*70)
    
    def cleanup(self):
        """Cleanup temporary directories"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception:
                pass


def create_parser():
    """Create argument parser"""
    
    parser = argparse.ArgumentParser(
        prog='atlan-code-analyze',
        description='Atlan Code Analysis - AI-powered code quality intelligence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze local directory
  atlan-code-analyze --path="/path/to/code" --detailed
  
  # Analyze GitHub repository
  atlan-code-analyze --repo="https://github.com/user/repo" --branch="main"
  
  # Start Q&A session for local code
  atlan-code-analyze --qa --path="/path/to/code"
  
  # Start Q&A session for GitHub repository
  atlan-code-analyze --qa --repo="https://github.com/user/repo"
  
  # Advanced analysis with specific agents
  atlan-code-analyze --path="." --agents="security,performance" --rag --detailed
        """
    )
    
    # Source specification (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--path', type=str, help='Local file or directory path to analyze')
    source_group.add_argument('--repo', type=str, help='GitHub repository URL to analyze')
    
    # Mode selection
    parser.add_argument('--qa', action='store_true', help='Start interactive Q&A session')
    
    # GitHub options
    parser.add_argument('--branch', type=str, default='main', help='GitHub branch to analyze (default: main)')
    
    # Analysis options
    parser.add_argument('--detailed', action='store_true', help='Show detailed issues report')
    parser.add_argument('--agents', type=str, help='Comma-separated list of agents (security,performance,complexity,documentation,testing,duplication)')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to analyze')
    
    # System options
    parser.add_argument('--rag', action='store_true', default=True, help='Enable RAG system (default: True)')
    parser.add_argument('--no-rag', action='store_true', help='Disable RAG system')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    
    # Utility commands
    parser.add_argument('--version', action='version', version='atlan-code-analyze 1.0.0')
    
    return parser


async def main():
    """Main CLI entry point"""
    
    # ASCII Art Banner
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                    ATLAN CODE ANALYZER                       ║
║           AI-Powered Code Quality Intelligence                ║
║                  Powered by LangGraph                        ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Parse agents
    selected_agents = None
    if args.agents:
        selected_agents = [agent.strip() for agent in args.agents.split(',')]
        valid_agents = {'security', 'performance', 'complexity', 'documentation', 'testing', 'duplication'}
        invalid_agents = [a for a in selected_agents if a not in valid_agents]
        if invalid_agents:
            print(f"❌ Invalid agents: {', '.join(invalid_agents)}")
            print(f"✅ Valid agents: {', '.join(sorted(valid_agents))}")
            return
    
    # Handle RAG settings
    enable_rag = args.rag and not args.no_rag
    enable_cache = not args.no_cache
    
    # Initialize analyzer
    analyzer = AtlanCodeAnalyzer()
    
    try:
        success = False
        
        if args.qa:
            # Q&A mode
            success = await analyzer.start_qa_session(
                path=args.path,
                repo_url=args.repo,
                branch=args.branch,
                enable_rag=enable_rag,
                run_analysis=True
            )
        
        elif args.path:
            # Local path analysis
            success = await analyzer.analyze_path(
                path=args.path,
                detailed=args.detailed,
                agents=selected_agents,
                enable_rag=enable_rag,
                enable_cache=enable_cache,
                max_files=args.max_files
            )
        
        elif args.repo:
            # GitHub repository analysis
            success = await analyzer.analyze_github_repo(
                repo_url=args.repo,
                branch=args.branch,
                detailed=args.detailed,
                agents=selected_agents,
                enable_rag=enable_rag,
                enable_cache=enable_cache,
                max_files=args.max_files
            )
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)
    finally:
        analyzer.cleanup()


if __name__ == '__main__':
    asyncio.run(main())