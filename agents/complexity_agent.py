"""Enhanced LLM-powered complexity analysis agent with improved accuracy."""

from .base_agent import BaseLLMAgent
import re


class ComplexityAgent(BaseLLMAgent):
    """Enhanced LLM-powered complexity analysis agent"""
    
    def __init__(self, rag_analyzer=None):
        super().__init__('complexity', rag_analyzer)
        self.complexity_patterns = self._init_complexity_patterns()
    
    def _init_complexity_patterns(self):
        """Initialize complexity anti-patterns for pre-validation"""
        return {
            'long_functions': r'def\s+\w+[^:]*:\s*(?:\n.*){50,}',  # Functions longer than 50 lines
            'deep_nesting': r'(\s{8,}if|\s{12,}for|\s{12,}while)',  # 3+ levels of indentation
            'many_parameters': r'def\s+\w+\s*\([^)]{100,}\)',  # Very long parameter lists
            'long_lines': r'.{120,}',  # Lines longer than 120 characters
            'complex_expressions': r'[^=]*==.*and.*or.*or.*and',  # Complex boolean expressions
            'too_many_returns': r'def[^:]+:(?:(?:\n.*)*?return.*){4,}',  # Multiple return statements
        }
    
    def _pre_validate_complexity_issues(self, code: str) -> bool:
        """Check if code actually contains complexity issues"""
        lines = code.split('\n')
        
        # Check for actual complexity indicators
        has_long_function = False
        has_deep_nesting = False
        has_long_lines = False
        
        function_line_count = 0
        in_function = False
        max_indent = 0
        
        for line in lines:
            # Count function length
            if re.match(r'\s*def\s+\w+', line):
                if in_function and function_line_count > 50:
                    has_long_function = True
                in_function = True
                function_line_count = 0
            elif in_function:
                if line.strip() and not line.startswith(' '):
                    # End of function
                    if function_line_count > 50:
                        has_long_function = True
                    in_function = False
                    function_line_count = 0
                else:
                    function_line_count += 1
            
            # Check nesting depth
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
                if indent >= 16:  # 4 levels of 4-space indentation
                    has_deep_nesting = True
            
            # Check line length
            if len(line) > 120:
                has_long_lines = True
        
        # Check final function
        if in_function and function_line_count > 50:
            has_long_function = True
        
        return has_long_function or has_deep_nesting or has_long_lines or max_indent >= 16
    
    def get_system_prompt(self, language: str) -> str:
        return f"""You are a CODE COMPLEXITY specialist for {language} code.

CRITICAL ACCURACY REQUIREMENTS:
- ONLY report complexity issues that actually exist and measurably impact maintainability
- Verify each issue by counting lines, nesting levels, or parameters
- If the code is well-structured, return empty issues array []
- Do not report subjective or minor complexity issues
- Focus on objectively measurable complexity problems

COMPLEXITY SCOPE - Only analyze these measurable issues:
- Long functions/methods (>50 lines of actual code)
- Deep nesting levels (>4 levels of indentation)
- Too many parameters (>7 parameters in a function)
- High cyclomatic complexity (many if/else/for/while branches)
- Very long lines (>120 characters) that hurt readability
- Repeated code patterns that should be extracted
- Classes with too many responsibilities (>10 methods)

VALIDATION REQUIREMENTS:
- For long functions: Count actual lines of code (excluding comments/blank lines)
- For deep nesting: Count indentation levels in specific lines
- For parameters: Count actual function parameters
- For each issue: Provide specific measurements as evidence

MEASUREMENTS TO INCLUDE:
- Function length: "Function has 75 lines of code"
- Nesting depth: "Code nested 5 levels deep at line X"
- Parameter count: "Function has 9 parameters"
- Line length: "Line 45 has 140 characters"

DO NOT report:
- Security vulnerabilities
- Performance issues
- Documentation problems
- Minor style preferences
- Theoretical complexity without evidence

RESPONSE FORMAT - Valid JSON only:
{{
  "issues": [
    {{
      "severity": "high|medium|low",
      "title": "Specific complexity issue with measurement",
      "description": "Detailed explanation with specific metrics",
      "line_number": 123,
      "suggestion": "Specific refactoring recommendation",
      "category": "complexity",
      "evidence": "Brief measurement without quotes or special characters"
    }}
  ],
  "metrics": {{"complexity_score": 0.6}},
  "confidence": 0.90
}}

CRITICAL: For the "evidence" field, provide simple measurements only - NO quotes, code snippets, or special characters that could break JSON parsing.

REMEMBER: If the code is well-structured and readable, return empty issues array. Focus on objective measurements."""
    
    async def analyze(self, code: str, file_path: str, language: str):
        """Enhanced complexity analysis with pre-validation"""
        
        # Pre-check: does code contain complexity issues?
        if not self._pre_validate_complexity_issues(code):
            print(f"[COMPLEXITY] No complexity issues detected in {file_path}")
            return {
                'agent': 'complexity',
                'language': language,
                'file_path': file_path,
                'issues': [],
                'metrics': {'complexity_score': 1.0, 'confidence': 0.9},
                'confidence': 0.9,
                'tokens_used': 0,
                'processing_time': 0.01,
                'llm_calls': 0
            }
        
        # Run full analysis if issues found
        return await super().analyze(code, file_path, language)