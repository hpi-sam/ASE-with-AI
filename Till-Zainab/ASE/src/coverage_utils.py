import subprocess
import tempfile
import os
import json
import ast
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

class CoverageAnalyzer:
    """Analyze code coverage using coverage.py"""
    
    def __init__(self):
        self.temp_dir: Optional[str] = None
        self.function_file: Optional[str] = None
        self.test_file: Optional[str] = None
    
    def setup_temp_files(self, function_code: str, test_code: str = "") -> None:
        """Setup temporary files for coverage analysis"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create function file
        self.function_file = os.path.join(self.temp_dir, "function_to_test.py")
        with open(self.function_file, 'w') as f:
            f.write(function_code)
        
        # Create test file
        self.test_file = os.path.join(self.temp_dir, "test_coverage.py")
        test_content = f"""
import sys
sys.path.append('{self.temp_dir}')
from function_to_test import *

{test_code}
"""
        with open(self.test_file, 'w') as f:
            f.write(test_content)
    
    def run_coverage_analysis(self, function_code: str, test_code: str) -> Dict:
        """Run coverage analysis and return coverage information"""
        self.setup_temp_files(function_code, test_code)
        
        if not self.temp_dir or not self.test_file or not self.function_file:
            return {"covered_lines": set(), "total_lines": set(), "missing_lines": set()}
        
        try:
            # Initialize coverage
            subprocess.run([
                'coverage', 'erase'
            ], cwd=self.temp_dir, capture_output=True)
            
            # Run tests with coverage
            result = subprocess.run([
                'coverage', 'run', '--source=.', self.test_file
            ], cwd=self.temp_dir, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Test execution failed: {result.stderr}")
                return {"covered_lines": set(), "total_lines": set(), "missing_lines": set()}
            
            # Generate coverage report
            report_result = subprocess.run([
                'coverage', 'json', '--omit=test_*'
            ], cwd=self.temp_dir, capture_output=True, text=True)
            
            if report_result.returncode != 0:
                print(f"Coverage report failed: {report_result.stderr}")
                return {"covered_lines": set(), "total_lines": set(), "missing_lines": set()}
            
            # Read coverage.json
            coverage_file = os.path.join(self.temp_dir, "coverage.json")
            if os.path.exists(coverage_file):
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                
                # Extract coverage for our function file
                function_filename = os.path.basename(self.function_file)
                file_coverage = None
                
                for filename, data in coverage_data['files'].items():
                    if filename.endswith(function_filename):
                        file_coverage = data
                        break
                
                if file_coverage:
                    executed_lines = set(file_coverage['executed_lines'])
                    missing_lines = set(file_coverage['missing_lines'])
                    total_lines = executed_lines | missing_lines
                    
                    return {
                        "covered_lines": executed_lines,
                        "total_lines": total_lines,
                        "missing_lines": missing_lines
                    }
            
            return {"covered_lines": set(), "total_lines": set(), "missing_lines": set()}
            
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
    
    def get_next_uncovered_line(self, function_code: str, existing_test_code: str = "") -> Optional[int]:
        """Get the next uncovered line number"""
        coverage_info = self.run_coverage_analysis(function_code, existing_test_code)
        missing_lines = coverage_info.get("missing_lines", set())
        
        if missing_lines:
            return min(missing_lines)  # Return the first uncovered line
        return None
    
    def get_line_content(self, function_code: str, line_number: int) -> str:
        """Get the content of a specific line"""
        lines = function_code.split('\n')
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1].strip()
        return ""

class TestGenerator:
    """Generate test inputs for functions"""
    
    def extract_function_signature(self, function_code: str) -> Dict:
        """Extract function name and parameters from function code"""
        try:
            tree = ast.parse(function_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    params = []
                    for arg in node.args.args:
                        params.append(arg.arg)
                    
                    # Handle default values
                    defaults = []
                    if node.args.defaults:
                        defaults = [ast.literal_eval(default) if isinstance(default, (ast.Constant, ast.Num, ast.Str)) 
                                  else None for default in node.args.defaults]
                    
                    return {
                        "name": node.name,
                        "params": params,
                        "defaults": defaults
                    }
        except Exception as e:
            print(f"Error parsing function: {e}")
        
        return {"name": "", "params": [], "defaults": []}
    
    def parse_test_inputs(self, llm_output: str, function_name: str) -> List[Dict]:
        """Parse LLM output to extract test inputs"""
        test_cases = []
        
        # Remove <think>...</think> sections from the output before parsing
        cleaned_output = re.sub(r'<think>.*?</think>', '', llm_output, flags=re.DOTALL | re.IGNORECASE)
        
        # Look for function calls or parameter specifications
        patterns = [
            rf"{function_name}\s*\((.*?)\)",  # function_name(args)
            r"args?\s*[:=]\s*\[(.*?)\]",     # args: [...]
            r"inputs?\s*[:=]\s*\[(.*?)\]",   # inputs: [...]
            r"parameters?\s*[:=]\s*\[(.*?)\]" # parameters: [...]
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, cleaned_output, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    # Try to evaluate the arguments
                    args_str = match.strip()
                    if args_str:
                        # Create a proper function call and extract args using ast
                        temp_func_call = f"temp_func({args_str})"
                        try:
                            # Parse as AST to safely extract arguments
                            tree = ast.parse(temp_func_call)
                            if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Call):
                                call_node = tree.body[0].value
                                args = []
                                for arg in call_node.args:
                                    args.append(ast.literal_eval(arg))
                                test_cases.append({"args": args})
                        except:
                            # Fallback to eval if AST fails
                            try:
                                args = eval(f"[{args_str}]") if ',' in args_str else [eval(args_str)]
                                test_cases.append({"args": args})
                            except:
                                pass
                except:
                    continue
        
        # If no patterns matched, try to extract from text
        if not test_cases:
            lines = cleaned_output.split('\n')
            for line in lines:
                if function_name in line and '(' in line and ')' in line:
                    try:
                        # Extract function call using more robust method
                        start = line.find(function_name + '(')
                        if start >= 0:
                            # Find matching closing parenthesis
                            open_count = 0
                            end = start + len(function_name) + 1
                            for i in range(end, len(line)):
                                if line[i] == '(':
                                    open_count += 1
                                elif line[i] == ')':
                                    if open_count == 0:
                                        end = i
                                        break
                                    open_count -= 1
                            
                            args_str = line[start + len(function_name) + 1:end]
                            if args_str.strip():
                                # Use AST to safely parse arguments
                                temp_func_call = f"temp_func({args_str})"
                                try:
                                    tree = ast.parse(temp_func_call)
                                    if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Call):
                                        call_node = tree.body[0].value
                                        args = []
                                        for arg in call_node.args:
                                            args.append(ast.literal_eval(arg))
                                        test_cases.append({"args": args})
                                except:
                                    # Fallback to eval
                                    try:
                                        args = eval(f"[{args_str}]") if ',' in args_str else [eval(args_str)]
                                        test_cases.append({"args": args})
                                    except:
                                        pass
                    except:
                        continue
        
        return test_cases if test_cases else [{"args": []}]  # Return empty args if nothing found
