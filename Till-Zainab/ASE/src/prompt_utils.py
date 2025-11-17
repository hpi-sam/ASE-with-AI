"""Utility functions for loading prompt templates"""
import os
import re
import ast
from pathlib import Path


def load_prompt_template(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory"""
    current_dir = Path(__file__).parent
    prompts_dir = current_dir / "prompts"
    prompt_file = prompts_dir / f"{prompt_name}.txt"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()


def add_line_numbers_to_code(code: str) -> str:
    """Add line numbers to code for better prompt context"""
    lines = code.split('\n')
    numbered_lines = []
    
    for i, line in enumerate(lines, 1):
        numbered_lines.append(f"{i:3d}: {line}")
    
    return '\n'.join(numbered_lines)


def extract_function_calls_from_tests(test_code: str, function_name: str) -> list:
    """Extract simple function calls from test code, excluding wrappers and comments"""
    if not test_code.strip():
        return []
    
    function_calls = []
    
    # Pattern to match function calls like function_name(args) - handle nested parentheses
    escaped_name = re.escape(function_name)
    
    # Find all function calls with proper parentheses matching
    lines = test_code.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Look for function name followed by opening parenthesis
        start_pattern = f'{escaped_name}\\s*\\('
        match = re.search(start_pattern, line)
        if match:
            start_pos = match.start()
            paren_start = match.end() - 1  # Position of opening parenthesis
            
            # Find matching closing parenthesis
            paren_count = 0
            pos = paren_start
            while pos < len(line):
                if line[pos] == '(':
                    paren_count += 1
                elif line[pos] == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        # Found matching closing parenthesis
                        function_call = line[start_pos:pos+1]
                        # Clean up whitespace
                        clean_call = re.sub(r'\s+', ' ', function_call.strip())
                        if clean_call not in [call for call in function_calls]:
                            function_calls.append(clean_call)
                        break
                pos += 1
    
    return function_calls


def format_test_generation_prompt(function_code: str, func_info: dict, 
                                target_line: int, target_line_content: str, 
                                existing_tests: str) -> str:
    """Format the test generation prompt with given parameters"""
    template = load_prompt_template("test_generation")
    
    # Add line numbers to the function code for better LLM understanding
    numbered_function_code = add_line_numbers_to_code(function_code)
    
    # Extract simple function calls from existing tests
    existing_tests_formatted = ""
    if existing_tests.strip():
        function_calls = extract_function_calls_from_tests(existing_tests, func_info['name'])
        if function_calls:
            existing_tests_formatted = "\n".join(function_calls)
        else:
            existing_tests_formatted = "No existing tests"
    else:
        existing_tests_formatted = "No existing tests"
    
    return template.format(
        function_code=numbered_function_code,
        func_info_name=func_info['name'],
        func_info_params=func_info['params'],
        target_line=target_line,
        target_line_content=target_line_content,
        existing_tests=existing_tests_formatted
    )


def format_perplexity_enhancement(top_influential_lines: list, failed_attempts = None, function_code = None, function_name = None, perplexity_enabled = True, failed_attempts_enabled = True, thinking_enabled = True) -> str:
    """Format the perplexity analysis prompt based on selected options"""
    
    # Determine which template to use based on enabled options
    if perplexity_enabled and failed_attempts_enabled and thinking_enabled:
        template = load_prompt_template("perplexity_failed_thinking")
    elif perplexity_enabled and failed_attempts_enabled and not thinking_enabled:
        template = load_prompt_template("perplexity_failed")
    elif perplexity_enabled and not failed_attempts_enabled and not thinking_enabled:
        template = load_prompt_template("perplexity")
    elif not perplexity_enabled and failed_attempts_enabled and thinking_enabled:
        template = load_prompt_template("failed_thinking")
    else:
        # This should not happen with validation, but provide fallback
        template = load_prompt_template("perplexity")
    
    # Handle empty perplexity lines explicitly
    if top_influential_lines and perplexity_enabled:
        lines_text = "\n".join(f"- {line}" for line in top_influential_lines[:3])
    else:
        lines_text = "(No perplexity analysis available - focus on failed attempts below)"
    
    # Format failed attempts information
    failed_attempts_text = ""
    if failed_attempts and failed_attempts_enabled:
        failed_attempts_text = "\n\nPrevious Failed Attempts:\n"
        for attempt in failed_attempts[-2:]:  # Show last 2 attempts
            iteration = attempt.get("iteration", "unknown")
            test_inputs = attempt.get("generated_test_inputs", [])
            covered_lines = attempt.get("coverage_info", {}).get("covered_lines", set())
            target_line = attempt.get("target_line", "unknown")
            
            failed_attempts_text += f"Attempt {iteration} (targeting line {target_line}):\n"
            
            # Format function calls instead of just args
            if test_inputs and function_name:
                function_calls = []
                for test_input in test_inputs:
                    args = test_input.get("args", [])
                    args_str = ", ".join(repr(arg) for arg in args)
                    function_calls.append(f"{function_name}({args_str})")
                failed_attempts_text += f"  Generated function calls:\n"
                for call in function_calls:
                    failed_attempts_text += f"    {call}\n"
            else:
                failed_attempts_text += f"  No valid function calls generated\n"
            
            # Format covered lines with actual content
            if covered_lines and function_code:
                failed_attempts_text += f"  Lines covered instead:\n"
                function_lines = function_code.split('\n')
                for line_num in sorted(list(covered_lines)):
                    if 1 <= line_num <= len(function_lines):
                        line_content = function_lines[line_num - 1].strip()
                        failed_attempts_text += f"    Line {line_num}: {line_content}\n"
                    else:
                        failed_attempts_text += f"    Line {line_num}: (content not available)\n"
            else:
                failed_attempts_text += f"  No lines covered\n"
            failed_attempts_text += "\n"
    
    # Format the template based on which parameters it expects
    if not perplexity_enabled and failed_attempts_enabled and thinking_enabled:
        # Failed attempts + thinking template (no perplexity)
        return template.format(
            failed_attempts_info=failed_attempts_text
        )
    elif failed_attempts_enabled or thinking_enabled:
        # Templates that include both perplexity and failed attempts info
        return template.format(
            top_influential_lines=lines_text,
            failed_attempts_info=failed_attempts_text
        )
    else:
        # Simple perplexity template
        return template.format(
            top_influential_lines=lines_text
        )
