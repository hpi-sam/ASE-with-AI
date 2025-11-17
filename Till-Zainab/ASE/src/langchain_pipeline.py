from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
import tempfile
import os
import ast
import traceback
import logging
import datetime
from pathlib import Path

from coverage_utils import CoverageAnalyzer, TestGenerator
from perplexity_analyzer import PerplexityAnalyzer
from utils import load_config, get_openai_key
from prompt_utils import format_test_generation_prompt, format_perplexity_enhancement

class TestGenerationState(TypedDict):
    function_code: str
    oracle_function_code: str  # Ground truth function
    existing_tests: str
    target_line: Optional[int]
    target_line_content: str
    generated_test_inputs: List[Dict]
    generated_test_code: str
    coverage_info: Dict
    test_passed: bool
    test_covers_target: bool
    perplexity_analysis: Optional[Dict]
    iteration_count: int
    max_iterations: int
    perplexity_enabled: bool
    failed_attempts_enabled: bool
    thinking_enabled: bool
    messages: Annotated[List, add_messages]
    final_result: Dict
    oracle_results: Optional[List[Dict]]
    failed_attempts: List[Dict]  # Track failed test generation attempts
    total_llm_calls: int  # Track total LLM calls made

class TestGenerationPipeline:
    """Main pipeline for intelligent test generation using coverage and perplexity"""
    
    def __init__(self):
        self.config = load_config()
        self.api_key = get_openai_key()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.llm = ChatOpenAI(
            model=self.config["model"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            api_key=self.api_key
        )
        
        self.coverage_analyzer = CoverageAnalyzer()
        self.test_generator = TestGenerator()
        self.perplexity_analyzer = PerplexityAnalyzer(self.api_key, self.config["model"])
        
        # Build the graph
        self.graph = self._build_graph()
        
        self.logger.info("TestGenerationPipeline initialized successfully")
    
    def setup_logging(self):
        """Setup logging for this pipeline run"""
        # Create logs directory if it doesn't exist
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create unique log filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_file = log_dir / f"test_generation_{timestamp}.log"
        
        # Setup logger
        self.logger = logging.getLogger(f"TestGeneration_{timestamp}")
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(message)s'  # '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")
        self.logger.info(f"Configuration: {self.config}")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(TestGenerationState)
        
        # Add nodes
        workflow.add_node("analyze_coverage", self.analyze_coverage_node)
        workflow.add_node("generate_test", self.generate_test_node)
        workflow.add_node("check_coverage", self.check_coverage_node)
        workflow.add_node("run_oracle", self.run_oracle_node)
        workflow.add_node("analyze_perplexity", self.analyze_perplexity_node)
        workflow.add_node("record_failed_attempt", self.record_failed_attempt_node)
        workflow.add_node("rewrite_test", self.rewrite_test_node)
        workflow.add_node("finalize", self.finalize_node)
        
        # Add edges
        workflow.set_entry_point("analyze_coverage")
        
        workflow.add_edge("analyze_coverage", "generate_test")
        workflow.add_edge("generate_test", "check_coverage")
        
        # Conditional edges from check_coverage
        workflow.add_conditional_edges(
            "check_coverage",
            self.coverage_decision,
            {
                "run_oracle": "run_oracle",
                "analyze_perplexity": "analyze_perplexity",
                "record_failed_attempt": "record_failed_attempt",
                "finalize": "finalize"
            }
        )
        
        workflow.add_edge("run_oracle", "finalize")
        workflow.add_edge("analyze_perplexity", "rewrite_test")
        workflow.add_edge("record_failed_attempt", "rewrite_test")
        
        # Conditional edges from rewrite_test
        workflow.add_conditional_edges(
            "rewrite_test",
            self.rewrite_decision,
            {
                "check_coverage": "check_coverage",
                "finalize": "finalize"
            }
        )
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def analyze_coverage_node(self, state: TestGenerationState) -> TestGenerationState:
        """Analyze current test coverage and find next uncovered line"""
        self.logger.info("Starting coverage analysis")
        
        function_code = state["function_code"]
        existing_tests = state.get("existing_tests", "")
        
        # Get next uncovered line
        target_line = self.coverage_analyzer.get_next_uncovered_line(function_code, existing_tests)
        target_line_content = ""
        
        if target_line:
            target_line_content = self.coverage_analyzer.get_line_content(function_code, target_line)
            self.logger.info(f"Found uncovered target line: {target_line} - '{target_line_content.strip()}'")
        else:
            self.logger.info("No uncovered lines found - all lines already covered")
        
        
        self.logger.info(f"Coverage analysis completed. Target line: {target_line}")
        return {
            **state,
            "target_line": target_line,
            "target_line_content": target_line_content,
            "iteration_count": 0,
            "max_iterations": self.config["max_rewrite_iterations"],
            "perplexity_enabled": self.config["perplexity_enabled"],
            "failed_attempts_enabled": self.config["failed_attempts_enabled"]
        }
    
    def generate_test_node(self, state: TestGenerationState) -> TestGenerationState:
        """Generate test inputs for the target line"""
        self.logger.info("Starting test generation")
        
        function_code = state["function_code"]
        target_line = state["target_line"]
        target_line_content = state["target_line_content"]
        existing_tests = state.get("existing_tests", "")
        
        if target_line is None:
            self.logger.info("No target line specified - marking as complete")
            return {**state, "final_result": {"status": "complete", "message": "All lines covered"}}
        
        self.logger.info(f"Generating test for line {target_line}: '{target_line_content.strip()}'")
        
        # Extract function signature
        func_info = self.test_generator.extract_function_signature(function_code)
        self.logger.info(f"Function info extracted: {func_info}")
        
        # Create prompt for test generation
        prompt = self._create_test_generation_prompt(
            function_code, func_info, target_line, target_line_content, existing_tests
        )
        
        # Enhance prompt with perplexity analysis and/or failed attempts information
        perplexity_data = state.get("perplexity_analysis") if state.get("perplexity_enabled", False) else None
        failed_attempts = state.get("failed_attempts", []) if state.get("failed_attempts_enabled", False) else []
        
        if perplexity_data or failed_attempts:
            func_name = func_info.get("name", "unknown_function")
            perplexity_enabled = state.get("perplexity_enabled", False)
            failed_attempts_enabled = state.get("failed_attempts_enabled", False)
            thinking_enabled = state.get("thinking_enabled", False)
            
            if perplexity_data:
                self.logger.info("Enhancing prompt with perplexity analysis and failed attempts")
                prompt = self._enhance_prompt_with_perplexity(
                    prompt, perplexity_data, failed_attempts, function_code, func_name,
                    perplexity_enabled, failed_attempts_enabled, thinking_enabled
                )
            elif failed_attempts:
                self.logger.info("Enhancing prompt with failed attempts (no perplexity analysis)")
                prompt = self._enhance_prompt_with_failed_attempts(
                    prompt, failed_attempts, function_code, func_name,
                    False, failed_attempts_enabled, thinking_enabled
                )
        
        self.logger.info("START OF PROMPT" + "="*100)
        self.logger.info(f"Generated prompt: {prompt}")
        self.logger.info("END OF PROMPT" + "="*100)

        print(prompt)
        
        # Track LLM call BEFORE making it
        current_calls = state.get("total_llm_calls", 0)
        updated_calls = current_calls + 1
        self.logger.info(f"ABOUT TO MAKE LLM CALL #{updated_calls} (Main test generation call)")
        
        # Generate test using LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        llm_output = response.content
        
        self.logger.info(f"LLM CALL #{updated_calls} COMPLETED - Response length: {len(llm_output)} chars")
        self.logger.info(f"TOTAL LLM CALLS SO FAR: {updated_calls}")
        print(llm_output)
        
        # Parse test inputs from LLM output
        test_inputs = self.test_generator.parse_test_inputs(llm_output, func_info["name"])
        self.logger.info(f"Parsed {len(test_inputs)} test inputs: {test_inputs}")
        
        return {
            **state,
            "generated_test_inputs": test_inputs,
            "messages": [HumanMessage(content=prompt), response],
            "total_llm_calls": updated_calls
        }
    
    def check_coverage_node(self, state: TestGenerationState) -> TestGenerationState:
        """Check if generated test covers the target line"""
        self.logger.info("Checking test coverage")
        
        function_code = state["function_code"]
        test_inputs = state["generated_test_inputs"]
        target_line = state["target_line"]
        existing_tests = state.get("existing_tests", "")
        
        # Create test code from inputs
        func_info = self.test_generator.extract_function_signature(function_code)
        test_code = self._create_test_code(func_info["name"], test_inputs, target_line)
        self.logger.info(f"Generated test code:\n{test_code}")
        
        # Combine with existing tests
        combined_tests = f"{existing_tests}\n\n{test_code}".strip()
        
        # Run coverage analysis
        coverage_info = self.coverage_analyzer.run_coverage_analysis(function_code, combined_tests)
        self.logger.info(f"Coverage info: {coverage_info}")
        
        # Check if target line is covered
        test_covers_target = target_line in coverage_info.get("covered_lines", set())
        self.logger.info(f"Target line {target_line} covered: {test_covers_target}")
        
        return {
            **state,
            "generated_test_code": test_code,
            "coverage_info": coverage_info,
            "test_covers_target": test_covers_target
        }
    
    def run_oracle_node(self, state: TestGenerationState) -> TestGenerationState:
        """Run oracle function to get expected outputs and validate test"""
        self.logger.info("Running oracle validation")
        
        oracle_code = state["oracle_function_code"]
        test_inputs = state["generated_test_inputs"]
        
        # Extract function info
        func_info = self.test_generator.extract_function_signature(oracle_code)
        function_name = func_info["name"]
        
        test_passed = True
        oracle_results = []
        
        # Execute oracle function for each test input
        for i, test_input in enumerate(test_inputs):
            self.logger.info(f"Running oracle test case {i+1}/{len(test_inputs)}")
            try:
                # Create a safe execution environment
                exec_globals = {}
                exec(oracle_code, exec_globals)
                oracle_func = exec_globals[function_name]
                
                # Call oracle function with test inputs
                args = test_input.get("args", [])
                # Create deep copy of args to prevent state mutation between calls
                import copy
                oracle_args = copy.deepcopy(args)
                expected_output = oracle_func(*oracle_args)
                self.logger.info(f"Oracle result for args {args}: {expected_output}")
                
                oracle_results.append({
                    "args": args,
                    "expected": expected_output
                })
                
                # Now test the original function with fresh copy of args
                test_globals = {}
                exec(state["function_code"], test_globals)
                test_func = test_globals[function_name]
                test_args = copy.deepcopy(args)
                actual_output = test_func(*test_args)
                self.logger.info(f"Function result for args {args}: {actual_output}")
                
                if actual_output != expected_output:
                    test_passed = False
                    self.logger.warning(f"Test case {i+1} failed: expected {expected_output}, got {actual_output}")
                else:
                    self.logger.info(f"Test case {i+1} passed")
                    
            except Exception as e:
                test_passed = False
                error_msg = str(e)
                self.logger.error(f"Test case {i+1} failed with exception: {error_msg}")
                oracle_results.append({
                    "args": test_input.get("args", []),
                    "error": error_msg
                })
        
        self.logger.info(f"Oracle validation completed. Test passed: {test_passed}")
        
        return {
            **state,
            "test_passed": test_passed,
            "oracle_results": oracle_results
        }
    
    def analyze_perplexity_node(self, state: TestGenerationState) -> TestGenerationState:
        """Analyze perplexity to understand why test generation failed"""
        if not state["perplexity_enabled"]:
            self.logger.info("Perplexity analysis disabled - skipping")
            return state
        
        self.logger.info("Starting perplexity analysis")
        
        # Record the failed attempt before analyzing (if enabled)
        failed_attempts = state.get("failed_attempts", [])
        if state.get("failed_attempts_enabled", True):
            failed_attempt = {
                "iteration": state["iteration_count"],
                "target_line": state["target_line"],
                "generated_test_inputs": state["generated_test_inputs"],
                "generated_test_code": state["generated_test_code"],
                "coverage_info": state["coverage_info"],
                "test_covers_target": state["test_covers_target"]
            }
            
            failed_attempts.append(failed_attempt)
            self.logger.info(f"Recorded failed attempt {state['iteration_count']}: inputs={state['generated_test_inputs']}, covered_lines={state['coverage_info'].get('covered_lines', set())}")
        else:
            self.logger.info("Failed attempts tracking is disabled")
        
        # Get the last prompt used
        messages = state.get("messages", [])
        if messages:
            last_prompt = messages[0].content
            current_total_calls = state.get("total_llm_calls", 0)
            self.logger.info(f"STARTING PERPLEXITY ANALYSIS - Current total LLM calls: {current_total_calls}")
            self.logger.info("Analyzing failed test generation with perplexity")
            
            analysis = self.perplexity_analyzer.analyze_failed_test_generation(
                last_prompt, 
                self.config["top_perplexity_lines"],
                max_workers=self.config.get("perplexity_max_workers", 5)
            )
            self.logger.info(f"Perplexity analysis completed: {analysis}")
            
            # Update total LLM call count with perplexity calls
            perplexity_calls = analysis.get("perplexity_calls_made", 0)
            updated_calls = current_total_calls + perplexity_calls
            self.logger.info(f"PERPLEXITY ANALYSIS MADE {perplexity_calls} LLM CALLS")
            self.logger.info(f"UPDATING TOTAL COUNT: {current_total_calls} + {perplexity_calls} = {updated_calls}")
            
            return {
                **state,
                "perplexity_analysis": analysis,
                "failed_attempts": failed_attempts,
                "total_llm_calls": updated_calls
            }
        
        self.logger.warning("No messages found for perplexity analysis")
        return {**state, "failed_attempts": failed_attempts}
    
    def record_failed_attempt_node(self, state: TestGenerationState) -> TestGenerationState:
        """Record a failed test generation attempt without perplexity analysis"""
        self.logger.info("Recording failed test generation attempt")
        
        failed_attempts = state.get("failed_attempts", [])
        if state.get("failed_attempts_enabled", False):
            failed_attempt = {
                "iteration": state["iteration_count"],
                "target_line": state["target_line"],
                "generated_test_inputs": state["generated_test_inputs"],
                "generated_test_code": state["generated_test_code"],
                "coverage_info": state["coverage_info"],
                "test_covers_target": state["test_covers_target"]
            }
            failed_attempts.append(failed_attempt)
            self.logger.info(f"Recorded failed attempt {state['iteration_count']}: inputs={state['generated_test_inputs']}, covered_lines={state['coverage_info'].get('covered_lines', set())}")
        else:
            self.logger.warning("Failed attempts tracking is disabled but record_failed_attempt_node was called")
        
        return {**state, "failed_attempts": failed_attempts}
    
    def rewrite_test_node(self, state: TestGenerationState) -> TestGenerationState:
        """Rewrite test based on perplexity analysis"""
        updated_iteration = state["iteration_count"] + 1
        self.logger.info(f"Starting test rewrite iteration {updated_iteration}")
        
        # Update state with new iteration count - use copy and modify
        updated_state = state.copy()
        updated_state["iteration_count"] = updated_iteration
        
        # Generate new test with perplexity insights
        return self.generate_test_node(updated_state)
    
    def finalize_node(self, state: TestGenerationState) -> TestGenerationState:
        """Finalize the results"""
        self.logger.info("Starting finalization")
        
        if state.get("final_result"):
            self.logger.info("Final result already exists - returning existing result")
            return state
        
        target_line = state["target_line"]
        test_covers_target = state["test_covers_target"]
        test_passed = state.get("test_passed", False)
        
        # Get total LLM calls made
        total_llm_calls = state.get("total_llm_calls", 0)
        self.logger.info(f"PIPELINE COMPLETED - TOTAL LLM CALLS MADE: {total_llm_calls}")
        
        if target_line is None:
            result = {
                "status": "complete", 
                "message": "All lines are already covered",
                "total_llm_calls": total_llm_calls
            }
            self.logger.info("All lines already covered")
        elif test_covers_target and test_passed:
            result = {
                "status": "success",
                "message": f"Successfully generated test covering line {target_line}",
                "test_code": state["generated_test_code"],
                "target_line": target_line,
                "target_line_content": state.get("target_line_content", ""),
                "oracle_results": state.get("oracle_results", []),
                "total_llm_calls": total_llm_calls
            }
            self.logger.info(f"Successfully generated test covering line {target_line}")
        elif test_covers_target and not test_passed:
            result = {
                "status": "bug_found",
                "message": f"Test covers line {target_line} but reveals a bug in the function",
                "test_code": state["generated_test_code"],
                "target_line": target_line,
                "target_line_content": state.get("target_line_content", ""),
                "oracle_results": state.get("oracle_results", []),
                "total_llm_calls": total_llm_calls
            }
            self.logger.warning(f"Bug found! Test covers line {target_line} but function behavior differs from oracle")
        else:
            result = {
                "status": "failed",
                "message": f"Could not generate test covering line {target_line} after {state['iteration_count']} iterations",
                "target_line": target_line,
                "target_line_content": state.get("target_line_content", ""),
                "iterations": state["iteration_count"],
                "total_llm_calls": total_llm_calls
            }
            self.logger.error(f"Failed to generate test covering line {target_line} after {state['iteration_count']} iterations")
        
        self.logger.info(f"Finalization completed with status: {result['status']}")
        self.logger.info(f"FINAL TOTAL LLM CALLS: {total_llm_calls}")
        return {**state, "final_result": result}
    
    def coverage_decision(self, state: TestGenerationState) -> str:
        """Decide next step based on coverage results"""
        if state["target_line"] is None:
            self.logger.info("Coverage decision: No target line -> finalize")
            return "finalize"
        
        if state["test_covers_target"]:
            self.logger.info("Coverage decision: Target covered -> run oracle")
            return "run_oracle"
        else:
            # Target not covered - need to decide on retry strategy
            perplexity_enabled = state.get("perplexity_enabled", False)
            failed_attempts_enabled = state.get("failed_attempts_enabled", False)
            
            if perplexity_enabled:
                self.logger.info("Coverage decision: Target not covered, perplexity enabled -> analyze perplexity")
                return "analyze_perplexity"
            elif failed_attempts_enabled:
                self.logger.info("Coverage decision: Target not covered, failed attempts enabled (no perplexity) -> record failed attempt")
                return "record_failed_attempt"
            else:
                self.logger.info("Coverage decision: Target not covered, no retry mechanisms enabled -> finalize")
                return "finalize"
    
    def rewrite_decision(self, state: TestGenerationState) -> str:
        """Decide whether to rewrite test or finalize"""
        if state["iteration_count"] < state["max_iterations"]:
            self.logger.info(f"Rewrite decision: Iteration {state['iteration_count']}/{state['max_iterations']} -> check coverage")
            return "check_coverage"
        else:
            self.logger.info(f"Rewrite decision: Max iterations ({state['max_iterations']}) reached -> finalize")
            return "finalize"
    
    def _create_test_generation_prompt(self, function_code: str, func_info: Dict, 
                                     target_line: int, target_line_content: str, 
                                     existing_tests: str) -> str:
        """Create prompt for test generation"""
        return format_test_generation_prompt(
            function_code, func_info, target_line, target_line_content, existing_tests
        )
    
    def _enhance_prompt_with_perplexity(self, original_prompt: str, perplexity_analysis: Dict, failed_attempts: Optional[List[Dict]] = None, function_code: Optional[str] = None, function_name: Optional[str] = None, perplexity_enabled: bool = True, failed_attempts_enabled: bool = True, thinking_enabled: bool = True) -> str:
        """Enhance prompt with perplexity analysis insights and failed attempt information"""
        top_lines = perplexity_analysis.get("top_influential_lines", [])
        
        enhancement = format_perplexity_enhancement(
            top_lines, 
            failed_attempts or [], 
            function_code, 
            function_name,
            perplexity_enabled,
            failed_attempts_enabled,
            thinking_enabled
        )
        return enhancement + "\n\n" + original_prompt
    
    def _enhance_prompt_with_failed_attempts(self, original_prompt: str, failed_attempts: List[Dict], function_code: Optional[str] = None, function_name: Optional[str] = None, perplexity_enabled: bool = False, failed_attempts_enabled: bool = True, thinking_enabled: bool = False) -> str:
        """Enhance prompt with failed attempt information only (no perplexity analysis)"""
        # Use empty perplexity lines but include failed attempts
        enhancement = format_perplexity_enhancement(
            [], 
            failed_attempts, 
            function_code, 
            function_name,
            perplexity_enabled,
            failed_attempts_enabled,
            thinking_enabled
        )
        return enhancement + "\n\n" + original_prompt
    
    def _create_test_code(self, function_name: str, test_inputs: List[Dict], target_line: Optional[int] = None) -> str:
        """Create executable test code from test inputs"""
        test_code_lines = []
        
        for i, test_input in enumerate(test_inputs):
            args = test_input.get("args", [])
            args_str = ", ".join(repr(arg) for arg in args)
            # Wrap each test call in try-except to handle exceptions
            test_code_lines.append(f"    try:")
            test_code_lines.append(f"        result_{i} = {function_name}({args_str})")
            test_code_lines.append(f"    except Exception as e:")
            test_code_lines.append(f"        pass")
        
        # Create unique function name based on target line
        if target_line is not None:
            test_function_name = f"test_line_{target_line}"
        else:
            test_function_name = "test_generated"
        
        if test_code_lines:
            test_function = f"def {test_function_name}():\n" + "\n".join(test_code_lines)
            return test_function + f"\n\n{test_function_name}()\n"
        else:
            return f"def {test_function_name}():\n    pass\n\n{test_function_name}()"
    
    def run_pipeline(self, function_code: str, oracle_function_code: str, 
                    existing_tests: str = "") -> Dict:
        """Run the complete test generation pipeline"""
        self.logger.info("="*60)
        self.logger.info("STARTING TEST GENERATION PIPELINE")
        self.logger.info("="*60)
        self.logger.info(f"Function code length: {len(function_code)} chars")
        self.logger.info(f"Oracle code length: {len(oracle_function_code)} chars")
        self.logger.info(f"Existing tests length: {len(existing_tests)} chars")
        
        initial_state: TestGenerationState = {
            "function_code": function_code,
            "oracle_function_code": oracle_function_code,
            "existing_tests": existing_tests,
            "target_line": None,
            "target_line_content": "",
            "generated_test_inputs": [],
            "generated_test_code": "",
            "coverage_info": {},
            "test_passed": False,
            "test_covers_target": False,
            "perplexity_analysis": None,
            "iteration_count": 0,
            "max_iterations": self.config["max_rewrite_iterations"],
            "perplexity_enabled": self.config["perplexity_enabled"],
            "failed_attempts_enabled": self.config["failed_attempts_enabled"],
            "thinking_enabled": self.config.get("thinking_enabled", False),
            "messages": [],
            "final_result": {},
            "oracle_results": None,
            "failed_attempts": [],
            "total_llm_calls": 0
        }
        
        try:
            self.logger.info("Invoking LangGraph workflow")
            final_state = self.graph.invoke(initial_state)
            result = final_state["final_result"]
            
            self.logger.info("="*60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Final status: {result.get('status', 'unknown')}")
            self.logger.info(f"Final message: {result.get('message', 'N/A')}")
            self.logger.info("="*60)
            
            return result
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            self.logger.error("="*60)
            self.logger.error("PIPELINE EXECUTION FAILED")
            self.logger.error(f"Error: {error_msg}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            self.logger.error("="*60)
            
            return {
                "status": "error",
                "message": error_msg,
                "traceback": traceback.format_exc()
            }
