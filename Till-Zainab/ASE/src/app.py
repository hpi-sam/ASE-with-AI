import streamlit as st
import json
import os
import sys
from pathlib import Path
from langchain_pipeline import TestGenerationPipeline
from utils import load_config

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

def main():
    st.set_page_config(
        page_title="Unit Test Generator",
        layout="wide"
    )
    
    st.title("Unit Test Generator")
    st.markdown("Generate unit tests using coverage analysis and perplexity-guided optimization.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Load current config
        config = load_config()
        
        # Configuration options
        model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "o1-mini"],
            index=0 if config["model"] == "gpt-4o-mini" else 1
        )
        
        st.subheader("Perplexity Analysis Options")
        st.caption("Select components to include in perplexity analysis:")
        
        perplexity_enabled = st.checkbox(
            "Perplexity",
            value=config.get("perplexity_enabled", False),
            help="Include perplexity analysis of influential prompt lines"
        )
        
        failed_attempts_enabled = st.checkbox(
            "Failed Attempts",
            value=config.get("failed_attempts_enabled", False),
            help="Include information about previous failed test attempts"
        )
        
        thinking_enabled = st.checkbox(
            "Thinking",
            value=config.get("thinking_enabled", False),
            help="Include step-by-step reasoning in the prompt"
        )
        
        max_iterations = st.slider(
            "Max Rewrite Iterations",
            min_value=1,
            max_value=10,
            value=config["max_rewrite_iterations"],
            help="Maximum number of times to rewrite tests using perplexity analysis"
        )
        
        top_perplexity_lines = st.slider(
            "Top Perplexity Lines",
            min_value=3,
            max_value=10,
            value=config["top_perplexity_lines"],
            help="Number of most influential lines to consider in perplexity analysis"
        )
        
        max_pipeline_iterations = st.slider(
            "Max Pipeline Iterations",
            min_value=1,
            max_value=20,
            value=config.get("max_pipeline_iterations", 10),
            help="Maximum number of pipeline runs for multiple test generation"
        )
        
        perplexity_max_workers = st.slider(
            "Perplexity Max Workers",
            min_value=1,
            max_value=40,
            value=config.get("perplexity_max_workers", 10),
            help="Maximum number of parallel workers for perplexity analysis"
        )
        
        # Validate perplexity configuration
        valid_combinations = [
            (True, False, False),  # perplexity only
            (True, True, False),   # perplexity + failed attempts
            (True, True, True),    # perplexity + failed attempts + thinking
            (False, True, True),   # failed attempts + thinking (no perplexity)
            (False, False, False)  # none (baseline)
        ]
        
        current_combination = (perplexity_enabled, failed_attempts_enabled, thinking_enabled)
        
        if current_combination not in valid_combinations:
            st.error("❌ Invalid combination! Valid options are:\n- None selected (baseline)\n- Perplexity only\n- Perplexity + Failed Attempts\n- Perplexity + Failed Attempts + Thinking\n- Failed Attempts + Thinking (no perplexity)")
        
        # Update config button
        if st.button("Update Configuration"):
            if current_combination not in valid_combinations:
                st.error("Cannot save invalid configuration. Please select a valid combination.")
            else:
                new_config = {
                    **config,
                    "model": model,
                    "perplexity_enabled": perplexity_enabled,
                    "failed_attempts_enabled": failed_attempts_enabled,
                    "thinking_enabled": thinking_enabled,
                    "max_rewrite_iterations": max_iterations,
                    "top_perplexity_lines": top_perplexity_lines,
                    "max_pipeline_iterations": max_pipeline_iterations,
                    "perplexity_max_workers": perplexity_max_workers
                }
                
                # Save updated config
                config_path = Path(__file__).parent / "config.json"
                with open(config_path, 'w') as f:
                    json.dump(new_config, f, indent=2)
                
                st.success("Configuration updated!")
                st.rerun()
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Function to Test")
        function_code = st.text_area(
            label="Function to test:",
            label_visibility="hidden",
            value="""def calculate_grade(score, max_score=100):

    if score > max_score:
        return "A+"

    percentage = (score / max_score) * 100

    if percentage >= 90:
        return "A"
    elif percentage >= 80:
        return "B"
    elif percentage >= 70:
        return "C"
    elif percentage >= 60:
        return "D"
    else:
        return "F\"""",
            height=300,
        )
    
    with col2:
        st.header("Correct Oracle Function")
        oracle_function_code = st.text_area(
            key=1,
            label="Oracle function:",
            label_visibility="hidden",
            value="""def calculate_grade(score, max_score=100):

    if score > max_score:
        return "A+"

    percentage = (score / max_score) * 100

    if percentage >= 90:
        return "A"
    elif percentage >= 80:
        return "B"
    elif percentage >= 70:
        return "C"
    elif percentage >= 60:
        return "D"
    else:
        return "F\"""",
            height=300,
        )
    
    # Action buttons
    col1, col2, col3, col4 = st.columns([1, 1.5, 1, 1])
    
    with col1:
        if st.button("Generate Test", type="secondary"):
            if function_code.strip() and oracle_function_code.strip():
                generate_test(function_code, oracle_function_code, "")
            else:
                st.error("Please provide both function code and oracle function code")
    
    with col2:
        if st.button("Generate Multiple Tests", type="secondary"):
            if function_code.strip() and oracle_function_code.strip():
                generate_multiple_tests(function_code, oracle_function_code, "")
            else:
                st.error("Please provide both function code and oracle function code")
    
    # Results section
    if "results" in st.session_state:
        st.header("Results")
        display_results(st.session_state.results)

def generate_test(function_code: str, oracle_function_code: str, existing_tests: str):
    """Generate test using the pipeline"""
    with st.spinner("Generating test... This will take a moment."):
        try:
            # Initialize pipeline
            pipeline = TestGenerationPipeline()
            
            # Run pipeline
            result = pipeline.run_pipeline(
                function_code=function_code,
                oracle_function_code=oracle_function_code,
                existing_tests=existing_tests
            )
            
            # Store results in session state
            st.session_state.results = result
            
        except Exception as e:
            st.error(f"Error generating test: {str(e)}")
            st.session_state.results = {
                "status": "error",
                "message": str(e)
            }

def generate_multiple_tests(function_code: str, oracle_function_code: str, existing_tests: str):
    """Generate multiple tests to achieve maximum coverage"""
    from coverage_utils import CoverageAnalyzer
    
    config = load_config()
    max_iterations = config.get("max_pipeline_iterations", 10)
    
    with st.spinner(f"Generating multiple tests... (max {max_iterations} iterations)"):
        try:
            # Initialize components
            pipeline = TestGenerationPipeline()
            coverage_analyzer = CoverageAnalyzer()
            
            # Track accumulated tests and results
            accumulated_tests = existing_tests
            all_results = []
            iteration = 0
            total_llm_calls = 0  # Track total LLM calls across all iterations
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while iteration < max_iterations:
                iteration += 1
                status_text.text(f"Running iteration {iteration}/{max_iterations}...")
                progress_bar.progress(iteration / max_iterations)
                
                # Check current coverage
                coverage_info = coverage_analyzer.run_coverage_analysis(function_code, accumulated_tests)
                missing_lines = coverage_info.get("missing_lines", set())
                total_lines = coverage_info.get("total_lines", set())
                covered_lines = coverage_info.get("covered_lines", set())
                
                # Calculate coverage percentage
                if total_lines:
                    coverage_percentage = len(covered_lines) / len(total_lines) * 100
                else:
                    coverage_percentage = 100
                
                status_text.text(f"Iteration {iteration}/{max_iterations} - Coverage: {coverage_percentage:.1f}%")
                
                # Check if we've reached 100% coverage
                if not missing_lines or coverage_percentage >= 100:
                    status_text.text(f"✅ 100% coverage achieved in {iteration-1} iterations!")
                    break
                
                # Get the next uncovered line
                target_line = min(missing_lines)
                target_content = coverage_analyzer.get_line_content(function_code, target_line)
                
                # Run pipeline for this target line
                result = pipeline.run_pipeline(
                    function_code=function_code,
                    oracle_function_code=oracle_function_code,
                    existing_tests=accumulated_tests
                )
                
                # Add target line information to result
                result["target_line"] = target_line
                result["target_line_content"] = target_content
                result["iteration"] = iteration
                result["coverage_before"] = coverage_percentage
                
                # Track LLM calls from this iteration
                iteration_calls = result.get("total_llm_calls", 0)
                previous_total = total_llm_calls
                total_llm_calls += iteration_calls
                
                print(f"ITERATION {iteration} COMPLETED:")
                print(f"  - Iteration LLM calls: {iteration_calls}")
                print(f"  - Previous total: {previous_total}")
                print(f"  - New total: {total_llm_calls}")
                
                # Check if the test was successful
                if result.get("status") == "success":
                    # Add comment with target line information
                    new_test = result.get("test_code", "")
                    if new_test.strip():
                        commented_test = f"\n# Test for line {target_line}: {target_content}\n{new_test}"
                        accumulated_tests += commented_test
                        
                        # Update coverage after adding the new test
                        new_coverage_info = coverage_analyzer.run_coverage_analysis(function_code, accumulated_tests)
                        new_covered_lines = new_coverage_info.get("covered_lines", set())
                        new_coverage_percentage = len(new_covered_lines) / len(total_lines) * 100 if total_lines else 100
                        result["coverage_after"] = new_coverage_percentage
                        
                        all_results.append(result)
                else:
                    # Failed to generate test for this line, break the loop
                    result["coverage_after"] = coverage_percentage
                    all_results.append(result)
                    status_text.text(f"❌ Failed to generate test for line {target_line}, stopping.")
                    break
            
            # Final coverage check
            final_coverage_info = coverage_analyzer.run_coverage_analysis(function_code, accumulated_tests)
            final_covered_lines = final_coverage_info.get("covered_lines", set())
            final_total_lines = final_coverage_info.get("total_lines", set())
            final_coverage_percentage = len(final_covered_lines) / len(final_total_lines) * 100 if final_total_lines else 100
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Completed! Final coverage: {final_coverage_percentage:.1f}%")
            
            print(f"\n" + "="*60)
            print(f"MULTIPLE TEST GENERATION SUMMARY")
            print(f"="*60)
            print(f"Total iterations: {iteration}")
            print(f"Final coverage: {final_coverage_percentage:.1f}%")
            print(f"TOTAL LLM CALLS ACROSS ALL ITERATIONS: {total_llm_calls}")
            print(f"="*60)
            
            # Store results in session state
            st.session_state.results = {
                "status": "multiple_tests_success",
                "accumulated_tests": accumulated_tests,
                "all_results": all_results,
                "total_iterations": iteration,
                "final_coverage": final_coverage_percentage,
                "final_coverage_info": final_coverage_info,
                "total_llm_calls": total_llm_calls
            }
            
        except Exception as e:
            st.error(f"Error generating multiple tests: {str(e)}")
            st.session_state.results = {
                "status": "error",
                "message": str(e)
            }

def display_results(result: dict):
    """Display the test generation results"""
    status = result.get("status", "unknown")
    
    if status == "success":
        st.success("✅ Successfully generated test covering the target line!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Generated Test")
            st.code(result.get("test_code", ""), language="python")
            
            st.subheader("Target Information")
            target_line = result.get('target_line', 'N/A')
            target_content = result.get('target_line_content', '').strip()
            if target_content:
                st.info(f"**Target Line {target_line}:** {target_content}")
            else:
                st.info(f"**Target Line:** {target_line}")
            
            # Show LLM calls
            total_calls = result.get("total_llm_calls", 0)
            st.metric("Total LLM Calls", total_calls)
        
        with col2:
            st.subheader("Oracle Results")
            oracle_results = result.get("oracle_results", [])
            if oracle_results:
                for i, oracle_result in enumerate(oracle_results):
                    with st.expander(f"Test Case {i+1}"):
                        st.write("**Arguments:**", oracle_result.get("args", []))
                        st.write("**Expected Output:**", oracle_result.get("expected", "N/A"))
    
    elif status == "bug_found":
        st.warning("⚠️ The test sucessfully generated. However it did not pass, therefore the function has a bug.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Generated Test")
            st.code(result.get("test_code", ""), language="python")
            
            st.subheader("Target Information")
            target_line = result.get('target_line', 'N/A')
            target_content = result.get('target_line_content', '').strip()
            if target_content:
                st.info(f"**Target Line {target_line}:** {target_content}")
            else:
                st.info(f"**Target Line:** {target_line}")
            
            # Show LLM calls
            total_calls = result.get("total_llm_calls", 0)
            st.metric("Total LLM Calls", total_calls)
        
        with col2:
            st.subheader("Bug Details")
            oracle_results = result.get("oracle_results", [])
            if oracle_results:
                for i, oracle_result in enumerate(oracle_results):
                    with st.expander(f"Failed Test Case {i+1}"):
                        st.write("**Arguments:**", oracle_result.get("args", []))
                        if "expected" in oracle_result:
                            st.write("**Expected Output:**", oracle_result.get("expected"))
                        if "error" in oracle_result:
                            st.error(f"**Error:** {oracle_result.get('error')}")
    
    elif status == "complete":
        st.info("ℹ️ All lines are already covered by existing tests!")
        st.write(result.get("message", ""))
        
        # Show LLM calls
        total_calls = result.get("total_llm_calls", 0)
        st.metric("Total LLM Calls", total_calls)
    
    elif status == "failed":
        st.error("❌ Failed to generate test covering the target line")
        st.write(f"**Message:** {result.get('message', '')}")
        target_line = result.get('target_line', 'N/A')
        target_content = result.get('target_line_content', '').strip()
        if target_content:
            st.write(f"**Target Line {target_line}:** {target_content}")
        else:
            st.write(f"**Target Line:** {target_line}")
        st.write(f"**Iterations Attempted:** {result.get('iterations', 0)}")
        
        # Show LLM calls
        total_calls = result.get("total_llm_calls", 0)
        st.metric("Total LLM Calls", total_calls)
    
    elif status == "error":
        st.error("💥 Pipeline execution error")
        st.write(f"**Error:** {result.get('message', '')}")
        
        # Show LLM calls (may be 0 if error occurred early)
        total_calls = result.get("total_llm_calls", 0)
        st.metric("Total LLM Calls", total_calls)
        
        if result.get("traceback"):
            with st.expander("Show Traceback"):
                st.code(result.get("traceback", ""), language="python")
    
    elif status == "multiple_tests_success":
        final_coverage = result.get("final_coverage", 0)
        total_iterations = result.get("total_iterations", 0)
        
        st.success(f"✅ Multiple test generation completed! Final coverage: {final_coverage:.1f}% in {total_iterations} iterations")
        
        # Show accumulated tests
        st.subheader("All Generated Tests")
        accumulated_tests = result.get("accumulated_tests", "")
        st.code(accumulated_tests, language="python")
        
        # Show iteration details
        st.subheader("Iteration Details")
        all_results = result.get("all_results", [])
        
        for i, iteration_result in enumerate(all_results):
            iteration_num = iteration_result.get("iteration", i+1)
            target_line = iteration_result.get("target_line", "N/A")
            target_content = iteration_result.get("target_line_content", "").strip()
            coverage_before = iteration_result.get("coverage_before", 0)
            coverage_after = iteration_result.get("coverage_after", 0)
            iteration_status = iteration_result.get("status", "unknown")
            
            with st.expander(f"Iteration {iteration_num}: Line {target_line} - {iteration_status.title()}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Target Line {target_line}:** {target_content}")
                    st.write(f"**Status:** {iteration_status}")
                    st.write(f"**Coverage Before:** {coverage_before:.1f}%")
                    st.write(f"**Coverage After:** {coverage_after:.1f}%")
                
                with col2:
                    if iteration_status == "success":
                        test_code = iteration_result.get("test_code", "")
                        if test_code:
                            st.code(test_code, language="python")
                    else:
                        st.write(f"**Error:** {iteration_result.get('message', 'Unknown error')}")
        
        # Show final coverage summary
        final_coverage_info = result.get("final_coverage_info", {})
        covered_lines = final_coverage_info.get("covered_lines", set())
        missing_lines = final_coverage_info.get("missing_lines", set())
        total_lines = final_coverage_info.get("total_lines", set())
        total_llm_calls = result.get("total_llm_calls", 0)
        
        st.subheader("Coverage Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Lines", len(total_lines))
        with col2:
            st.metric("Covered Lines", len(covered_lines))
        with col3:
            st.metric("Missing Lines", len(missing_lines))
        with col4:
            st.metric("Total LLM Calls", total_llm_calls)
        
        if missing_lines:
            st.write("**Uncovered Lines:**", sorted(list(missing_lines)))
    
    else:
        st.warning(f"Unknown status: {status}")
        st.json(result)

if __name__ == "__main__":
    main()