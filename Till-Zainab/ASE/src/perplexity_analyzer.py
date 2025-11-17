import numpy as np
from openai import OpenAI
from typing import List, Tuple, Dict
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class PerplexityAnalyzer:
    """Analyze perplexity drops using our (line) occlusion methodology"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.call_count = 0  # Track number of API calls made
    
    def get_response_with_logprobs(self, prompt: str, max_retries: int = 3) -> Dict:
        """Get response with log probabilities, with retry logic for rate limits"""
        for attempt in range(max_retries + 1):
            try:
                print(f"PERPLEXITY: Making LLM call #{self.call_count + 1}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    top_p=1e-10,
                    logprobs=True,
                    top_logprobs=1,
                    temperature=0.1
                )
                
                # Track successful API call
                self.call_count += 1
                print(f"PERPLEXITY: Completed LLM call #{self.call_count}")
                
                choice = response.choices[0]
                if choice.logprobs and choice.logprobs.content:
                    # Extract tokens and logprobs from the chat completion response
                    tokens = []
                    logprobs = []
                    
                    for token_data in choice.logprobs.content:
                        if token_data.token:
                            tokens.append(token_data.token)
                            logprobs.append(token_data.logprob)
                    
                    text = choice.message.content.strip() if choice.message.content else ""
                    
                    # Filter out None values from logprobs
                    valid_logprobs = [lp for lp in logprobs if lp is not None]
                    
                    return {
                        "text": text,
                        "tokens": tokens,
                        "logprobs": valid_logprobs,
                        "avg_logprob": np.mean(valid_logprobs) if valid_logprobs else 0,
                        "perplexity": np.exp(-np.mean(valid_logprobs)) if valid_logprobs else float('inf')
                    }
                else:
                    return {
                        "text": choice.message.content.strip() if choice.message.content else "",
                        "tokens": [],
                        "logprobs": [],
                        "avg_logprob": 0,
                        "perplexity": float('inf')
                    }
            except Exception as e:
                error_str = str(e)
                # Check if it's a rate limit error
                if "rate_limit_exceeded" in error_str or "429" in error_str:
                    if attempt < max_retries:
                        # Exponential backoff: 2s, 4s, 8s, ...
                        sleep_time = 2 ** (attempt+1)
                        print(f"Rate limit hit, sleeping for {sleep_time}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(sleep_time)
                        continue
                    else:
                        print(f"Rate limit exceeded after {max_retries} retries: {e}")
                else:
                    print(f"Error getting response: {e}")
                
                # If we've exhausted retries or it's not a rate limit error, return default
                return {
                    "text": "",
                    "tokens": [],
                    "logprobs": [],
                    "avg_logprob": 0,
                    "perplexity": float('inf')
                }
        
        # This shouldn't be reached, but just in case
        return {
            "text": "",
            "tokens": [],
            "logprobs": [],
            "avg_logprob": 0,
            "perplexity": float('inf')
        }
    
    def mask_line_and_get_perplexity(self, lines: List[str], line_idx: int, 
                                   mask_token: str = "[THIS LINE IS OCCLUDED]") -> Tuple[float, str]:
        """Mask a specific line and compute perplexity"""
        masked_lines = lines.copy()
        masked_lines[line_idx] = mask_token
        masked_prompt = "\n".join(masked_lines)
        
        response_data = self.get_response_with_logprobs(masked_prompt)
        return response_data["perplexity"], response_data["text"]
    
    def compute_line_perplexity_drops(self, prompt: str, max_workers: int = 5) -> List[Tuple[str, float]]:
        """Compute perplexity drops for each line in the prompt using parallel processing"""
        lines = [line for line in prompt.strip().split('\n') if line.strip()]
        
        # Get original perplexity
        original_response = self.get_response_with_logprobs(prompt)
        original_perplexity = original_response["perplexity"]
        
        # Create a function for parallel execution
        def compute_single_line_drop(line_idx_and_line):
            line_idx, line = line_idx_and_line
            perplexity_masked, _ = self.mask_line_and_get_perplexity(lines, line_idx)
            drop = perplexity_masked - original_perplexity
            return line_idx, line.strip(), drop
        
        line_drops = []
        
        # Run masked perplexity calculations in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_line = {
                executor.submit(compute_single_line_drop, (i, line)): (i, line) 
                for i, line in enumerate(lines)
            }
            
            # Collect results as they complete
            results = {}
            for future in as_completed(future_to_line):
                try:
                    line_idx, line_text, drop = future.result()
                    results[line_idx] = (line_text, drop)
                except Exception as e:
                    line_idx, line = future_to_line[future]
                    print(f"Error processing line {line_idx}: {e}")
                    results[line_idx] = (line.strip(), 0.0)  # Default value on error
            
            # Sort results by original line order
            for i in range(len(lines)):
                if i in results:
                    line_drops.append(results[i])
        
        return line_drops
    
    def get_top_influential_lines(self, prompt: str, top_n: int = 5, max_workers: int = 5) -> List[str]:
        """Get the top N most influential lines based on perplexity drop"""
        line_drops = self.compute_line_perplexity_drops(prompt, max_workers=max_workers)
        
        # Sort by drop value (descending - higher drop = more influential)
        sorted_drops = sorted(line_drops, key=lambda x: x[1], reverse=True)
        
        # Return just the line text
        return [line for line, _ in sorted_drops[:top_n]]
    
    def analyze_failed_test_generation(self, prompt: str, top_n: int = 5, max_workers: int = 5) -> Dict:
        """Analyze why test generation failed and return insights"""
        # Track call count at start of analysis
        calls_before = self.call_count
        
        # Count lines to predict number of calls
        lines = [line for line in prompt.strip().split('\n') if line.strip()]
        expected_calls = 1 + len(lines)  # 1 for original + N for each line
        print(f"PERPLEXITY ANALYSIS: Starting with {len(lines)} lines")
        print(f"PERPLEXITY ANALYSIS: Expecting {expected_calls} LLM calls (1 original + {len(lines)} masked)")
        print(f"PERPLEXITY ANALYSIS: Current total calls before: {calls_before}")
        
        # Compute line drops only once (instead of twice)
        line_drops = self.compute_line_perplexity_drops(prompt, max_workers=max_workers)
        
        # Sort by drop value (descending - higher drop = more influential) and get top N
        sorted_drops = sorted(line_drops, key=lambda x: x[1], reverse=True)
        influential_lines = [line for line, _ in sorted_drops[:top_n]]
        
        # Calculate calls made during this analysis
        calls_made = self.call_count - calls_before
        print(f"PERPLEXITY ANALYSIS: Completed - Made {calls_made} LLM calls (expected {expected_calls})")
        print(f"PERPLEXITY ANALYSIS: Current total calls after: {self.call_count}")
        
        return {
            "top_influential_lines": influential_lines,
            "all_line_drops": line_drops,
            "analysis_summary": f"Top {top_n} lines that most influenced the model's incorrect output",
            "perplexity_calls_made": calls_made
        }
