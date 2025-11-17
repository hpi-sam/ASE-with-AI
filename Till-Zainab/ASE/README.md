# Unit Test Generator

An LLM based unit test generation pipeline with coverage analysis.
To regenerate erroneous tests, it uses perplexity drops (which lines are important for the LLM, similar to attention) as input to the test regeneration node to analyze what the LLM focussed on when producing a wrong test that did not cover the specified lines.

## Getting Started

1. Create a `src/.env` file with your OpenAI key: `OPENAI_API_KEY=sk-proj...`.
2. Run the app via `python -m streamlit run src/app.py`

This will automatically create a log for each run in the `logs/` directory.