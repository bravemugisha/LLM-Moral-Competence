# Moral Competence Test (MCT) for LLMs

This project implements the Moral Competence Test (MCT) for Large Language Models (LLMs) using various AI providers' APIs. It presents two moral dilemmas (Workers and Doctor) and collects responses to various arguments for and against the actions taken in each dilemma.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

## Usage

The script supports testing multiple LLM providers and can be run in different ways:

1. Test a specific model:
```bash
python mct_test.py --model openai
```

Available model options:
- `openai` (o3-2025-04-16)
- `claude` (claude-3-sonnet-20240229)
- `gemini` (gemini-1.5-pro)
- `deepseek` (deepseek-chat)

2. Test all models sequentially:
```bash
python mct_test.py
```

3. Specify the number of trials:
```bash
python mct_test.py --model openai --trials 3
```

## Output

The script generates two CSV files:

1. Main Results (`mct_results_YYYYMMDD_HHMMSS.csv`):
- trial_id: Unique identifier for each trial
- timestamp: When the response was recorded
- model: Which LLM provided the response
- dilemma: Which dilemma was presented (workers/doctor)
- question_type: Whether it's the main question or a follow-up
- question_number: The original question number from the MCT
- question: The actual question text
- response: The LLM's response (-3 to +3 for main questions, -4 to +4 for follow-ups)
- argument_type: Whether the argument is pro, contra, or NA (for main questions)

2. C-Scores (`mct_c_scores_YYYYMMDD_HHMMSS.csv`):
- trial_id: Unique identifier for each trial
- timestamp: When the trial was run
- model: Which LLM was tested
- c_index: The calculated C-index score
- Additional metrics used in C-index calculation

## Notes

- The script includes delays between trials and models to avoid API rate limiting
- Responses are designed to be numerical only, matching the original MCT scale
- When testing multiple models, all results are combined into a single CSV file for easier comparison
- The C-index score provides a measure of moral competence based on the responses
- Log files are generated for each run with detailed information about the testing process 