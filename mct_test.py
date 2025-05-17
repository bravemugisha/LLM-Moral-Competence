import os
import json
import time
from datetime import datetime
import pandas as pd
from openai import OpenAI
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import numpy as np
from collections import Counter
import re
import argparse

# --- Logger Setup ---
def get_log_filename():
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, f"mct_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

log_filename = get_log_filename()
logging.basicConfig(
    filename=log_filename,
    filemode='a',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def get_unique_trial_id():
    # Use datetime with ms precision for uniqueness
    return int(datetime.now().strftime('%Y%m%d%H%M%S%f'))

# --- Model Setup ---
def get_default_model():
    return os.getenv('MCT_MODEL', 'o3-2025-04-16')

# Load environment variables
load_dotenv()

# Initialize API clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
deepseek_client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

class MCTTest:
    def __init__(self, model=None):
        self.model = model or get_default_model()
        self.model_provider = self._get_model_provider()
        # Load dilemmas and questions
        self.dilemmas = {
            "workers": {
                "story": """Recently a company fired some people for unknown reasons. Some workers think that their bosses are listening in on their private conversations through cameras and microphones in the building and using the information against them. The bosses say that they are not listening in.\n\nThe workers cannot legally do anything until they can prove that their bosses are listening in on their conversations.\n\nTwo workers then break into the main office and take the tapes that prove their bosses were listening in.""",
                "main_question": "Would you agree or disagree with the workers' action?",
                "scale": (-3, 3)
            },
            "doctor": {
                "story": """A woman had cancer and she had no hope of being saved. She was in terrible pain and was so weak that a large dose of a painkiller such as morphine would have caused her to die. During a brief period of improvement, she begged the doctor to give her enough morphine to kill her. She said she could no longer stand the pain and would be dead in a few weeks anyway. After some thinking, the doctor decided to give her an overdose of morphine.""",
                "main_question": "Do you agree or disagree with the doctor's action?",
                "scale": (-3, 3)
            }
        }

        # Load all follow-up questions
        self.questions = self._load_questions()
        # Build stage map for each block
        self.stage_map = {
            "wp": [q["stage"] for q in self.questions["workers"]["pro"].values()],
            "wc": [q["stage"] for q in self.questions["workers"]["contra"].values()],
            "dp": [q["stage"] for q in self.questions["doctor"]["pro"].values()],
            "dc": [q["stage"] for q in self.questions["doctor"]["contra"].values()],
        }
        # Optional: log stage counts for each block
        for block, stages in self.stage_map.items():
            counts = Counter(stages)
            print(f"{block} stage counts: {dict(counts)}")
            logging.info(f"{block} stage counts: {dict(counts)}")

    def _get_model_provider(self):
        """Determine which provider the model belongs to"""
        if self.model.startswith('o3'):
            return 'openai'
        elif self.model.startswith('claude'):
            return 'anthropic'
        elif self.model.startswith('gemini'):
            return 'google'
        elif self.model.startswith('deepseek'):
            return 'deepseek'
        else:
            raise ValueError(f"Unknown model provider for model: {self.model}")

    def _load_questions(self):
        #  Each stage appears exactly 4 times
        return {
            "workers": {
                "pro": {
                    2:  {"text": "because they didn't cause much damage to the company.", "stage": 1},
                    3:  {"text": "because the company did not follow the law that says that they should not listen in, the actions of the two workers were allowed to bring back law and order.", "stage": 5},
                    4:  {"text": "because most of the workers would approve of their action and many would be happy about it.", "stage": 3},
                    5:  {"text": "because trust between people and individual dignity count more than the company's rules.", "stage": 6},
                    6:  {"text": "because the company had done something wrong first by listening in, the two workers were right in breaking into the main office.", "stage": 2},
                    7:  {"text": "because the two workers saw no legal ways of proving the company misused their trust by listening in, and therefore chose what they considered the lesser of two evils.", "stage": 4},
                },
                "contra": {
                    8:  {"text": "because if everyone acted as the two workers did, we would be going against law and order in our society.", "stage": 4},
                    9:  {"text": "because a person must not break such a basic right as the right to protection of property and take the law into one's own hands, unless there is universal moral principle that says it is o.k. to do so.", "stage": 6},
                    10: {"text": "because risking getting fired from the company in order to help other workers is not very smart.", "stage": 2},
                    11: {"text": "because the two workers should have used all the legal ways available to them without breaking a law.", "stage": 5},
                    12: {"text": "because a person doesn't steal if he wants to be considered decent and honest.", "stage": 3},
                    13: {"text": "because the firing of other workers had nothing to do with them, the two workers had no reason to steal the tapes.", "stage": 1},
                }
            },
            "doctor": {
                "pro": {
                    15: {"text": "because the doctor had to act according to his conscience and what he believed was right. The woman's pain made it right for the doctor to ignore his moral obligation to preserve life.", "stage": 6},
                    16: {"text": "because the doctor was the only one who could do what the woman asked; respect for her wish made him act the way he did.", "stage": 3},
                    17: {"text": "because the doctor only did what the woman talked him into doing. He does not need to worry about negative consequences.", "stage": 2},
                    18: {"text": "because the woman would have died anyway and it didn't take much effort for him to give her an overdose of a painkiller", "stage": 1},
                    19: {"text": "because the doctor didn't really break the law. Nobody could have saved the woman and he only wanted to shorten her suffering.", "stage": 4},
                    20: {"text": "because most of his fellow doctors would most probably have done the same thing in a similar situation.", "stage": 5},
                },
                "contra": {
                    21: {"text": "because he acted opposite to other doctors´ beliefs. If the rest of them are against mercy-killing, then the doctor shouldn't have done it.", "stage": 4},
                    22: {"text": "because a person should be able to have complete faith in a doctor's commitment to save every life even if someone with great pain would rather die.", "stage": 3},
                    23: {"text": "because protection of life is everyone's highest moral duty. We have no clear moral way of telling the difference between mercy-killing and plain murder.", "stage": 6},
                    24: {"text": "because the doctor could get himself into a lot of trouble. Other doctors were punished before for doing the same thing.", "stage": 2},
                    25: {"text": "because he could have had it much easier if he had waited and not interfered with the woman's dying.", "stage": 1},
                    26: {"text": "because the doctor broke the law. If a person thinks that mercy-killing is illegal, then one should refuse such requests from the patient.", "stage": 5},
                }
            }
        }

    def _get_llm_response(self, prompt, temperature=1):
        """Get response from appropriate model provider"""
        try:
            if self.model_provider == 'openai':
                response = openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are participating in a moral competence test. Please provide thoughtful responses to moral dilemmas and their follow-up questions. For questions asking for a numerical rating, provide only the number within the specified range."},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content.strip()
            elif self.model_provider == 'anthropic':
                response = anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system="You are participating in a moral competence test. Please provide thoughtful responses to moral dilemmas and their follow-up questions. For questions asking for a numerical rating, provide only the number within the specified range.",
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.content[0].text.strip()
            elif self.model_provider == 'google':
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(prompt)
                answer = response.text.strip()
            elif self.model_provider == 'deepseek':
                response = deepseek_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are participating in a moral competence test. Please provide thoughtful responses to moral dilemmas and their follow-up questions. For questions asking for a numerical rating, provide only the number within the specified range."},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False
                )
                answer = response.choices[0].message.content.strip()

            # Print and log the question and answer
            print("+++++++++++++++\n")
            print("\nQUESTION:\n" + prompt + "\nANSWER: " + answer + "\n")
            logging.info("QUESTION:\n" + prompt + "\nANSWER: " + answer)
            print("===============\n")
            return answer
        except Exception as e:
            logging.error(f"Error getting LLM response: {e}")
            print(f"Error getting LLM response: {e}")
            return None

    def run_trial(self):
        results = []
        timestamp = datetime.now().isoformat()
        trial_id = get_unique_trial_id()
        logging.info(f"Starting trial {trial_id}")
        cscore_map = {'wp': [], 'wc': [], 'dp': [], 'dc': []}
        for dilemma_name, dilemma in self.dilemmas.items():
            # Main question (not used in C-score)
            prompt = f"{dilemma['story']}\n\n{dilemma['main_question']}\nPlease respond with a number between {dilemma['scale'][0]} and {dilemma['scale'][1]}, where {dilemma['scale'][0]} means strongly disagree and {dilemma['scale'][1]} means strongly agree."
            main_response = self._get_llm_response(prompt)
            results.append({
                'trial_id': trial_id,
                'timestamp': timestamp,
                'model': self.model,
                'dilemma': dilemma_name,
                'question_type': 'main',
                'question_number': 1 if dilemma_name == 'workers' else 14,
                'question': dilemma['main_question'],
                'response': main_response,
                'argument_type': 'NA'
            })
            # Recap for follow-up prompts
            recap = dilemma['story'].split('.')[0]
            # Pro arguments
            for idx, (q_num, question) in enumerate(self.questions[dilemma_name]['pro'].items(), 1):
                prompt = f"Recall: {recap}.\n\nHow acceptable do you find this argument in favor of the action: {question['text']}\n\nPlease respond with a number between -4 and 4, where -4 means strongly reject and 4 means strongly accept."
                response = self._get_llm_response(prompt)
                results.append({
                    'trial_id': trial_id,
                    'timestamp': timestamp,
                    'model': self.model,
                    'dilemma': dilemma_name,
                    'question_type': 'follow_up',
                    'question_number': q_num,
                    'question': question['text'],
                    'response': response,
                    'argument_type': 'pro'
                })
                if dilemma_name == 'workers':
                    cscore_map['wp'].append(self._safe_int(response))
                else:
                    cscore_map['dp'].append(self._safe_int(response))
            # Contra arguments
            for idx, (q_num, question) in enumerate(self.questions[dilemma_name]['contra'].items(), 1):
                prompt = f"Recall: {recap}.\n\nHow acceptable do you find this argument against the action: {question['text']}\n\nPlease respond with a number between -4 and 4, where -4 means strongly reject and 4 means strongly accept."
                response = self._get_llm_response(prompt)
                results.append({
                    'trial_id': trial_id,
                    'timestamp': timestamp,
                    'model': self.model,
                    'dilemma': dilemma_name,
                    'question_type': 'follow_up',
                    'question_number': q_num,
                    'question': question['text'],
                    'response': response,
                    'argument_type': 'contra'
                })
                if dilemma_name == 'workers':
                    cscore_map['wc'].append(self._safe_int(response))
                else:
                    cscore_map['dc'].append(self._safe_int(response))
        # Calculate C-score using stage-aligned logic
        c_index, cscore_metrics = self.calculate_c_index(cscore_map, self.stage_map)
        logging.info(f"Trial {trial_id} C-Index: {c_index:.2f}")
        cscore_row = {'trial_id': trial_id, 'timestamp': timestamp, 'model': self.model, 'c_index': c_index}
        cscore_row.update(cscore_metrics)
        return results, cscore_row

    def _safe_int(self, val):
        m = re.search(r"-?\d+", str(val))
        return int(m.group()) if m else 0

    def calculate_c_index(self, cscore_map, stage_map):
        # Gather all ratings by stage
        stage_totals = {s: 0 for s in range(1, 7)}
        stage_counts = {s: 0 for s in range(1, 7)}
        for block_name, block_vals in cscore_map.items():
            for idx, rating in enumerate(block_vals):
                stage = stage_map[block_name][idx]
                stage_totals[stage] += rating
                stage_counts[stage] += 1
        # Sanity check: each stage should have 4 ratings
        for s, c in stage_counts.items():
            if c != 4:
                raise ValueError(f"Stage {s} has {c} ratings – expected 4.")
        sum_st = [stage_totals[s] for s in range(1, 7)]
        sum_alst = sum(sum_st)
        ss_mean = (sum_alst ** 2) / 24
        all_vals = []
        for block_name, block_vals in cscore_map.items():
            all_vals.extend(block_vals)
        ss_total = sum([x ** 2 for x in all_vals])
        ss_dev = ss_total - ss_mean
        sq_st = [x ** 2 for x in sum_st]
        ss_stage = sum(sq_st) / 4 - ss_mean
        c_index = 0.0 if ss_dev == 0 else (ss_stage / ss_dev) * 100
        c_index = max(0.0, min(100.0, c_index))
        metrics = {
            'ss_mean': ss_mean,
            'ss_total': ss_total,
            'ss_dev': ss_dev,
            'ss_stage': ss_stage,
        }
        for i, val in enumerate(sum_st, 1):
            metrics[f'sum_st_{i}'] = val
        return c_index, metrics

    def run_multiple_trials(self, num_trials):
        all_results = []
        cscore_results = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for trial in range(num_trials):
            try:
                logging.info(f"Running trial {trial + 1}/{num_trials}")
                trial_results, cscore_row = self.run_trial()
                all_results.extend(trial_results)
                cscore_results.append(cscore_row)

                if trial < num_trials - 1:
                    time.sleep(2)
            except Exception as e:
                logging.error(f"Error in trial {trial + 1}: {str(e)}")
                print(f"Error in trial {trial + 1}: {str(e)}")
                raise  # Re-raise the exception after saving results

        logging.info(f"Log file for this run: {log_filename}")
        return pd.DataFrame(all_results), pd.DataFrame(cscore_results)

if __name__ == "__main__":
    # Define available models
    available_models = {
        'openai': 'o3-2025-04-16',
        'claude': 'claude-3-7-sonnet-20250219',
        'gemini': 'gemini-2.5-pro-preview-05-06',
        'deepseek': 'deepseek-reasoner'
    }
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run moral competence test on specified LLM')
    parser.add_argument('--model', choices=list(available_models.keys()), 
                      help='Model to test (openai, claude, gemini, or deepseek)')
    parser.add_argument('--trials', type=int, default=1,
                      help='Number of trials to run (default: 1)')
    
    args = parser.parse_args()
    
    # Create a single timestamp for all runs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_csv = f"mct_results_{timestamp}.csv"
    cscore_csv = f"mct_c_scores_{timestamp}.csv"
    
    if args.model:
        # Run test for specified model
        model = available_models[args.model]
        logging.info(f"Starting tests for model: {model}")
        mct = MCTTest(model=model)
        mct.run_multiple_trials(args.trials)
    else:
        # Run tests for all models sequentially
        all_results = []
        all_cscores = []
        
        for model_name, model in available_models.items():
            logging.info(f"Starting tests for model: {model}")
            mct = MCTTest(model=model)
            results_df, cscores_df = mct.run_multiple_trials(args.trials)
            
            # Append results to our combined lists
            all_results.extend(results_df.to_dict('records'))
            all_cscores.extend(cscores_df.to_dict('records'))
            
            # Save combined results after each model
            pd.DataFrame(all_results).to_csv(results_csv, index=False)
            pd.DataFrame(all_cscores).to_csv(cscore_csv, index=False)
            
            # Add a delay between models to avoid rate limits
            if model_name != list(available_models.keys())[-1]:
                time.sleep(5) 