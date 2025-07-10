"""Conversation information extraction and analysis for evaluation in the Arklex framework.

This module provides functionality for analyzing conversations and extracting metrics for
evaluation purposes. It includes utilities for building intent graphs, tracking conversation
flows, checking goal completion, and calculating various performance metrics such as task
completion rates and efficiency. The module supports both user and bot goal tracking,
conversation filtering, and statistical analysis of conversation patterns.
"""

import json
import networkx as nx
from typing import List, Dict, Any, Optional, Union

from arklex.evaluation.chatgpt_utils import (
    chatgpt_chatbot,
    format_chat_history_str,
    flip_hist_content_only,
    filter_convo,
)

import traceback


def get_edges_and_counts(data: List[Dict[str, Any]]) -> Dict[tuple[str, str], int]:
    edge_counts: Dict[tuple[str, str], int] = {}
    for convo in data:
        convo = filter_convo(convo)
        for i in range(len(convo)):
            if convo[i]["role"] == "assistant":
                continue
            prev_intent: str = "start" if i == 0 else convo[i - 2]["intent"]
            current_intent: str = convo[i]["intent"]
            edge_counts[(prev_intent, current_intent)] = (
                edge_counts.get((prev_intent, current_intent), 0) + 1
            )
    return edge_counts


def build_intent_graph(data: List[Dict[str, Any]]) -> nx.DiGraph:
    G: nx.DiGraph = nx.DiGraph()
    edge_counts: Dict[tuple[str, str], int] = get_edges_and_counts(data)
    for key in edge_counts.keys():
        G.add_edge(key[0], key[1], weight=edge_counts[key])
    return G


def check_bot_goal_enhanced(convo: List[Dict[str, Any]], bot_goal: str, client: Any) -> bool:
    """
    Enhanced bot goal evaluation with detailed scoring system.
    Uses a 100-point scoring system to better assess goal achievement across different types of bots.
    """
    convo_str: str = format_chat_history_str(flip_hist_content_only(convo))
    
    prompt: str = f"""Here is a conversation between a user and a chatbot assistant:
{convo_str}

The chatbot's goal is: {bot_goal}

Evaluate whether the bot successfully achieved its goal using this 100-point scoring system:

**Bot Goal Execution (35 points)**:
- Did the bot successfully execute the specific tasks outlined in its goal? (20 points)
- Did the bot demonstrate the expected behavior patterns defined in its goal? (15 points)

**Strategic Objective Achievement (25 points)**:
- Did the bot achieve its primary strategic objective as defined in the goal? (15 points)
- Were secondary objectives or sub-goals also accomplished when applicable? (10 points)

**Role & Function Adherence (25 points)**:
- Did the bot stay within its defined role and function boundaries? (15 points)
- Did the bot consistently follow its programmed guidelines and constraints? (10 points)

**Goal-Driven Conversation Management (15 points)**:
- Did the bot actively guide the conversation toward achieving its goal? (10 points)
- Did the bot maintain focus on goal-relevant topics and activities? (5 points)

Calculate the total score out of 100 points. The bot succeeds if it scores 70 points or higher.

Focus on evaluating the bot's performance from the perspective of its own goals and objectives, not user satisfaction.

Output format:
SCORE: [total points]
PASS: [True/False based on >= 85 points]
ANALYSIS: [Brief explanation of scoring focusing on bot goal achievement]

Only output the SCORE, PASS, and ANALYSIS lines."""

    output: str = chatgpt_chatbot([{"role": "user", "content": prompt}], client)
    
    # Parse the enhanced evaluation result
    try:
        lines = output.strip().split('\n')
        score_line = next((line for line in lines if line.startswith('SCORE:')), None)
        pass_line = next((line for line in lines if line.startswith('PASS:')), None)
        analysis_line = next((line for line in lines if line.startswith('ANALYSIS:')), None)
        
        if score_line and pass_line and analysis_line:
            score = float(score_line.split(':', 1)[1].strip())
            pass_result = pass_line.split(':', 1)[1].strip().lower() == 'true'
            analysis = analysis_line.split(':', 1)[1].strip()
            return score, pass_result, analysis
    except Exception as e:
        print("Error parsing output:", e)
        traceback.print_exc()
        pass

def check_bot_goal(convo: List[Dict[str, Any]], bot_goal: str, client: Any) -> bool:
    convo_str: str = format_chat_history_str(flip_hist_content_only(convo))
    prompt: str = f"Here is a conversation between a user and a customer service chatbot assistant:\n{convo_str}\n\nThe chatbot's goal is the following: {bot_goal}\nOutput True if the bot was able to achieve its goal. Output False otherwise. Only output True or False and nothing else."
    output: str = chatgpt_chatbot([{"role": "user", "content": prompt}], client)
    return output == "True"


def num_user_turns(convo: List[Dict[str, Any]]) -> int:
    user_turns: int = 0
    for turn in convo:
        if turn.get("role", None) == "user":
            user_turns += 1
    return user_turns


def extract_task_completion_metrics(
    data: List[Dict[str, Any]], client: Any, bot_goal: Optional[str] = None
) -> Union[Dict[str, float], str]:
    num_convos: int = len(data)
    if num_convos == 0:
        return "Error while extracting task completion metrics"
    goal_completions: int = 0
    bot_goal_completions: int = 0
    completion_efficiency: int = 0
    user_goal_scores = []
    user_goal_analyses = []
    bot_goal_scores = []
    bot_goal_analyses = []
    for convo in data:
        convo_history: List[Dict[str, Any]] = convo["convo"]
        completion_efficiency += num_user_turns(convo_history)
        if convo["goal_completion"]:
            goal_completions += 1
        user_goal_scores.append(convo["score"])
        user_goal_analyses.append(convo["analysis"])
        try:
            score, passed, analysis = check_bot_goal_enhanced(convo_history, bot_goal, client)
        except Exception as e:
            print("Error checking bot goal:", e)
            traceback.print_exc()
            score = None
            passed = check_bot_goal(convo_history, bot_goal, client)
            analysis = None
        bot_goal_scores.append(score)
        bot_goal_analyses.append(analysis)
        if bot_goal is not None and passed:
            bot_goal_completions += 1
    metrics: Dict[str, float] = {
        "user_task_completion": goal_completions / num_convos,
        "user_task_completion_efficiency": completion_efficiency / num_convos,
        "user_goal_score": user_goal_scores,
        "user_goal_analysis": user_goal_analyses,
    }
    if bot_goal is not None:
        metrics["bot_goal_completion"] = bot_goal_completions / num_convos
        metrics["bot_goal_score"] = bot_goal_scores
        metrics["bot_goal_analysis"] = bot_goal_analyses
    return metrics


if __name__ == "__main__":
    # with open('files/p1_sample_convos.json') as f:
    #     data = json.load(f)

    # model_api = "http://adaptation.cs.columbia.edu:55131/predict"
    # model_params = {'bot_id' : 'richtech', 'bot_version': 'v1alpha1'}
    # convos  = get_nlu_labels(data, model_api, model_params)
    # with open('files/p1_sample_convos_labeled.json', 'w') as f:
    #     json.dump(convos, f, indent=5)

    with open("files/p1_sample_convos_labeled.json") as f:
        data: List[Dict[str, Any]] = json.load(f)
    G: nx.DiGraph = build_intent_graph(data)
    for e in list(G.edges()):
        print(f"Weight for edge {e}: {G.get_edge_data(e[0], e[1])['weight']}")
