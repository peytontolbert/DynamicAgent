from typing import Dict, Any, List
import json
import numpy as np
import logging
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.chat_with_ollama import ChatGPT
import time
from app.analysis.tool_analyzer import ToolAnalyzer
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class RewardModel:
    def __init__(self, llm: ChatGPT, knowledge_graph: KnowledgeGraph):
        self.llm = llm
        self.knowledge_graph = knowledge_graph
        self.evaluation_history: List[Dict[str, Any]] = []
        self.learning_rate = 0.01
        self.feature_weights = {
            "accuracy": 0.25,
            "efficiency": 0.25,
            "completeness": 0.25,
            "knowledge_utilization": 0.25
        }
        self.tool_analyzer = ToolAnalyzer(llm, knowledge_graph)

    async def evaluate_task(self, task: str, task_context: Dict[str, Any], result: str) -> float:
        relevant_knowledge = await self.knowledge_graph.get_relevant_knowledge(task)
        
        prompt = f"""
        Task: {task}
        Task Context: {json.dumps(task_context, indent=2)}
        Result: {result}
        Relevant Knowledge: {json.dumps(relevant_knowledge, indent=2)}
        
        Evaluate the performance of the task on a scale from 0 to 1 for each of the following criteria:
        1. Accuracy of the result
        2. Efficiency of the solution
        3. Completeness of the task
        4. Utilization of relevant knowledge
        
        Provide your evaluation as a JSON object with the following structure:
        {{
            "accuracy": float,
            "efficiency": float,
            "completeness": float,
            "knowledge_utilization": float,
            "reasoning": string,
            "improvement_suggestions": [string]
        }}
        """
        evaluation = await self.llm.chat_with_ollama("You are an expert in task evaluation and improvement.", prompt)
        try:
            eval_data = json.loads(evaluation)
            
            # Calculate weighted score
            score = sum(eval_data[key] * self.feature_weights[key] for key in self.feature_weights)
            
            # Store evaluation data for continuous learning
            self.evaluation_history.append({
                "task": task,
                "evaluation": eval_data,
                "final_score": score
            })
            
            # Store improvement suggestions in the knowledge graph
            for suggestion in eval_data["improvement_suggestions"]:
                await self.knowledge_graph.add_improvement_suggestion(suggestion)
            
            # Trigger continuous learning after every 10 evaluations
            if len(self.evaluation_history) % 10 == 0:
                await self.continuous_learn()
            
            return max(0.0, min(1.0, score))  # Ensure the score is between 0 and 1
        except (ValueError, KeyError, json.JSONDecodeError):
            return 0.5  # Return a default mid-range score if parsing fails

    async def continuous_learn(self):
        # Analyze past evaluations to adjust feature weights
        feature_scores = {key: [] for key in self.feature_weights}
        final_scores = []
        
        for evaluation in self.evaluation_history[-50:]:  # Consider last 50 evaluations
            for key in self.feature_weights:
                feature_scores[key].append(evaluation['evaluation'][key])
            final_scores.append(evaluation['final_score'])
        
        # Calculate correlations between feature scores and final scores
        correlations = {}
        for key in self.feature_weights:
            correlations[key] = np.corrcoef(feature_scores[key], final_scores)[0, 1]
        
        # Adjust weights based on correlations
        total_correlation = sum(abs(corr) for corr in correlations.values())
        for key in self.feature_weights:
            self.feature_weights[key] += self.learning_rate * (abs(correlations[key]) / total_correlation - self.feature_weights[key])
        
        # Normalize weights
        total_weight = sum(self.feature_weights.values())
        for key in self.feature_weights:
            self.feature_weights[key] /= total_weight
        
        # Store updated weights in knowledge graph
        await self.knowledge_graph.store_meta_learning_weights(self.feature_weights)
        
        # Analyze improvement suggestions
        await self.analyze_improvement_suggestions()

    async def analyze_improvement_suggestions(self):
        # Fetch recent improvement suggestions from the knowledge graph
        recent_suggestions = await self.knowledge_graph.get_recent_improvement_suggestions(50)
        
        # Analyze patterns in suggestions
        suggestion_prompt = f"""
        Analyze the following improvement suggestions and identify common patterns or themes:
        {json.dumps(recent_suggestions, indent=2)}
        
        Provide your analysis as a JSON object with the following structure:
        {{
            "common_themes": [string],
            "priority_improvements": [string],
            "long_term_strategies": [string]
        }}
        """
        analysis = await self.llm.chat_with_ollama("You are an expert in analyzing improvement patterns and strategies.", suggestion_prompt)
        
        try:
            analysis_data = json.loads(analysis)
            # Store analysis results in the knowledge graph
            await self.knowledge_graph.store_improvement_analysis(analysis_data)
        except json.JSONDecodeError:
            print("Error parsing improvement suggestion analysis")

    async def get_learning_insights(self) -> Dict[str, Any]:
        return {
            "current_weights": self.feature_weights,
            "evaluation_history_size": len(self.evaluation_history),
            "recent_scores": [eval_data['final_score'] for eval_data in self.evaluation_history[-10:]]
        }
    
    
    async def _extract_insights(self, text: str) -> List[Dict[str, Any]]:
        prompt = f"""
        Extract key insights and learnings from the following text:
        {text}
        Format the output as a JSON array of objects, where each object has 'name' and 'value' keys.
        Example:
        [
            {{"name": "Insight1", "value": "Description of Insight1"}},
            {{"name": "Insight2", "value": "Description of Insight2"}}
        ]
        Ensure that the output is valid JSON and contains at least one insight.
        If unsure or unknown, use the respond tool to gather more information.
        """
        insights = await self.llm.chat_with_ollama("You are an AI tasked with extracting insights from feedback.", prompt)
        parsed_insights = self._parse_json_or_text(insights)
        
        if not parsed_insights:
            raise ValueError("Failed to extract valid insights. Retrying with more specific instructions.")
        
        return parsed_insights
    
    
    def _parse_json_or_text(self, text: str) -> List[Dict[str, Any]]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and all(isinstance(item, dict) and 'name' in item and 'value' in item for item in parsed):
                return parsed
            else:
                raise ValueError("Invalid JSON structure")
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON, falling back to text parsing: {text}")
            logger.error(f"JSON parsing error: {text}", exc_info=True)
            return self._parse_text(text)

    async def store_learned_insight(self, insight: Dict[str, Any]):
        await self.knowledge_graph.add_or_update_node("LearnedInsight", {
            "content": json.dumps(insight),
            "timestamp": time.time()
        })

    async def retrieve_learned_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        insights = await self.knowledge_graph.get_recent_nodes("LearnedInsight", limit)
        return [json.loads(insight['content']) for insight in insights]

    async def update_knowledge_with_patterns(self):
        recent_evaluations = self.evaluation_history[-50:]
        patterns = await self.llm.extract_patterns(recent_evaluations)
        for pattern in patterns:
            await self.knowledge_graph.add_or_update_node("LearningPattern", {
                "pattern": json.dumps(pattern),
                "timestamp": time.time()
            })

    async def evaluate_tool_usage(self, tool_name: str, subtask: Dict[str, Any], result: Dict[str, Any]) -> float:
        prompt = f"""
        Tool: {tool_name}
        Subtask: {json.dumps(subtask, indent=2)}
        Result: {json.dumps(result, indent=2)}
        
        Evaluate the performance of the tool usage on a scale from 0 to 1 for each of the following criteria:
        1. Effectiveness (how well the tool accomplished its task)
        2. Efficiency (how quickly or resource-efficiently the tool performed)
        3. Appropriateness (how suitable the tool was for the given subtask)
        
        Provide your evaluation as a JSON object with the following structure:
        {{
            "effectiveness": float,
            "efficiency": float,
            "appropriateness": float,
            "reasoning": string,
            "improvement_suggestions": [string]
        }}
        """
        evaluation = await self.llm.chat_with_ollama("You are an expert in evaluating AI tool usage.", prompt)
        try:
            eval_data = json.loads(evaluation)
            score = (eval_data["effectiveness"] + eval_data["efficiency"] + eval_data["appropriateness"]) / 3
            
            # Store evaluation data for continuous learning
            self.evaluation_history.append({
                "type": "tool_usage",
                "tool": tool_name,
                "evaluation": eval_data,
                "final_score": score
            })
            
            # Store improvement suggestions in the knowledge graph
            for suggestion in eval_data["improvement_suggestions"]:
                await self.knowledge_graph.add_improvement_suggestion(f"Tool '{tool_name}': {suggestion}")
            
            return max(0.0, min(1.0, score))  # Ensure the score is between 0 and 1
        except (ValueError, KeyError, json.JSONDecodeError):
            return 0.5  # Return a default mid-range score if parsing fails

    async def analyze_tool_usage_patterns(self):
        return await self.tool_analyzer.analyze_tool_usage_patterns()
    
    async def generate_meta_insights(self, recent_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompt = f"""
        Generate meta-learning insights from the following tasks:
        {json.dumps(recent_tasks, indent=2)}
        
        Provide your insights as a JSON object with the following structure:
        {{
            "insights": [{{"name": string, "value": string}}]
        }}
        """
        insights = await self.llm.chat_with_ollama("You are an AI tasked with generating meta-learning insights.", prompt)
        return self._parse_json_or_text(insights)
