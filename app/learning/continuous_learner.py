from app.knowledge.knowledge_graph import KnowledgeGraph
from app.chat_with_ollama import ChatGPT
from typing import Dict, Any, List
import logging
import json
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class ContinuousLearner:
    def __init__(self, knowledge_graph: KnowledgeGraph, llm: ChatGPT):
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.learning_rate = 0.1
        self.novelty_threshold = 0.5
        logger.info("ContinuousLearner initialized")

    async def learn(self, task: Dict[str, Any], result: Dict[str, Any]):
        extracted_knowledge = await self._extract_knowledge(task, result)
        novelty_score = await self._calculate_novelty(extracted_knowledge)
        
        if novelty_score > self.novelty_threshold:
            await self._update_knowledge_graph(extracted_knowledge)
            await self._generate_new_connections(extracted_knowledge)
        
        await self._update_task_success_rate(task, result)
        
        if self._should_optimize_model():
            await self._optimize_llm_model()

    async def _generate_new_connections(self, new_knowledge: List[Dict[str, Any]]):
        existing_knowledge = await self.knowledge_graph.get_all_knowledge()
        prompt = f"""
        Analyze the new knowledge and existing knowledge to identify potential new connections or insights:
        New Knowledge: {json.dumps(new_knowledge, indent=2)}
        Existing Knowledge: {json.dumps(existing_knowledge, indent=2)}
        
        Provide your analysis as a JSON array of objects, where each object has 'source', 'target', and 'relationship' keys.
        """
        connections = await self.llm.chat_with_ollama("You are an expert in knowledge synthesis and connection identification.", prompt)
        parsed_connections = json.loads(connections)
        for connection in parsed_connections:
            await self.knowledge_graph.add_relationship(connection['source'], connection['target'], connection['relationship'])

    async def _update_task_success_rate(self, task: Dict[str, Any], result: Dict[str, Any]):
        # Implement logic to track and update task success rates
        pass

    async def _should_optimize_model(self) -> bool:
        # Implement logic to decide when to optimize the LLM model
        pass

    async def _optimize_llm_model(self):
        # Implement logic to fine-tune or optimize the LLM model based on accumulated data
        pass

    async def _calculate_novelty(self, knowledge: List[Dict[str, Any]]) -> float:
        knowledge_str = json.dumps(knowledge, indent=2)
        prompt = f"""
        Analyze the following extracted knowledge and rate its novelty on a scale from 0 to 1,
        where 0 is completely familiar and 1 is entirely new and groundbreaking:

        {knowledge_str}

        Consider factors such as:
        1. Uniqueness of concepts
        2. Unexpected connections between ideas
        3. Potential for new applications or insights

        Provide your response as a single float value between 0 and 1.
        """
        
        novelty_score = await self.llm.chat_with_ollama("You are an AI expert in assessing the novelty of information.", prompt)
        
        try:
            novelty = float(novelty_score.strip())
            return max(0.0, min(1.0, novelty))  # Ensure the score is between 0 and 1
        except ValueError:
            logger.error(f"Failed to parse novelty score: {novelty_score}")
            return 0.5  # Return a default mid-range score if parsing fails
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10),
           retry=retry_if_exception_type(ValueError))
    async def _extract_knowledge(self, task: Dict[str, Any], result: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = f"""
        Task: {task['content']}
        Result: {result['result']}
        Extract key concepts and relationships from this task-result pair.
        Format the output as a JSON array of objects, where each object has 'name' and 'value' keys.
        Example:
        [
            {{"name": "Concept1", "value": "Description of Concept1"}},
            {{"name": "Concept2", "value": "Description of Concept2"}}
        ]
        """
        extracted_knowledge = await self.llm.chat_with_ollama("You are a knowledge extraction expert.", prompt)
        parsed_knowledge = self._parse_json_or_text(extracted_knowledge)
        
        if not parsed_knowledge:
            raise ValueError("Failed to extract valid knowledge. Retrying with more specific instructions.")
        
        return parsed_knowledge

    async def _update_knowledge_graph(self, knowledge: List[Dict[str, Any]]):
        for item in knowledge:
            try:
                item['id'] = str(uuid.uuid4())  # Add an 'id' field if it doesn't exist
                await self.knowledge_graph.add_or_update_node("Concept", item)
            except Exception as e:
                logger.error(f"Error updating knowledge graph: {str(e)}", exc_info=True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _apply_concepts(self, task: Dict[str, Any], concepts: List[Dict[str, Any]]) -> str:
        concepts_str = json.dumps(concepts, indent=2)
        prompt = f"""
        Task: {task['content']}
        Relevant concepts:
        {concepts_str}
        Apply these concepts to improve or solve the task. Provide a clear and concise solution.
        If unsure or unknown, use the respond tool to gather more information.
        """
        response = await self.llm.chat_with_ollama(prompt)
        
        if len(response.split()) < 10:  # Simple check for response quality
            raise ValueError("Response too short or unclear. Retrying with more specific instructions.")
        
        return response

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10),
           retry=retry_if_exception_type(ValueError))
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

    def _parse_text(self, text: str) -> List[Dict[str, Any]]:
        insights = []
        lines = text.split('\n')
        for line in lines:
            match = re.match(r'^(.*?):\s*(.*)$', line.strip())
            if match:
                name, value = match.groups()
                insights.append({"name": name.strip(), "value": value.strip()})
        return insights

    async def improve_llm_response(self, initial_prompt: str, initial_response: str, max_attempts: int = 3) -> str:
        for attempt in range(max_attempts):
            if self._is_response_satisfactory(initial_response):
                return initial_response

            feedback_prompt = f"""
            The previous response was not satisfactory. Please improve upon it.
            Original prompt: {initial_prompt}
            Previous response: {initial_response}
            Provide a more detailed and accurate response, ensuring it meets the required format and contains sufficient information.
            If unsure or unknown, use the respond tool to gather more information.
            """
            improved_response = await self.llm.chat_with_ollama(feedback_prompt)
            
            if self._is_response_satisfactory(improved_response):
                return improved_response
            
            initial_response = improved_response

        logger.warning(f"Failed to get a satisfactory response after {max_attempts} attempts.")
        return initial_response

    def _is_response_satisfactory(self, response: str) -> bool:
        # Implement logic to determine if a response is satisfactory
        # This is a placeholder implementation
        return len(response.split()) >= 50 and '{"' in response and '}' in response

    async def learn_from_improvements(self, improvement: str):
        prompt = f"""
        Analyze the following improvement and extract key learnings:
        {improvement}
        
        Provide your analysis as a JSON object with the following structure:
        {{
            "key_concepts": ["concept1", "concept2", ...],
            "potential_applications": ["application1", "application2", ...],
            "suggested_updates": [
                {{
                    "target": "knowledge_graph|model|workflow",
                    "update_type": "add|modify|remove",
                    "details": "Detailed description of the update"
                }}
            ]
        }}
        If unsure or unknown, use the respond tool to gather more information.
        """
        analysis = await self.llm.chat_with_ollama("You are an AI specializing in continuous learning and improvement.", prompt)
        parsed_analysis = json.loads(analysis)
        
        for update in parsed_analysis['suggested_updates']:
            if update['target'] == 'knowledge_graph':
                await self._update_knowledge_graph(update)
            elif update['target'] == 'model':
                await self._update_model(update)
            elif update['target'] == 'workflow':
                await self._update_workflow(update)

    async def _update_knowledge_graph(self, update: Dict[str, Any]):
        # Implement logic to update the knowledge graph
        pass

    async def _update_model(self, update: Dict[str, Any]):
        # Implement logic to update the model (e.g., fine-tuning)
        pass

    async def _update_workflow(self, update: Dict[str, Any]):
        # Implement logic to update the workflow
        pass

    async def _recommend_collaboration(self, task: Dict[str, Any]):
        prompt = f"""
        The following task appears to be new or unfamiliar:
        Task: {task['content']}
        
        Recommend a collaboration strategy with the respond tool to accomplish this task.
        Provide your response as a JSON object with the following structure:
        {{
            "collaboration_strategy": "Detailed description of the collaboration strategy",
            "respond_tool_involvement": "Description of how the respond tool will be involved"
        }}
        """
        collaboration_response = await self.llm.chat_with_ollama("You are an AI specializing in recommending collaboration strategies.", prompt)
        try:
            collaboration_strategy = json.loads(collaboration_response)
            logger.info(f"Recommended collaboration strategy: {collaboration_strategy}")
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response for collaboration recommendation.")
            logger.error(f"JSON parsing error: {collaboration_response}", exc_info=True)

    async def apply_learned_skill(self, task):
        relevant_skills = await self.knowledge_graph.get_relevant_skills(task)
        if relevant_skills:
            skill_prompt = f"""
            Task: {task}
            
            Relevant skills:
            {json.dumps(relevant_skills, indent=2)}
            
            Apply the most appropriate skill to complete the task.
            Provide the steps to execute the skill as a list of actions.
            """
            skill_application = await self.llm.chat_with_ollama("You are an expert in applying learned skills.", skill_prompt)
            return skill_application
        return None

    async def learn_new_skill(self, task, steps):
        skill_name = await self.generate_skill_name(task)
        new_skill = {
            "name": skill_name,
            "steps": steps,
            "created_at": datetime.now().isoformat()
        }
        await self.knowledge_graph.add_skill(new_skill)
        return new_skill

    async def generate_skill_name(self, task):
        name_prompt = f"Generate a short, descriptive name for a skill that accomplishes the following task: {task}"
        skill_name = await self.llm.chat_with_ollama("You are an expert in naming and categorizing skills.", name_prompt)
        return skill_name.strip()

