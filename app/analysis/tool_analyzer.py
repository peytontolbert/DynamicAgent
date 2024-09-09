import json
from app.chat_with_ollama import ChatGPT
from app.knowledge.knowledge_graph import KnowledgeGraph

class ToolAnalyzer:
    def __init__(self, llm: ChatGPT, knowledge_graph: KnowledgeGraph):
        self.llm = llm
        self.knowledge_graph = knowledge_graph

    async def analyze_tool_usage_patterns(self):
        tool_usage_history = await self.knowledge_graph.get_tool_usage_history(limit=10)
        
        prompt = f"""
        Analyze the following tool usage history and identify patterns or trends:
        {json.dumps(tool_usage_history, indent=2)}
        
        Provide your analysis as a JSON object with the following structure:
        {{
            "common_patterns": [string],
            "tool_effectiveness": {{
                "tool_name": float  // effectiveness score
            }},
            "improvement_suggestions": [string]
        }}
        """
        analysis = await self.llm.chat_with_ollama("You are an expert in analyzing AI tool usage patterns.", prompt)
        
        try:
            analysis_data = json.loads(analysis)
            await self.knowledge_graph.store_tool_usage_analysis(analysis_data)
            return analysis_data
        except json.JSONDecodeError:
            print("Error parsing tool usage analysis")
            return None