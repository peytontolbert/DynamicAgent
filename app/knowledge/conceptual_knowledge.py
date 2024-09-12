import json
from app.chat_with_ollama import ChatGPT
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.knowledge.community_manager import CommunityManager
from app.knowledge.embedding_manager import EmbeddingManager

"""
The `ConceptualKnowledgeSystem` manages conceptual knowledge and relationships between concepts.

Key Features:
- **Concept Storage**: Stores and retrieves related concepts.
- **Concept Enhancement**: Enhances concept understanding through analysis.
- **Integration with Knowledge Graph**: Adds and updates nodes in the knowledge graph.
- **Community Management**: Manages communities of related concepts.
- **LLM Integration**: Uses a language model for concept analysis and enhancement.
"""


class ConceptualKnowledgeSystem:
    def __init__(
        self, knowledge_graph: KnowledgeGraph, embedding_manager: EmbeddingManager
    ):
        self.knowledge_graph = knowledge_graph
        self.embedding_manager = embedding_manager
        self.llm = ChatGPT()
        self.community_manager = CommunityManager(knowledge_graph, embedding_manager)

    async def add_concept(self, concept, related_concepts):
        if concept not in self.concept_graph:
            self.concept_graph[concept] = []
        self.concept_graph[concept].extend(related_concepts)
        await self.knowledge_graph.add_or_update_node(
            "ConceptGraph", {"concept": concept, "related_concepts": related_concepts}
        )

    async def get_related_concepts(self, concept):
        related_concepts = self.concept_graph.get(concept, [])
        if not related_concepts:
            related_concepts = await self.enhance_concept(concept)
        return related_concepts

    def export(self, path):
        with open(path, "w") as f:
            json.dump(self.concept_graph, f)

    def import_data(self, path):
        with open(path, "r") as f:
            self.concept_graph = json.load(f)
            for concept, related_concepts in self.concept_graph.items():
                self.knowledge_graph.add_or_update_node(
                    "ConceptGraph",
                    {"concept": concept, "related_concepts": related_concepts},
                )

    async def enhance_concept(self, concept):
        prompt = f"""
        Analyze the following concept and provide related concepts and detailed information:
        Concept: {concept}
        """
        related_concepts = await self.llm.chat_with_ollama(
            "You are a concept analysis expert.", prompt
        )
        await self.add_concept(concept, related_concepts.strip().split(","))
        return related_concepts.strip()

    async def analyze_concepts(self, concept: str):
        prompt = f"""
        Analyze the following concept and provide detailed insights:
        Concept: {concept}

        Provide insights on:
        1. What are the related concepts?
        2. How does this concept relate to other concepts?
        3. Are there any patterns or relationships that could be applied to similar concepts?

        Format your response as:
        Insights: <Your analysis>
        Related Concepts: <Specific related concepts>
        """
        analysis = await self.llm.chat_with_ollama(
            "You are a conceptual analysis expert.", prompt
        )
        return analysis.strip()
