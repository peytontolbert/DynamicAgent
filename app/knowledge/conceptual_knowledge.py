import json
from app.chat_with_ollama import ChatGPT
from app.knowledge.knowledge_graph import KnowledgeGraph


class ConceptualKnowledgeSystem:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.concept_graph = {}
        self.knowledge_graph = knowledge_graph
        self.llm = ChatGPT()

    def add_concept(self, concept, related_concepts):
        if concept not in self.concept_graph:
            self.concept_graph[concept] = []
        self.concept_graph[concept].extend(related_concepts)
        self.knowledge_graph.add_or_update_node(
            "ConceptGraph", {"concept": concept, "related_concepts": related_concepts}
        )

    def get_related_concepts(self, concept):
        return self.concept_graph.get(concept, [])

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
        self.add_concept(concept, related_concepts.strip().split(","))
        return related_concepts.strip()
