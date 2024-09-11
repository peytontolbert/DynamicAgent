import json
from app.chat_with_ollama import ChatGPT
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.knowledge.embedding_manager import EmbeddingManager
from app.knowledge.community_manager import CommunityManager
from app.utils.logger import StructuredLogger
from typing import Dict, Any, List, Optional
import asyncio

logger = StructuredLogger("SemanticKnowledge")

class SemanticKnowledgeSystem:
    def __init__(self, knowledge_graph: KnowledgeGraph, embedding_manager: EmbeddingManager):
        self.language_understanding = {}
        self.llm = ChatGPT()
        self.knowledge_graph = knowledge_graph
        self.embedding_manager = embedding_manager
        self.community_manager = CommunityManager(knowledge_graph, embedding_manager)

    async def log_language_data(self, phrase: str, meaning: str):
        self.language_understanding[phrase] = meaning
        await self.knowledge_graph.add_or_update_node("LanguageUnderstanding", {
            "phrase": phrase,
            "meaning": meaning,
            "embedding": self.embedding_manager.encode(phrase).tolist()
        })
        logger.info(f"Logged language data for phrase: {phrase}")

    async def retrieve_language_meaning(self, phrase: str) -> Optional[str]:
        # First, try to retrieve from local cache
        if phrase in self.language_understanding:
            return self.language_understanding[phrase]
        
        # If not in cache, query the knowledge graph
        result = await self.knowledge_graph.get_node("LanguageUnderstanding", {"phrase": phrase})
        if result:
            meaning = result.get('meaning')
            self.language_understanding[phrase] = meaning  # Update local cache
            return meaning
        
        return None

    async def export(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.language_understanding, f)
        logger.info(f"Exported language understanding data to {path}")

    async def import_data(self, path: str):
        with open(path, 'r') as f:
            self.language_understanding = json.load(f)
        for phrase, meaning in self.language_understanding.items():
            await self.log_language_data(phrase, meaning)
        logger.info(f"Imported language understanding data from {path}")

    async def enhance_language_understanding(self, phrase: str, context: Optional[str] = None) -> str:
        prompt = f"""
        Analyze the following phrase and provide a detailed understanding of its meaning:
        Phrase: {phrase}
        Context: {context or 'No additional context provided'}
        
        Please include:
        1. Literal meaning
        2. Possible figurative or idiomatic meanings
        3. Contextual implications
        4. Related concepts or phrases
        """
        meaning = await self.llm.chat_with_ollama("You are a language understanding expert.", prompt)
        await self.log_language_data(phrase, meaning.strip())
        return meaning.strip()

    async def link_related_phrases(self, phrase1: str, phrase2: str):
        await self.knowledge_graph.add_relationship(
            {"phrase": phrase1},
            {"phrase": phrase2},
            "RELATED_TO",
            {"type": "semantic_relation"}
        )
        logger.info(f"Linked related phrases: {phrase1} and {phrase2}")

    async def refine_language_meaning(self, phrase: str, additional_context: str) -> str:
        existing_meaning = await self.retrieve_language_meaning(phrase)
        prompt = f"""
        The phrase "{phrase}" currently means: {existing_meaning or 'No existing meaning found'}.
        With additional context: {additional_context}, update or refine the understanding of this phrase.
        
        Please provide:
        1. Updated meaning
        2. Explanation of how the additional context affects the understanding
        3. Any new related concepts or phrases
        """
        refined_meaning = await self.llm.chat_with_ollama("You are a language understanding expert.", prompt)
        await self.log_language_data(phrase, refined_meaning.strip())
        return refined_meaning.strip()

    async def find_similar_phrases(self, phrase: str, k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.encode(phrase)
        similar_nodes = await self.knowledge_graph.get_similar_nodes(query_embedding, "LanguageUnderstanding", k)
        return [{"phrase": node["phrase"], "meaning": node["meaning"], "similarity": similarity} 
                for node, similarity in similar_nodes]

    async def generate_example_sentences(self, phrase: str, num_examples: int = 3) -> List[str]:
        prompt = f"""
        Generate {num_examples} example sentences using the phrase: "{phrase}"
        Ensure the examples showcase different contexts or meanings of the phrase.
        """
        response = await self.llm.chat_with_ollama("You are a language expert.", prompt)
        return [sentence.strip() for sentence in response.split('\n') if sentence.strip()]

    async def analyze_semantic_relations(self, phrase: str) -> Dict[str, Any]:
        related_phrases = await self.knowledge_graph.get_related_nodes("LanguageUnderstanding", {"phrase": phrase})
        prompt = f"""
        Analyze the semantic relations between "{phrase}" and the following related phrases:
        {', '.join([node['phrase'] for node in related_phrases])}
        
        Provide:
        1. Type of relation (e.g., synonym, antonym, hypernym, hyponym)
        2. Explanation of the relationship
        """
        analysis = await self.llm.chat_with_ollama("You are a semantic analysis expert.", prompt)
        return {
            "phrase": phrase,
            "related_phrases": [node['phrase'] for node in related_phrases],
            "analysis": analysis.strip()
        }

    async def batch_process_phrases(self, phrases: List[str]):
        tasks = [self.enhance_language_understanding(phrase) for phrase in phrases]
        await asyncio.gather(*tasks)
        logger.info(f"Batch processed {len(phrases)} phrases")

    async def organize_semantic_communities(self):
        await self.community_manager.initialize()
        logger.info("Semantic concepts organized into communities")

    async def get_semantic_community_summary(self, community_id: str) -> str:
        return self.community_manager.community_summaries.get(community_id, "Community not found")

    async def query_semantic_communities(self, query: str) -> str:
        return await self.community_manager.query_communities(query)

    async def update_semantic_knowledge(self, phrase: str, meaning: str):
        await self.log_language_data(phrase, meaning)
        new_knowledge = {
            "id": phrase,
            "label": "LanguageUnderstanding",
            "content": f"{phrase}: {meaning}"
        }
        await self.community_manager.update_knowledge(new_knowledge)

    async def get_related_semantic_concepts(self, phrase: str, k: int = 5) -> List[Dict[str, Any]]:
        community_id = self.community_manager.communities.get(phrase)
        if community_id is None:
            return []
        
        community_nodes = [node for node in self.community_manager.graph.nodes 
                           if self.community_manager.communities[node] == community_id]
        
        related_concepts = []
        for node in community_nodes:
            if node != phrase:
                meaning = await self.retrieve_language_meaning(node)
                related_concepts.append({
                    "phrase": node,
                    "meaning": meaning,
                    "community_id": community_id
                })
        
        return related_concepts[:k]

    async def generate_community_based_examples(self, phrase: str, num_examples: int = 3) -> List[str]:
        related_concepts = await self.get_related_semantic_concepts(phrase)
        related_phrases = ", ".join([concept["phrase"] for concept in related_concepts])
        
        prompt = f"""
        Generate {num_examples} example sentences using the phrase: "{phrase}"
        Consider using these related phrases for context: {related_phrases}
        Ensure the examples showcase different contexts or meanings of the phrase within its semantic community.
        """
        response = await self.llm.chat_with_ollama("You are a language expert.", prompt)
        return [sentence.strip() for sentence in response.split('\n') if sentence.strip()]

    async def analyze_semantic_community(self, phrase: str) -> Dict[str, Any]:
        community_id = self.community_manager.communities.get(phrase)
        if community_id is None:
            return {"error": "Phrase not found in any semantic community"}
        
        community_summary = await self.get_semantic_community_summary(community_id)
        related_concepts = await self.get_related_semantic_concepts(phrase)
        
        prompt = f"""
        Analyze the semantic community containing the phrase "{phrase}".
        Community summary: {community_summary}
        Related concepts: {', '.join([concept['phrase'] for concept in related_concepts])}
        
        Provide:
        1. Overall theme or domain of this semantic community
        2. How "{phrase}" fits within this community
        3. Potential applications or contexts where this semantic community is relevant
        """
        analysis = await self.llm.chat_with_ollama("You are a semantic analysis expert.", prompt)
        
        return {
            "phrase": phrase,
            "community_id": community_id,
            "community_summary": community_summary,
            "related_concepts": related_concepts,
            "analysis": analysis.strip()
        }
