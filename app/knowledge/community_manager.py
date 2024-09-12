import networkx as nx
import community as community_louvain
from typing import List, Dict, Any
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.knowledge.embedding_manager import EmbeddingManager
from app.utils.logger import StructuredLogger
from app.chat_with_ollama import ChatGPT

"""
This module manages communities within the knowledge graph.
It builds a graph from knowledge, detects communities, generates summaries, and handles queries by generating partial answers from relevant communities.

Key features:
- Builds a graph from knowledge data
- Detects communities using the Louvain algorithm
- Generates summaries for each community
- Handles queries by generating partial answers from relevant communities
"""

logger = StructuredLogger("CommunityManager")


class CommunityManager:
    def __init__(
        self, knowledge_graph: KnowledgeGraph, embedding_manager: EmbeddingManager
    ):
        self.knowledge_graph = knowledge_graph
        self.embedding_manager = embedding_manager
        self.llm = ChatGPT()
        self.graph = nx.Graph()
        self.communities = {}
        self.community_summaries = {}

    async def build_graph_from_knowledge(self):
        """Build a NetworkX graph from the knowledge graph."""
        nodes = await self.knowledge_graph.get_all_knowledge()
        for node in nodes:
            self.graph.add_node(node["id"], **node)

        # Add edges based on relationships in the knowledge graph
        # This is a simplified version and may need to be adapted based on your specific graph structure
        relationships = await self.knowledge_graph.execute_query(
            "MATCH (a)-[r]->(b) RETURN a.id, b.id, type(r)"
        )
        for rel in relationships:
            self.graph.add_edge(rel["a.id"], rel["b.id"], type=rel["type(r)"])

        logger.info(
            f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
        )

    def detect_communities(self):
        """Detect communities in the graph using the Louvain algorithm."""
        self.communities = community_louvain.best_partition(self.graph)
        logger.info(f"Detected {len(set(self.communities.values()))} communities")

    async def summarize_communities(self):
        """Generate summaries for each community."""
        for community_id in set(self.communities.values()):
            community_nodes = [
                node
                for node in self.graph.nodes
                if self.communities[node] == community_id
            ]
            community_content = " ".join(
                [self.graph.nodes[node].get("content", "") for node in community_nodes]
            )

            # Use the generate_summary method to create a summary
            summary = await self.generate_summary(community_content)
            self.community_summaries[community_id] = summary

        logger.info(
            f"Generated summaries for {len(self.community_summaries)} communities"
        )

    async def query_communities(self, query: str) -> str:
        """
        Handle a query by generating partial answers from relevant communities and combining them.

        Args:
            query (str): The user's query.

        Returns:
            str: The combined answer to the query.
        """
        query_embedding = self.embedding_manager.encode(query)
        relevant_communities = []

        for community_id, summary in self.community_summaries.items():
            summary_embedding = self.embedding_manager.encode(summary)
            similarity = self.embedding_manager.cosine_similarity(
                query_embedding, summary_embedding
            )
            if similarity > 0.5:  # Adjust this threshold as needed
                relevant_communities.append((community_id, similarity))

        relevant_communities.sort(key=lambda x: x[1], reverse=True)

        partial_answers = []
        for community_id, _ in relevant_communities[
            :3
        ]:  # Consider top 3 most relevant communities
            community_nodes = [
                node
                for node in self.graph.nodes
                if self.communities[node] == community_id
            ]
            community_content = " ".join(
                [self.graph.nodes[node].get("content", "") for node in community_nodes]
            )

            partial_answer = await self.generate_partial_answer(
                community_content, query
            )
            partial_answers.append(partial_answer)

        final_answer = await self.combine_partial_answers(partial_answers, query)

        logger.info(f"Generated answer for query: {query[:50]}...")
        return final_answer

    async def generate_partial_answer(self, community_content: str, query: str) -> str:
        """
        Generate a partial answer based on community content and the query.

        Args:
            community_content (str): The content of the community.
            query (str): The user's query.

        Returns:
            str: A partial answer generated from the community content.
        """
        # Prepare the prompt for the LLM
        prompt = f"""
        Given the following community content and user query, generate a relevant partial answer:

        Community Content:
        {community_content}

        User Query:
        {query}

        Partial Answer:
        """

        # Use the LLM to generate the partial answer
        partial_answer = await self.llm.chat_with_ollama(
            "You are an AI assistant tasked with generating a partial answer based on the given community content and user query.",
            prompt,
        )

        return partial_answer.strip()

    async def combine_partial_answers(
        self, partial_answers: List[str], query: str
    ) -> str:
        """
        Combine partial answers into a final, coherent answer.

        Args:
            partial_answers (List[str]): List of partial answers from different communities.
            query (str): The original user query.

        Returns:
            str: A combined, coherent answer to the user's query.
        """
        # Prepare the prompt for the LLM
        prompt = f"""
        Given the following partial answers to the user's query, combine them into a coherent, comprehensive answer:

        User Query:
        {query}

        Partial Answers:
        {' '.join([f'- {answer}' for answer in partial_answers])}

        Combined Answer:
        """

        # Use the LLM to generate the combined answer
        combined_answer = await self.llm.chat_with_ollama(
            system_prompt="You are an AI assistant tasked with combining and refining partial answers into a coherent answer.",
            user_prompt=prompt,
        )

        return combined_answer.strip()

    async def generate_summary(self, content: str) -> str:
        """
        Generate a summary for a community's content.

        Args:
            content (str): The content of the community to summarize.

        Returns:
            str: A summary of the community content.
        """
        # Prepare the prompt for the LLM
        prompt = f"""
        Please provide a concise summary of the following community content. 
        Focus on the main topics, key ideas, and any notable relationships or patterns:

        Community Content:
        {content}

        Summary:
        """

        # Use the LLM to generate the summary
        summary = await self.llm.chat_with_ollama(
            "You are an AI assistant tasked with summarizing community content.", prompt
        )

        return summary.strip()

    async def update_knowledge(self, new_knowledge: Dict[str, Any]):
        """
        Update the knowledge graph with new information and update affected communities.

        Args:
            new_knowledge (Dict[str, Any]): The new knowledge to be added.
        """
        # Add the new knowledge to the graph
        await self.knowledge_graph.add_or_update_node(
            new_knowledge["label"], new_knowledge
        )

        # Update the NetworkX graph
        self.graph.add_node(new_knowledge["id"], **new_knowledge)

        # Re-detect communities (this could be optimized for incremental updates in a more advanced implementation)
        self.detect_communities()

        # Update summaries for affected communities
        affected_community = self.communities[new_knowledge["id"]]
        await self.summarize_communities()

        logger.info(
            f"Updated knowledge and communities for new node: {new_knowledge['id']}"
        )

    async def initialize(self):
        """Initialize the CommunityManager by building the graph and detecting initial communities."""
        await self.build_graph_from_knowledge()
        self.detect_communities()
        await self.summarize_communities()
        logger.info("Initialized CommunityManager")

    async def get_relevant_communities(self, query: str) -> List[int]:
        """
        Get the most relevant communities for a given query.

        Args:
            query (str): The query to find relevant communities for.

        Returns:
            List[int]: A list of community IDs sorted by relevance.
        """
        query_embedding = self.embedding_manager.encode(query)
        relevant_communities = []

        for community_id, summary in self.community_summaries.items():
            summary_embedding = self.embedding_manager.encode(summary)
            similarity = self.embedding_manager.cosine_similarity(
                query_embedding, summary_embedding
            )
            relevant_communities.append((community_id, similarity))

        # Sort communities by similarity (descending) and return top 3
        relevant_communities.sort(key=lambda x: x[1], reverse=True)
        return [community_id for community_id, _ in relevant_communities[:3]]

    # Note: You'll need to implement or mock these methods in the EmbeddingManager:
    # - generate_summary
    # - generate_partial_answer
    # - combine_partial_answers
