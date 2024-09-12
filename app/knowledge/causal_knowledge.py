from typing import Dict, Any, List, Tuple
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.knowledge.embedding_manager import EmbeddingManager
from app.knowledge.community_manager import CommunityManager

"""
This module manages knowledge about cause-and-effect relationships.
It logs causal relationships, retrieves causes and outcomes, generates causal chains, and calculates success rates for actions.

Key features:
- Logs causal relationships between actions and outcomes
- Retrieves causes and outcomes based on context
- Generates causal chains for actions
- Calculates success rates for actions
"""


class CausalKnowledgeSystem:
    def __init__(
        self, knowledge_graph: KnowledgeGraph, embedding_manager: EmbeddingManager
    ):
        self.knowledge_graph = knowledge_graph
        self.embedding_manager = embedding_manager
        self.community_manager = CommunityManager(knowledge_graph, embedding_manager)

    async def log_causal_relationship(self, action: str, outcome: str, context: str):
        """
        Log a causal relationship between an action and its outcome.
        """
        action_node = {"type": "Action", "content": action, "context": context}
        outcome_node = {"type": "Outcome", "content": outcome, "context": context}

        await self.knowledge_graph.add_or_update_node("CausalAction", action_node)
        await self.knowledge_graph.add_or_update_node("CausalOutcome", outcome_node)
        await self.knowledge_graph.add_relationship(action_node, outcome_node, "CAUSED")

        new_knowledge = {
            "id": action_node["id"],
            "label": "CausalAction",
            "content": action,
        }
        await self.community_manager.update_knowledge(new_knowledge)

    async def retrieve_causes(self, outcome: str, context: str) -> List[str]:
        """
        Retrieve actions that led to a specific outcome in a similar context.
        """
        query = f"""
        MATCH (a:CausalAction)-[:CAUSED]->(o:CausalOutcome)
        WHERE o.content = $outcome AND o.context = $context
        RETURN a.content AS action
        """
        results = await self.knowledge_graph.execute_query(
            query, {"outcome": outcome, "context": context}
        )
        return [result["action"] for result in results]

        relevant_communities = await self.community_manager.get_relevant_communities(
            outcome
        )
        # Use relevant_communities to refine the query or results

    async def retrieve_outcomes(self, action: str, context: str) -> List[str]:
        """
        Retrieve outcomes that resulted from a specific action in a similar context.
        """
        query = f"""
        MATCH (a:CausalAction)-[:CAUSED]->(o:CausalOutcome)
        WHERE a.content = $action AND a.context = $context
        RETURN o.content AS outcome
        """
        results = await self.knowledge_graph.execute_query(
            query, {"action": action, "context": context}
        )
        return [result["outcome"] for result in results]

        relevant_communities = await self.community_manager.get_relevant_communities(
            action
        )
        # Use relevant_communities to refine the query or results

    async def get_causal_chain(
        self, initial_action: str, depth: int = 3
    ) -> List[Tuple[str, str]]:
        """
        Retrieve a causal chain starting from an initial action up to a specified depth.
        """
        query = f"""
        MATCH path = (a:CausalAction {{content: $initial_action}})-[:CAUSED*1..{depth}]->(o:CausalOutcome)
        RETURN [node in nodes(path) | node.content] AS chain
        """
        results = await self.knowledge_graph.execute_query(
            query, {"initial_action": initial_action}
        )

        causal_chains = []
        for result in results:
            chain = result["chain"]
            causal_chains.extend(zip(chain[::2], chain[1::2]))

        return causal_chains

    async def get_most_common_outcome(self, action: str) -> str:
        """
        Retrieve the most common outcome for a given action.
        """
        query = """
        MATCH (a:CausalAction {content: $action})-[:CAUSED]->(o:CausalOutcome)
        RETURN o.content AS outcome, COUNT(*) AS frequency
        ORDER BY frequency DESC
        LIMIT 1
        """
        results = await self.knowledge_graph.execute_query(query, {"action": action})
        return results[0]["outcome"] if results else None

    async def get_action_success_rate(self, action: str) -> float:
        """
        Calculate the success rate of a given action.
        """
        query = """
        MATCH (a:CausalAction {content: $action})-[:CAUSED]->(o:CausalOutcome)
        WITH a, o,
             CASE WHEN o.content CONTAINS 'success' OR o.content CONTAINS 'completed' THEN 1 ELSE 0 END AS success
        RETURN toFloat(SUM(success)) / COUNT(*) AS success_rate
        """
        results = await self.knowledge_graph.execute_query(query, {"action": action})
        return results[0]["success_rate"] if results else 0.0
