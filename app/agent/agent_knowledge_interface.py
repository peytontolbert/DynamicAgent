from app.execution.code_execution_manager import CodeExecutionManager
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.virtual_env.virtual_environment import VirtualEnvironment
from app.knowledge.procedural_knowledge import ProceduralKnowledgeSystem
from app.knowledge.episodic_knowledge import EpisodicKnowledgeSystem
from app.knowledge.conceptual_knowledge import ConceptualKnowledgeSystem
from app.knowledge.contextual_knowledge import ContextualKnowledgeSystem
from app.knowledge.meta_cognitive_knowledge import MetaCognitiveKnowledgeSystem
from app.knowledge.semantic_knowledge import SemanticKnowledgeSystem
from app.knowledge.embedding_manager import EmbeddingManager


class AgentKnowledgeInterface:
    def __init__(self, uri, user, password, base_path):
        self.knowledge_graph = KnowledgeGraph(uri, user, password)
        self.embedding_manager = EmbeddingManager()
        self.procedural_memory = ProceduralKnowledgeSystem(
            self.knowledge_graph, self.embedding_manager
        )
        self.episodic_memory = EpisodicKnowledgeSystem(
            self.knowledge_graph, self.embedding_manager
        )
        self.conceptual_knowledge = ConceptualKnowledgeSystem(self.knowledge_graph)
        self.contextual_knowledge = ContextualKnowledgeSystem(self.knowledge_graph)
        self.meta_cognitive = MetaCognitiveKnowledgeSystem(self.knowledge_graph)
        self.semantic_knowledge = SemanticKnowledgeSystem(
            self.knowledge_graph, self.embedding_manager
        )

    async def gather_knowledge(self, task: str, context: str) -> dict:
        procedural_info = await self.procedural_memory.retrieve_tool_usage(task)

        # Retrieve related concepts from conceptual knowledge
        related_concepts = self.conceptual_knowledge.get_related_concepts(task)

        # Reflect on past performance using meta-cognitive knowledge
        performance_data = self.meta_cognitive.get_performance(task)

        # Ensure the task is understood using semantic knowledge
        interpreted_task = await self.semantic_knowledge.retrieve_language_meaning(task)
        if not interpreted_task:
            interpreted_task = (
                await self.semantic_knowledge.enhance_language_understanding(task)
            )

        # Retrieve generalized knowledge for the task
        concepts = await self.meta_cognitive.extract_concepts(task)
        generalized_knowledge = await self.meta_cognitive.get_generalized_knowledge(
            concepts
        )

        return {
            "procedural_info": procedural_info,
            "related_concepts": related_concepts,
            "context_info": context,
            "performance_data": performance_data,
            "interpreted_task": interpreted_task,
            "generalized_knowledge": generalized_knowledge,
        }

    async def update_knowledge_step(
        self, task: str, result: str, action: str, context: str, thoughts: str
    ):
        await self.episodic_memory.log_task(task, result, context)
        await self.meta_cognitive.log_performance(
            task, {"result": result, "action": action}
        )

        if action == "code_execute":
            insights, tool_usage = (
                await self.procedural_memory.enhance_procedural_knowledge(
                    task, result, context
                )
            )
            self.logging_manager.log_info(f"Procedural Knowledge Insights: {insights}")
            self.logging_manager.log_info(f"Tool Usage Recommendations: {tool_usage}")

        # Extract concepts for this step
        concepts = await self.meta_cognitive.extract_concepts(task)

        self.logging_manager.log_info(f"Step Extracted Concepts: {concepts}")
        self.task_history.append(
            {
                "task": task,
                "result": result,
                "concepts": concepts,
                "context": context,
                "action": action,
                "thoughts": thoughts,
                "procedural_insights": insights if action == "code_execute" else None,
                "tool_usage": tool_usage if action == "code_execute" else None,
            }
        )

    async def update_knowledge_complete(
        self, task: str, result: str, action: str, context: str, thoughts: str
    ):
        # Extract concepts and generalize knowledge for the complete task
        concepts = await self.meta_cognitive.extract_concepts(task)
        generalized_knowledge = await self.meta_cognitive.generalize_knowledge(concepts)

        self.logging_manager.log_info(
            f"Task Completed - Extracted Concepts: {concepts}"
        )
        self.logging_manager.log_info(
            f"Task Completed - Generalized Knowledge: {generalized_knowledge}"
        )

        # Update the final task history entry with the generalized knowledge
        if self.task_history:
            self.task_history[-1]["generalized_knowledge"] = generalized_knowledge

        # Additional updates for task completion
        await self.conceptual_knowledge.update_concept_relations(concepts)
        await self.contextual_knowledge.update_context(task, context, result)
        await self.semantic_knowledge.update_language_understanding(task, result)

        if action == "code_execute":
            await self.procedural_memory.enhance_tool_usage(task)
