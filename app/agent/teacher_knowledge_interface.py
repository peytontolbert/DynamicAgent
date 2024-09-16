from app.agent.agent_knowledge_interface import AgentKnowledgeInterface

class TeacherKnowledgeInterface:
    def __init__(self, agent_knowledge_interface: AgentKnowledgeInterface):
        self.agent_knowledge_interface = agent_knowledge_interface

    async def fetch_knowledge(self, task: str):
        return await self.agent_knowledge_interface.gather_knowledge(task)

    async def update_knowledge(self, event_type: str, *args):
        await self.agent_knowledge_interface.update_knowledge_step(event_type, *args)
