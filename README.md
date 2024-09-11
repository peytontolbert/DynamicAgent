# Dynamic Agent

Dynamic Agent is an intelligent, adaptive system designed to process tasks using a sophisticated knowledge graph, community detection, and decision-making workflow. It leverages various knowledge systems and tools to accomplish a wide range of tasks, with the ability to learn and improve over time.

## Features

- **Multi-Tool System**: Utilizes various tools for information retrieval, task execution, and knowledge management.
- **Knowledge-Driven Decision Making**: Uses a robust knowledge graph and community detection to inform tool selection and task execution.
- **Continuous Learning**: Improves its knowledge and capabilities over time through task results and user feedback.
- **Adaptive Code Execution**: Generates and executes Python, JavaScript, or Bash code to perform complex tasks.
- **Virtual Environment Management**: Creates isolated environments for safe code execution.
- **Performance Monitoring**: Tracks and logs the performance of various operations.
- **User-Driven Task Completion**: Confirms task completion directly with the user, ensuring satisfaction with the results.
- **Error Analysis and Recovery**: Analyzes execution errors and suggests fixes or next steps.
- **Community Detection**: Uses graph-based community detection to organize and summarize knowledge.
- **Embedding-based Similarity Search**: Utilizes embeddings for efficient similarity search and knowledge retrieval.

## Components

1. **ChatGPT (LLM)**: Handles natural language processing and generation.
2. **CodeExecutionManager**: Manages the generation and execution of code in multiple languages.
3. **KnowledgeGraph**: Stores and retrieves relevant knowledge using Neo4j.
4. **EmbeddingManager**: Manages text embeddings for efficient similarity search and knowledge retrieval.
5. **CommunityManager**: Detects and manages communities within the knowledge graph for better organization and summarization.
6. **VirtualEnvironment**: Creates isolated environments for safe code execution.
7. **LoggingManager**: Handles logging of system operations.
8. **ContextManager**: Manages task history and working memory.
9. **Specialized Knowledge Systems**: Including Semantic, MetaCognitive, Conceptual, Procedural, Episodic, and Contextual knowledge systems.

## Workflow

1. The user inputs a task.
2. The system analyzes the task, relevant knowledge, and context to decide on the best course of action.
3. The system may:
   a. Generate a response using its knowledge systems.
   b. Execute code to perform the task.
   c. Query the user for more information.
4. The system updates its knowledge graph and communities based on the task result.
5. The system asks the user to confirm if the task is complete.
6. Steps 2-5 repeat until the user confirms the task is fully accomplished.

## Key Features in Detail

### Knowledge Graph and Community Detection
- Uses Neo4j for storing and retrieving knowledge.
- Implements community detection to organize knowledge into related clusters.
- Supports adding task results, improvement suggestions, and tool usage information.
- Allows for complex querying and relationship management.

### Embedding-based Similarity Search
- Utilizes sentence transformers to generate embeddings for efficient similarity search.
- Supports both in-memory and FAISS-based indexing for scalable similarity search.

### Continuous Learning
- Evaluates task performance and updates the knowledge graph.
- Adapts decision-making based on past experiences and community structure.

### Error Handling
- Analyzes execution errors and provides suggested fixes.
- Supports code adaptation based on error analysis.

## Setup

1. Clone the repository.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up a Neo4j database and configure the connection details in a `.env` file:
   ```
   NEO4J_URI=<your-neo4j-uri>
   NEO4J_USER=<your-neo4j-username>
   NEO4J_PASSWORD=<your-neo4j-password>
   VIRTUAL_ENV_BASE_PATH=./virtual_env
   ```
4. Run the main script:
   ```
   python main.py
   ```

## Usage

Once the Dynamic Agent is running, you can interact with it by entering tasks. The agent will automatically determine the best course of action based on the task and its current knowledge.

Example tasks:
- "What is the capital of France?"
- "Calculate the factorial of 5"
- "Create a list of prime numbers up to 100"
- "Summarize the main themes in the knowledge graph"
- "Find similar concepts to 'machine learning' in the knowledge base"

The agent will process the task, potentially executing code, querying its knowledge graph, or generating responses based on its specialized knowledge systems. It will ask for confirmation when the task is complete.

To exit the program, simply type 'exit'.

## Advanced Features

### Knowledge Import/Export
The system supports importing and exporting knowledge, allowing for:
- Bootstrapping the agent with initial knowledge
- Sharing knowledge between different instances of the agent
- Backing up and restoring the agent's knowledge base

### Performance Analytics
The system tracks various performance metrics, which can be used to:
- Identify areas for improvement
- Optimize decision-making processes
- Track the agent's learning progress over time

## Contributing

Contributions to the Dynamic Agent project are welcome. Please ensure to follow the coding standards and submit pull requests for any new features or bug fixes.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgements

- Ollama for the GPT model used in natural language processing
- Neo4j for the graph database used in knowledge management
- Sentence Transformers for generating text embeddings
- FAISS for efficient similarity search
