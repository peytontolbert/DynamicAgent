# Dynamic Agent

Dynamic Agent is an intelligent, adaptive system designed to process tasks using two primary tools: Respond and Code Execute. It leverages a sophisticated knowledge graph and decision-making workflow to accomplish a wide range of tasks, with the ability to confirm task completion directly with the user.

## Features

- **Dual-Tool System**: Utilizes 'Respond' for information retrieval, clarification, and task completion confirmation, and 'Code Execute' for task execution.
- **Knowledge-Driven Decision Making**: Uses a robust knowledge graph to inform tool selection and task execution.
- **Continuous Learning**: Improves its knowledge and capabilities over time through task results and user feedback.
- **Adaptive Code Execution**: Generates and executes Python, JavaScript, or Bash code to perform complex tasks.
- **Virtual Environment Management**: Creates isolated environments for safe code execution.
- **Performance Monitoring**: Tracks and logs the performance of various operations.
- **User-Driven Task Completion**: Confirms task completion directly with the user, ensuring satisfaction with the results.
- **Error Analysis and Recovery**: Analyzes execution errors and suggests fixes or next steps.

## Components

1. **ChatGPT (LLM)**: Handles natural language processing and generation.
2. **CodeExecutionManager**: Manages the generation and execution of code in multiple languages.
3. **KnowledgeGraph**: Stores and retrieves relevant knowledge for tasks using Neo4j.
4. **VirtualEnvironment**: Creates isolated environments for safe code execution.
5. **ContinuousLearner**: Improves the system's knowledge over time.
6. **LoggingManager**: Handles logging of system operations.
7. **ContextManager**: Manages task history and working memory.
8. **RewardModel**: Evaluates task performance and provides feedback for learning.

## Workflow

1. The user inputs a task.
2. The system analyzes the task, relevant knowledge, and context to decide whether to use 'Respond' or 'Code Execute'.
3. If 'Respond' is chosen:
   a. The system formulates a question or request for more information.
   b. The user provides the requested information.
   c. The system may ask follow-up questions or confirm task completion.
4. If 'Code Execute' is chosen:
   a. The system generates thoughts on how to approach the task.
   b. Code is generated in the appropriate language (Python, JavaScript, or Bash).
   c. The code is executed in a virtual environment.
   d. The system presents the results to the user.
5. The system asks the user to confirm if the task is complete.
6. The system learns from the task result, user feedback, and updates its knowledge graph.
7. Steps 2-6 repeat until the user confirms the task is fully accomplished.

## Key Features in Detail

### Code Execution
- Supports Python, JavaScript, and Bash execution.
- Uses a virtual environment for safe execution.
- Monitors execution progress and provides real-time status updates.

### Knowledge Graph
- Uses Neo4j for storing and retrieving knowledge.
- Supports adding task results, improvement suggestions, and tool usage information.
- Allows for complex querying and relationship management.

### Continuous Learning
- Evaluates task performance using a reward model.
- Updates the knowledge graph based on task results and user feedback.
- Adapts decision-making based on past experiences.

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

Once the Dynamic Agent is running, you can interact with it by entering tasks. The agent will automatically determine whether to provide a response or execute code based on the task.

Example tasks:
- "What is the capital of France?" (Response)
- "Calculate the factorial of 5" (Code Execution)
- "Create a list of prime numbers up to 100" (Code Execution)
- "Explain the concept of recursion" (Response)

The agent will guide you through the process, asking for additional information if needed and confirming when the task is complete.

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

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- OpenAI for the GPT model used in natural language processing
- Neo4j for the graph database used in knowledge management
- All contributors who have helped to improve and expand this project
