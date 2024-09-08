# Dynamic Agent

Dynamic Agent is an intelligent, adaptive system designed to process tasks, execute code, and learn continuously. It leverages natural language processing and code execution capabilities to provide a versatile interface for various operations.

## Features

- **Task Analysis**: Automatically determines whether to respond with information or execute code based on the given task.
- **Natural Language Responses**: Provides informative answers to queries using its knowledge base.
- **Code Execution**: Generates and executes Python or JavaScript code to perform complex tasks.
- **Continuous Learning**: Improves its knowledge and capabilities over time through task results.
- **Error Analysis**: Provides suggestions for fixing errors in code execution.
- **Virtual Environment Management**: Creates isolated environments for code execution.
- **Workspace Management**: Handles project workspaces efficiently.
- **Performance Monitoring**: Tracks and logs the performance of various operations.

## Components

1. **ChatGPT**: Handles natural language processing and generation.
2. **CodeExecutionManager**: Manages the execution of generated code.
3. **KnowledgeGraph**: Stores and retrieves relevant knowledge for tasks.
4. **VirtualEnvironment**: Creates isolated environments for code execution.
5. **ContinuousLearner**: Improves the system's knowledge over time.
6. **LoggingManager**: Handles logging of system operations.
7. **PerformanceMonitor**: Tracks the performance of various system components.
8. **WorkspaceManager**: Manages project workspaces.
9. **ProjectManager**: Handles project-related operations.

## Setup

1. Clone the repository.
2. Install the required dependencies (list them in a `requirements.txt` file).
3. Set up a Neo4j database and configure the connection details in a `.env` file:

NEO4J_URI=<your-neo4j-uri>
NEO4J_USER=<your-neo4j-username>
NEO4J_PASSWORD=<your-neo4j-password>
VIRTUAL_ENV_BASE_PATH=./virtual_env


4. Run the main script:

python main.py

## Usage

Once the Dynamic Agent is running, you can interact with it by entering tasks. The agent will automatically determine whether to provide a response or execute code based on the task.

Example tasks:
- "What is the capital of France?" (Response)
- "Calculate the factorial of 5" (Code Execution)
- "Create a list of prime numbers up to 100" (Code Execution)

To exit the program, simply type 'exit'.

## Contributing

Contributions to the Dynamic Agent project are welcome. Please ensure to follow the coding standards and submit pull requests for any new features or bug fixes.
