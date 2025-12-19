import os
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import MessageRole, FilePurpose, FunctionTool, ToolSet, FileSearchTool
from tools import calculate_pizza_for_people
from dotenv import load_dotenv


load_dotenv(override=True)

# Create Project Client Instance
project_client = AIProjectClient(
    endpoint=os.environ["PROJECT_CONNECTION_STRING"],
    credential=DefaultAzureCredential()
)

DOCS_DIR = "./documents"

# 1. Upload the files (you already have this part, but here it is for context)
file_ids = []
for fname in os.listdir(DOCS_DIR):
    fpath = os.path.join(DOCS_DIR, fname)
    if not os.path.isfile(fpath) or fname.startswith('.'):
        continue
    print(f"Uploading: {fname}")
    uploaded = project_client.agents.files.upload_and_poll(
        file_path=fpath,
        purpose=FilePurpose.AGENTS
    )
    file_ids.append(uploaded.id)
    print(f"Uploaded {fname}, ID: {uploaded.id}")

if not file_ids:
    raise RuntimeError("No files uploaded. Put files in ./documents and re-run.")

print(f"Total files uploaded: {len(file_ids)}")

vector_store = project_client.agents.vector_stores.create_and_poll(
    file_ids=file_ids,
    name="contoso-pizza-store-information"
)

print(f"Created vector store, ID: {vector_store.id}")

file_search = FileSearchTool(vector_store_ids=[vector_store.id])


# Create the function tool
function_tool = FunctionTool(functions={calculate_pizza_for_people})

# Creating the toolset
toolset = ToolSet()
toolset.add(file_search)
toolset.add(function_tool)

# Creating the agent
agent = project_client.agents.create_agent(
    model="gpt-4o-mini",
    name="my-agent",
    instructions=open("instructions.txt").read(),
    top_p=0.7,
    temperature=0.7,
    toolset=toolset  # Add the toolset to the agent
)
print(f"Created agent, ID: {agent.id}")

# Creating the thread
thread = project_client.agents.threads.create()
print(f"Created thread, ID: {thread.id}")

try:
    while True:
        # Get the user input
        user_input = input("You: ")

        # Break out of the loop
        if user_input.lower() in ["exit", "quit"]:
            break

        # Add a message to the thread
        message = project_client.agents.messages.create(
            thread_id=thread.id,
            role=MessageRole.USER,
            content=user_input
        )

        # Process the agent run
        run = project_client.agents.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent.id
        )

        # List messages and print the first text response from the agent
        messages = project_client.agents.messages.list(thread_id=thread.id)
        first_message = next(iter(messages), None)
        if first_message:
            print(next((item["text"]["value"] for item in first_message.content if item.get("type") == "text"), "")) 

finally:
    # Clean up the agent when done
    project_client.agents.delete_agent(agent.id)
    print("Deleted agent")

