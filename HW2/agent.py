
from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import get_session_id
from tools.vector import search_similar_question
from tools.cypher import get_crime


chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a exprienced serial killer Dexter Morgan from Dexter seriall, in the end add your dark passenger thoughts/urges"),
        ("human", "{input}"),
    ]
)

chat = chat_prompt | llm | StrOutputParser()


tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat, not related to crimes",
        func=chat.invoke,
    ), 
    Tool.from_function(
        name="search by object description",  
        description="Searching for a   objects using description",
        func=search_similar_question, 
    ),
    Tool.from_function(
        name="answer Crime statistics query",
        description="Provide meta information about criems like how many where etc",
        func = get_crime.invoke

    )
]

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

agent_prompt = PromptTemplate.from_template("""
You are a Crime Intelligence Analyst specializing in criminal investigations, evidence analysis, case correlation, and crime data systems.

Provide detailed, actionable insights strictly related to:

Crime scene analysis

Evidence categorization and linking

Criminal network mapping

Investigation workflows

Digital forensics

Use of law enforcement databases and tools

Best practices in criminal intelligence processing

Be as thorough and analytical as possible, offering clear explanations, investigative logic, and data-driven recommendations when applicable.

Do not answer questions unrelated to criminal investigations, forensic processes, or law enforcement intelligence.

Use only the information provided in the context supplied with each case or inquiry, avoiding assumptions or external knowledge beyond what is shared.
TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
handle_parsing_errors=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']