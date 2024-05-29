from langchain.tools import tool
import requests
from pydantic import BaseModel, Field
import datetime

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.agent import AgentFinish
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from django.conf import settings


from langchain.tools import Tool

from langchain.utilities import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()

tool_search = Tool(
    name="Google Search",
    description="Search Google for recent results",
    func=search.run,
)


@tool
def search_tool(query: str) -> str:
    """This function is used to get answers from Google Search. Only to be used when the answer is unknown"""
    answer = tool_search.run(query)
    return answer


import param


OPENAI_API_KEY = settings.OPENAI_API_KEY


class cbfs(param.Parameterized):
    def __init__(self, tools, system_message="", **params):
        super(cbfs, self).__init__(**params)
        self.panels = []
        self.functions = [format_tool_to_openai_function(f) for f in tools]
        self.model = ChatOpenAI(
            temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-16k"
        ).bind(functions=self.functions)
        self.memory = ConversationBufferMemory(
            return_messages=True, memory_key="chat_history"
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        self.chain = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_to_openai_functions(
                    x["intermediate_steps"]
                )
            )
            | self.prompt
            | self.model
            | OpenAIFunctionsAgentOutputParser()
        )
        self.qa = AgentExecutor(
            agent=self.chain, tools=tools, verbose=False, memory=self.memory
        )


class ChatbotService:
    def __init__(self, analysis="", analysis_context=""):
        # Initialize the chatbot from chatbot.py
        tools = [search_tool]

        # Set up the prompt
        system_message = f"""
        The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. The AI is also a financial expert that specializes in news about {analysis_context}. If the AI does not know the answer to a question, it truthfully says it does not know. Following is an analysis of the recent news that the AI can use to answer the Human's questions:
        
        ###############################################################
        {analysis}

    """

        self.chatbot = cbfs(tools, system_message=system_message)

    def get_response(self, question, chat_history_tuples):
        # Saving the messages to memory
        for user, bot in chat_history_tuples:
            self.chatbot.memory.save_context({"input": user}, {"output": bot})

        response = self.chatbot.qa.invoke({"input": question})

        return response["output"]
