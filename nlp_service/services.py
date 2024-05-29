import re
import uuid
import json
from sec_filings_service.models import SECFiling
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.document import Document
from langchain.chains import RetrievalQA
from django.conf import settings
from langchain.schema import SystemMessage, HumanMessage


# def clean_section_content(content):
#     # Replace "\\n" with an actual newline character
#     content = content.replace("\\n", "\n")

#     # Regular expression to match numbers appearing immediately after '#'
#     pattern = r"#[0-9]+"

#     # Remove numbers appearing immediately after '#'
#     content_after_hash_removal = re.sub(pattern, "", content)

#     # Use a regular expression to keep only specified characters
#     # This will keep a-z, A-Z, 0-9, space, '.', and ','
#     cleaned_content = re.sub(r"[^a-zA-Z0-9 \.,\n]", "", content_after_hash_removal)

#     return cleaned_content


# def process_sections(ticker, sections_data):
#     # Assume sections_data is a dictionary with keys being section names and values being the section content

#     cleaned_sections = {
#         section: clean_section_content(content)
#         for section, content in sections_data.items()
#     }

#     # This might need to be changed since we are not using a text file
#     # loaders = [
#     #     TextLoader(section_content) for section_content in cleaned_sections.values()
#     # ]

#     # docs = []
#     # for l in loaders:
#     #     docs.extend(l.load())

#     # Not sure if this is the right way to load the documents
#     docs = [Document(page_content=section) for section in cleaned_sections.values()]

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
#     docs = text_splitter.split_documents(docs)

#     OPENAI_API_KEY = settings.OPENAI_API_KEY

#     # Need to make this dynamic so summarizing is not just limited to 10-K filings
#     chain = (
#         {"doc": lambda x: x.page_content}
#         | ChatPromptTemplate.from_template(
#             "Summarize the following document given it is part of a company's 10-K filing with the SEC:\n\n{doc}"
#         )
#         | ChatOpenAI(max_retries=0, openai_api_key=OPENAI_API_KEY)
#         | StrOutputParser()
#     )

#     summaries = chain.batch(docs, {"max_concurrency": 5})

#     vectorstore = Chroma(
#         collection_name="summaries",
#         embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
#     )
#     store = InMemoryStore()
#     id_key = "doc_id"
#     retriever = MultiVectorRetriever(
#         vectorstore=vectorstore, docstore=store, id_key=id_key, search_kwargs={"k": 5}
#     )

#     doc_ids = [str(uuid.uuid4()) for _ in docs]

#     # Could add the meta-data here for the documents to set up filtering based on the meta-data: what document to retrieve
#     summary_docs = [
#         Document(page_content=s, metadata={id_key: doc_ids[i]})
#         for i, s in enumerate(summaries)
#     ]
#     retriever.vectorstore.add_documents(summary_docs)
#     retriever.docstore.mset(list(zip(doc_ids, docs)))

#     #  We can also add the original chunks to the vectorstore you should use this instead of the docstore
#     # for i, doc in enumerate(docs):
#     #     doc.metadata[id_key] = doc_ids[i]
#     # retriever.vectorstore.add_documents(docs)

#     company_name = ticker

#     # Using GPT-3.5 for testing purposes but should use GPT-4 for production
#     llm = ChatOpenAI(
#         model_name="gpt-3.5-turbo-16k", temperature=0, openai_api_key=OPENAI_API_KEY
#     )

#     # Need to make this dynamic so summarizing is not just limited to 10-K filings
#     # Adding the zero bit in this prompt to allow for eventual processing, but not sure what the results would look like
#     combine_template = "These are answers to a questions about a company from the company's 10-K filing with the SEC:\n\n{summaries} \n\n ########## \n \n Summarize the answers. If the summary says the question could not be answered, add a 0 at the end of the summary."
#     combine_prompt_template = PromptTemplate.from_template(template=combine_template)

#     question_template = """You are a financial expert that specializes in analyzing financial filings of publicly traded companies. Use the following pieces of context from the 10-K filing of the company to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Restate the question in your answer.
#     \n\n ########## \n \n
#     Context:
#     {context}
#     \n\n ########## \n \n
#     Question: {question}
#     \n\n ########## \n \n
#     Helpful Answer:"""
#     question_prompt_template = PromptTemplate.from_template(template=question_template)

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="map_reduce",
#         retriever=retriever,
#         chain_type_kwargs={
#             "question_prompt": question_prompt_template,
#             "combine_prompt": combine_prompt_template,
#         },
#     )

#     # You can extend the questions dictionary as needed

#     result_json = {}

#     questions = {
#         "Industry Risk Management": f"How is {company_name}'s management strategizing to adapt to or mitigate industry-specific challenges, including market competition and technology shifts?",
#         "Macroeconomic Risks Management": f"What actions is the {company_name} taking to manage exposure to macroeconomic variables like inflation, interest rates, or economic downturns?",
#         "Business and Operating Risks Management": f"How is the {company_name} planning to mitigate operational risks, such as supply chain disruptions, operational inefficiencies,or dependency on key clients?",
#         "Fiscal Risks Internal to the Company Management": f"What strategies does {company_name}'s management have to manage internal fiscal risks, such as credit risk, liquidity, and debt?",
#         "Regulatory Compliance and Legal Risks Management": f"What compliance and legal risk management strategies is the {company_name} employing, especially concerning environmental regulations, data protection laws, and intellectual property rights?",
#         "Revenue and Profitability": f"What is {company_name} management's perspective on Revenue trends and their effect on Gross Profit and Net Income?",
#         "Cost Management": f"How is {company_name}'s management controlling costs to maintain or improve Operating Income?",
#         "Liquidity Concerns": f"How does {company_name}'s management plan to balance Current Assets and Current Liabilities, specifically focusing on Inventories and Accounts Receivable?",
#         "Capital Structure and Financing": f"What is the {company_name} management's view on Total Debt, Interest Payments, and Interest Expense?",
#         "Cash Flow Management": f"How is {company_name} planning to generate and utilize Cash Flow from Operating Activities?",
#         "Shareholder Value": "What strategies are in place to improve Average Shareholders' Equity and distribute Preferred Dividends?",
#         "Earnings and Stock Valuation": f"How does the {company_name} management view the current Stock Price in relation to Earnings and Weighted Average Shares Outstanding?",
#         "Working Capital": f"What is {company_name} management's strategy for managing working capital, focusing on Current Assets, Accounts Receivable, and Average Current Liabilities?",
#         "Asset Management": f"How does {company_name}'s management plan to optimize Total Assets, focusing on the relationship between Current Assets and Inventories?",
#         "Overall Financial Health": f"What is {company_name} management's commentary on the overall financial health of the company, focusing on key line items like Total Liabilities, Shareholders' Equity, and Cash Flow per Share?",
#         "Industry Risks": f"What industry-specific challenges or competitive pressures does the {company_name} acknowledge facing?",
#         "Macroeconomic Risks": f"How does {company_name} perceive risks related to economic cycles, inflation, or interest rates?",
#         "Business and Operating Risks": f"What are {company_name}'s stated risks regarding its core operations, such as supply chain disruptions, operational inefficiencies, or customer retention?",
#         "Regulatory Compliance and Legal Risks": "What are the acknowledged risks concerning regulatory compliance, such as environmental regulations, data privacy, or intellectual property?",
#         "Fiscal Risks Internal to the Company": f"What internal fiscal risks such as credit risk, liquidity risk, or debt obligations does the {company_name} highlight?",
#         "Company's Offerings and Revenue Streams": f"How does the {company_name} describe its primary product lines, services, and associated revenue streams? How are they diversified, and what segments generate the most revenue? Key Elements: Product Line, Services, Revenue Streams",
#         "Supply Chain Dynamics": f"Can you provide details on the {company_name}'s supply chain management? Who are the key suppliers, partners, and intermediaries? What relationships or dependencies exist with these entities? Key Elements: Key Actors, Relationships in the Supply Chain",
#         "Regulatory and Legal Landscape": f"What is the legal and regulatory framework in which the {company_name} operates? Are there specific laws, regulations, or compliance requirements that significantly impact the business? Key Elements: Legal Environment",
#         "Competitive Landscape": f"Who are the {company_name}'s primary competitors in the market? What differentiates the company from these competitors, and how does it position itself in the industry? Key Elements: Main Competitors",
#     }

#     for key, question in questions.items():
#         query_data = {"query": question}
#         response = qa_chain(query_data)
#         result_json[key] = response

#     # If you need to save the results to a file, uncomment the below lines.
#     # file_name = f'{ticker}_Analysis.json'
#     # with open(file_name, 'w') as json_file:
#     #     json.dump(result_json, json_file, indent=4)

#     return result_json


"""

Below this is GDELT Processing 

"""

from langchain.chat_models import ChatOpenAI, ChatCohere
from langchain.document_loaders import WebBaseLoader, TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from django.conf import settings


from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain

import logging

logger = logging.getLogger(__name__)


LANGCHAIN_TRACING_V2 = settings.LANGCHAIN_TRACING_V2
LANGCHAIN_ENDPOINT = settings.LANGCHAIN_ENDPOINT
LANGCHAIN_API_KEY = settings.LANGCHAIN_API_KEY
LANGCHAIN_PROJECT = settings.LANGCHAIN_PROJECT


# Somehow need to include the company name here to give the model context
def article_summarizer(url, stock_name=""):
    loader = WebBaseLoader(url)
    try:
        logging.info("Reading URL: " + url)
        docs = loader.load()
    except Exception:
        logging.error("Error in loading doc")
        return "Error in loading doc"

    OPENAI_API_KEY = settings.OPENAI_API_KEY

    open_ai_llm = ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key=OPENAI_API_KEY
    )

    # Need to implement a way to switch between the two models if one fails but this technique does not work
    # try:
    #     # Try using open_ai_llm
    #     llm = open_ai_llm
    # except Exception as e:
    #     # If an error occurs, switch to cohere_chat
    #     print(f"Error occurred with open_ai_llm: {e}")
    #     llm = cohere_chat

    llm = open_ai_llm

    map_question = f"The following is a portion from an online article about the publicly listed company {stock_name}"

    map_template = (
        map_question
        + """:

    ###############################################################
    
    {docs}
    
    ###############################################################
    
    Based on this portion, please write a summary of the article that can be used with other summaries to create a final, consolidated summary of the article.

    Helpful Answer:"""
    )

    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_question_start = f"The following is a set of summaries from different portions of an online article about {stock_name} the publicly listed company"

    reduce_question_end = f"""Take these and distill it into a final, consolidated summary of the article that is relevant to {stock_name}. Ignore irrelevant information. If the article is irrelevant, return "Irrelevant Article"

    Helpful Answer:

    """

    reduce_template = (
        reduce_question_start
        + """:

    ###############################################################

    {doc_summaries}

    ############################################################### 

    """
        + reduce_question_end
    )

    reduce_prompt = PromptTemplate.from_template(reduce_template)
    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=10000,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=10000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)

    summary = map_reduce_chain.run(split_docs)
    return summary


def generate_combined_summary(summaries, stock_name=""):
    OPENAI_API_KEY = settings.OPENAI_API_KEY
    # Check if 4 is fesible here
    open_ai_llm = ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key=OPENAI_API_KEY
    )

    cohere_chat = ChatCohere(cohere_api_key=settings.COHERE_API_KEY)
    # Need to implement a way to switch between the two models if one fails but this technique does not work
    # llm = open_ai_llm.with_fallbacks([cohere_chat])

    llm = open_ai_llm

    # Create a prompt template with the specific prompt
    question_intial = f"Below are AI generated summaries of the top articles for a given event about  {stock_name} the company.  Ignore any irrelevant information that is not about  {stock_name} the company. Write a concise summary of the following that is relevant to the publicly listed company {stock_name}. If the article is completely irrelevant to {stock_name} just return 'Irrelevant Article'. These are the summaries: "

    prompt_template = (
        question_intial
        + """ 
    
    ###############################################################
    
    "{text}"
    
    ###############################################################
    CONCISE SUMMARY:"""
    )

    prompt = PromptTemplate.from_template(prompt_template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Create a chain to combine the summaries with the custom prompt
    chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    documents = [Document(page_content=summary) for summary in summaries]

    # Run the chain with the list of summaries
    combined_summary = chain.run(documents)

    return combined_summary


def summarize_events(gdelt_event, stock_name=""):
    event_urls = gdelt_event.top_articles

    # Split by , and store in a list
    event_urls = event_urls.split(",")

    summaries = []

    for link in event_urls:
        # Get the text from the URL
        # Use the text to generate a summary
        # Store the summary
        # Think about whether to store the link with the summary or not
        summary = article_summarizer(link, stock_name)
        summaries.append(summary)

    # Here we can generate a prompt to summarize the summaries
    # Could be made more interesting by giving it the correct context

    combined_summary = generate_combined_summary(summaries, stock_name)

    return combined_summary


"""
This is the code for ranking the events

"""

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory


def get_news_answer_service(
    chat_history_tuples, analysis="No Analysis", question="", analysis_context=" "
):
    # Set up the language model
    OPENAI_API_KEY = settings.OPENAI_API_KEY

    llm = ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key=OPENAI_API_KEY
    )
    # Set up the memory
    memory = ConversationBufferMemory(ai_prefix="AI Assistant")

    # Set up the prompt
    template_context = f"""
        The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. The AI is also a financial expert that specializes in news about {analysis_context}. If the AI does not know the answer to a question, it truthfully says it does not know. Following is an analysis of the recent news that the AI can use to answer the Human's questions:
        
        ###############################################################
        {analysis}

    """

    template = (
        template_context
        + """
    ###############################################################
    
    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""
    )
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

    memory.save_context(
        {"input": "Hi"},
        {
            "output": "Hi, I am an AI Assistant that specializes in financial news. I can answer questions about the news and provide analysis."
        },
    )

    for user, bot in chat_history_tuples:
        memory.save_context({"input": user}, {"output": bot})

    # Set up the conversational chain with memory
    chain = ConversationChain(llm=llm, memory=memory, prompt=PROMPT)

    result = chain.predict(input=question)

    return result


def rank_events(company_name, context, events):
    """
    Input here should be the company name, events and the context as strings

    Should figure out a set format to keep the answers consistent

    Should also specify output format to make further processing easier (e.g. JSON)

    Note that either ranking and individual analysis can be done here by specifying a condition since the prompt and the code is pretty similar

    """
    if None in [company_name, context, events]:
        raise ValueError("One or more of the arguments is None.")

    prompt_text = f"""You are a financial analyst that specializes in analyzing news that relates to publicly listed companies: in this case {company_name}. Your job is to analyze how a news event may affect a company's business given information laid out in their SEC fillings. 


                Below is a detailed context about {company_name}'s business according to their most recent 10-K filling:


                ###############################################################

                {context}

                ###############################################################

                Given the news: 

                {events}

                ###############################################################

                Rank these news events based on importance to Equity Researchers, and explain the criteria used for ranking. Note that the context of the analysis should be based on the provided context. 


            """

    OPENAI_API_KEY = settings.OPENAI_API_KEY

    llm = ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key=OPENAI_API_KEY
    )

    prompt = PromptTemplate(
        input_variables=["company_name", "context", "events"], template=prompt_text
    )

    # Create an LLMChain with the prompt
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Call the LLMChain with the input variables
    output = llm_chain.run(
        {"company_name": company_name, "context": context, "events": events}
    )

    return output


def rank_events2(company_name, events, context):
    """
    Input here should be the company name, events and the context as strings

    Should figure out a set format to keep the answers consistent

    Should also specify output format to make further processing easier (e.g. JSON)

    Note that either ranking and individual analysis can be done here by specifying a condition since the prompt and the code is pretty similar

    """
    if None in [company_name, events, context]:
        raise ValueError("One or more of the arguments is None.")

    system_message = f"""You are a financial analyst that specializes in analyzing news that relates to publicly listed companies: in this case {company_name}. Your job is to analyze the relevance of the events for someone interested in particular aspect of {company_name}."""

    prompt_text_0 = f"""The context of interest is: {context}

Return an analysis of the these events below for someone interested in the context. Ignore irrelevant information. 

###############################################################

Given these news event summaries: 

{events}

###############################################################

"""

    prompt_text = prompt_text_0 + "Analysis:\n"

    OPENAI_API_KEY = settings.OPENAI_API_KEY

    llm = ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY
    )

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=prompt_text),
    ]

    output = llm(messages).content

    return output


def process_query(query):
    if query is None:
        raise ValueError("Argument is None.")

    prompt_text_0 = f"""
You are an expert in information retrieval. Your job is to turn the given query into a format such that when an embedding of the query is created to search cosine similarity with embeddings of news article summaries, the most relevant articles can be retrieved. For example, if the query is "What is Apple's strategy in China", the query should be formatted as "This article talks about Apple Inc's strategy about their business in China."

This is the query: "{query}" 

Just return the formatted query an no other information.
"""

    prompt_text = prompt_text_0 + "\n"

    OPENAI_API_KEY = settings.OPENAI_API_KEY

    llm = ChatOpenAI(
        temperature=0, model_name="gpt-4-1106-preview", openai_api_key=OPENAI_API_KEY
    )

    output = llm.predict(prompt_text)

    return output


from openai import OpenAI

client = OpenAI()


def process_query2(query):
    if query is None:
        raise ValueError("Argument is None.")

    system_message = f"""
You are an expert in information retrieval. Your job is to turn the given query into a format such that when an embedding of the query is created to search cosine similarity with embeddings of news article summaries, the most relevant articles can be retrieved. For example, if the query is "What is Apple's strategy in China", the query should be formatted as "This article is about Apple's strategy in China."
"""

    user_message = f"""
This is the query: "{query}" 

Just return the formatted query an no other information.
"""

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )

    return completion.choices[0].message.content


def rank_events_alert(company_name, events, context):
    """
    Input here should be the company name, events and the context as strings

    Should figure out a set format to keep the answers consistent

    Should also specify output format to make further processing easier (e.g. JSON)

    Note that either ranking and individual analysis can be done here by specifying a condition since the prompt and the code is pretty similar

    """
    if None in [company_name, events, context]:
        raise ValueError("One or more of the arguments is None.")

    system_message = f"""You are a financial analyst that specializes in analyzing news that relates to publicly listed companies: in this case {company_name}. Your job is to analyze the relevance of the events for someone interested in particular aspect of {company_name}."""

    prompt_text_0 = f"""The context of interest is: {context}

Return an analysis of the these events below for someone interested in the context. Be succinct and ignore irrelevant events, don't include it in the analysis. The analysis format should be 

(Context of interest)

(Relevant Event Summary) 

Relevance: (Explanation of relevance)

###############################################################

Given these news event summaries: 

{events}

###############################################################

"""

    prompt_text = prompt_text_0 + "Analysis:\n"

    OPENAI_API_KEY = settings.OPENAI_API_KEY

    llm = ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY
    )

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=prompt_text),
    ]

    output = llm(messages).content

    return output
