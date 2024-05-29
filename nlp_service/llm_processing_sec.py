from langchain.retrievers import ParentDocumentRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader


"""

This code is what we will use to process the SEC filings.
Currently, it was taken from a notebook so the code is not organized. We want to 
make this into various functions that we can use in the Django application. Primarily, this should
deal with the following:

- Use the fetched SEC to answer the list of questions that can be used to analyze news events
- Return the answers to the questions in a JSON format that can be used by the Django application

"""


# Cleaning the data to remove all special characters
import re


def clean_doc(content):
    # Replace "\\n" with an actual newline character
    content = content.replace("\\n", "\n")

    # Regular expression to match numbers appearing immediately after '#'
    pattern = r"#[0-9]+"

    # Remove numbers appearing immediately after '#'
    content_after_hash_removal = re.sub(pattern, "", content)

    # Use a regular expression to keep only specified characters
    # This will keep a-z, A-Z, 0-9, space, '.', and ','
    cleaned_content = re.sub(r"[^a-zA-Z0-9 \.,\n]", "", content_after_hash_removal)

    return cleaned_content


# Note that this was done in a Jupyter Notebook, so the file paths are relative to the notebook.
# For the application this will be the files returned by the SEC Filings Service


loaders = [
    TextLoader("tesla_data/section1.txt"),
    TextLoader("tesla_data/section1A.txt"),
    TextLoader("tesla_data/section7.txt"),
]

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import uuid
from langchain.schema.document import Document


def init_vector_store(loaders):
    docs = []
    for l in loaders:
        docs.extend(l.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
    docs = text_splitter.split_documents(docs)

    chain = (
        {"doc": lambda x: x.page_content}  # Add the additional variable here
        | ChatPromptTemplate.from_template(
            "Summarize the following document given it is part of a company's 10-K filing with the SEC:\n\n{doc}"
        )  # Include the variable in the template
        | ChatOpenAI(max_retries=0)
        | StrOutputParser()
    )

    summaries = chain.batch(docs, {"max_concurrency": 5})

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="summaries", embedding_function=OpenAIEmbeddings()
    )
    # The storage layer for the parent documents
    store = InMemoryStore()
    id_key = "doc_id"
    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, docstore=store, id_key=id_key, search_kwargs={"k": 5}
    )
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    return retriever


from langchain.prompts import PromptTemplate


def init_qa_chain(retriever):
    prompt_template = """You are a financial expert that specializes in analyzing financial filings of publicly traded companies. Use the following pieces of context from the 10-K filing of the company to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    \n\n ______________________ \n \n

    Context:

    {context}

    \n\n ______________________ \n \n
    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    # This was again done in a Jupyter Notebook, where the example was Tesla, we want to make this dynamic based on the stock
    company_name = "Tesla"

    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    combine_template = "These are answers to a questions about a company from the company's 10-K filing with the SEC:\n\n{summaries} \n\n ########## \n \n Summarize the answers."
    combine_prompt_template = PromptTemplate.from_template(template=combine_template)

    question_template = """You are a financial expert that specializes in analyzing financial filings of publicly traded companies. Use the following pieces of context from the 10-K filing of the company to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Restate the question in your answer.
    \n\n ########## \n \n
    Context:
    {context}
    \n\n ########## \n \n
    Question: {question}
    \n\n ########## \n \n
    Helpful Answer:"""
    question_prompt_template = PromptTemplate.from_template(template=question_template)

    qa_chain_mr_test = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        # return_source_documents=True,
        chain_type_kwargs={
            "question_prompt": question_prompt_template,
            "combine_prompt": combine_prompt_template,
        },
    )

    return qa_chain_mr_test


import json

# Initialize an empty dictionary to store the JSON object


def answer_questions(qa_chain, company_name):
    result_json = {}

    questions = {
        "Industry Risk Management": f"How is {company_name}'s management strategizing to adapt to or mitigate industry-specific challenges, including market competition and technology shifts?",
        "Macroeconomic Risks Management": f"What actions is the {company_name} taking to manage exposure to macroeconomic variables like inflation, interest rates, or economic downturns?",
        "Business and Operating Risks Management": f"How is the {company_name} planning to mitigate operational risks, such as supply chain disruptions, operational inefficiencies,or dependency on key clients?",
        "Fiscal Risks Internal to the Company Management": f"What strategies does {company_name}'s management have to manage internal fiscal risks, such as credit risk, liquidity, and debt?",
        "Regulatory Compliance and Legal Risks Management": f"What compliance and legal risk management strategies is the {company_name} employing, especially concerning environmental regulations, data protection laws, and intellectual property rights?",
        "Revenue and Profitability": f"What is {company_name} management's perspective on Revenue trends and their effect on Gross Profit and Net Income?",
        "Cost Management": f"How is {company_name}'s management controlling costs to maintain or improve Operating Income?",
        "Liquidity Concerns": f"How does {company_name}'s management plan to balance Current Assets and Current Liabilities, specifically focusing on Inventories and Accounts Receivable?",
        "Capital Structure and Financing": f"What is the {company_name} management's view on Total Debt, Interest Payments, and Interest Expense?",
        "Cash Flow Management": f"How is {company_name} planning to generate and utilize Cash Flow from Operating Activities?",
        "Shareholder Value": "What strategies are in place to improve Average Shareholders' Equity and distribute Preferred Dividends?",
        "Earnings and Stock Valuation": f"How does the {company_name} management view the current Stock Price in relation to Earnings and Weighted Average Shares Outstanding?",
        "Working Capital": f"What is {company_name} management's strategy for managing working capital, focusing on Current Assets, Accounts Receivable, and Average Current Liabilities?",
        "Asset Management": f"How does {company_name}'s management plan to optimize Total Assets, focusing on the relationship between Current Assets and Inventories?",
        "Overall Financial Health": f"What is {company_name} management's commentary on the overall financial health of the company, focusing on key line items like Total Liabilities, Shareholders' Equity, and Cash Flow per Share?",
        "Industry Risks": f"What industry-specific challenges or competitive pressures does the {company_name} acknowledge facing?",
        "Macroeconomic Risks": f"How does {company_name} perceive risks related to economic cycles, inflation, or interest rates?",
        "Business and Operating Risks": f"What are {company_name}'s stated risks regarding its core operations, such as supply chain disruptions, operational inefficiencies, or customer retention?",
        "Regulatory Compliance and Legal Risks": "What are the acknowledged risks concerning regulatory compliance, such as environmental regulations, data privacy, or intellectual property?",
        "Fiscal Risks Internal to the Company": f"What internal fiscal risks such as credit risk, liquidity risk, or debt obligations does the {company_name} highlight?",
        "Company's Offerings and Revenue Streams": f"How does the {company_name} describe its primary product lines, services, and associated revenue streams? How are they diversified, and what segments generate the most revenue? Key Elements: Product Line, Services, Revenue Streams",
        "Supply Chain Dynamics": f"Can you provide details on the {company_name}'s supply chain management? Who are the key suppliers, partners, and intermediaries? What relationships or dependencies exist with these entities? Key Elements: Key Actors, Relationships in the Supply Chain",
        "Regulatory and Legal Landscape": f"What is the legal and regulatory framework in which the {company_name} operates? Are there specific laws, regulations, or compliance requirements that significantly impact the business? Key Elements: Legal Environment",
        "Competitive Landscape": f"Who are the {company_name}'s primary competitors in the market? What differentiates the company from these competitors, and how does it position itself in the industry? Key Elements: Main Competitors",
    }

    # Loop through the dictionary of questions and collect responses
    for key, question in questions.items():
        query_data = {"query": question}
        response = qa_chain(query_data)
        result_json[key] = response

    # Save the JSON object to a local file
    # Remember to change the file name each time you run the code
    return result_json
