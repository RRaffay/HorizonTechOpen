from django.db import models

# Create your models here.
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
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
from django.conf import settings
import uuid
from langchain.chains.question_answering import (
    load_qa_chain,
)
from langchain.memory import ConversationBufferWindowMemory

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
)

OPENAI_API_KEY = settings.OPENAI_API_KEY


from langchain.retrievers.multi_vector import MultiVectorRetriever
from typing import List, Optional
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Field
from langchain.schema import BaseRetriever, BaseStore, Document


class CustomRetriever(MultiVectorRetriever):
    docstore: Optional[BaseStore[str, Document]] = Field(None)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        # Extract original documents directly from sub_docs
        return [
            Document(
                page_content=d.metadata["original_content"],
                metadata={"source": d.metadata["filing_section"]},
            )
            for d in sub_docs
        ]


class ChromaManager:
    def __init__(self, chroma_openai_key="DummyKey"):
        self.chroma = None
        self.retriever = None
        self.persist_dir = "chroma_db"
        self.chroma_openai_key = chroma_openai_key
        self.init_chroma()

    def init_chroma(self):
        self.chroma = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=OpenAIEmbeddings(openai_api_key=self.chroma_openai_key),
        )

        self.chroma.persist()

    def load_data(self, docs, meta_data):
        """Load documents into Chroma

        This should add documents to the Chroma database with associated meta-data. We should
        assume that the documents come in the following format:

        {

            'section_name': (This will hold the actual text of the section),


        }

        Meta data should be in the following format:

        {
            'ticker': (Company Name),
            'filing_type': (Filing Type),
            'filing_date': (Filing Date),
            'filing_section': (Filing Section),
            'filing_description': (Filing Description),
        }

        This should be the format nlp_service sends the data

        """

        docs = [
            Document(
                page_content=doc_section_text,
                metadata={
                    "ticker": meta_data["ticker"],
                    "filing_type": meta_data["filing_type"],
                    "filing_date": str(meta_data["filing_date"]),
                    "filing_section": doc_section_name,
                    "filing_description": meta_data["filing_description"],
                },
            )
            for doc_section_name, doc_section_text in docs.items()
        ]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
        docs = text_splitter.split_documents(docs)

        OPENAI_API_KEY = settings.OPENAI_API_KEY

        chain = (
            {
                "doc": lambda x: x.page_content,
                "filing_type": lambda x: x.metadata["filing_type"],
                "ticker": lambda x: x.metadata["ticker"],
            }
            | ChatPromptTemplate.from_template(
                "Summarize the following document given it is part of a {ticker}'s {filing_type} filing with the SEC:\n\n{doc}"
            )
            | ChatOpenAI(max_retries=0, openai_api_key=OPENAI_API_KEY)
            | StrOutputParser()
        )

        summaries = chain.batch(docs, {"max_concurrency": 5})

        id_key = "doc_id"
        self.retriever = CustomRetriever(
            vectorstore=self.chroma, id_key=id_key, search_kwargs={"k": 5}
        )
        doc_ids = [str(uuid.uuid4()) for _ in docs]

        summary_docs_with_original = [
            Document(
                page_content=s,
                metadata={
                    **docs[
                        i
                    ].metadata,  # Extracting and spreading the metadata from the original doc
                    id_key: doc_ids[i],
                    "original_content": docs[i].page_content,
                },
            )
            for i, s in enumerate(summaries)
        ]

        self.retriever.vectorstore.add_documents(summary_docs_with_original)

    def retrieve_data(self, query, search_kwargs={}):
        """Retrieve relevant documents for a query"""
        return self.chroma.similarity_search(query, k=5, **search_kwargs)

    def filter_data(self, metadata):
        """Filter documents in Chroma by metadata"""
        return self.chroma.get(where=metadata)

    def delete_data(self, doc_ids):
        """Delete documents from Chroma by ID"""
        self.chroma._collection.delete(doc_ids)

    def update_doc(self, doc_id, doc):
        """Update a document in Chroma"""
        self.chroma.update_document(doc_id, doc)

    def get_answer(self, question, ticker, filing_type, try_refine=0):
        """Get an answer to a question from the database"""

        # if self.retriever is None:
        #     raise ValueError("Retriever Not setup")

        # One option here is to add the meta data to the query
        # Extract the ticker and filing_type from the query
        # then filter the vectorstore for those documents

        id_key = "doc_id"
        self.retriever = CustomRetriever(
            vectorstore=self.chroma,
            id_key=id_key,
            search_kwargs={
                "k": 5,
                "filter": {"$and": [{"filing_type": filing_type}, {"ticker": ticker}]},
            },
        )

        llm = ChatOpenAI(
            model_name="gpt-4-1106-preview",
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
        )

        # Need to make this dynamic so summarizing is not just limited to 10-K filings
        # Adding the zero bit in this prompt to allow for eventual processing, but not sure what the results would look like
        combine_template = "Below are the answers generated to the question:\n \n {question} \n \n Each of answers used some part of the company's 10-K filing with the SEC as context. \n\n Answers:\n\n##########\n\n{summaries} \n\n ########## \n \n Using these answers, generated an answer to the question. If an answer can not be generated, say that the question could not be answered, but suggest a question that could be answered using the context from a 10-K filing with the SEC."
        combine_prompt_template = PromptTemplate.from_template(
            template=combine_template
        )

        question_template = """You are a financial expert that specializes in analyzing financial filings of publicly traded companies. Use the following pieces of context from the 10-K filing of the company to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Restate the question in your answer.
        \n\n ########## \n \n
        Context:
        {context}
        \n\n ########## \n \n
        Question: {question}
        \n\n ########## \n \n
        Helpful Answer:"""
        question_prompt_template = PromptTemplate.from_template(
            template=question_template
        )

        # qa_chain = RetrievalQA.from_chain_type(
        #     llm=llm,
        #     chain_type="map_reduce",
        #     retriever=self.retriever,
        #     chain_type_kwargs={
        #         "question_prompt": question_prompt_template,
        #         "combine_prompt": combine_prompt_template,
        #     },
        # )

        # question_chain = load_qa_chain(
        #     llm,
        #     chain_type="map_reduce",
        #     question_prompt=question_prompt_template,
        #     combine_prompt=combine_prompt_template,
        # )

        if try_refine == 0:
            question_chain = load_qa_with_sources_chain(
                llm,
                chain_type="map_reduce",
                question_prompt=question_prompt_template,
                combine_prompt=combine_prompt_template,
            )

        else:
            question_chain = load_qa_with_sources_chain(llm, chain_type="refine")

        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

        chain = ConversationalRetrievalChain(
            retriever=self.retriever,
            combine_docs_chain=question_chain,
            question_generator=question_generator,
        )

        return chain

        # Start a conversation
        result = chain({"question": question, "chat_history": chat_history})
        print(result)
        return result["answer"]

        # query_data = {"query": question}
        # response = qa_chain(query_data)
        # print(response)
        return response["answer"]

    def get_answer_parent_questions(self, ticker, filing_type):
        """Get an answer to a question from the database"""

        # if self.retriever is None:
        #     raise ValueError("Retriever Not setup")

        # One option here is to add the meta data to the query
        # Extract the ticker and filing_type from the query
        # then filter the vectorstore for those documents

        id_key = "doc_id"
        self.retriever = CustomRetriever(
            vectorstore=self.chroma,
            id_key=id_key,
            search_kwargs={
                "k": 5,
                "filter": {"$and": [{"filing_type": filing_type}, {"ticker": ticker}]},
            },
        )

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k", temperature=0, openai_api_key=OPENAI_API_KEY
        )

        # Need to make this dynamic so summarizing is not just limited to 10-K filings
        # Adding the zero bit in this prompt to allow for eventual processing, but not sure what the results would look like
        combine_template = "These are answers to a questions about a company from the company's 10-K filing with the SEC:\n\n{summaries} \n\n ########## \n \n Summarize the answers. If the summary says the question could not be answered, add a 0 at the end of the summary."
        combine_prompt_template = PromptTemplate.from_template(
            template=combine_template
        )

        question_template = """You are a financial expert that specializes in analyzing financial filings of publicly traded companies. Use the following pieces of context from the 10-K filing of the company to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Restate the question in your answer.
        \n\n ########## \n \n
        Context:
        {context}
        \n\n ########## \n \n
        Question: {question}
        \n\n ########## \n \n
        Helpful Answer:"""
        question_prompt_template = PromptTemplate.from_template(
            template=question_template
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",
            retriever=self.retriever,
            chain_type_kwargs={
                "question_prompt": question_prompt_template,
                "combine_prompt": combine_prompt_template,
            },
        )

        result_json = {}

        questions = {
            "Industry Risk Management": f"How is {ticker}'s management strategizing to adapt to or mitigate industry-specific challenges, including market competition and technology shifts?",
            "Macroeconomic Risks Management": f"What actions is the {ticker} taking to manage exposure to macroeconomic variables like inflation, interest rates, or economic downturns?",
            "Business and Operating Risks Management": f"How is the {ticker} planning to mitigate operational risks, such as supply chain disruptions, operational inefficiencies,or dependency on key clients?",
            "Fiscal Risks Internal to the Company Management": f"What strategies does {ticker}'s management have to manage internal fiscal risks, such as credit risk, liquidity, and debt?",
            "Regulatory Compliance and Legal Risks Management": f"What compliance and legal risk management strategies is the {ticker} employing, especially concerning environmental regulations, data protection laws, and intellectual property rights?",
            "Revenue and Profitability": f"What is {ticker} management's perspective on Revenue trends and their effect on Gross Profit and Net Income?",
            "Cost Management": f"How is {ticker}'s management controlling costs to maintain or improve Operating Income?",
            "Liquidity Concerns": f"How does {ticker}'s management plan to balance Current Assets and Current Liabilities, specifically focusing on Inventories and Accounts Receivable?",
            "Capital Structure and Financing": f"What is the {ticker} management's view on Total Debt, Interest Payments, and Interest Expense?",
            "Cash Flow Management": f"How is {ticker} planning to generate and utilize Cash Flow from Operating Activities?",
            "Shareholder Value": "What strategies are in place to improve Average Shareholders' Equity and distribute Preferred Dividends?",
            "Earnings and Stock Valuation": f"How does the {ticker} management view the current Stock Price in relation to Earnings and Weighted Average Shares Outstanding?",
            "Working Capital": f"What is {ticker} management's strategy for managing working capital, focusing on Current Assets, Accounts Receivable, and Average Current Liabilities?",
            "Asset Management": f"How does {ticker}'s management plan to optimize Total Assets, focusing on the relationship between Current Assets and Inventories?",
            "Overall Financial Health": f"What is {ticker} management's commentary on the overall financial health of the company, focusing on key line items like Total Liabilities, Shareholders' Equity, and Cash Flow per Share?",
            "Industry Risks": f"What industry-specific challenges or competitive pressures does the {ticker} acknowledge facing?",
            "Macroeconomic Risks": f"How does {ticker} perceive risks related to economic cycles, inflation, or interest rates?",
            "Business and Operating Risks": f"What are {ticker}'s stated risks regarding its core operations, such as supply chain disruptions, operational inefficiencies, or customer retention?",
            "Regulatory Compliance and Legal Risks": "What are the acknowledged risks concerning regulatory compliance, such as environmental regulations, data privacy, or intellectual property?",
            "Fiscal Risks Internal to the Company": f"What internal fiscal risks such as credit risk, liquidity risk, or debt obligations does the {ticker} highlight?",
            "Company's Offerings and Revenue Streams": f"How does the {ticker} describe its primary product lines, services, and associated revenue streams? How are they diversified, and what segments generate the most revenue? Key Elements: Product Line, Services, Revenue Streams",
            "Supply Chain Dynamics": f"Can you provide details on the {ticker}'s supply chain management? Who are the key suppliers, partners, and intermediaries? What relationships or dependencies exist with these entities? Key Elements: Key Actors, Relationships in the Supply Chain",
            "Regulatory and Legal Landscape": f"What is the legal and regulatory framework in which the {ticker} operates? Are there specific laws, regulations, or compliance requirements that significantly impact the business? Key Elements: Legal Environment",
            "Competitive Landscape": f"Who are the {ticker}'s primary competitors in the market? What differentiates the company from these competitors, and how does it position itself in the industry? Key Elements: Main Competitors",
        }

        for key, question in questions.items():
            query_data = {"query": question}
            response = qa_chain(query_data)
            result_json[key] = response

        # If you need to save the results to a file, uncomment the below lines.
        # file_name = f'{ticker}_Analysis.json'
        # with open(file_name, 'w') as json_file:
        #     json.dump(result_json, json_file, indent=4)

        return result_json

    def get_graph_entities(self, ticker, filing_type):
        id_key = "doc_id"
        self.retriever = CustomRetriever(
            vectorstore=self.chroma,
            id_key=id_key,
            search_kwargs={
                "k": 5,
                "filter": {"$and": [{"filing_type": filing_type}, {"ticker": ticker}]},
            },
        )

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k", temperature=0, openai_api_key=OPENAI_API_KEY
        )

        # Need to make this dynamic so summarizing is not just limited to 10-K filings
        # Adding the zero bit in this prompt to allow for eventual processing, but not sure what the results would look like
        combine_template = "These are answers to a questions about a company from the company's 10-K filing with the SEC:\n\n{summaries} \n\n ########## \n \n Summarize the answers. If the summary says the question could not be answered, add a 0 at the end of the summary."
        combine_prompt_template = PromptTemplate.from_template(
            template=combine_template
        )

        question_template = """You are a financial expert that specializes in analyzing financial filings of publicly traded companies. Use the following pieces of context from the 10-K filing of the company to answer the question at the end. Return the answer in following format: (Important Entity, Reason for importance). Note important entities should be proper nouns. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Restate the question in your answer.
        \n\n ########## \n \n
        Context:
        {context}
        \n\n ########## \n \n
        Question: {question}
        \n\n ########## \n \n
        Helpful Answer:"""
        question_prompt_template = PromptTemplate.from_template(
            template=question_template
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",
            retriever=self.retriever,
            chain_type_kwargs={
                "question_prompt": question_prompt_template,
                "combine_prompt": combine_prompt_template,
            },
        )

        result_json = {}

        questions = {
            "Industry_and_Competition": {
                "Key_Industry_Players": f"Who are the key competitors and stakeholders that directly affect {ticker}'s industry risk management strategies?",
                "Differentiators_and_Vulnerabilities": f"What competitive advantages or vulnerabilities does {ticker} identify in comparison to its key competitors?",
            },
            # "Macroeconomic_Variables": {
            #     "Influential_Macro_Entities": f"Which government bodies or international organizations' policies most significantly impact {ticker}'s macroeconomic risk management?"
            # },
            # "Operational_Aspects": {
            #     "Supply_Chain_Entities": f"Who are the key suppliers, partners, and intermediaries crucial to {ticker}'s operations? What relationships or dependencies exist with these entities?",
            #     "Client_Dependence": f"Does {ticker} identify any key clients or customer segments that contribute significantly to its revenue? How is this managed?",
            # },
            # "Fiscal_Management": {
            #     "Debt_and_Credit_Relations": f"Who are the key creditors or debt holders of {ticker}, and how does the company plan on managing these relationships to mitigate fiscal risks?"
            # },
            # "Regulatory_Compliance": {
            #     "Regulatory_Bodies_and_Legislations": f"What specific legal and regulatory bodies and legislations most affect {ticker}'s compliance and legal strategies?"
            # },
            # "Revenue_and_Profitability": {
            #     "Revenue_Streams_and_Stakeholders": f"How does {ticker} categorize its revenue streams? Are there key stakeholders involved in the most significant revenue-generating segments?"
            # },
            # "Financial_Management": {
            #     "Liquidity_Partners": f"Who are the key financial partners or institutions helping {ticker} in managing its liquidity?",
            #     "Capital_Providers": f"Who are the primary financiers contributing to {ticker}'s capital structure? Any key stakeholders in debt financing or equity?",
            # },
            # "Shareholder_Relations": {
            #     "Major_Shareholders": f"Who are the major shareholders of {ticker}? What influence do they have on shareholder value strategies?"
            # },
            # "Valuation_and_Market_Perception": {
            #     "Market_Analysts_and_Influencers": f"What key market analysts or financial influencers' views does {ticker} management consider in relation to stock valuation?"
            # },
            # "Overall_Financial_Health": {
            #     "Advisory_Entities": f"Does {ticker} rely on any advisory boards or consultancies for financial health assessments?"
            # },
        }

        for keys, questions in questions.items():
            for key, question in questions.items():
                query_data = {"query": question}
                response = qa_chain(query_data)
                result_json[key] = response

        # If you need to save the results to a file, uncomment the below lines.
        # file_name = f'{ticker}_Analysis.json'
        # with open(file_name, 'w') as json_file:
        #     json.dump(result_json, json_file, indent=4)

        return result_json
