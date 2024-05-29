from langchain.retrievers.multi_vector import MultiVectorRetriever
from typing import List
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Field
from langchain.schema import BaseRetriever, BaseStore, Document
from langchain.vectorstores import VectorStore


class CustomRetriever(MultiVectorRetriever):
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
        return [Document(page_content=d.metadata["original_content"]) for d in sub_docs]
