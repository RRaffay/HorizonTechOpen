from .models import ChromaManager
from django.http import HttpResponse
from sec_filings_service.services import extract_key_items
from django.http import JsonResponse
from django.shortcuts import render
import json
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required


def process_request(request):
    if request.method == "POST":
        chroma_manager = ChromaManager()
        if request.POST.get("action") == "load":
            docs = request.POST.get("docs")
            chroma_manager.load_data(docs)

        elif request.POST.get("action") == "retrieve":
            query = request.POST.get("query")
            search_kwargs = request.POST.get("search_kwargs", {})
            results = chroma_manager.retrieve_data(query, search_kwargs)
            return results

        # Other actions like delete, update, filter

    return HttpResponse("Request processed")


def get_documents(request, ticker="AAPL"):
    # Check if Post Request
    if request.method == "POST":
        # Extract the filing_type and ticker from the request
        filing_type = request.POST.get("filing_type")
        ticker = request.POST.get("ticker")
        sections = request.POST.get("sections")

    # Just for testing, remove later and replace with the above code: ticker from request
    filing_type = "10-K"

    sections = ["1", "1A", "7"]
    # Currently extract_key_items just uses the latest of [10k,10Q,8k] but need to be able to
    # specify the filing_type
    ret, filing = extract_key_items([ticker], sections)

    # Ret has the following structure:
    # {
    #     "AAPL": {
    #         "text": {
    #         "1": "(Text for Item 1)",
    #         "1A": "(Text for Item 1A)",
    #         "7": "(Text for Item 7)",
    #         },
    #         "filing_date": "(Filing Date)",
    #         "filing_type": "10-K",
    #         "filing_description": "Form 10-K - Annual report [Section 13 and 15(d), not S-K Item 405]",
    #     }

    #

    docs = ret[ticker]["text"]

    meta_data = {
        "ticker": ticker,
        "filing_type": filing_type,
        "filing_date": ret[ticker]["filing_date"],
        "filing_description": ret[ticker]["filing_description"],
    }
    print(f"Loading {ticker} {filing_type} into Chroma")
    # Load the documents into Chroma
    chroma_manager = ChromaManager()
    chroma_manager.load_data(docs, meta_data)

    return filing


@login_required  # This decorator is used to exempt the view from CSRF verification. Consider CSRF protection for production.
def get_answer(request):
    if request.method == "POST":
        user = request.user
        if not user.profile.is_paying_customer:
            return JsonResponse(
                {"answer": "Please subscribe to  Pro plan to use this feature"}
            )
        chroma_manager = ChromaManager()

        data = json.loads(request.body)
        ticker = data.get("ticker", "TSLA")
        filing_type = data.get("filing_type", "10-K")
        question = data.get("question", "What is the revenue of Apple?")
        chat_history = data.get("chat_history", [("Hello", "Hi")])

        # Checking if the document already exists
        results = chroma_manager.chroma.get(
            where={"$and": [{"filing_type": filing_type}, {"ticker": ticker}]},
            include=["metadatas"],
        )

        # If the document already loaded
        if len(results["ids"]) > 0:
            # Return the answer
            answer_chain = chroma_manager.get_answer(
                question=question,
                ticker=ticker,
                filing_type=filing_type,
            )

            chat_history = [tuple(item) for item in chat_history]

            result = answer_chain({"question": question, "chat_history": chat_history})

            return JsonResponse({"answer": result["answer"]})

        # If the document does not exist
        else:
            answer = "The document does not exist in the database"
            return JsonResponse({"Error": "Document Does Not Exist"})
            get_documents(request)
            # Get the answer
            answer = chroma_manager.get_answer(
                question=question, ticker=ticker, filing_type=filing_type
            )

        return JsonResponse({"answer": answer})

    else:
        return JsonResponse({"Error": "Invalid request"}, status=400)


# Currently has stock_ticker argument, but should be removed and replaced with request
def process_sec_document(request, stock_ticker):
    # For testing, these should come from the request
    ticker = stock_ticker
    filing_type = "10-K"

    chroma_manager = ChromaManager()
    # Checking if the document already exists
    results = chroma_manager.chroma.get(
        where={"$and": [{"filing_type": filing_type}, {"ticker": ticker}]},
        include=["metadatas"],
    )

    # If the document already loaded
    if len(results["ids"]) > 0:
        # Return the answer
        processed_answers = chroma_manager.get_answer_parent_questions(
            ticker, filing_type
        )

    # If the document does not exist
    else:
        print("Vector DB did not have the document. Loading the document into the DB")
        # Add the documents to the database
        filing = get_documents(request, ticker)
        # Get the answer
        processed_answers = chroma_manager.get_answer_parent_questions(
            ticker, filing_type
        )

    return processed_answers, filing


def graph_questions(request, stock_ticker="AAPL"):
    return JsonResponse({"error": "Feature not implemented"})

    # For testing, these should come from the request
    ticker = stock_ticker
    filing_type = "10-K"

    # Checking if the document already exists
    results = chroma_manager.chroma.get(
        where={"$and": [{"filing_type": filing_type}, {"ticker": ticker}]},
        include=["metadatas"],
    )

    # If the document already loaded
    if len(results["ids"]) > 0:
        # Return the answer
        processed_answers = chroma_manager.get_graph_entities(ticker, filing_type)

    # If the document does not exist
    else:
        print("Vector DB did not have the document. Loading the document into the DB")
        # Add the documents to the database
        get_documents(request, ticker)
        # Get the answer
        processed_answers = chroma_manager.get_graph_entities(ticker, filing_type)

    print(processed_answers)

    return JsonResponse(processed_answers)


@login_required
def chat_with_docs(request, ticker, filing_type):
    chroma_manager = ChromaManager()
    results = chroma_manager.chroma.get(
        where={"$and": [{"filing_type": filing_type}, {"ticker": ticker}]},
        include=["metadatas"],
    )

    # If the document already loaded
    if len(results["ids"]) > 0:
        # Render the page
        return render(
            request,
            "retriever_processing/chat_with_docs.html",
            {"ticker": ticker, "filing_type": filing_type},
        )

    # If the document does not exist
    else:
        print("Vector DB did not have the document. Loading the document into the DB")
        # Add the documents to the database
        get_documents(request, ticker)
        # Render the page
        return render(
            request,
            "retriever_processing/chat_with_docs.html",
            {"ticker": ticker, "filing_type": filing_type},
        )
