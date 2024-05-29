from django.urls import path
from . import views

urlpatterns = [
    # ... other URL patterns ...
    path(
        "retriever_processing",
        views.get_documents,
        name="retriever_processing",
    ),
    path(
        "get_answer",
        views.get_answer,
        name="get_answer",
    ),
    path(
        "get_graph",
        views.graph_questions,
        name="get_graph",
    ),
    path(
        "chat_with_docs/<str:ticker>/<str:filing_type>/",
        views.chat_with_docs,
        name="chat_with_docs",
    ),
]
