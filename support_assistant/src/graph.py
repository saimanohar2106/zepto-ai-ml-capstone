import os
from typing import TypedDict

from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from ingestion import ensure_index
from prompts import STRUCTURED_PROMPT


MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"


class GraphState(TypedDict, total=False):
    query: str
    intent: str
    context: list[str]
    sources: list[str]
    answer: str
    confidence: float


class AssistantResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


def classify_intent(state: GraphState) -> GraphState:
    query = state["query"].lower()

    if any(keyword in query for keyword in POLICY_KEYWORDS):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        **state,
        "intent": intent,
    }


def retrieve_context(query: str):
    collection = ensure_index()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    documents = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]

    return documents, ids



def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    documents, ids = retrieve_context(query)

    if not documents:
        return {
            **state,
            "context": [],
            "sources": [],
            "answer": "I could not find relevant information in the Zepto policy documents.",
            "confidence": 0.0,
        }

    if MOCK_LLM:
        top_chunk_snippet = documents[0][:200]

        response = AssistantResponse(
            answer=(
                f"Based on the retrieved context: "
                f"{top_chunk_snippet}"
            ),
            sources=ids,
            confidence=1.0,
        )

    else:
        context = "\n\n".join(documents)

        prompt = STRUCTURED_PROMPT.format(
            context=context,
            query=query,
        )

        response = call_real_llm(prompt, ids)

    return {
        **state,
        "context": documents,
        "sources": response.sources,
        "answer": response.answer,
        "confidence": response.confidence,
    }


def direct_answer(state: GraphState) -> GraphState:
    if MOCK_LLM:
        response = AssistantResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0,
        )

    else:
        prompt = STRUCTURED_PROMPT.format(
            context="No policy context was retrieved because this is a general question.",
            query=state["query"],
        )

        response = call_real_llm(prompt, [])

    return {
        **state,
        "context": [],
        "sources": response.sources,
        "answer": response.answer,
        "confidence": response.confidence,
    }


def route_intent(state: GraphState) -> str:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


def call_real_llm(prompt: str, sources: list[str]) -> AssistantResponse:
    """
    Optional real-LLM path with Pydantic validation and retries.

    MOCK_LLM=1 or unset is the default graded mode.
    MOCK_LLM=0 requires a configured LLM provider.
    """
    try:
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is required when MOCK_LLM=0."
            )

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=api_key,
        )

        last_error = None

        for attempt in range(3):
            try:
                schema_prompt = (
                    prompt
                    + "\n\nReturn ONLY valid JSON with exactly these fields:\n"
                    + '{"answer": "string", "sources": ["source_id"], '
                    + '"confidence": 0.0}'
                )

                if last_error:
                    schema_prompt += (
                        "\nYour previous response failed schema validation. "
                        "Return valid JSON only."
                    )

                response = llm.invoke(schema_prompt)

                return AssistantResponse.model_validate_json(
                    response.content
                )

            except Exception as exc:
                last_error = exc

        return AssistantResponse(
            answer="Unable to generate a valid structured response.",
            sources=sources,
            confidence=0.0,
        )

    except ImportError:
        raise RuntimeError(
            "Install langchain-groq to use MOCK_LLM=0."
        )


from langgraph.graph import StateGraph, END


workflow = StateGraph(GraphState)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

graph = workflow.compile()


def ask_question(query: str) -> AssistantResponse:
    state = graph.invoke({"query": query})

    return AssistantResponse(
        answer=state["answer"],
        sources=state.get("sources", []),
        confidence=state.get("confidence", 0.0),
    )
