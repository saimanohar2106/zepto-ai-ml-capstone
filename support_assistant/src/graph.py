import os
from typing import TypedDict

from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

from ingestion import ensure_index
from prompts import STRUCTURED_PROMPT


# ============================================================
# Configuration
# ============================================================

# Default behavior is MOCK_LLM=1.
# Real LLM is used only when MOCK_LLM=0.
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"


# Load the embedding model once when the application starts.
# This avoids downloading/loading it for every API request.
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# Graph State
# ============================================================

class GraphState(TypedDict, total=False):
    query: str
    intent: str
    context: list[str]
    sources: list[str]
    answer: str
    confidence: float


# ============================================================
# Structured Response
# ============================================================

class AssistantResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# ============================================================
# Intent Classification
# ============================================================

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
    """
    Required mock-mode intent classification.

    A query containing one of the policy keywords is classified
    as a policy question. Everything else is classified as a
    general question.
    """

    query = state["query"].lower()

    if any(keyword in query for keyword in POLICY_KEYWORDS):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        **state,
        "intent": intent,
    }


# ============================================================
# Retrieval
# ============================================================

def retrieve_context(query: str) -> tuple[list[str], list[str]]:
    """
    Embed the query locally and retrieve the top-3 chunks
    from the ChromaDB collection.
    """

    collection = ensure_index()

    query_embedding = EMBEDDING_MODEL.encode(
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


# ============================================================
# Policy Question Node
# ============================================================

def retrieve_and_answer(state: GraphState) -> GraphState:
    """
    Retrieve policy context and generate an answer.

    In MOCK_LLM mode:
        No LLM is called. A deterministic answer is generated
        from the most relevant retrieved chunk.

    In real-LLM mode:
        The retrieved context is passed to the structured prompt
        and the response is validated with Pydantic.
    """

    query = state["query"]

    documents, ids = retrieve_context(query)

    # No relevant documents found.
    if not documents:
        response = AssistantResponse(
            answer=(
                "I could not find relevant information in the "
                "Zepto policy documents."
            ),
            sources=[],
            confidence=0.0,
        )

        return {
            **state,
            "context": [],
            "sources": [],
            "answer": response.answer,
            "confidence": response.confidence,
        }

    # --------------------------------------------------------
    # Required graded MOCK_LLM path
    # --------------------------------------------------------

    if MOCK_LLM:
        top_chunk_snippet = documents[0][:200]

        response = AssistantResponse(
            answer=(
                "Based on the retrieved context: "
                f"{top_chunk_snippet}"
            ),
            sources=ids,
            confidence=1.0,
        )

    # --------------------------------------------------------
    # Optional real-LLM path
    # --------------------------------------------------------

    else:
        context = "\n\n".join(documents)

        prompt = STRUCTURED_PROMPT.format(
            context=context,
            query=query,
        )

        response = call_real_llm(
            prompt=prompt,
            sources=ids,
        )

    return {
        **state,
        "context": documents,
        "sources": response.sources,
        "answer": response.answer,
        "confidence": response.confidence,
    }


# ============================================================
# General Question Node
# ============================================================

def direct_answer(state: GraphState) -> GraphState:
    """
    Handle questions that are not classified as Zepto policy
    questions.

    In MOCK_LLM mode, return the required deterministic response.

    In real-LLM mode, call the configured LLM without retrieval.
    """

    # --------------------------------------------------------
    # Required graded MOCK_LLM path
    # --------------------------------------------------------

    if MOCK_LLM:
        response = AssistantResponse(
            answer=(
                "I can only answer questions about Zepto "
                "policies right now."
            ),
            sources=[],
            confidence=1.0,
        )

    # --------------------------------------------------------
    # Optional real-LLM path
    # --------------------------------------------------------

    else:
        prompt = STRUCTURED_PROMPT.format(
            context=(
                "No policy context was retrieved because this "
                "is a general question."
            ),
            query=state["query"],
        )

        response = call_real_llm(
            prompt=prompt,
            sources=[],
        )

    return {
        **state,
        "context": [],
        "sources": response.sources,
        "answer": response.answer,
        "confidence": response.confidence,
    }


# ============================================================
# Conditional Routing
# ============================================================

def route_intent(state: GraphState) -> str:
    """
    Route the graph according to the intent determined by
    classify_intent.
    """

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# Optional Real LLM
# ============================================================

def call_real_llm(
    prompt: str,
    sources: list[str],
) -> AssistantResponse:
    """
    Optional real-LLM implementation.

    MOCK_LLM=1 or unset:
        This function is not called.

    MOCK_LLM=0:
        Uses Groq and validates the response against the
        AssistantResponse Pydantic schema.

    The function retries up to three total attempts if the
    returned response does not satisfy the required JSON schema.
    """

    try:
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is required when MOCK_LLM=0."
            )

        llm = ChatGroq(
            model=os.getenv(
                "GROQ_MODEL",
                "llama-3.1-8b-instant",
            ),
            temperature=0,
            api_key=api_key,
        )

        last_error = None

        # Initial attempt + two retries = 3 total attempts.
        for attempt in range(3):

            schema_prompt = (
                prompt
                + "\n\n"
                + "Return ONLY valid JSON with exactly these "
                "three fields:\n"
                + '{"answer": "string", '
                '"sources": ["source_id"], '
                '"confidence": 0.0}'
            )

            if last_error is not None:
                schema_prompt += (
                    "\n\nYour previous response failed schema "
                    "validation. Correct the response and return "
                    "ONLY valid JSON matching the required schema."
                )

            try:
                llm_response = llm.invoke(schema_prompt)

                return AssistantResponse.model_validate_json(
                    llm_response.content
                )

            except Exception as exc:
                last_error = exc

        # All three attempts failed.
        return AssistantResponse(
            answer=(
                "Unable to generate a valid structured response."
            ),
            sources=sources,
            confidence=0.0,
        )

    except ImportError:
        raise RuntimeError(
            "Install langchain-groq to use MOCK_LLM=0."
        )


# ============================================================
# LangGraph Construction
# ============================================================

def build_graph():
    """
    Build and compile the LangGraph workflow.

    Graph:

        START
          |
          v
    classify_intent
       /        \
      /          \
 policy        general
   |              |
   v              v
retrieve       direct
   |              |
   v              v
  END            END
    """

    workflow = StateGraph(GraphState)

    # Three required nodes.
    workflow.add_node(
        "classify_intent",
        classify_intent,
    )

    workflow.add_node(
        "retrieve_and_answer",
        retrieve_and_answer,
    )

    workflow.add_node(
        "direct_answer",
        direct_answer,
    )

    # Start -> classify.
    workflow.add_edge(
        START,
        "classify_intent",
    )

    # Conditional routing.
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    # Both answer nodes terminate the graph.
    workflow.add_edge(
        "retrieve_and_answer",
        END,
    )

    workflow.add_edge(
        "direct_answer",
        END,
    )

    return workflow.compile()


# Compile the graph once when the application starts.
graph = build_graph()


# ============================================================
# Public Helper
# ============================================================

def ask_question(query: str) -> AssistantResponse:
    """
    Run a user query through the complete LangGraph workflow
    and return the validated AssistantResponse object.
    """

    state = graph.invoke(
        {
            "query": query,
        }
    )

    return AssistantResponse(
        answer=state["answer"],
        sources=state.get("sources", []),
        confidence=state.get("confidence", 0.0),
    )
