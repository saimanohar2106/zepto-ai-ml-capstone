\# Module 3 — Zepto Support Assistant



A RAG-based customer-support assistant for answering questions about Zepto policies.



The application uses local sentence-transformer embeddings, ChromaDB vector search, LangGraph routing, Pydantic validation, and FastAPI.



\---



\## 1. Architecture



The system follows this flow:



```text

Zepto Policy Documents

&#x20;       |

&#x20;       v

Document Ingestion

&#x20;       |

&#x20;       v

Chunking

&#x20;       |

&#x20;       v

all-MiniLM-L6-v2 Embeddings

&#x20;       |

&#x20;       v

ChromaDB Vector Store

&#x20;       |

&#x20;       v

User Question

&#x20;       |

&#x20;       v

LangGraph classify\_intent

&#x20;       |

&#x20;       +-----------------------------+

&#x20;       |                             |

&#x20;       v                             v

policy\_question                general\_question

&#x20;       |                             |

&#x20;       v                             v

retrieve\_and\_answer             direct\_answer

&#x20;       |                             |

&#x20;       v                             |

Top-3 ChromaDB Retrieval              |

&#x20;       |                             |

&#x20;       v                             v

Mock/Real Generation --------> Pydantic Response

&#x20;                                     |

&#x20;                                     v

&#x20;                                 FastAPI /ask









support\_assistant/

├── chroma\_db/

├── docs/

│   ├── doc\_01.txt

│   ├── doc\_02.txt

│   ├── doc\_03.txt

│   ├── doc\_04.txt

│   ├── doc\_05.txt

│   ├── doc\_06.txt

│   ├── doc\_07.txt

│   └── doc\_08.txt

├── src/

│   ├── ingestion.py

│   ├── prompts.py

│   ├── graph.py

│   └── main.py

├── tests/

│   └── test\_support\_assistant.py

├── Dockerfile

├── requirements.txt

└── README.md







Document Ingestion



The eight Zepto policy documents are stored in:



support\_assistant/docs/



The src/ingestion.py file provides:



load\_documents() — loads all policy documents.

get\_collection() — creates or opens the ChromaDB collection.

build\_index() — creates embeddings and stores them.

ensure\_index() — makes sure the vector index exists.



The embedding model is:



all-MiniLM-L6-v2



Embeddings are generated locally using Sentence Transformers.



ChromaDB



The persistent vector database is stored in:



support\_assistant/chroma\_db/



Collection name:



zepto\_policy



The eight documents are indexed using:



doc\_01

doc\_02

doc\_03

doc\_04

doc\_05

doc\_06

doc\_07

doc\_08



For policy questions, the query is embedded using the same model and the top 3 relevant documents are retrieved using cosine similarity.



Structured Prompt



src/prompts.py contains STRUCTURED\_PROMPT.



The prompt includes:



ROLE

CONTEXT

TASK

FORMAT

LENGTH

NEGATIVE CONSTRAINT

FEW-SHOT EXAMPLE

RETRIEVED POLICY CONTEXT

CUSTOMER QUESTION



The negative constraint tells the assistant:



Do not answer using information that is not present in the provided context.



The prompt also contains a few-shot example showing how a policy question should be answered.



The structured prompt is used by the optional real-LLM path.



LangGraph Workflow



src/graph.py defines the typed graph state using TypedDict.



The LangGraph workflow contains three nodes.



1\. classify\_intent



The query is classified as:



policy\_question



or:



general\_question



The mock/default mode checks these policy keywords:



delivery

return

refund

membership

tracking

cancel

gift card

support hours



The routing decision is independent of MOCK\_LLM.



2\. retrieve\_and\_answer



For policy questions:



Embed the user query.

Search ChromaDB.

Retrieve the top 3 documents.

Generate the answer.

Return source document IDs.

Return confidence.



In mock mode the answer starts with:



Based on the retrieved context:



and uses approximately 200 characters from the top retrieved document.



3\. direct\_answer



For general questions, mock mode returns:



I can only answer questions about Zepto policies right now.



General questions return an empty source list.



MOCK\_LLM



The default mode is deterministic mock mode.



If MOCK\_LLM is unset or:



MOCK\_LLM=1



no external LLM API is required.



This mode is used for deterministic evaluation.



An optional real-LLM path is available when:



MOCK\_LLM=0



and a valid GROQ\_API\_KEY is configured.



The baseline project does not require an API key.



Pydantic Response Schema



The response uses the AssistantResponse Pydantic model:



answer: str

sources: list\[str]

confidence: float



Confidence is constrained between:



0.0 and 1.0



Mock responses use:



confidence = 1.0



Policy questions return retrieved document IDs.



General questions return:



\[]



for sources.



FastAPI



src/main.py provides:



POST /ask



Request:



{

&#x20; "query": "What is the delivery fee?"

}



The application runs using:



uvicorn main:app --reload --host 127.0.0.1 --port 7860

API Demonstration

Policy / Retrieval Question



Request:



{

&#x20; "query": "What is the delivery fee?"

}



Response:



{

&#x20; "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",

&#x20; "sources": \[

&#x20;   "doc\_01",

&#x20;   "doc\_05",

&#x20;   "doc\_02"

&#x20; ],

&#x20; "confidence": 1.0

}

General Question



Request:



{

&#x20; "query": "What is your favorite food?"

}



Response:



{

&#x20; "answer": "I can only answer questions about Zepto policies right now.",

&#x20; "sources": \[],

&#x20; "confidence": 1.0

}

Testing



Automated tests are located at:



support\_assistant/tests/test\_support\_assistant.py



Run:



pytest support\_assistant/tests



Current test result:



6 passed



The tests verify:



8 policy documents exist.

ChromaDB contains 8 documents.

Policy questions use retrieval.

General questions use direct answering.

FastAPI policy endpoint works.

FastAPI general endpoint works.

Response confidence is valid.

Docker



The project includes:



support\_assistant/Dockerfile



The application exposes port:



7860



The Docker container starts FastAPI using:



uvicorn src.main:app --host 0.0.0.0 --port 7860

End-to-End Data Flow

1\. Load 8 Zepto policy documents

&#x20;       |

2\. Generate local embeddings

&#x20;  using all-MiniLM-L6-v2

&#x20;       |

3\. Store vectors in ChromaDB

&#x20;  collection: zepto\_policy

&#x20;       |

4\. Receive POST /ask request

&#x20;       |

5\. LangGraph classifies the intent

&#x20;       |

6\. Policy question:

&#x20;     embed query

&#x20;     retrieve top 3 documents

&#x20;     generate answer

&#x20;       |

7\. General question:

&#x20;     return direct answer

&#x20;       |

8\. Validate response with Pydantic

&#x20;       |

9\. Return JSON through FastAPI

Technologies

Python

Sentence Transformers

all-MiniLM-L6-v2

ChromaDB

LangGraph

Pydantic

FastAPI

Uvicorn

Docker



