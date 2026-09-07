from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst.\n\n"
        "Extract all action items, follow-up tasks, commitments, promises, "
        "or next steps mentioned in the conversation.\n\n"
        "Action items may be explicit or implied.\n\n"
        "Examples:\n"
        "- I will send the catalog.\n"
        "- We can arrange a demo next week.\n"
        "- Let's discuss pricing later.\n"
        "- Sarah will prepare the report.\n\n"
        "For each action item provide:\n"
        "- Task\n"
        "- Owner (if known, otherwise 'Not specified')\n"
        "- Deadline (if mentioned, otherwise 'Not specified')\n\n"
        "Format as a numbered list.\n"
        "If absolutely no action items exist, return exactly:\n"
        "'No action items found.'"
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst.\n\n"
        "Extract all decisions, agreements, conclusions, approvals, "
        "recommended approaches, business directions, or commitments.\n\n"
        "Treat agreements, accepted proposals, approvals, planned collaborations, "
        "and agreed next steps as decisions.\n\n"
        "Decisions may be explicit or implied.\n\n"
        "Examples:\n"
        "- We will proceed with the partnership.\n"
        "- We agreed to launch next month.\n"
        "- We will use AWS.\n"
        "- Product demonstration will be arranged.\n"
        "- Eco-friendly products are a suitable fit for the market.\n\n"
        "Format as a numbered list.\n"
        "If absolutely no decisions exist, return exactly:\n"
        "'No key decisions found.'"
    )

    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst.\n\n"
        "Extract all open questions, unresolved issues, concerns, "
        "or discussion points requiring follow-up.\n\n"
        "Questions may be explicit or implied.\n\n"
        "Examples:\n"
        "- Should we use PostgreSQL or MongoDB?\n"
        "- What deadline should we choose?\n"
        "- Delivery schedule still needs confirmation.\n\n"
        "Format as a numbered list.\n"
        "If absolutely no open questions exist, return exactly:\n"
        "'No open questions found.'"
    )

    return chain.invoke(transcript)