import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever
)


def get_llm():

    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )


def format_docs(docs):

    return "\n\n".join(
        [doc.page_content for doc in docs]
    )


def create_rag_chain(retriever):

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant.

Answer the user's question using the meeting transcript context.

You must answer the user's question whenever the transcript contains information related to it.

For broad questions such as "What is this video about?", summarize the main topic of the transcript.

Do not say "I could not find this information in the meeting transcript" if the transcript contains information that can answer or summarize the question.

Only say:

"I could not find this information in the meeting transcript."

when the meeting transcript contains no relevant information at all.

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:

{context}"""
        ),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def build_rag_chain(transcript: str):

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(
        vector_store,
        k=4
    )

    return create_rag_chain(retriever)


def load_rag_chain():

    vector_store = load_vector_store()

    retriever = get_retriever(
        vector_store,
        k=4
    )

    return create_rag_chain(retriever)


def ask_question(rag_chain, question: str) -> str:

    print(f"Question: {question}")

    answer = rag_chain.invoke(question)

    print(f"Answer: {answer}")

    return answer