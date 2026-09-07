import time

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize this portion of a meeting transcript concisely.",
            ),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    # chunk_summaries = [
    #     map_chain.invoke({"text": chunk})
    #     for chunk in chunks
    # ]

    chunk_summaries = []

    for chunk in chunks:
      max_retries = 5

      for attempt in range(max_retries):
        try:
            result = map_chain.invoke({"text": chunk})
            chunk_summaries.append(result)

            # Har successful request ke baad thoda gap
            time.sleep(2)
            break

        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt == max_retries - 1:
                    raise Exception(
                        "Mistral API rate limit reached. Please wait a minute and try again."
                    )

                wait_time = 10 * (attempt + 1)
                time.sleep(wait_time)
            else:
                raise e

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer.\n\n"
                "Generate a concise professional meeting summary.\n"
                "Write in 2-4 short paragraphs.\n"
                "Use normal business language.\n"
                "Avoid markdown headings.\n"
                "Avoid bullet points.\n"
                "Avoid excessive blank lines.\n"
                "Do not repeat information.\n"
                "Focus on the purpose of the meeting, key discussion points, "
                "important outcomes, and next steps.\n"
                "Keep the summary easy to read and under 200 words."
            ),
            ("human", "{text}"),
        ]
    )

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Based on the meeting transcript, generate a short professional meeting title "
                    "(max 8 words). Only return the title, nothing else.",
                ),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])