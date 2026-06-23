from core.summarizer import get_llm

def generate_speaker_summary(transcript: str):

    llm = get_llm()

    prompt = f"""
Analyze the meeting transcript.

Identify each speaker and provide:

- Speaker Name
- Key contributions
- Important points discussed

Keep each speaker summary concise (max 3 bullet points).

If speakers cannot be identified, return exactly:
Speaker information not available.

Transcript:
{transcript}
"""

    response = llm.invoke(prompt)

    return response.content