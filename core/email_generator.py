from core.summarizer import get_llm

def generate_followup_email(transcript: str):

    llm = get_llm()

    prompt = f"""
You are an executive meeting assistant.

Generate a professional follow-up email.

STRICT RULES:
- Use ONLY information explicitly present in the transcript.
- Never invent names, dates, deadlines, contact information, job titles, or follow-up calls.
- Never use placeholders like:
  [Date]
  [Name]
  [Position]
  [Contact Information]
  TBD
- If information is not mentioned, omit it completely.
- Do not guess future actions.
- Do not add recommendations.
- Keep the email concise and professional.
- Maximum 150 words.

Structure:

Subject:
Greeting:
Key Points:
Decisions:
Action Items:
Next Steps:
Closing:

Transcript:
{transcript}
"""

    response = llm.invoke(prompt)
    return response.content