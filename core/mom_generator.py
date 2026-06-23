from core.summarizer import get_llm


def generate_mom(transcript: str):

    llm = get_llm()

    prompt = f"""
Generate professional Meeting Minutes (MOM) from the transcript.

IMPORTANT:
- Never use placeholders like [Insert Date], [Insert Time], [Name], [Date].
- If information is unavailable, write "Not Mentioned".
- Only use information present in the transcript.
- Do not invent attendees, dates, deadlines, or action items.
- Do not create tables.
- Use bullet points only.
- Do not repeat the entire summary.
- If no action items exist, write exactly: No action items identified.
- If no decisions exist, write exactly: No key decisions identified.
- Keep Meeting Overview maximum 2 bullet points.
- Keep Discussion Summary maximum 3 bullet points.
- Focus mainly on Key Decisions, Action Items and Next Steps.

Output format:

Meeting Overview
- bullet point
- bullet point

Discussion Summary
- bullet point
- bullet point

Key Decisions
- bullet point

Action Items
- bullet point

Next Steps
- bullet point

Transcript:
{transcript}
"""

    response = llm.invoke(prompt)

    mom = response.content.strip()

    return mom