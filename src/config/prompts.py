"""
Prompt configurations for the RAG pipeline.
"""

# Summary generation prompt
def get_summary_prompt(query: str, text: str) -> str:
    return (
        "You are a professional resume analyst preparing a summary for a recruiter with limited technical background. "
        "Your task is to write a clear, one-paragraph overview of the candidate based on the resume provided.\n\n"
        
        "Instructions:\n"
        "1. Use plain, easy-to-understand language — avoid technical jargon and acronyms.\n"
        "2. Structure the summary as follows:\n"
        "   - Begin with the candidate's full name.\n"
        "   - State their primary professional role and total years of experience.\n"
        "   - Highlight key skills, focus areas, or notable achievements.\n"
        "   - Mention any relevant certifications, awards, or unique contributions (if applicable).\n"
        "3. Keep the tone professional, concise, and informative.\n"
        "4. Do NOT use headings like 'Summary'. Start directly with the paragraph.\n\n"
        
        f"Job Requirement:\n{query}\n\n"
        f"Resume:\n{text}\n\n"
        "Candidate Summary:"
    )

# Analysis prompt
def get_analysis_prompt(query: str, context_text: str) -> str:
    return (
        "You are a senior hiring analyst. Analyze the following 5 candidate profiles in response to the job requirement.\n\n"
        f"Job Requirement:\n{query}\n\n"
        f"Candidate Profiles:\n{context_text}\n\n"
        "Instructions:\n"
        "- For each candidate, evaluate the following:\n"
        "  1. Relevant Experience: How well their experience aligns with the job.\n"
        "  2. Technical Skills: Proficiency with required tools and technologies.\n"
        "  3. Project Contributions: Noteworthy projects that demonstrate capability.\n"
        "  4. Soft Skills or Unique Traits if evident.\n\n"
        "- Provide your evaluation for each candidate clearly and objectively.\n"
        "- Avoid vague praise or irrelevant details.\n"
        "- Do not use matching scores. Focus on relevance and quality of experience.\n\n"
        "Output Format:\n"
        "Candidate 1:\n- Relevant Experience: ...\n- Technical Skills: ...\n- Projects: ...\n- Soft Skills: ...\n\n"
        "...\n\n"
    )

# Final analysis prompt
def get_final_analysis_prompt(query: str, detailed_comparison_text: str) -> str:
    return (
        "You are a senior technical recruiter reviewing detailed comparative evaluations of 5 candidates for a job.\n\n"
        f"Job Requirement:\n{query}\n\n"
        f"Candidate Comparative Analysis:\n{detailed_comparison_text}\n\n"
        "Task:\n"
        "Based on the above detailed analysis, select the single best candidate who most closely fits the job requirement.\n"
        "Your decision should be based on:\n"
        "- Depth and relevance of experience\n"
        "- Specific technical skills matching the job\n"
        "- Complexity and impact of projects\n"
        "- Any standout qualifications or traits\n\n"
        "Important:\n"
        "- Do NOT compare or mention other candidates.\n"
        "- Provide a clear, 2-3 sentence justification citing concrete examples.\n\n"
        "Output Format:\n"
        "Best Candidate: <Candidate Name>\n"
        "Reason: <Justification>\n"
    )

# Unwanted prefixes to remove from summaries
UNWANTED_SUMMARY_PREFIXES = [
    "Summary:",
    "Here is a concise summary for the recruiter:",
    "Here is a summary of the candidate's resume for a recruiter:",
    "Here's a concise summary of the candidate for the recruiter:",
    "Key elements used from the resume:",
    "Summary:"
] 