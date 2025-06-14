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
            "You are a senior technical hiring expert.\n\n"
            "You are given a job requirement and 5 candidate resumes. Your task is to review the candidate profiles and provide a clear, comparative, bullet-point-based evaluation focused only on the relevance to the job.\n\n"
            f"Job Requirement:\n{query}\n\n"
            f"Candidate Profiles:\n{context_text}\n\n"
            "Instructions:\n"
            "- For each candidate, summarize in 4-6 bullet points:\n"
            "  ✅ Key relevant experience, skills, or tools aligned with the job\n"
            "  ⚠️ Any notable gaps, missing experience, or partial alignment\n"
            "- Focus specifically on data engineering relevance (if that's the job role).\n"
            "- Do not add personal opinions or use scores.\n"
            "- Keep bullet points short, precise, and meaningful.\n\n"
            "Output Format:\n"
            "🔹 1. [Candidate Name]\n"
            "Title: [Job Title]\n"
            "Experience: [Total Years, Relevant Years if possible]\n"
            "Relevant to [Job Role]:\n"
            "✅ ...\n"
            "✅ ...\n"
            "⚠️ ...\n"
            "\n"
            "🔹 2. ...\n"
        )

# Final analysis prompt
def get_final_analysis_prompt(query: str, detailed_comparison_text: str) -> str:
    return (
            "You are an expert technical recruiter tasked with selecting the best-fit candidate for a specific job role based on a comparative analysis.\n\n"
            f"Job Requirement:\n{query}\n\n"
            f"Candidate Comparative Analysis:\n{detailed_comparison_text}\n\n"
            "Task:\n"
            "Review the analysis and select ONLY ONE candidate who is the best fit for the job.\n\n"
            "Instructions:\n"
            "- Focus only on how each candidate’s experience, skills, and contributions align with the job requirements.\n"
            "- Do NOT compare candidates explicitly in your answer.\n"
            "- Only select a candidate if they clearly meet or exceed the job expectations.\n"
            "- If none of the candidates are a strong match, clearly state that.\n\n"
            "Output Format:\n"
            "Best Candidate: <Candidate Name>\n"
            "Reason: <2–3 sentence explanation highlighting specific qualifications and experiences that make this candidate a strong fit for the role>\n"
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