import asyncio
import logging
import chainlit as cl
from core.resume_indexer import ResumeIndexer, Config as IndexerConfig
from core.rag_pipeline import RAGPipeline, RAGConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

indexer = ResumeIndexer(IndexerConfig())
pipeline = RAGPipeline(RAGConfig(), indexer)

async def animate_steps_loop(msg, steps):
    """Animate step messages for UI feedback."""
    try:
        current_step_index = 0
        while True:
            label, emoji = steps[current_step_index]
            dot_states = [".", "..", "..."]
            for dots in dot_states:
                animated_text = f"*{label}{dots}* {emoji}"
                msg.content = animated_text
                await msg.update()
                await asyncio.sleep(0.4)
            await asyncio.sleep(0.4)
            current_step_index = (current_step_index + 1) % len(steps)
    except asyncio.CancelledError:
        pass

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages and orchestrate pipeline response with animation."""
    steps_for_loop = [
        ("Thinking", "🧠"),
        ("Fetching data", "🔍"),
        ("Analyzing input", "📊"),
        ("Summarizing", "📝"),
    ]
    msg = cl.Message(content="Starting...")
    await msg.send()
    animation_task = asyncio.create_task(animate_steps_loop(msg, steps_for_loop))
    try:
        result = await asyncio.to_thread(pipeline.search_and_analyze, message.content)
        animation_task.cancel()
        await asyncio.sleep(0.1)
        finalizing_label, finalizing_emoji = ("Finalizing response", "✅")
        dot_states = [".", "..", "..."]
        for dots in dot_states:
            animated_text = f"*{finalizing_label}{dots}* {finalizing_emoji}"
            msg.content = animated_text
            await msg.update()
            await asyncio.sleep(0.4)
        await asyncio.sleep(0.4)
        response = f"## 📝 AI Analysis\n"
        best_candidate_name = 'N/A'
        reason_text = 'No analysis available'
        if result.get('analysis'):
            analysis_lines = result['analysis'].split('\n')
            for line in analysis_lines:
                if line.startswith('Best Candidate:'):
                    best_candidate_name = line.replace('Best Candidate:', '').strip()
                elif line.startswith('Reason:'):
                    reason_text = line.replace('Reason:', '').strip()
        response += f"**Best Candidate:** {best_candidate_name}\n\n"
        response += f"**Reason:** {reason_text}\n\n"
        response += f"---\n"
        response += f"## 👥 Matching Candidates\n"
        if result.get('matches'):
            for match in result['matches']:
                response += (
                    f"**Candidate ID:** `{match.get('candidate_id', 'N/A')}`\n"
                    f"**Score:** `{round(match.get('score', 0), 1)}%`\n"
                    f"**Summary:** {match.get('summary', '')}\n\n"
                    "---\n"
                )
        else:
            response += "No matching resumes found."
        full_response_text = response
        msg.content = ""
        await msg.update()
        for char in full_response_text:
            msg.content += char
            await msg.update()
            await asyncio.sleep(0.005)
    except Exception as e:
        animation_task.cancel()
        msg.content = f"Error: {str(e)}"
        await msg.update()