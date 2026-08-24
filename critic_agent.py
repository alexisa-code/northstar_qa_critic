# critic_agent.py

# 1️ Import libraries
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

#  Logging counters
pass_count = 0
fail_count = 0


# 2️ Load environment variables
load_dotenv()

# 3️ Initialize DeepSeek client (OpenAI-compatible)
llm = ChatOpenAI(
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)

# 4️ Define your critic prompt
critic_prompt = PromptTemplate(
    input_variables=["ticket", "triage_output"],
    template=(
        "You are a QA Critic Engineer reviewing a customer support triage result.\n"
        "Ticket: {ticket}\n"
        "Triage Output: {triage_output}\n\n"
        "Evaluate the triage for:\n"
        "- Accuracy (does it correctly address the ticket intent?)\n"
        "- Clarity (is the response detailed and actionable?)\n"
        "- Safety (is it professional and avoids dismissive or vague language?)\n\n"
        "If the triage is vague, dismissive, incomplete, or unhelpful, respond with FAIL.\n"
        "Otherwise, respond with PASS.\n\n"
        "Format your answer as:\n"
        "PASS or FAIL\nReason: <short explanation>"
    )
)

# 5️ Create the LangChain pipeline
critic_chain = critic_prompt | llm | StrOutputParser()

# 6️ Example test run
def evaluate_ticket(ticket, triage_output):
    global pass_count, fail_count
    result = critic_chain.invoke({"ticket": ticket, "triage_output": triage_output})
    print("Critic Result:\n", result)

    if "FAIL" in result:
        fail_count += 1
        retry_triage(ticket, triage_output)
    else:
        pass_count += 1

    print(f"Summary → PASS: {pass_count} | FAIL: {fail_count}")


def retry_triage(ticket, triage_output):
    print("Retry triggered: sending back to triage agent for correction...")
    feedback_prompt = (
        f"The critic marked this triage as FAIL.\n"
        f"Ticket: {ticket}\n"
        f"Triage Output: {triage_output}\n"
        "Revise the triage to fix accuracy, clarity, or safety issues."
    )
    revised_output = llm.invoke(feedback_prompt)
    print("Revised Triage:\n", revised_output.content)
    return revised_output


if __name__ == "__main__":
    sample_ticket = "Customer reports that their login credentials are not working."
    sample_output = "Tell them to try again later."
    evaluate_ticket(sample_ticket, sample_output)
