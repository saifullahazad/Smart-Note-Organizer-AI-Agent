import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables from .env
load_dotenv()

# Initialize the LLM (ensure OPENAI_API_KEY is set in your .env)
llm = OpenAI(temperature=0.5)

# Define prompt templates for summarization and keyword extraction
summary_template = PromptTemplate(
    input_variables=["note"],
    template="Summarize the following note in two sentences:\n\n{note}\n\nSummary:"
)

keywords_template = PromptTemplate(
    input_variables=["note"],
    template="Extract 5 key keywords or tags from the following note:\n\n{note}\n\nKeywords:"
)

# Initialize chains for summarization and keyword extraction
summary_chain = LLMChain(llm=llm, prompt=summary_template)
keywords_chain = LLMChain(llm=llm, prompt=keywords_template)

def organize_note(note: str):
    """Process the note to generate a summary and extract keywords."""
    summary = summary_chain.run(note)
    keywords = keywords_chain.run(note)
    return summary.strip(), keywords.strip()

if __name__ == "__main__":
    note = input("Enter your note: ")
    summary, keywords = organize_note(note)
    print("\n--- Organized Note ---")
    print(f"Summary:\n{summary}")
    print(f"\nKeywords:\n{keywords}")
