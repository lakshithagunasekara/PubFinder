import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

keywords = ['responsible generative AI']
mandatory_filters = ['responsible,responsibility,ethical,ethics,trustworthy,trust,'
                     'explainable,fair,accountable,accountability,governance,health,education,creative,impact,'
                     'Standardization,Standardize,risk,challenges,issues,challenge,problem',
                     'Generative,large language model,llm,AI,GPT,artificial intelligence,machine']

start_year = 2020
end_year = 2025
sources = ['crossref', 'scopus', 'semscholar', 'gscholar', 'wos', 'masv2']

allow_filter_by_optional_keywords = False
optional_filter = ["principles", "strategies", "challenges", "issues", "limitations"]

filter_by_citations = False
most_recent_year_citation = 20
min_citations = 500

consider_year_for_filter = False

# Load sensitive information from environment variables
elsevier_api_keys = os.getenv("ELSEVIER_API_KEYS").split(',')
openai_key = os.getenv("OPENAI_KEY")

reuse = {
    "sources": True,
    "abstract_responses": False,
    "topics": False,
    "embeddings": False,
}
