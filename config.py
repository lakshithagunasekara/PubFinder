import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

keywords = ['Responsible Generative AI']
allow_filter_by_mandatory_keywords = True
mandatory_filters = ['responsible,ethical,ethics,governance,safety,regulation,transparent,explainable,XAI,fair,'
                     'accountable,bias,fairness,transparency,accountability,oversight,auditing,risk,standards,'
                     'ethical,interpretability,human-centered,trust,good,sustainable,inclusive,policy,compliance,'
                     'harm,responsibility,transparency,robustness,trustworthiness,privacy,security,human in the loop,'
                     'fairness',
                     'generative,large language model,LLM,GPT,'
                     'transformer,text generation,language model,GAN,LLAMA,Gemini'
                     'generative modeling,diffusion model']

start_year = 2020
end_year = 2025
sources = ['crossref', 'scopus', 'semscholar', 'gscholar', 'wos', 'masv2']

allow_filter_by_optional_keywords = True
optional_filter = ["responsible Generative AI development", "challenges of Generative AI",
                   "issues/problems with generative AI", "usecases of responsible generative AI developement"
                   "principles of responsible Generative AI development",
                   "Applications of responsible Generative AI development", "policies for responsible generative AI"
                   "generative ai governance"]

filter_by_citations = False
most_recent_year_citation = 0
min_citations = 20

consider_year_for_filter = False

# Load sensitive information from environment variables
elsevier_api_keys = os.getenv("ELSEVIER_API_KEYS").split(',')
openai_key = os.getenv("OPENAI_KEY")

only_cluster = True

reuse = {
    "sources": True,
    "abstract_responses": True,
    "topics": True,
    "embeddings": True,
}
