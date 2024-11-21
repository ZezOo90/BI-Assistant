from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Verifies whether the question is related to the database schema.
class SchemaVerifier:
    """Verifies if the serious question relates to the schema of the database."""
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=self.gemini_api_key)

    def verify_schema_relevance(self, question: str, schema_info: str):
        """Verifies if the serious question is related to the database schema."""
        template = """
        You are a highly intelligent system designed to classify user questions as either **related** or **not related** to a database schema. 
        The database is a data warehouse that contains information about the company's operations, including data about employees, products, sales, and other related business processes. 
        The schema follows specific naming conventions where tables may start with prefixes like 'dim' (e.g., dimEmployee, dimDate, dimProduct) for dimension tables and 'fact' (e.g., factResellerSales) for fact tables.

        The question must be classified as "Related" only if the data requested can be extracted from the database based on the provided schema. 
        User questions may be in English or Arabic. Consider both the schema and the database context when making your determination.
        
        <SCHEMA>{schema_info}</SCHEMA>
        
        Question: {question}
        
        If the question is related to the schema and data can be extracted from the database, respond with "Related".
        
        Otherwise, respond with "Not related".
        """

        prompt = ChatPromptTemplate.from_template(template)
        
        prompt_text = prompt.format(question=question, schema_info=schema_info)
        response = self.llm.invoke(prompt_text)

        return response.content.strip().lower()  # "related" or "not related"