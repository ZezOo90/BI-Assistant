from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Classifies user questions as serious or non-serious.
class QuestionClassifier:
    """Classifies user questions as serious or non-serious."""
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=self.gemini_api_key)
        
    def classify_question(self, question: str):
        """Classify the question as serious or non-serious."""
        template = """
        Classify the following question as either "Serious" or "Non-Serious." 
        Serious questions are related to database queries, business matters, or technical inquiries, such as questions about the company's operations, employees, products, sales, or other business processes. 
        Non-serious questions may be humorous, light-hearted, or irrelevant to these topics.

        The user can ask in either Arabic or English. Your response must match the language used by the user.

        Question: {question}
        
        Respond with either "Serious" or "Non-Serious".
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        prompt_text = prompt.format(question=question)
        response = self.llm.invoke(prompt_text)

        return response.content.strip().lower()  # Return "serious" or "non-serious"