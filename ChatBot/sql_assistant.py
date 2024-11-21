from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from schema_describer import SchemaDescriber
import streamlit as st
import time
import os

class SQLAssistant:
    """Manages LLM-driven SQL query generation and response handling."""
    def __init__(self, gemini_api_key_1: str, gemini_api_key_2: str):
        self.gemini_api_key_1 = gemini_api_key_1
        self.gemini_api_key_2 = gemini_api_key_2
        self.gemini_api_key_3 = os.getenv('GEMINI_API_KEY_2')
        self.describer = SchemaDescriber(self.gemini_api_key_3)

    def get_sql_chain(self, db: SQLDatabase):
        """Creates a chain to generate SQL queries from user input."""
        template = """
        You are a data analyst at a company, working with a data warehouse. The database schema follows a naming convention where column and table names may start with prefixes like 'dim' for dimension tables (e.g., dimEmployee, dimDate, dimProduct) and 'fact' for fact tables (e.g., factResellerSales). You are interacting with a user who is asking questions about this data warehouse.

        The user's question may be in English or Arabic. Based on the table schema provided below and the conversation history, write an SQL query that would answer the user's question. Take the conversation history into account.

        <SCHEMA>{schema}</SCHEMA>
        
        Conversation History: {chat_history}
        
        Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
        """
        prompt = ChatPromptTemplate.from_template(template)
        llm_1 = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=self.gemini_api_key_1)

        def get_schema(_):
            schema_info = db.get_table_info()
            descriptions = self.describer.describe_and_save_all(schema_info)
            return descriptions

        return (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt
            | llm_1
            | StrOutputParser()
        )

    def generate_response(self, user_query: str, db: SQLDatabase, chat_history: list):
        """Generates a natural language response to the SQL query."""
        sql_chain = self.get_sql_chain(db)

        response_template = """
        You are a data analyst at a company.Response in the same language as the user's question. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, SQL query, and SQL response, write a natural language response.

        if user ask in Arabic,write a natural language explanation in Arabic only.
        if user ask in English, write it in English only. 

        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}
        """
        prompt = ChatPromptTemplate.from_template(response_template)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=self.gemini_api_key_2)

        def get_schema(_):
            schema_info = db.get_table_info()
            descriptions = self.describer.describe_and_save_all(schema_info)
            return descriptions

        chain = (
            RunnablePassthrough.assign(query=sql_chain).assign(
                schema=lambda _: get_schema,
                response=lambda vars: db.run(vars["query"]),
            )
            | prompt
            | llm
            | StrOutputParser()
        )


        try:
            # Attempt to invoke the chain
            time.sleep(10)
            return chain.invoke({
                "question": user_query,
                "chat_history": chat_history,
            })
        except Exception as e:
            # Handle API-related errors or any other exceptions
            error_message = (
                "An error occurred while processing your request. "
                "This could be due to hitting the API limit or other issues. Please try again later."
            )
            st.error(error_message)  # Display the error message in the Streamlit app
            return error_message