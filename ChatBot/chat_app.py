import streamlit as st
from dotenv import load_dotenv
from db_connector import DatabaseConnector
from question_classifier import QuestionClassifier
from schema_verifier import SchemaVerifier
from non_serious_assistant import NonSeriousAssistant
from sql_assistant import SQLAssistant
from schema_describer import SchemaDescriber
from langchain.schema import AIMessage, HumanMessage
import os

# The main Streamlit app to interact with the user.
class ChatApp:
    """Streamlit app to interact with the SQL assistant."""
    def __init__(self):
        load_dotenv()
        self.gemini_api_key_1 = os.getenv('GEMINI_API_KEY_1')
        self.gemini_api_key_2 = os.getenv('GEMINI_API_KEY_2')
        self.gemini_api_key_3 = os.getenv('GEMINI_API_KEY_3')
        self.gemini_api_key_4 = os.getenv('GEMINI_API_KEY_4')
        self.gemini_api_key_5 = os.getenv('GEMINI_API_KEY_5')
        self.gemini_api_key_6 = os.getenv('GEMINI_API_KEY_6')
        self.db_connector = DatabaseConnector()
        self.describer = SchemaDescriber(self.gemini_api_key_2)
        self.sql_assistant = SQLAssistant(self.gemini_api_key_4, self.gemini_api_key_1)
        self.schema_verifier = SchemaVerifier(self.gemini_api_key_3)
        self.non_serious_assistant = NonSeriousAssistant(self.gemini_api_key_5)
        self.question_classifier = QuestionClassifier(self.gemini_api_key_6)
        self.init_session_state()

    def init_session_state(self):
        """Initialize Streamlit session state variables."""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello! I'm SQL assistant. Ask me anything about your database."),
            ]
        if "db" not in st.session_state:
            st.session_state.db = None

    def sidebar(self):
        """Render the sidebar for database connection settings."""
        st.sidebar.subheader("Settings")
        st.sidebar.text_input("Host", value="localhost", key="Host")
        st.sidebar.text_input("Port", value="3306", key="Port")
        st.sidebar.text_input("User", value="root", key="User")
        st.sidebar.text_input("Password", type="password", value="admin", key="Password")
        st.sidebar.text_input("Database", value="AdventureWorksDW2022_copy", key="Database")

        if st.sidebar.button("Connect"):
            with st.spinner("Connecting to database..."):
                try:
                    st.session_state.db = self.db_connector.connect(
                        st.session_state["User"],
                        st.session_state["Password"],
                        st.session_state["Host"],
                        st.session_state["Port"],
                        st.session_state["Database"]
                    )
                    st.success("Connected to the database!")
                except Exception as e:
                    st.error(f"Failed to connect: {e}")

    def display_chat_history(self):
        """Render the chat history in the app."""
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
    def handle_user_query(self):
        """Process user input and generate responses."""
        user_query = st.chat_input("Type a message...")
        if user_query:
            st.session_state.chat_history.append(HumanMessage(user_query))

            with st.chat_message("Human"):
                st.markdown(user_query)

            with st.chat_message("AI"):
                if st.session_state.db:
                    # First classify the question
                    classification = self.question_classifier.classify_question(user_query)

                    if classification == "serious":
                        # Check if the serious question is related to the schema
                        schema_info = st.session_state.db.get_table_info()  # Retrieve schema information
                        descriptions = self.describer.describe_and_save_all(schema_info)
                        relevance = self.schema_verifier.verify_schema_relevance(user_query, descriptions)

                        if relevance == "related":
                            response = self.sql_assistant.generate_response(
                                user_query,
                                st.session_state.db,
                                st.session_state.chat_history[-2:]
                            )
                            st.markdown(response)
                            st.session_state.chat_history.append(AIMessage(content=response))

                        else:
                            response = "Thank you for your question! It doesn't seem to be related to the database. Feel free to ask anything related to the database, such as queries about employees, products, sales, or other business data, and I'll be happy to assist you!"
                            st.markdown(response)
                            st.session_state.chat_history.append(AIMessage(content=response))

                    else:
                        # Handle non-serious questions with the playful assistant
                        response = self.non_serious_assistant.get_non_serious_response(user_query)
                        st.markdown(response)
                        st.session_state.chat_history.append(AIMessage(content=response))


    def run(self):
        """Run the Streamlit app."""
        st.set_page_config(page_title="Chat with Database", page_icon=":speech_balloon:")
        st.title("Chat with Database")
        self.sidebar()
        self.display_chat_history()
        self.handle_user_query()