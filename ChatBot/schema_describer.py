from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

class SchemaDescriber:
    def __init__(self, gemini_api_key, output_file="schema_descriptions.txt"):
        """
        Initialize the SchemaDescriber.

        Parameters:
        - gemini_api_key: API key for Google Gemini API.
        - output_file: Path to the single output file for all schema descriptions.
        """
        self.gemini_api_key = gemini_api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=self.gemini_api_key)
        self.output_file = output_file
    
    def generate_description(self, schema_info):
        """
        Generate a concise description for the entire schema. The LLM will infer the table structure from the full schema.
        The description should include:
        - Columns, data types, and constraints (primary/foreign keys).
        - An example row for each table in the schema.
        - Keep it concise, yet informative, ensuring all necessary details are captured.
        
        Parameters:
        - schema_info: The raw schema information of the database, which includes all tables.

        Returns:
        - A string containing the concise description for the schema.
        """
        template = f"""
        You are given a database schema. For each table, provide only the following:
        - A list of columns with their data types.
        - Primary and foreign key constraints, if any.
        - One example row of data.

        Keep the description as brief as possible while including all critical details.

        Schema:
        {schema_info}

        Description:
        """ 
        prompt = ChatPromptTemplate.from_template(template)
        prompt_text = prompt.format(schema_info=schema_info)
        response = self.llm.invoke(prompt_text)
        return response.content.strip()
    
    def save_descriptions(self, descriptions):
        """
        Save all schema descriptions to a single file.

        Parameters:
        - descriptions: The full description of all tables.
        """
        with open(self.output_file, "a", encoding="utf-8") as file:
            file.write(descriptions)
            file.write("\n" + "-" * 80 + "\n")  # Separator for readability
        
        print(f"All descriptions saved to {self.output_file}")
    
    def load_existing_descriptions(self):
        """
        Load existing descriptions from the output file if it exists.

        Returns:
        - The content of the existing description file or an empty string if the file doesn't exist.
        """
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding="utf-8") as file:
                return file.read()
        return ""

    def describe_and_save_all(self, database_schema):
        """
        Generate and save descriptions for all tables in the database schema.
        This method will only generate new descriptions if the output file doesn't exist.

        Parameters:
        - database_schema: The complete schema information for all tables in the database.

        Returns:
        - The description of all tables in the schema.
        """
        # Check if the output file already exists and return its content if it does
        existing_descriptions = self.load_existing_descriptions()
        if existing_descriptions:
            print(f"Descriptions already exist in {self.output_file}. Returning existing descriptions.")
            return existing_descriptions

        # Generate the description for the full schema
        print("Generating descriptions for the entire schema...")
        description = self.generate_description(database_schema)

        # Save the description to a file
        self.save_descriptions(description)
        return description