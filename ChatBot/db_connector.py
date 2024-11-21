from langchain_community.utilities import SQLDatabase

class DatabaseConnector:
    """Handles the connection to the SQL database."""
    def __init__(self):
        self.db = None

    def connect(self, user: str, password: str, host: str, port: str, database: str):
        """Initialize database connection."""
        db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        self.db = SQLDatabase.from_uri(db_uri)
        return self.db
