import os
import pyodbc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseUtil:
    def __init__(self):
        self.db_host = os.getenv('DB_HOST')
        self.db_name = os.getenv('DB_NAME')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_driver = os.getenv('DB_DRIVER')
        self.connection = self.create_connection()

    def create_connection(self):
        try:
            connection = pyodbc.connect(
                f"DRIVER={self.db_driver};"
                f"SERVER={self.db_host};"
                f"DATABASE={self.db_name};"
                f"UID={self.db_user};"
                f"PWD={self.db_password}"
            )
            print("Database connection successful.")
            return connection
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None

    def execute_query(self, query):
        """Executes a given SQL query via the current database connection."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            self.connection.commit()
            print("Query executed successfully.")
        except Exception as e:
            print(f"Error executing query: {e}")

    def fetch_data(self, query, params=None):
        """Fetches data from the database using a given SQL query."""
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            result = cursor.fetchall()
            return result
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def insert_data(self, query, params, return_id=False):
        """Inserts data into the database using a parameterized query."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            self.connection.commit()
            print("Data inserted successfully.")
            if return_id:
                # Assuming the table has an identity column and you're using SQL Server
                id_query = "SELECT @@IDENTITY AS ID;"
                cursor.execute(id_query)
                row_id = cursor.fetchone()[0]
                return row_id
        except Exception as e:
            print(f"Error inserting data: {e}")
            return None



# Example usage
if __name__ == "__main__":
    db_util = DatabaseUtil()
    # Example query to fetch data
    result = db_util.fetch_data("SELECT * FROM recording")
    print(result)
