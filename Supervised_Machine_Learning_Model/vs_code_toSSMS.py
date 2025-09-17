from sqlalchemy import create_engine
from urllib.parse import quote_plus # Wasn't used
import pandas as pd
import pyodbc

# Configure connection details
server_name = 'localhost'
database_name = 'Email_Classify'
trusted_connection = 'yes' # or 'no' if using username and password

# Read csv into pandas DF
try:
    df = pd.read_csv('spam.csv', usecols= [0, 1], names= ['class', 'text'], header= 0) # To read the dataset
    print("CSV file 'spam.csv' loaded successfully")
except FileNotFoundError:
    print("Error: 'spam.csv' file path not found")
    exit() # # Exits the script if file is not found

# Build the connection string
conn_string = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={server_name};"
    f"DATABASE={database_name};"
    f"Trusted_Connection={trusted_connection}"
)
# Encode the connection string
quoted_conn_str = quote_plus(conn_string)
# Create the SQLAlchemy engine 
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quoted_conn_str}")

table_name = 'Spam' # Define table name #NOTE We don't have to create the Spam table in SSMS
# Establish the connection
try:
    # Use the to_sql() method to insert the DataFrame into the SQL table, 
    df.to_sql(table_name, con= engine, if_exists= 'replace', index= False) #NOTE 'if_exists= replace, relaces the existing table if there is one
    print(f"Data from '{'spam.csv'}' successfuly loaded into '{table_name}'.")

except pyodbc.Error as ex:
    sqlstate = ex.args[0]
    print(f"Database error: {sqlstate}")
    
finally: 
    if 'engine' in locals(): # Closing the database connection, this is best practice
        engine.dispose()
        print('Engine disposed and Connection closed')
        

#-------------------------------------------------------------------------------------------------
#NOTE No longer neccessary
# Trouble-shooting
# Create a SQLAlchemy Engine for SQL Server
# Construct the connection string using urllib.parse.quote_plus for proper encoding
# params = quote_plus(
#     f'DRIVER={{ODBC Driver 17 for SQL Server}};'  # Adjust driver version if needed
#     f'SERVER={server_name};'
#     f'DATABASE={database_name}'
#     f'UID={'SesethuMBango'};'
#     f'PWD={'0844bango'}'
# )

# # Create the SQLAlchemy engine
# engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# # Use the SQLAlchemy Engine with to_sql()
# table_name = 'Spam'

# try:
#     df.to_sql(table_name, con=engine, if_exists='replace', index=False)
#     print(f"DataFrame successfully written to {table_name} in SQL Server.")
# except Exception as e:
#     print(f"An error occurred: {e}")