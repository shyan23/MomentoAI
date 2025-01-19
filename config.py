from dotenv import load_dotenv
import os
from supabase import Client,create_client

load_dotenv()

def Database_connection() :   
    secret_key = os.getenv("API_KEY")
    database_url = os.getenv("DATABASE_URL")
    supabase:Client = create_client(supabase_url=database_url,supabase_key=secret_key)
    return supabase