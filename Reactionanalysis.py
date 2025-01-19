
import pandas as pd
from config import Database_connection

client = Database_connection()

response = client.table('reviews').select("*").execute()

Data = (response.data)

df= pd.DataFrame(Data)

print(df)


# TODO : Preprocess The Data



