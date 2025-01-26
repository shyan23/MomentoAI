from typing import List
from config import Database_connection

client = Database_connection()
from aiocache import cached, Cache
@cached(ttl=300, cache=Cache.MEMORY)

async def getReview(event_id: int, review_field: str) -> List[str]:
    client = Database_connection()
    try:
        # Fetch feedback for the given review_field and event_id from the Supabase table
        response = client.from_("reviews").select(review_field).eq("event_id", event_id).execute()
        
        # Access the 'data' attribute directly from the response object
        if response.data:
            # Dynamically access the review_field from the response
            review_data = [item.get(review_field) for item in response.data if review_field in item]
        
        return review_data
    except Exception as e:
        return {"error": str(e)}