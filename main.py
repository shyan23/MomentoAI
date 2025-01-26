from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
import asyncio
from getitems import getReview
from Summarization import summarization
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import Database_connection

nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()
client = Database_connection()


router = APIRouter()


@router.get('/Food/{event_id}', response_model=List[str])
async def food(event_id: int) -> List[str]:
    try:
        response = client.from_("reviews").select("food_feedback").eq("event_id", event_id).execute()
        return [item["food_feedback"] for item in response.data if "food_feedback" in item]
    except Exception as e:
        print(f"Food error: {str(e)}")
        return []

@router.get('/Accomodation/{event_id}', response_model=List[str])
async def accomodation(event_id: int) -> List[str]:
    try:
        return await getReview(event_id, "accommodation_feedback")
    except Exception as e:
        print(f"Accommodation error: {str(e)}")
        return []

@router.get('/Management/{event_id}', response_model=List[str])
async def management(event_id: int) -> List[str]:
    try:
        return await getReview(event_id, "management_feedback")
    except Exception as e:
        print(f"Management error: {str(e)}")
        return []

@router.get('/AppUsage/{event_id}', response_model=List[str])
async def app_usage(event_id: int) -> List[str]:
    try:
        return await getReview(event_id, "app_usage_feedback")
    except Exception as e:
        print(f"App usage error: {str(e)}")
        return []

@router.get('/summarization/{event_id}', response_model=Dict[str, Any])
async def Summarize(event_id: int) -> Dict[str, Any]:
    try:
        
        food_rev, accom_rev, mgmt_rev, app_rev = await asyncio.gather(
            food(event_id),
            accomodation(event_id),
            management(event_id),
            app_usage(event_id)
        )

        
        food_sum, accom_sum, mgmt_sum, app_sum = await asyncio.gather(
            summarization(food_rev, nlp, sia),
            summarization(accom_rev, nlp, sia),
            summarization(mgmt_rev, nlp, sia),
            summarization(app_rev, nlp, sia)
        )

        return {
            "event_id": event_id,
            "summaries": {
                "food": food_sum,
                "accommodation": accom_sum,
                "management": mgmt_sum,
                "app_usage": app_sum
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")