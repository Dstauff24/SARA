# seller_memory_service.py

import os
import json
from datetime import datetime
from supabase import create_client, Client

# Environment variables (make sure these are in your .env)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_seller_memory(phone_number: str):
    """
    Retrieves seller memory by phone number from the 'seller_memory' table.
    Returns a dictionary of all useful fields, or None if not found.
    """
    try:
        response = supabase.table("seller_memory").select("*").eq("phone_number", phone_number).limit(1).execute()
        if response.data:
            seller = response.data[0]
            return {
                "conversation_log": json.loads(seller.get("conversation_log", "[]")),
                "asking_price": seller.get("asking_price"),
                "estimated_arv": seller.get("estimated_arv"),
                "repair_cost": seller.get("repair_cost"),
                "property_address": seller.get("property_address"),
                "condition_notes": seller.get("condition_notes"),
                "offer_history": seller.get("offer_history", []),
                "strategy_flags": seller.get("strategy_flags", []),
                "tags": seller.get("tags", []),
                "disposition_status": seller.get("disposition_status"),
                "follow_up_date": seller.get("follow_up_date"),
                "follow_up_reason": seller.get("follow_up_reason"),
                "follow_up_set_by": seller.get("follow_up_set_by"),
                "call_summary": seller.get("call_summary"),
                "summary_history": seller.get("summary_history", []),
                "intent_level": seller.get("intent_level"),
                "is_deal": seller.get("is_deal"),
                "last_offer_amount": seller.get("last_offer_amount"),
                "bedrooms": seller.get("bedrooms"),
                "bathrooms": seller.get("bathrooms"),
                "square_footage": seller.get("square_footage"),
                "year_built": seller.get("year_built"),
                "lead_source": seller.get("lead_source")
            }
        return None
    except Exception as e:
        print(f"[Supabase Error] Failed to retrieve seller memory: {e}")
        return None

def update_seller_memory(phone_number: str, updates: dict):
    """
    Updates or inserts seller memory fields in the 'seller_memory' table.
    Accepts a dictionary of fields to update.
    """
    try:
        # Convert conversation_log to JSON string if necessary
        if "conversation_log" in updates and isinstance(updates["conversation_log"], list):
            updates["conversation_log"] = json.dumps(updates["conversation_log"])

        if "summary_history" in updates and isinstance(updates["summary_history"], list):
            updates["summary_history"] = json.dumps(updates["summary_history"])

        # Always update the last_updated timestamp
        updates["last_updated"] = datetime.utcnow().isoformat()

        # Ensure phone number is included for upsert
        updates["phone_number"] = phone_number

        # Perform upsert with conflict resolution on phone_number
        response = supabase.table("seller_memory").upsert(updates, on_conflict="phone_number").execute()
        return response
    except Exception as e:
        print(f"[Supabase Error] Failed to update seller memory: {e}")
        return None
