# seller_memory_service.py

import os
import json
from datetime import datetime
from supabase import create_client, Client

# === Load environment ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Allowed Supabase columns ===
ALLOWED_FIELDS = {
    "phone_number", "property_address", "conversation_log", "call_summary",
    "last_updated", "follow_up_reason", "follow_up_set_by", "asking_price",
    "estimated_arv", "repair_cost", "condition_notes", "strategy_flags", "offer_history",
    "verbal_offer_amount", "min_offer_amount", "max_offer_amount", "bedrooms",
    "bathrooms", "square_footage", "year_built", "lead_source", "is_deal",
    "summary_history", "conversation_stage", "next_follow_up_date", "lead_status",
    "motivation_score", "personality_tag", "timeline_to_sell", "contradiction_flags",
    "lead_score", "valuation_range_low", "valuation_range_high", "price_per_sqft",
    "estimated_rent", "cap_rate", "arv_source"
}

def get_seller_memory(phone_number: str):
    """
    Retrieves seller memory by phone number from the 'seller_memory' table.
    Returns a dictionary or None if not found.
    """
    try:
        response = supabase.table("seller_memory").select("*").eq("phone_number", phone_number).limit(1).execute()
        if response.data:
            return response.data[0]
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
        # Ensure list-type fields are properly formatted as lists, not strings
for key in ["conversation_log", "summary_history", "offer_history", "strategy_flags", "contradiction_flags"]:
    if key in updates:
        # If it's a string that looks like JSON, parse it back to list
        if isinstance(updates[key], str):
            try:
                updates[key] = json.loads(updates[key])
            except json.JSONDecodeError:
                updates[key] = []
        # If it's not a list or dict at this point, make it an empty list
        elif not isinstance(updates[key], (list, dict)):
            updates[key] = []

        # Always update the last_updated timestamp
        updates["last_updated"] = datetime.utcnow().isoformat()

        # Ensure phone number is included for upsert
        updates["phone_number"] = phone_number

        # === DEBUG: Print the payload ===
        print("\n📤 Attempting Supabase Upsert with payload:")
        print(json.dumps(updates, indent=2, default=str))

        # Perform upsert
        response = supabase.table("seller_memory").upsert(updates, on_conflict="phone_number").execute()

        # Debug response
        if hasattr(response, "error") and response.error:
            print("❌ Supabase Upsert Error:")
            print(response.error)
            return False
        elif hasattr(response, "status_code") and response.status_code >= 400:
            print(f"❌ Supabase HTTP {response.status_code} Error:")
            print(response)
            return False
        else:
            print("✅ Supabase update successful.")
            return True
    except Exception as e:
        print("🚨 Exception during Supabase update:")
        print(str(e))
        return False





