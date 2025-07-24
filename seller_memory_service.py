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
    Updates the seller_memory table with filtered fields and debug logs.
    """
    try:
        # Serialize list fields
        if "conversation_log" in updates and isinstance(updates["conversation_log"], list):
            updates["conversation_log"] = json.dumps(updates["conversation_log"])
        if "summary_history" in updates and isinstance(updates["summary_history"], list):
            updates["summary_history"] = json.dumps(updates["summary_history"])
        if "offer_history" in updates and isinstance(updates["offer_history"], list):
            updates["offer_history"] = json.dumps(updates["offer_history"])
        if "strategy_flags" in updates and isinstance(updates["strategy_flags"], list):
            updates["strategy_flags"] = json.dumps(updates["strategy_flags"])

        # Add metadata
        updates["phone_number"] = phone_number
        updates["last_updated"] = datetime.utcnow().isoformat()

        # Only keep allowed fields
        filtered_updates = {k: v for k, v in updates.items() if k in ALLOWED_FIELDS}

        print("\n📤 Attempting Supabase Upsert...")
        print(json.dumps(filtered_updates, indent=2, default=str))

        response = supabase.table("seller_memory").upsert(filtered_updates, on_conflict="phone_number").execute()

        if response.error:
            print("\n❌ Supabase returned an error:")
            print(f"Message: {response.error.get('message')}")
            print(f"Code: {response.error.get('code')}")
            print(f"Details: {response.error.get('details')}")
            print(f"Hint: {response.error.get('hint')}")
            print("\n🔎 Tip: This is often caused by sending an unexpected or misspelled field not in your table schema.")
        else:
            print("✅ Supabase update successful.")
            print(f"🔁 Response: {response.data}")

        return response
    except Exception as e:
        print("\n🚨 Exception during Supabase update")
        print(str(e))
        return None





