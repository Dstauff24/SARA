# seller_memory_service.py

import os
import json
from datetime import datetime
from supabase import create_client, Client

# Load environment variables
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
                "phone_number": seller.get("phone_number"),
                "property_address": seller.get("property_address"),
                "conversation_log": json.loads(seller.get("conversation_log", "[]")),
                "call_summary": seller.get("call_summary"),
                "last_updated": seller.get("last_updated"),
                "follow_up_reason": seller.get("follow_up_reason"),
                "follow_up_set_by": seller.get("follow_up_set_by"),
                "asking_price": seller.get("asking_price"),
                "estimated_arv": seller.get("estimated_arv"),
                "repair_cost": seller.get("repair_cost"),
                "condition_notes": seller.get("condition_notes"),
                "strategy_flags": seller.get("strategy_flags", []),
                "offer_history": seller.get("offer_history", []),
                "verbal_offer_amount": seller.get("verbal_offer_amount"),
                "min_offer_amount": seller.get("min_offer_amount"),
                "max_offer_amount": seller.get("max_offer_amount"),
                "bedrooms": seller.get("bedrooms"),
                "bathrooms": seller.get("bathrooms"),
                "square_footage": seller.get("square_footage"),
                "year_built": seller.get("year_built"),
                "lead_source": seller.get("lead_source"),
                "is_deal": seller.get("is_deal"),
                "summary_history": json.loads(seller.get("summary_history", "[]")),
                "conversation_stage": seller.get("conversation_stage"),
                "next_follow_up_date": seller.get("next_follow_up_date"),
                "lead_status": seller.get("lead_status"),
                "motivation_score": seller.get("motivation_score"),
                "personality_tag": seller.get("personality_tag"),
                "timeline_to_sell": seller.get("timeline_to_sell"),
                "contradiction_flags": seller.get("contradiction_flags", []),
                "lead_score": seller.get("lead_score"),
                "valuation_range_low": seller.get("valuation_range_low"),
                "valuation_range_high": seller.get("valuation_range_high"),
                "price_per_sqft": seller.get("price_per_sqft"),
                "estimated_rent": seller.get("estimated_rent"),
                "cap_rate": seller.get("cap_rate"),
                "arv_source": seller.get("arv_source")
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
        # Convert to JSON strings where needed
        if "conversation_log" in updates and isinstance(updates["conversation_log"], list):
            updates["conversation_log"] = json.dumps(updates["conversation_log"])

        if "summary_history" in updates and isinstance(updates["summary_history"], list):
            updates["summary_history"] = json.dumps(updates["summary_history"])

        if "strategy_flags" in updates and isinstance(updates["strategy_flags"], list):
            updates["strategy_flags"] = json.dumps(updates["strategy_flags"])

        if "offer_history" in updates and isinstance(updates["offer_history"], list):
            updates["offer_history"] = json.dumps(updates["offer_history"])

        if "contradiction_flags" in updates and isinstance(updates["contradiction_flags"], list):
            updates["contradiction_flags"] = json.dumps(updates["contradiction_flags"])

        # Always update last_updated timestamp
        updates["last_updated"] = datetime.utcnow().isoformat()

        # Ensure phone number is included
        updates["phone_number"] = phone_number

        # === DEBUG: Log payload ===
        print("\n📤 Attempting Supabase Upsert...")
        print(json.dumps(updates, indent=2, default=str))

        # Perform upsert
        response = supabase.table("seller_memory").upsert(updates, on_conflict="phone_number").execute()

        if response.error:
            print("\n❌ Supabase returned an error:")
            print(response.error)
        else:
            print("✅ Supabase update successful.")
            print(f"🔁 Response: {response.data}")

        return response
    except Exception as e:
        print("\n🚨 Exception during Supabase update")
        print(str(e))
        return None





