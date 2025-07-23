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
                "phone_number": phone,
                "property_address": address or memory.get("property_address"),
                "conversation_log": conversation_memory["history"],
                "call_summary": summary,
                "summary_history": memory.get("summary_history", []),
                "last_updated": datetime.utcnow().isoformat(),
                "conversation_stage": memory.get("conversation_stage"),
                "next_follow_up_date": memory.get("next_follow_up_date"),
                "lead_status": memory.get("lead_status"),
                "follow_up_reason": memory.get("follow_up_reason"),
                "follow_up_set_by": memory.get("follow_up_set_by"),
                "asking_price": safe_number(asking_price or memory.get("asking_price")),
                "estimated_arv": safe_number(arv),
                "repair_cost": safe_number(repair_cost),
                "verbal_offer_amount": safe_number(verbal_offer),
                "min_offer_amount": safe_number(min_offer),
                "max_offer_amount": safe_number(max_offer),
                "bedrooms": memory.get("bedrooms"),
                "bathrooms": memory.get("bathrooms"),
                "square_footage": memory.get("square_footage"),
                "year_built": memory.get("year_built"),
                "motivation_score": memory.get("motivation_score"),
                "personality_tag": memory.get("personality_tag"),
                "timeline_to_sell": memory.get("timeline_to_sell"),
                "strategy_flags": memory.get("strategy_flags", []),
                "offer_history": memory.get("offer_history", []),
                "contradiction_flags": memory.get("contradiction_flags", []),
                "lead_score": score_lead(tone, intent),
                "valuation_range_low": safe_number(valuation_data.get("valuation_range_low")),
                "valuation_range_high": safe_number(valuation_data.get("valuation_range_high")),
                "price_per_sqft": safe_number(valuation_data.get("price_per_sqft")),
                "estimated_rent": safe_number(rent_data.get("estimated_rent")),
                "cap_rate": safe_number(rent_data.get("cap_rate")),
                "arv_source": valuation_data.get("arv_source", "manual" if data.get("estimated_arv") else None),
                "condition_notes": memory.get("condition_notes"),
                "lead_source": memory.get("lead_source"),
                "is_deal": memory.get("is_deal", False)
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





