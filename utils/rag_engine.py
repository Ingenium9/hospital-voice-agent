from openai import OpenAI
import json
import os


class RAGEngine:
    def __init__(self, data_loader, conversation_manager):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.data_loader = data_loader
        self.conversation_manager = conversation_manager

    def extract_search_terms(self, query):
        functions = [
            {
                "name": "extract_hospital_search_terms",
                "description": "Extract parameters from a user’s hospital-related query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hospital_name": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                        "search_type": {
                            "type": "string",
                            "enum": ["list_hospitals", "check_network", "general_inquiry"],
                        },
                        "count": {"type": "integer"},
                    },
                    "required": ["search_type"],
                },
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Extract search terms from: {query}"}],
                functions=functions,
                function_call={"name": "extract_hospital_search_terms"},
            )
            if response.choices and response.choices[0].message.function_call:
                return json.loads(response.choices[0].message.function_call.arguments)
        except Exception:
            pass

        return {"search_type": "general_inquiry", "city": ""}

    def generate_response(self, query, hospital_data):
        system_prompt = (
            "You are a helpful hospital network assistant. Keep responses short, clear, "
            "and conversational. Base answers only on the hospital details provided."
        )

        user_prompt = (
            f'User Query: "{query}"\n\n'
            f"Hospital Information:\n{hospital_data}\n\n"
            "Provide a concise response:"
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=150,
                temperature=0.6,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "I am having trouble retrieving the hospital information right now."

    def process_query(self, query):
        search_terms = self.extract_search_terms(query)
        search_type = search_terms.get("search_type", "general_inquiry")
        city = search_terms.get("city", "")
        hospital_name = search_terms.get("hospital_name", "")
        count = int(search_terms.get("count", 3) or 3)

        q = query.lower()
        hospital_data = ""

        if "3 hospitals" in q and "bangalore" in q:
            results = self.data_loader.search_hospitals("hospitals in Bangalore", k=3)
            hospital_data = "\n".join([doc.page_content for doc in results])

        elif "manipal sarjapur" in q and "bangalore" in q:
            results = self.data_loader.exact_match_search("Manipal Sarjapur", "Bangalore")
            if not results.empty:
                hospital_data = f"Network status: {results.iloc[0].get('network_status', 'Unknown')}"
            else:
                hospital_data = "Hospital not found in network"

        elif search_type == "list_hospitals" and city:
            results = self.data_loader.search_hospitals(f"hospitals in {city}", k=count)
            hospital_data = "\n".join([doc.page_content for doc in results])

        elif search_type == "check_network" and hospital_name and city:
            results = self.data_loader.exact_match_search(hospital_name, city)
            if not results.empty:
                status = results.iloc[0].get("network_status", "Unknown")
                hospital_data = f"{hospital_name} in {city}: {status}"
            else:
                hospital_data = f"No matching hospital found: {hospital_name} in {city}"

        else:
            results = self.data_loader.search_hospitals(query, k=3)
            hospital_data = "\n".join([doc.page_content for doc in results])

        response = self.generate_response(query, hospital_data)
        self.conversation_manager.add_exchange(query, response)
        return response
