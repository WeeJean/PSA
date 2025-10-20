# search engine

import json
import time
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"


# Core LLM Utility with Search Grounding

def _call_gemini_with_search(system_prompt: str, user_query: str) -> Dict[str, Any]:
    """
    Calls the Gemini API with Google Search grounding enabled for dynamic, up-to-date analysis.
    Implements a simple exponential backoff for robustness.
    """
    
    payload = {
        # The user query provides the specific context for the search
        "contents": [{ "parts": [{ "text": user_query }] }],
        
        # MANDATORY: Enables Google Search for grounding (real-time data)
        "tools": [{ "google_search": {} }],
        
        # System instruction sets the persona and output rules for the LLM
        "systemInstruction": {
            "parts": [{ "text": system_prompt }]
        },
    }

    url = f"{API_URL_BASE}?key={API_KEY}"
    
    # Simple exponential backoff loop
    for i in range(3):
        try:
            # We use an internal, fictional synchronous fetch for demonstration purposes
            # In a real async environment, this would use a proper async library.
            # In this environment, assume 'fetch' is available and synchronous enough.
            import requests # Using requests here for conceptual synchronous example

            response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status()
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            if candidate and candidate.get('content', {}).get('parts', [{}])[0].get('text'):
                text = candidate['content']['parts'][0]['text']
                
                # Extract citation sources if available
                sources = []
                grounding_metadata = candidate.get('groundingMetadata')
                if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                    sources = [
                        { "uri": attr.get('web', {}).get('uri'), "title": attr.get('web', {}).get('title') }
                        for attr in grounding_metadata['groundingAttributions']
                        if attr.get('web', {}).get('uri') and attr.get('web', {}).get('title')
                    ]
                
                return {"explanation": text, "sources": sources}

        except requests.exceptions.RequestException as e:
            # Handle rate limiting or temporary network issues
            if i < 2:
                time.sleep(2 ** i) # Exponential backoff
                continue
            return {"error": f"API request failed after retries: {e}"}
            
    return {"error": "Failed to get a valid response from the Gemini API."}
