#!/usr/bin/env python3
"""
A simple multi‑turn chatbot that reads a predefined list of user messages,
sends each turn to the Cerebras chat completion API, and prints the assistant
responses. No interactive input is required.
"""

import json
import os
import sys
import time
from typing import List, Dict

import requests

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
API_ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"
API_KEY_ENV = "CEREBRAS_API_KEY"

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def get_api_key() -> str:
    """Retrieve the Cerebras API key from environment variables."""
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        print(f"Error: environment variable '{API_KEY_ENV}' is not set.", file=sys.stderr)
        sys.exit(1)
    return api_key


def call_cerebras_chat(api_key: str,
                       messages: List[Dict[str, str]]) -> str:
    """
    Send a chat completion request to the Cerebras API.

    Parameters
    ----------
    api_key : str
        The bearer token for authentication.
    messages : list of dict
        The conversation history, each dict contains 'role' and 'content'.

    Returns
    -------
    str
        The assistant's reply content.

    Raises
    ------
    SystemExit
        If the request fails or the response format is unexpected.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        print(f"Network error while contacting Cerebras API: {e}", file=sys.stderr)
        sys.exit(1)

    if response.status_code != 200:
        print(f"API returned non‑200 status code {response.status_code}: {response.text}",
              file=sys.stderr)
        sys.exit(1)

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}\nResponse text: {response.text}",
              file=sys.stderr)
        sys.exit(1)

    # Expected shape: {"choices": [{"message": {"role": "...", "content": "..."}}, ...]}
    try:
        assistant_message = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"Unexpected response format: {e}\nFull response: {json.dumps(data, indent=2)}",
              file=sys.stderr)
        sys.exit(1)

    return assistant_message.strip()


def main() -> None:
    api_key = get_api_key()

    # Initial system prompt (optional but good practice)
    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a helpful, concise assistant."}
    ]

    # Predefined user messages for the multi‑turn conversation
    user_messages = [
        "Hello! How are you today?",
        "Can you tell me a short joke?",
        "What is the capital city of France?",
        "Explain the concept of recursion in programming in two sentences."
    ]

    for user_input in user_messages:
        # Append the user's message to the history
        conversation.append({"role": "user", "content": user_input})

        # Call the API and get assistant's response
        assistant_reply = call_cerebras_chat(api_key, conversation)

        # Append the assistant's reply to preserve context for next turn
        conversation.append({"role": "assistant", "content": assistant_reply})

        # Output the turn
        print(f"User: {user_input}")
        print(f"Assistant: {assistant_reply}\n")

        # Optional: small pause to be polite to the API (remove if speed is critical)
        time.sleep(0.5)


if __name__ == "__main__":
    main()