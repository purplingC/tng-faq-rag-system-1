"""This file contains the refusal messages shown when a request is blocked."""

from __future__ import annotations

REFUSAL_MESSAGES = {
    "prompt_injection": (
        "I can only answer questions using the official Touch 'n Go eWallet FAQ, and I "
        "cannot change those instructions or reveal how I am configured. Ask me about "
        "an eWallet feature and I will help."
    ),
    "pii_request": (
        "I cannot access or share anyone's personal information, account details or "
        "transaction history. For account-specific help, please contact TNG Digital "
        "support directly through the eWallet app."
    ),
    "credential_request": (
        "I will never ask for or reveal passwords, PINs, OTPs or TACs, and neither will "
        "TNG Digital staff. If you have lost access to your account, use the in-app "
        "recovery options or contact official support."
    ),
    "payment_data": (
        "Please do not share card numbers, CVV codes or other payment details here. I "
        "cannot process or store them. Use the official TNG eWallet app for anything "
        "involving your card."
    ),
    "illicit_activity": (
        "I cannot help with that. I am the TNG eWallet FAQ assistant and I can only "
        "answer questions about using the eWallet safely and legitimately."
    ),
    "self_harm": (
        "I am sorry you are going through this, and I am not the right kind of help. "
        "Please reach out to someone who can support you right now - in Malaysia you "
        "can call Talian Kasih on 15999, or Befrienders KL on 03-7627 2929, which is "
        "open 24 hours. If you are in immediate danger, please call 999."
    ),
    "": (
        "I cannot help with that request. I can answer questions about the Touch 'n Go "
        "eWallet using the official FAQ."
    ),
}


def refusal_message(category: str) -> str:
    return REFUSAL_MESSAGES.get(category, REFUSAL_MESSAGES[""])
