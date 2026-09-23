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


REFUSAL_MESSAGES_MS = {
    "prompt_injection": (
        "Saya hanya boleh menjawab menggunakan FAQ rasmi Touch 'n Go eWallet, dan saya "
        "tidak boleh mengubah arahan itu atau mendedahkan tetapan saya. Tanya saya tentang "
        "ciri eWallet dan saya akan membantu."
    ),
    "pii_request": (
        "Saya tidak boleh mengakses atau berkongsi maklumat peribadi, butiran akaun atau "
        "sejarah transaksi sesiapa. Untuk bantuan berkaitan akaun, sila hubungi sokongan "
        "TNG Digital melalui aplikasi eWallet."
    ),
    "credential_request": (
        "Saya tidak akan sesekali meminta atau mendedahkan kata laluan, PIN, OTP atau TAC, "
        "dan kakitangan TNG Digital juga tidak akan berbuat demikian. Jika anda hilang akses "
        "kepada akaun anda, gunakan pilihan pemulihan dalam aplikasi atau hubungi sokongan rasmi."
    ),
    "payment_data": (
        "Sila jangan kongsikan nombor kad, kod CVV atau butiran pembayaran lain di sini. Saya "
        "tidak boleh memproses atau menyimpannya. Gunakan aplikasi rasmi TNG eWallet untuk "
        "sebarang perkara berkaitan kad anda."
    ),
    "illicit_activity": (
        "Saya tidak boleh membantu dengan perkara itu. Saya ialah pembantu FAQ TNG eWallet dan "
        "saya hanya boleh menjawab soalan tentang penggunaan eWallet secara selamat dan sah."
    ),
    "self_harm": (
        "Saya bersimpati dengan apa yang anda lalui, dan saya bukan bantuan yang sesuai. Sila "
        "hubungi seseorang yang boleh menyokong anda sekarang - di Malaysia anda boleh "
        "menghubungi Talian Kasih di 15999, atau Befrienders KL di 03-7627 2929, yang dibuka "
        "24 jam. Jika anda dalam bahaya segera, sila hubungi 999."
    ),
    "": (
        "Saya tidak boleh membantu dengan permintaan itu. Saya boleh menjawab soalan tentang "
        "Touch 'n Go eWallet menggunakan FAQ rasmi."
    ),
}


def refusal_message(category: str, language: str = "en") -> str:
    """The refusal for a blocked request, in the language the question was asked in."""
    messages = REFUSAL_MESSAGES_MS if language == "ms" else REFUSAL_MESSAGES
    return messages.get(category, messages[""])
