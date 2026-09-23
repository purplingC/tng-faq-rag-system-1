"""This file contains the blocking rules, see docs/guardrails.md for the design."""

from __future__ import annotations
from ..models import Action, Rule
from .normalization import _rx

_ASK_VERB = r"""(?:show|reveal|print|display|output|repeat|recite|echo|expose|leak|give|tell|
              share|dump|list|write|return|translate|summari[sz]e|paraphrase|disclose|state)"""


_INSTRUCTION_NOUN = r"""(?:system\s+)?(?:prompt|prompts|instruction|instructions|rule|rules|
                     directive|directives|guideline|guidelines|guardrail|guardrails|
                     configuration|persona|context\s+window|sources?\s+block)"""


_CREDENTIAL = r"""(?:password|passcode|pass\s?word|pin(?:\s?number|\s?code)?|otp|one[\s-]?time
                \s?(?:password|code)|tac(?:\s?number|\s?code)?|cvv|cvc|security\s?code|
                secret\s?key|api\s?key|access\s?token|private\s?key|seed\s?phrase|
                recovery\s?(?:code|phrase)|mpin)"""


_CRED_BENIGN = _rx(r"""\b(reset|resetting|change|changing|update|updating|forgot|forgotten|
                        recover|recovering|retrieve\s+my\s+own|set\s?up|setting\s?up|create|
                        creating|enable|disable|secure|protect|lock|unlock|expired?|
                        wrong|incorrect|not\s+working|didn't\s+receive|did\s+not\s+receive|
                        never\s+received|request\s+a\s+new|resend)\b
                        | ^\s*(?:what|what's|apakah|apa)\s*(?:is|are|itu|maksud)?\s*
                          (?:a|an|the)?\s*["\u2018\u2019\u201c\u201d']?\s*
                          (?:tng\s+|ewallet\s+|tng\s+ewallet\s+)?
                          (?:reload\s+pin|strong\s+6-digit\s+pin|6-digit\s+pin|pin|otp|tac)\b""")

# A question about a jailbroken or rooted phone is a real support topic, not a jailbreak attempt
_JAILBREAK_BENIGN = _rx(r"""\b(?:di-?)?jailbreak(?:ed|en)?\b[^.?!]{0,30}
                            \b(device|devices|phone|iphone|android|peranti|telefon|aplikasi|app)\b
                          | \b(device|devices|phone|iphone|android|peranti|telefon|aplikasi|app)\b
                            [^.?!]{0,30}\b(?:di-?)?jailbreak(?:ed|en)?\b
                          | \brooted?\b[^.?!]{0,25}\b(device|phone|peranti|telefon)\b""")


INPUT_RULES: list[Rule] = [
    # Prompt injection
    Rule(
        "override_instructions",
        "prompt_injection",
        0.95,
        on_normalized=True,
        reason="attempt to override system instructions",
        pattern=_rx(r"""\b(ignore|disregard|forget|discard|drop|override|bypass|circumvent|
                       skip|abandon)\b[^.?!]{0,45}\b(previous|prior|above|earlier|preceding|
                       initial|original|all|any|your|the|these|those)\b[^.?!]{0,45}
                       \b(instruction|instructions|prompt|prompts|rule|rules|guideline|
                       guidelines|restriction|restrictions|constraint|constraints|
                       polic\w+|guardrail\w*|direction\w*|training|context)\b
                       | \b(ignore|disregard|forget)\s+(?:the\s+|all\s+)?
                         (?:previous|prior|above|preceding|earlier|foregoing)\b
                       | \bignore\s*all\s*previous\b"""),
    ),
    # De-spaced forms, so the missing word boundaries below are deliberate
    Rule(
        "obfuscated_injection",
        "prompt_injection",
        0.9,
        on_normalized=True,
        exempt=_JAILBREAK_BENIGN,
        reason="obfuscated instruction-override or prompt-extraction attempt",
        pattern=_rx(r"""ignore(?:all|the|any)?(?:previous|prior|above|earlier|preceding)
                       | disregard(?:all|the|any)?(?:previous|prior|above|earlier)
                       | forget(?:all|the|any)?(?:previous|prior|above|earlier)
                       | (?:reveal|show|print|output|repeat|tell|give)(?:me)?(?:your|the)?
                         (?:system)?(?:prompt|instruction)
                       | youarenow | doanythingnow | developermode | jailbreak
                       | withoutanyrestriction | norestriction"""),
    ),
    Rule(
        "exfiltrate_prompt",
        "prompt_injection",
        0.95,
        on_normalized=True,
        reason="attempt to extract the system prompt or internal instructions",
        pattern=_rx(rf"""\b{_ASK_VERB}\b[^.?!]{{0,60}}\b(your|the|his|its|system|initial|
                       original|hidden|secret|internal|underlying|full)\b[^.?!]{{0,30}}
                       \b{_INSTRUCTION_NOUN}\b
                       | \bwhat\s+(?:were|was|are|is)\s+(?:you|your)\b[^.?!]{{0,40}}
                         \b(told|instructed|given|programmed|configured|set\s?up|
                         {_INSTRUCTION_NOUN})\b
                       | \b(?:your|the)\s+{_INSTRUCTION_NOUN}\b[^.?!]{{0,25}}
                         \b(?:say|says|contain|contains|is|are)\b
                       | \bat\s+the\s+(?:start|beginning|top)\s+of\s+
                         (?:this|the|our)\s+(?:conversation|chat|prompt|session)\b"""),
    ),
    Rule(
        "repeat_context",
        "prompt_injection",
        0.9,
        on_normalized=True,
        reason="attempt to make the model dump its context verbatim",
        pattern=_rx(r"""\b(repeat|echo|print|output|copy|reproduce|render|show)\b
                       [^.?!]{0,45}\b(everything|all\s+(?:of\s+)?(?:the\s+)?
                       (?:text|words|content|above)|verbatim|word[\s-]for[\s-]word|
                       character\s+for\s+character|exactly\s+as\s+(?:written|given))\b
                       | \b(print|show|output|display|give)\b[^.?!]{0,35}\b(?:the\s+)?
                         (?:text|content|lines?|block)\b[^.?!]{0,25}
                         \b(between|before|above|preceding|prior\s+to)\b"""),
    ),
    Rule(
        "persona_jailbreak",
        "prompt_injection",
        0.92,
        on_normalized=True,
        exempt=_JAILBREAK_BENIGN,
        reason="jailbreak persona or unrestricted-mode request",
        pattern=_rx(r"""\b(do\s+anything\s+now|developer\s+mode|jailbreak\w*|god\s?mode|
                       sudo\s+mode|root\s+mode|unfiltered|uncensored|opposite\s+day|
                       evil\s+(?:mode|assistant)|no\s+longer\s+bound|
                       without\s+(?:any\s+)?(?:restriction|restrictions|filter|filters|
                       rule|rules|limit|limits|guardrail|guardrails|censorship)|
                       with\s+no\s+(?:restriction|restrictions|filter|filters|rules?))\b
                       | \b(?:you\s+are|you're)\s+(?:now|no\s+longer)\b
                       | \bfrom\s+now\s+on,?\s+you\b
                       | \b(?:act\s+as|pretend\s+(?:to\s+be|that\s+you|you\s+are)|
                         role[\s-]?play(?:ing)?\s+as|simulate\s+(?:being|that\s+you))\b
                       | \byou\s+are\s+dan\b | \bhi\s+dan\b"""),
    ),
    Rule(
        "fake_authority",
        "prompt_injection",
        0.85,
        on_normalized=True,
        reason="false claim of privileged access used to unlock behaviour",
        pattern=_rx(r"""\b(?:as|i\s+am|i'm|this\s+is)\s+(?:an?|the)\s+
                       (?:admin|administrator|developer|engineer|operator|auditor|
                       tester|owner|supervisor|staff\s+member)\b[^.?!]{0,60}
                       \b(debug|debugging|override|unlock|disable|bypass|internal|
                       output|reveal|show|test\s+mode|maintenance)\b
                       | \b(?:debug|maintenance|test|developer)\s+mode\s+
                         (?:on|enabled|activated)\b"""),
    ),
    Rule(
        "delimiter_injection",
        "prompt_injection",
        0.9,
        reason="control-token or delimiter injection",
        pattern=_rx(r"""(<\|?\s*(?:im_start|im_end|endoftext|system|assistant|user)\s*\|?>)
                       | (\[/?\s*INST\s*\])
                       | (\#{2,}\s*(?:system|instruction|new\s+instruction))
                       | (\bBEGIN\s+(?:SYSTEM|PROMPT|INSTRUCTION))
                       | (</?\s*(?:system|instructions?)\s*>)"""),
    ),
    Rule(
        "encoded_payload",
        "prompt_injection",
        0.8,
        reason="instruction hidden in an encoded payload",
        pattern=_rx(r"""\b(decode|base64|b64|rot13|hex\s?decode|from\s?hex|reverse\s+this)\b
                       [^.?!]{0,40}\b(and|then|to)\b[^.?!]{0,40}
                       \b(execute|run|follow|obey|do|answer|reply|output)\b"""),
    ),
    # Third party data and PII
    Rule(
        "third_party_pii",
        "pii_request",
        0.95,
        reason="request for another person's personal data",
        pattern=_rx(r"""\b(reveal|show|tell|give|list|provide|find|look\s?up|share|
                       what\s+is|who\s+is|get\s+me)\b[^.?!]{0,50}
                       \b(?:other|another|someone\s+else'?s?|the\s+last|previous|last|
                       every|all)\s+(?:user|customer|person|caller|client|account\s?holder)
                       (?:'s|s')?\b
                       | \b(?:customer|user|another\s+person|someone\s+else|other\s+people)
                         (?:'s|s')\s+(?:email|phone|address|name|ic|nric|mykad|account|
                         balance|transaction|password|pin|otp|details|data|information)\b
                       | \b(list|dump|export|show)\s+(?:me\s+)?(?:all\s+)?
                         (?:the\s+)?(?:users?|customers?|accounts?|emails?|phone\s+numbers?)
                         \b[^.?!]{0,25}\b(database|system|records?|table)\b"""),
    ),
    Rule(
        "credential_request",
        "credential_request",
        0.95,
        exempt=_CRED_BENIGN,
        reason="request to disclose a secret credential",
        pattern=_rx(rf"""\b(what\s+(?:is|are|was)|tell\s+me|give\s+me|show\s+me|reveal|
                        send\s+me|share|provide|type\s+out|repeat)\b[^.?!]{{0,35}}
                        \b(?:my|your|the|his|her|their|a|an|users?'?)?\s*{_CREDENTIAL}\b"""),
    ),
    Rule(
        "payment_data",
        "payment_data",
        1.0,
        reason="message contains or requests raw payment card data",
        pattern=_rx(r"""\b(?:cvv|cvc|card\s?verification\s?(?:value|code))\b[^.?!]{0,30}
                       \b(?:is|was|=|:|\d)
                       | \b(?:my|the)\s+(?:full\s+)?card\s+number\s+is\b
                       | \b(what\s+is|tell\s+me|give\s+me|show\s+me|reveal)\b[^.?!]{0,30}
                         \b(?:my|the|his|her|their)\s+(?:full\s+)?
                         (?:card|pan|credit\s?card|debit\s?card)\s+number\b"""),
    ),
    # Illicit activity
    Rule(
        "illicit_howto",
        "illicit_activity",
        0.95,
        reason="request for instructions to commit an offence",
        exempt=_rx(r"""\b(report|reporting|victim|protect|prevent|avoid|recognise|recognize|
                        spot|was|were|got|been|suspect|suspicious|scammed|happened)\b"""),
        pattern=_rx(r"""\b(how\s+(?:do|can|could|would)\s+(?:i|we|you|one)|how\s+to|
                       teach\s+me|show\s+me\s+how|help\s+me|ways?\s+to|steps?\s+to|
                       guide\s+(?:me\s+)?(?:to|on)|best\s+way\s+to|method\s+to)\b
                       [^.?!]{0,70}\b(hack|hacking|crack|cracking|exploit|exploiting|
                       bypass|circumvent|defraud|scam|scamming|phish|phishing|launder|
                       laundering|steal|stealing|counterfeit|forge|forging|clone|cloning|
                       brute[\s-]?force|sim[\s-]?swap|skim|skimming)\b"""),
    ),
    Rule(
        "illicit_target",
        "illicit_activity",
        0.9,
        reason="request to attack or compromise a system or account",
        exempt=_rx(r"""\b(my\s+account\s+(?:was|has\s+been|got)|i\s+(?:was|got|have\s+been)|
                        report|victim|suspicious|prevent|protect|recover)\b"""),
        pattern=_rx(r"""\b(hack|breach|compromise|break\s+into|gain\s+access\s+to|
                       take\s+over|steal\s+from|drain|siphon)\b[^.?!]{0,45}
                       \b(?:someone|another|other|other\s+people'?s?|a\s+user'?s?|
                       an?\s+account|accounts|the\s+system|your\s+system|the\s+database|
                       the\s+server|their)\b"""),
    ),
    Rule(
        "weapons_violence",
        "illicit_activity",
        0.95,
        reason="request relating to weapons, explosives or violence against a person",
        pattern=_rx(r"""\b(make|build|construct|assemble|buy|obtain|get)\b[^.?!]{0,35}
                       \b(bomb|explosive|ied|firearm|gun|silencer|nerve\s+agent|ricin)\b
                       | \bhow\s+to\s+(?:kill|murder|poison|assault|stab|shoot)\s+
                         (?:a\s+)?(?:someone|person|people|him|her|them|my)\b"""),
    ),
    # Self harm, which is refused with care rather than a bare block
    Rule(
        "self_harm",
        "self_harm",
        0.9,
        action=Action.SAFE_COMPLETE,
        reason="message indicates possible self-harm",
        pattern=_rx(r"""\b(kill\s+myself|killing\s+myself|end\s+my\s+life|
                       take\s+my\s+own\s+life|commit\s+suicide|suicidal|
                       want\s+to\s+die|self[\s-]harm|hurt\s+myself)\b"""),
    ),
]


# Everyday support questions that must never be blocked, checked before soft rules

# Malay wording for the same attacks: the patterns above are English and match none of it
INPUT_RULES += [
    Rule(
        "override_instructions_ms",
        "prompt_injection",
        0.95,
        on_normalized=True,
        reason="attempt to override system instructions, in Malay",
        pattern=_rx(r"""\b(abaikan|lupakan|ketepikan|langkau|buang)\b[^.?!]{0,45}
                       \b(arahan|peraturan|panduan|prompt|sekatan|polisi|had)\b"""),
    ),
    Rule(
        "exfiltrate_prompt_ms",
        "prompt_injection",
        0.95,
        on_normalized=True,
        reason="attempt to extract the system prompt, in Malay",
        pattern=_rx(r"""\b(tunjukkan|tunjuk|paparkan|papar|beritahu|dedahkan|cetak|ulangi|
                       kongsikan|bocorkan)\b[^.?!]{0,40}
                       \b(prompt|arahan\s+sistem|peraturan\s+anda|rahsia\s+sistem|
                       arahan\s+asal|arahan\s+tersembunyi)\b
                       | \bprompt\s+sistem\b"""),
    ),
    Rule(
        "illicit_activity_ms",
        "illicit_activity",
        0.9,
        on_normalized=True,
        reason="request for illicit activity, in Malay",
        pattern=_rx(r"""\b(godam|menggodam|meretas|retas)\b
                       | \b(curi|mencuri|memalsukan|palsukan|menyamar\s+sebagai)\b
                         [^.?!]{0,30}\b(akaun|wang|duit|data|maklumat|identiti|kad)\b"""),
    ),
    Rule(
        "third_party_pii_ms",
        "pii_request",
        0.9,
        on_normalized=True,
        reason="request for another person's data, in Malay",
        pattern=_rx(r"""\b(pengguna|orang|pelanggan|akaun)\s+lain\b[^.?!]{0,40}
                       \b(maklumat|data|butiran|transaksi|sejarah|baki|nombor|alamat)\b
                       | \b(maklumat|data|butiran|transaksi|sejarah|baki)\b[^.?!]{0,30}
                         \b(pengguna|orang|pelanggan)\s+lain\b"""),
    ),
]

BENIGN_CONTEXT = _rx(r"""
    \b(how\s+do\s+i\s+(?:update|change|verify|reset|report|renew|link|unlink|check|enable))\b
  | \b(?:report(?:ing)?|reported)\s+(?:an?\s+)?(?:unauthorised|unauthorized|fraudulent|
      suspicious|scam)\b
  | \bmy\s+(?:account|card|wallet|phone)\s+(?:was|has\s+been|got|is)\s+
      (?:hacked|compromised|stolen|lost|blocked|suspended)\b
  | \b(?:force\s+)?(?:kill|close|quit|restart|reinstall|clear)\s+(?:the\s+)?
      (?:app|application|process|cache|session)\b
  | \bis\s+it\s+(?:legal|illegal|safe|allowed|permitted)\s+to\b
""")
