"""
AURA Security & Guardrails Engine
Implements Criterion 8:
- Input sanitization & prompt injection / toxic language defense
- Adversarial persona classification
- Tool return type validation & Pydantic schema checking
- Hard budget ceiling spend verification
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

# =====================================================================
# Pydantic Schemas for Guardrail Decision & Output
# =====================================================================

class GuardrailCheckResult(BaseModel):
    is_safe: bool = Field(..., description="True if prompt passed all safety checks; False otherwise")
    violation_category: Optional[str] = Field(None, description="'toxicity', 'prompt_injection', 'budget_tampering', 'jailbreak', or None")
    risk_score: float = Field(0.0, description="Risk level between 0.0 (safe) and 1.0 (malicious)")
    reason: str = Field("Passed safety filters.", description="Detailed reason for the guardrail decision")
    sanitized_prompt: str = Field(..., description="Sanitized version of prompt safe for LLM context")
    detected_persona: str = Field("polite_shopper", description="Identified user persona: polite, slang, toxic, adversarial")


class ToolValidationResult(BaseModel):
    is_valid: bool
    tool_name: str
    errors: List[str] = []
    validated_data: Optional[Any] = None


# =====================================================================
# Injection & Toxicity Knowledge Base Patterns
# =====================================================================

INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"(?i)you\s+are\s+now\s+(in\s+developer\s+mode|dan|unrestricted)",
    r"(?i)\b(dan\s+mode(\s+enabled)?|bypass\s+(all\s+)?safety(\s+policies)?)\b",
    r"(?i)(system\s+prompt|reveal\s+your\s+prompt|show\s+secret\s+key)",
    r"(?i)(discount\s+(everything|all)\s+by\s+100%|make\s+price\s+(0|zero|free))",
    r"(?i)(override\s+budget|bypass\s+payment|force\s+order)",
    r"(?i)(drop\s+database|eval\(|exec\(|\$where|;\s*drop\s+table)",
    r"(?i)(sudo\s+mode|root\s+access|jailbreak)"
]

TOXICITY_PATTERNS = [
    r"(?i)\b(idiot|stupid|trash|scam|garbage|useless|shut\s+up|dumb|hate\s+you|fraud)\b",
    r"(?i)\b(fuck|shit|bitch|bastard|asshole|bullshit|scammer)\b"
]

SLANG_PATTERNS = [
    r"(?i)\b(drip|cop|lit|fire|fit|vibe|cozy|fresh|flex|steals|budget\s+friendly)\b",
    r"(?i)\b(kicks|sneaks|threads|slay|aesthetic|snag)\b"
]


# =====================================================================
# Core Guardrail Validator Class
# =====================================================================

class SecurityGuardrail:
    """
    Multi-layer defense protecting the AURA multi-agent system from
    adversarial attacks, malicious injections, abusive content, and data corruption.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.injection_regexes = [re.compile(p) for p in INJECTION_PATTERNS]
        self.toxic_regexes = [re.compile(p) for p in TOXICITY_PATTERNS]
        self.slang_regexes = [re.compile(p) for p in SLANG_PATTERNS]

    def validate_user_prompt(self, prompt: str) -> GuardrailCheckResult:
        """
        Parses user input for toxic language, jailbreaks, prompt injection,
        and identifies persona variances (Polite, Slang, Toxic, Adversarial).
        """
        if not prompt or not prompt.strip():
            return GuardrailCheckResult(
                is_safe=False,
                violation_category="empty_input",
                risk_score=0.5,
                reason="Empty user message cannot be processed.",
                sanitized_prompt="",
                detected_persona="ambiguous"
            )

        cleaned = prompt.strip()

        # 1. Check Prompt Injections / Jailbreak Attacks (Critical Risk)
        for rx in self.injection_regexes:
            if rx.search(cleaned):
                return GuardrailCheckResult(
                    is_safe=False,
                    violation_category="prompt_injection",
                    risk_score=0.95,
                    reason="Security violation: Detected attempted prompt injection or instruction override.",
                    sanitized_prompt="[Blocked malicious instruction]",
                    detected_persona="adversarial_injector"
                )

        # 2. Check Toxic & Abusive Language
        for rx in self.toxic_regexes:
            if rx.search(cleaned):
                return GuardrailCheckResult(
                    is_safe=False,
                    violation_category="toxicity",
                    risk_score=0.85,
                    reason="Concierge standard violation: Abusive or uncivil language is prohibited in luxury atelier.",
                    sanitized_prompt="[Blocked toxic language]",
                    detected_persona="toxic_user"
                )

        # 3. Detect Slang / Streetwear Persona vs Polite Patron
        is_slang = any(rx.search(cleaned) for rx in self.slang_regexes)
        persona = "streetwear_slang" if is_slang else "polite_shopper"

        # 4. Sanitize potentially dangerous HTML/Script tags
        sanitized = re.sub(r"<[^>]*>", "", cleaned)

        return GuardrailCheckResult(
            is_safe=True,
            violation_category=None,
            risk_score=0.05 if persona == "polite_shopper" else 0.15,
            reason="Prompt passed all security filters.",
            sanitized_prompt=sanitized,
            detected_persona=persona
        )

    def validate_tool_output(self, tool_name: str, output: Any, expected_type: type = list) -> ToolValidationResult:
        """
        Runtime callback validator ensuring tool outputs strictly conform to
        expected data contracts and types before consumption by subsequent agents.
        """
        errors = []
        if output is None:
            errors.append(f"Tool '{tool_name}' returned None where data was expected.")
            return ToolValidationResult(is_valid=False, tool_name=tool_name, errors=errors)

        if not isinstance(output, expected_type):
            errors.append(f"Tool '{tool_name}' returned type {type(output).__name__}, expected {expected_type.__name__}.")
            return ToolValidationResult(is_valid=False, tool_name=tool_name, errors=errors)

        # Specialized catalog tool item validation
        if tool_name in ["catalog_hybrid_search", "aura_search_catalog"] and isinstance(output, list):
            for i, item in enumerate(output):
                if not isinstance(item, dict):
                    errors.append(f"Item #{i} in catalog output is not a dictionary.")
                    continue
                # Ensure essential fields exist
                for req_key in ["product_id", "name"]:
                    if req_key not in item:
                        errors.append(f"Item #{i} missing required key '{req_key}'.")
                # Price key validation: accept 'price', 'price_inr', or 'price_usd'
                has_price = ("price" in item) or ("price_inr" in item) or ("price_usd" in item)
                if not has_price:
                    errors.append(f"Item #{i} missing required price key ('price', 'price_inr', or 'price_usd').")
                # Price sanity check
                price_val = item.get("price") if "price" in item else (item.get("price_usd") if "price_usd" in item else item.get("price_inr"))
                if price_val is not None and (not isinstance(price_val, (int, float)) or price_val < 0):
                    errors.append(f"Item #{i} has invalid negative or non-numeric price: {price_val}.")

        return ToolValidationResult(
            is_valid=len(errors) == 0,
            tool_name=tool_name,
            errors=errors,
            validated_data=output if len(errors) == 0 else None
        )

    def validate_spend_ceiling(self, order_total_inr: float, max_authorized_budget_inr: float) -> Tuple[bool, str]:
        """
        Hard budget gating guardrail: Prevents autonomous agents from exceeding
        the human principal's authorized spending limit.
        """
        if order_total_inr <= 0:
            return False, "Order total must be greater than zero."

        if max_authorized_budget_inr <= 0:
            return False, "Authorized budget ceiling must be greater than zero."

        if order_total_inr > max_authorized_budget_inr:
            excess = order_total_inr - max_authorized_budget_inr
            return False, f"Gated Spend Ceiling Violation: Order total ₹{order_total_inr:,.2f} exceeds human authorized ceiling of ₹{max_authorized_budget_inr:,.2f} by ₹{excess:,.2f} INR."

        return True, "Spend ceiling authorized."


# Singleton instance
_security_guardrail = SecurityGuardrail()

def get_security_guardrail() -> SecurityGuardrail:
    """Returns global SecurityGuardrail singleton."""
    return _security_guardrail

