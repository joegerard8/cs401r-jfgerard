"""
CS 401R — Lab 3: Model Development
Track C: Customer Service Agent — Tool Definitions

Tools implement the business logic that the Bedrock Converse agent can call.

Design principles:
  1. All tools return structured dicts (not strings) — the agent formats responses.
  2. Every tool call is logged: tool name, inputs, output, latency, and estimated cost.
  3. Tools enforce business rules internally (POL-RET-004 return windows: 30 days,
     60 for Platinum, holiday extension, final sale never; $50 credit cap).
     This means the agent cannot circumvent policy by passing boundary inputs.
  4. Tools simulate DynamoDB / API Gateway calls with mock data for demo purposes.
     In production, replace the mock data with real boto3 DynamoDB calls.
  5. query_policy retrieves from the actual policy corpus (policy_docs/, a copy of
     the Lab 3 starter kit's northstar-policy-docs/; found automatically). It never answers from
     hardcoded text: until 2026-10-05 it did, and that text contradicted the
     corpus (60-day window for everyone, tiers in points, a $5.99 return fee).

Cost tracking:
  Claude Haiku 4.5 via the us. geographic inference profile (AWS Price List,
  AmazonBedrockFoundationModels, us-east-1, "Regional", checked 2026-10-05):
    Input:  $1.10 per 1M tokens
    Output: $5.50 per 1M tokens
  The global. profile is ~10% cheaper ($1.00 / $5.00).

  Tool call latency and cost are tracked in session_logger.
  Call session_logger.report() at the end of a conversation to see totals.
"""

import json
import logging
import math
import os
import random
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import boto3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

# Claude Haiku 4.5, us. inference profile (update when model pricing changes)
COST_PER_INPUT_TOKEN = 0.0000011     # $1.10 per 1M input tokens
COST_PER_OUTPUT_TOKEN = 0.0000055    # $5.50 per 1M output tokens


class ToolCallLogger:
    """
    Logs all tool calls for a single agent session and accumulates cost.

    Usage:
        session_logger.log("lookup_order", {"order_id": "ORD-123"}, result, latency_ms=45)
        session_logger.report()  # prints summary at end of conversation
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.total_cost: float = 0.0
        self.session_start: datetime = datetime.utcnow()

    def log(
        self,
        tool_name: str,
        inputs: dict,
        output: dict,
        latency_ms: float,
        cost: float = 0.0,
    ) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "tool": tool_name,
            "inputs": inputs,
            "output_keys": list(output.keys()) if isinstance(output, dict) else type(output).__name__,
            "latency_ms": round(latency_ms, 1),
            "cost_usd": cost,
        }
        self.calls.append(entry)
        self.total_cost += cost
        logger.info(
            "[TOOL] %s | latency=%.1fms | cost=$%.6f",
            tool_name, latency_ms, cost,
        )

    def report(self) -> str:
        """Return a formatted session summary."""
        duration = (datetime.utcnow() - self.session_start).total_seconds()
        lines = [
            "",
            "=" * 50,
            "Agent Session Report",
            "=" * 50,
            f"  Session duration: {duration:.1f}s",
            f"  Tool calls:       {len(self.calls)}",
            f"  Total cost:       ${self.total_cost:.6f}",
            "",
            "Tool call breakdown:",
        ]
        for i, call in enumerate(self.calls, 1):
            lines.append(
                f"  {i}. {call['tool']} | {call['latency_ms']}ms | ${call['cost_usd']:.6f}"
            )
        lines.append("=" * 50)
        report_str = "\n".join(lines)
        print(report_str)
        return report_str


# Global logger for the current agent session.
# Reset between sessions: session_logger = ToolCallLogger()
session_logger = ToolCallLogger()


# ---------------------------------------------------------------------------
# Mock data (replace with DynamoDB calls in production)
# ---------------------------------------------------------------------------

_MOCK_ORDERS = {
    # The order the Lab 3 evaluation harness asks about in TC-001 (happy path).
    "ORD-2847301": {
        "order_id": "ORD-2847301",
        "customer_id": "CUST-001",
        "status": "Shipped",
        "order_date": (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d"),
        "estimated_delivery": (datetime.utcnow() + timedelta(days=3)).strftime("%Y-%m-%d"),
        "tracking_number": "1Z999AA10198765432",
        "carrier": "UPS",
        "items": [
            {"sku": "SKU-0063", "name": "NorthStar Outerwear Item 63", "qty": 1, "price": 189.00},
        ],
        "total": 189.00,
        "payment_method": "Visa ending 4242",
    },
    "ORD-1234567": {
        "order_id": "ORD-1234567",
        "customer_id": "CUST-001",
        "status": "Shipped",
        "order_date": (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%d"),
        "estimated_delivery": (datetime.utcnow() + timedelta(days=2)).strftime("%Y-%m-%d"),
        "tracking_number": "1Z999AA10123456784",
        "carrier": "UPS",
        "items": [
            {"sku": "SKU-0041", "name": "NorthStar Footwear Item 41", "qty": 1, "price": 129.99},
            {"sku": "SKU-0012", "name": "NorthStar Accessories Item 12", "qty": 2, "price": 24.99},
        ],
        "total": 179.97,
        "payment_method": "Visa ending 4242",
    },
    "ORD-9876543": {
        "order_id": "ORD-9876543",
        "customer_id": "CUST-002",
        "status": "Delivered",
        "order_date": (datetime.utcnow() - timedelta(days=75)).strftime("%Y-%m-%d"),
        "estimated_delivery": (datetime.utcnow() - timedelta(days=68)).strftime("%Y-%m-%d"),
        "tracking_number": "9400111899223397626001",
        "carrier": "USPS",
        "items": [
            {"sku": "SKU-0088", "name": "NorthStar Activewear Item 88", "qty": 1, "price": 215.00},
        ],
        "total": 215.00,
        "payment_method": "Mastercard ending 5555",
    },
    "ORD-5551234": {
        "order_id": "ORD-5551234",
        "customer_id": "CUST-003",
        "status": "Processing",
        "order_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "estimated_delivery": (datetime.utcnow() + timedelta(days=5)).strftime("%Y-%m-%d"),
        "tracking_number": None,
        "carrier": "FedEx",
        "items": [
            {"sku": "SKU-0007", "name": "NorthStar Accessories Item 7", "qty": 3, "price": 18.99},
        ],
        "total": 56.97,
        "payment_method": "PayPal",
    },
}

_MOCK_CUSTOMERS = {
    "CUST-001": {
        "customer_id": "CUST-001",
        "name": "Jordan Martinez",
        "loyalty_tier": "Gold",
        "loyalty_points": 2847,
        "email": "jordan.m@example.com",
    },
    "CUST-002": {
        "customer_id": "CUST-002",
        "name": "Alex Chen",
        "loyalty_tier": "Platinum",
        "loyalty_points": 7420,
        "email": "alex.c@example.com",
    },
    "CUST-003": {
        "customer_id": "CUST-003",
        "name": "Sam Williams",
        "loyalty_tier": "Bronze",
        "loyalty_points": 312,
        "email": "sam.w@example.com",
    },
}

_MOCK_RETURNS = {}  # Populated by initiate_return()
_MOCK_TICKETS = {}  # Populated by escalate_to_human()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def lookup_order(order_id: str) -> dict:
    """
    Return the current status and details of a customer order.

    In production: calls DynamoDB GetItem on the northstar-orders table.
    Simulates ~20ms DynamoDB lookup latency.

    Args:
        order_id: Order ID in format ORD-XXXXXXX

    Returns:
        Order dict with status, items, tracking info, or an error dict.

    Error codes:
        ORDER_NOT_FOUND: order_id does not exist
    """
    t0 = time.monotonic()
    time.sleep(0.02)  # Simulate DynamoDB latency

    order_id = order_id.strip().upper()
    result = _MOCK_ORDERS.get(order_id)

    if result is None:
        output = {
            "error": "ORDER_NOT_FOUND",
            "message": f"No order found with ID {order_id}. Please verify the order ID and try again.",
        }
    else:
        output = dict(result)

    latency_ms = (time.monotonic() - t0) * 1000
    session_logger.log("lookup_order", {"order_id": order_id}, output, latency_ms)
    return output


# ---------------------------------------------------------------------------
# Policy corpus retrieval (used by query_policy)
# ---------------------------------------------------------------------------

def _find_policy_docs() -> Path:
    """policy_docs/ beside this file (inside the Lambda zip), else the repo's northstar-policy-docs/."""
    if os.environ.get("NORTHSTAR_POLICY_DOCS"):
        return Path(os.environ["NORTHSTAR_POLICY_DOCS"])
    here = Path(__file__).resolve().parent
    for cand in (here / "policy_docs", here.parent / "northstar-policy-docs"):
        if cand.is_dir():
            return cand
    return here / "policy_docs"


POLICY_DOCS_DIR = _find_policy_docs()

_STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it
its me my of on or our than that the their them then there these this to was we what when
which who will with you your am im i'm want would could should get got about any not no""".split())

_CHUNK_CHARS = 1200      # passages longer than this are split on paragraph breaks
_POLICY_INDEX = None     # built lazily on first query


def _terms(text: str) -> list:
    words = re.findall(r"[a-z0-9$%]+(?:[.,][0-9]+)?", text.lower())
    out = []
    for w in words:
        if w in _STOPWORDS or len(w) < 2:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]                        # returns -> return, members -> member
        out.append(w)
    return out


def _build_policy_index() -> dict:
    """Split every corpus document into heading-scoped chunks and index them."""
    chunks = []
    files = sorted(POLICY_DOCS_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(
            f"No policy documents in {POLICY_DOCS_DIR}. Copy the Lab 3 starter kit's "
            "northstar-policy-docs/ there or set NORTHSTAR_POLICY_DOCS.")
    for path in files:
        text = path.read_text(encoding="utf-8")
        m = re.search(r"\*\*Document ID:\*\*\s*(\S+)", text)
        doc_id = m.group(1) if m else path.stem
        parts = re.split(r"(?m)^(#{2,3} .+)$", text)
        for i in range(1, len(parts), 2):
            heading = parts[i].lstrip("#").strip()
            body = parts[i + 1].strip().strip("-").strip()
            if not body:
                continue
            num = re.match(r"(\d+(?:\.\d+)?)\.?\s+(.*)", heading)
            source = (f"{doc_id} §{num.group(1)} {num.group(2)}" if num
                      else f"{doc_id} — {heading}")
            # Keep short sections whole; split long ones on blank lines.
            paras, buf = re.split(r"\n\s*\n", body), ""
            pieces = []
            for p in paras:
                if buf and len(buf) + len(p) > _CHUNK_CHARS:
                    pieces.append(buf)
                    buf = p
                else:
                    buf = f"{buf}\n\n{p}" if buf else p
            if buf:
                pieces.append(buf)
            for piece in pieces:
                tf = Counter(_terms(heading) * 2 + _terms(piece))   # heading counts double
                chunks.append({"source": source, "text": piece, "tf": tf,
                               "len": sum(tf.values())})
    n = len(chunks)
    df = Counter(t for c in chunks for t in c["tf"])
    idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    avg_len = sum(c["len"] for c in chunks) / n
    return {"chunks": chunks, "idf": idf, "avg_len": avg_len}


def _retrieve_policy(question: str, k: int = 3) -> list:
    """BM25 over corpus chunks. Returns up to k hits sharing >= 2 query terms."""
    global _POLICY_INDEX
    if _POLICY_INDEX is None:
        _POLICY_INDEX = _build_policy_index()
    idx = _POLICY_INDEX
    q = set(_terms(question))
    k1, b = 1.2, 0.75
    scored = []
    # A single incidental word in common ("store", "refund") is not evidence.
    min_terms = min(2, len(q))
    for c in idx["chunks"]:
        matched = [t for t in q if t in c["tf"]]
        if not matched or len(matched) < min_terms:
            continue
        norm = k1 * (1 - b + b * c["len"] / idx["avg_len"])
        score = sum(idx["idf"][t] * c["tf"][t] * (k1 + 1) / (c["tf"][t] + norm)
                    for t in matched)
        scored.append((score, len(matched), c))
    scored.sort(key=lambda s: -s[0])
    return [{"source": c["source"], "text": c["text"], "score": round(s, 3),
             "matched_terms": m} for s, m, c in scored[:k]]


def query_policy(
    question: str,
    bedrock_client: Optional[object] = None,
    faiss_index: Optional[object] = None,
) -> dict:
    """
    Answer a policy question by retrieving passages from the NorthStar policy corpus.

    Retrieval is lexical (BM25 over section-level chunks of policy_docs/*.md),
    so it needs no Bedrock call, no index build, and returns the corpus text
    verbatim with its document ID and section. The agent composes the answer;
    this tool only supplies grounded evidence. Swap in the Track B FAISS index
    by passing faiss_index if you want semantic retrieval.

    Args:
        question:       Natural language question about NorthStar policy
        bedrock_client: Unused by the lexical retriever (kept for interface parity)
        faiss_index:    Unused by the lexical retriever (kept for interface parity)

    Returns:
        Dict with answer (retrieved passages), source_documents, confidence.
        confidence is "low" when nothing in the corpus matches: the agent should
        say it cannot find a policy and escalate rather than improvise one.
    """
    t0 = time.monotonic()

    hits = _retrieve_policy(question, k=3)

    if hits:
        answer = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        sources = [h["source"] for h in hits]
        # High when the best passage matches several distinct query terms.
        confidence = "high" if hits[0]["matched_terms"] >= 3 else "medium"
    else:
        answer = (
            "No NorthStar policy document addresses this question. Do not answer from "
            "general knowledge; tell the customer you will connect them with a person."
        )
        sources = []
        confidence = "low"

    output = {
        "answer": answer,
        "source_documents": sources,
        "confidence": confidence,
    }

    latency_ms = (time.monotonic() - t0) * 1000
    session_logger.log("query_policy", {"question": question[:100]}, output, latency_ms)
    return output


def _return_window_end(order: dict):
    """Last returnable date under POL-RET-004 §1, or None if not yet delivered."""
    if order.get("status") != "Delivered":
        return None
    delivered = datetime.strptime(order["estimated_delivery"], "%Y-%m-%d").date()
    purchased = datetime.strptime(order["order_date"], "%Y-%m-%d").date()
    tier = _MOCK_CUSTOMERS.get(order["customer_id"], {}).get("loyalty_tier", "Bronze")
    end = delivered + timedelta(days=60 if tier == "Platinum" else 30)        # §1, §1.1
    if (purchased.month, purchased.day) >= (11, 1) and (purchased.month, purchased.day) <= (12, 24):
        end = max(end, datetime(purchased.year + 1, 1, 31).date())             # §1.2
    return end


def initiate_return(order_id: str, reason: str) -> dict:
    """
    Initiate a return for a customer order.

    Business rules enforced internally (not left to agent judgment), per POL-RET-004:
      - Order must exist
      - Orders in "Processing" status cannot be returned — must be cancelled instead
      - Final sale items are never returnable, at any tier (§4 overrides everything)
      - Window (§1): 30 calendar days from delivery (online), 60 for a customer who
        was Platinum at purchase (§1.1); purchases Nov 1 – Dec 24 are returnable
        through Jan 31, whichever is longer (§1.2). The mock treats the customer's
        current tier as the tier at purchase.
      - Return shipping (§6.2): free for Gold/Platinum, otherwise $7.95 deducted
      - Each return gets a unique return ID (RTN-XXXXXXX)

    In production: writes to northstar-returns DynamoDB table and triggers
    a return label generation Lambda.

    Args:
        order_id: Order ID to return
        reason:   Customer-provided reason for return

    Returns:
        Success: {"return_id": str, "status": str, "instructions": str, "label_url": str}
        Error:   {"error": str, "message": str}
    """
    t0 = time.monotonic()
    time.sleep(0.03)  # Simulate write latency

    order_id = order_id.strip().upper()
    order = _MOCK_ORDERS.get(order_id)

    if order is None:
        output = {
            "error": "ORDER_NOT_FOUND",
            "message": f"No order found with ID {order_id}.",
        }
        session_logger.log("initiate_return", {"order_id": order_id, "reason": reason}, output,
                           (time.monotonic() - t0) * 1000)
        return output

    # POL-RET-004 §4: final sale overrides every tier benefit and window.
    final_sale = [i["name"] for i in order.get("items", []) if i.get("final_sale")]
    if final_sale:
        output = {
            "error": "FINAL_SALE",
            "message": (
                f"Order {order_id} contains final sale merchandise ({', '.join(final_sale)}). "
                "Per POL-RET-004 §4, final sale items cannot be returned or exchanged at any "
                "tier. Exceptions may only be granted by the Director of Customer Experience "
                "(§9); this system has no authority to grant one."
            ),
        }
        session_logger.log("initiate_return", {"order_id": order_id, "reason": reason}, output,
                           (time.monotonic() - t0) * 1000)
        return output

    # POL-RET-004 §1: window from delivery date (online orders), tier at purchase.
    window_end = _return_window_end(order)
    if window_end is not None and datetime.utcnow().date() > window_end:
        tier = _MOCK_CUSTOMERS.get(order["customer_id"], {}).get("loyalty_tier", "Bronze")
        output = {
            "error": "OUTSIDE_RETURN_WINDOW",
            "message": (
                f"Order {order_id}'s return window closed on {window_end.isoformat()}. "
                f"Per POL-RET-004 §1, the window is {60 if tier == 'Platinum' else 30} days "
                f"from delivery for a {tier} member. An exception requires a person: store "
                "staff may approve window exceptions up to $150 of merchandise (§9)."
            ),
        }
        session_logger.log("initiate_return", {"order_id": order_id, "reason": reason}, output,
                           (time.monotonic() - t0) * 1000)
        return output

    # Cannot return Processing orders
    if order["status"] == "Processing":
        output = {
            "error": "ORDER_NOT_ELIGIBLE",
            "message": (
                f"Order {order_id} is currently being processed and cannot be returned yet. "
                "Please contact us after the order ships to initiate a return."
            ),
        }
        session_logger.log("initiate_return", {"order_id": order_id, "reason": reason}, output,
                           (time.monotonic() - t0) * 1000)
        return output

    # Create return record
    return_id = f"RTN-{random.randint(1000000, 9999999)}"
    _MOCK_RETURNS[return_id] = {
        "return_id": return_id,
        "order_id": order_id,
        "reason": reason,
        "created_at": datetime.utcnow().isoformat(),
        "status": "Initiated",
    }

    tier = _MOCK_CUSTOMERS.get(order["customer_id"], {}).get("loyalty_tier", "Bronze")
    shipping_fee = 0.0 if tier in ("Gold", "Platinum") else 7.95   # POL-RET-004 §6.2
    output = {
        "return_id": return_id,
        "status": "Initiated",
        "instructions": (
            f"Your return for order {order_id} has been initiated (Return ID: {return_id}). "
            "A prepaid return label has been emailed to the address on file. "
            + ("Return shipping is free for your tier. " if shipping_fee == 0 else
               "A $7.95 return shipping fee will be deducted from the refund; returning "
               "to any NorthStar store instead is free. ")
            + "Card refunds reach the original card in 5-7 business days."
        ),
        "label_url": f"https://returns.northstar.com/label/{return_id}",
        "return_shipping_fee": shipping_fee,
        "eligible_refund": round(order["total"] - shipping_fee, 2),
    }

    latency_ms = (time.monotonic() - t0) * 1000
    session_logger.log("initiate_return", {"order_id": order_id, "reason": reason}, output, latency_ms)
    return output


def apply_loyalty_credit(customer_id: str, amount: float, reason: str) -> dict:
    """
    Apply a loyalty credit to a customer's account.

    IMPORTANT: This tool must NEVER be called without explicit policy justification.
    The agent must cite the specific policy reason before calling this tool.
    Credits > $50 require manager approval (this is a hard system limit, not advisory).

    Business rules:
      - Credits <= $50: applied immediately, returns success
      - Credits > $50: not applied; returns approval_required=True with ticket ID
      - Customer must exist in the system

    In production: calls northstar-loyalty-service API, which writes to
    DynamoDB and triggers a notification to the customer.

    Args:
        customer_id: Customer account ID
        amount:      Credit amount in USD (positive)
        reason:      Policy-based justification for the credit

    Returns:
        {"success": bool, "new_balance": float, "approval_required": bool, ...}
    """
    t0 = time.monotonic()
    time.sleep(0.025)

    if amount <= 0:
        output = {
            "success": False,
            "error": "INVALID_AMOUNT",
            "message": "Credit amount must be positive.",
        }
        session_logger.log("apply_loyalty_credit",
                           {"customer_id": customer_id, "amount": amount, "reason": reason},
                           output, (time.monotonic() - t0) * 1000)
        return output

    customer = _MOCK_CUSTOMERS.get(customer_id)
    if customer is None:
        output = {
            "success": False,
            "error": "CUSTOMER_NOT_FOUND",
            "message": f"No customer found with ID {customer_id}.",
        }
        session_logger.log("apply_loyalty_credit",
                           {"customer_id": customer_id, "amount": amount, "reason": reason},
                           output, (time.monotonic() - t0) * 1000)
        return output

    # Hard limit: credits > $50 require manager approval
    if amount > 50.0:
        ticket_id = f"MGR-{random.randint(10000, 99999)}"
        output = {
            "success": False,
            "approval_required": True,
            "ticket_id": ticket_id,
            "message": (
                f"A loyalty credit of ${amount:.2f} exceeds the $50 self-service limit. "
                f"A manager approval request (Ticket {ticket_id}) has been submitted. "
                "The credit will be applied within 1 business day if approved."
            ),
        }
        logger.warning(
            "Credit of $%.2f for %s requires manager approval (reason: %s)",
            amount, customer_id, reason,
        )
        session_logger.log("apply_loyalty_credit",
                           {"customer_id": customer_id, "amount": amount, "reason": reason},
                           output, (time.monotonic() - t0) * 1000)
        return output

    # Apply credit
    current_balance = customer["loyalty_points"]
    # Convert dollar credit to points (100 points = $1)
    points_added = int(amount * 100)
    new_balance = current_balance + points_added
    _MOCK_CUSTOMERS[customer_id]["loyalty_points"] = new_balance

    output = {
        "success": True,
        "approval_required": False,
        "customer_id": customer_id,
        "amount_credited": amount,
        "points_added": points_added,
        "new_balance": new_balance,
        "reason_recorded": reason,
        "message": (
            f"A loyalty credit of ${amount:.2f} ({points_added} points) has been applied "
            f"to {customer.get('name', 'your')} account. "
            f"New balance: {new_balance} points."
        ),
    }

    latency_ms = (time.monotonic() - t0) * 1000
    session_logger.log("apply_loyalty_credit",
                       {"customer_id": customer_id, "amount": amount, "reason": reason},
                       output, latency_ms)
    return output


def escalate_to_human(
    customer_id: str,
    issue_summary: str,
    priority: str = "normal",
) -> dict:
    """
    Escalate the conversation to a human customer service agent.

    Required for:
      - Policy exceptions (anything outside documented policy)
      - Credits > $50
      - Customer expressed frustration or used abusive language
      - Complex multi-issue complaints
      - Any situation where the AI agent is uncertain

    Priority levels: "normal", "urgent"
      - normal: standard queue, estimated wait 5-15 minutes
  - urgent: priority queue, estimated wait 1-3 minutes

    In production: writes a ticket to Zendesk/ServiceNow and initiates
    a call-back queue entry.

    Args:
        customer_id:   Customer account ID
        issue_summary: Concise summary of the issue for the human agent
        priority:      "normal" or "urgent"

    Returns:
        {"ticket_id": str, "estimated_wait_minutes": int, "message": str}
    """
    t0 = time.monotonic()
    time.sleep(0.02)

    priority = priority.lower()
    if priority not in ("normal", "urgent"):
        priority = "normal"

    ticket_id = f"TKT-{random.randint(1000000, 9999999)}"
    wait_minutes = random.randint(1, 3) if priority == "urgent" else random.randint(5, 15)

    _MOCK_TICKETS[ticket_id] = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "issue_summary": issue_summary,
        "priority": priority,
        "created_at": datetime.utcnow().isoformat(),
        "status": "Queued",
    }

    output = {
        "ticket_id": ticket_id,
        "priority": priority,
        "estimated_wait_minutes": wait_minutes,
        "message": (
            f"I've connected you with a NorthStar customer service specialist "
            f"(Ticket: {ticket_id}). "
            f"Estimated wait time: {wait_minutes} minutes. "
            "You'll receive a callback at the number on file, or you can continue "
            "waiting in this chat. Thank you for your patience."
        ),
    }

    logger.info(
        "Escalated customer %s to human (priority=%s, ticket=%s): %s",
        customer_id, priority, ticket_id, issue_summary[:80],
    )

    latency_ms = (time.monotonic() - t0) * 1000
    session_logger.log("escalate_to_human",
                       {"customer_id": customer_id, "priority": priority},
                       output, latency_ms)
    return output
