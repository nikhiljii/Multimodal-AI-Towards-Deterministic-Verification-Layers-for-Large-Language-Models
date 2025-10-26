# Multimodal-AI-Towards-Deterministic-Verification-Layers-for-Large-Language-Models
<img width="2546" height="1418" alt="image" src="https://github.com/user-attachments/assets/7419d525-561b-4f27-9a58-8fba7c9e7fa9" />

This white paper introduces the Deterministic Verification Layer (DVL) — a symbolic, rule-based framework that brings verifiable reasoning to Large Language Models. While LLMs predict likely words, DVL computes, validates, and explains each claim before narration, enabling trustworthy multimodal AI.

Large Language Models (LLMs) can now interpret and generate text, images, and code fluently — but their reasoning remains probabilistic, not proven.
Humans verify through perception, logic, and experience; LLMs predict words based on statistical likelihood.
To close this gap, we introduce the Deterministic Verification Layer (DVL) — a symbolic, rule-based layer that computes, validates, and explains before the LLM is allowed to narrate.
The DVL fuses:
•	RDF knowledge graphs for structured semantics
•	SHACL constraint reasoning for logical validation
•	Deterministic computation for proof-based accuracy
•	Provenance tracking for auditability
•	A policy of refusal for impossible statements
This approach gives machines a human-like verification sense — they “check before believing.


# ============================================================
# Gradio App – Deterministic Math Validator + (Optional) DeepSeek
# ============================================================

!pip -q install gradio rdflib pyshacl requests

import re, os, json, ast, operator, traceback, requests
from decimal import Decimal, getcontext
from rdflib import Graph, Namespace, URIRef, Literal, RDF, XSD
from pyshacl import validate
import gradio as gr
from datetime import datetime, timezone

# ---------------- CONFIG ----------------
os.environ["DEEPSEEK_API_KEY"] = " "   # <<< ENTER YOUR KEY HERE (optional)
os.environ.setdefault("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-chat")
getcontext().prec = 30  # High precision math

# ---------------- RDF Ontology + SHACL ----------------
MATH = Namespace("https://example.org/math/")
MATH_TTL = """
@prefix math: <https://example.org/math/> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
math:Equation a rdf:Class .
math:operator a rdf:Property .
math:operand1 a rdf:Property .
math:operand2 a rdf:Property .
math:claimedResult a rdf:Property .
math:computedResult a rdf:Property .
"""
SHACL_TTL = """
@prefix math: <https://example.org/math/> .
@prefix sh:   <http://www.w3.org/ns/shacl#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
math:EquationShape a sh:NodeShape ;
  sh:targetClass math:Equation ;
  sh:property [ sh:path math:operator ; sh:datatype xsd:string ; sh:minCount 1 ] ;
  sh:property [ sh:path math:operand1 ; sh:datatype xsd:decimal ; sh:minCount 1 ] ;
  sh:property [ sh:path math:operand2 ; sh:datatype xsd:decimal ; sh:minCount 1 ] ;
  sh:property [ sh:path math:computedResult ; sh:datatype xsd:decimal ; sh:minCount 1 ] .
"""

# ---------------- Deterministic Evaluator ----------------
_ALLOWED = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}
def _eval_ast(node):
    if isinstance(node, ast.Constant):
        return Decimal(str(node.value))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_eval_ast(node.operand))
    raise ValueError("Unsupported expression (use + - * / **, integers/decimals).")

def safe_compute(expr:str)->Decimal:
    return _eval_ast(ast.parse(expr, mode="eval").body)

# ---------------- RDF & Validation ----------------
def build_graph():
    g=Graph(); g.parse(data=MATH_TTL,format="turtle"); return g

def insert_equation(g, expr, claimed):
    eq=URIRef(MATH["eq-1"])
    g.add((eq,RDF.type,MATH.Equation))
    parts=re.split(r"\s*([\+\-\*/\^])\s*",expr)
    if len(parts)>=3:
        g.add((eq,MATH.operand1,Literal(Decimal(parts[0]),datatype=XSD.decimal)))
        g.add((eq,MATH.operator,Literal(parts[1])))
        g.add((eq,MATH.operand2,Literal(Decimal(parts[2]),datatype=XSD.decimal)))
    computed=safe_compute(expr)
    g.add((eq,MATH.computedResult,Literal(computed,datatype=XSD.decimal)))
    if claimed: g.add((eq,MATH.claimedResult,Literal(Decimal(str(claimed)),datatype=XSD.decimal)))
    return computed

def validate_equation(expr, claimed, precision=30):
    getcontext().prec = precision
    g=build_graph()
    computed=insert_equation(g,expr,claimed)
    conforms,_,_=validate(data_graph=g,shacl_graph_text=SHACL_TTL,inference='rdfs')
    mismatch=[]
    if claimed:
        mismatch=list(g.query("""
        PREFIX math:<https://example.org/math/>
        SELECT ?eq ?c ?r WHERE {
          ?eq a math:Equation ;
              math:computedResult ?c ;
              math:claimedResult ?r .
          FILTER(?c != ?r)
        }"""))
    status=("unverified" if not claimed else ("mismatch" if mismatch else "valid"))
    badge={"valid":"✅ Validated","mismatch":"❌ Mismatch","unverified":"ℹ️ Computed"}[status]
    expl=f"Computed {expr} = {computed}"
    if status=="mismatch": expl+=f"; claimed {claimed} mismatches."
    return {
        "status":status,
        "answer":{"computed":str(computed),"claimed":claimed,"precision":precision},
        "evidence":{"constraints_passed":bool(conforms),"mismatch_fields":["claimed"] if mismatch else []},
        "provenance":{"timestamp_utc":datetime.now(timezone.utc).isoformat()},
        "ui":{"badge":badge,"explanation":expl}
    }

# ---------------- DeepSeek (optional) ----------------
def deepseek_math_answer(prompt:str):
    key=os.getenv("DEEPSEEK_API_KEY")
    if not key: raise RuntimeError("DeepSeek key not set.")
    base=os.getenv("DEEPSEEK_API_BASE"); model=os.getenv("DEEPSEEK_MODEL")
    headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"}
    sys_prompt=('Return JSON only: {"expression":"...","claimed_result":"..."}')
    payload={"model":model,"messages":[{"role":"system","content":sys_prompt},
                                       {"role":"user","content":prompt}]}
    r=requests.post(f"{base}/chat/completions",headers=headers,json=payload,timeout=90)
    if not r.ok: raise Exception(f"{r.status_code}: {r.text}")
    content=r.json().get("choices",[{}])[0].get("message",{}).get("content","")
    m=re.search(r"\{.*\}",content,re.S)
    if not m: raise ValueError("DeepSeek did not return JSON: "+content)
    obj=json.loads(m.group(0))
    return obj["expression"],obj["claimed_result"]

# ---------------- Pipeline for Gradio ----------------
def validate_math(expr_or_prompt, claimed, precision, use_deepseek):
    try:
        if use_deepseek:
            expr, claimed = deepseek_math_answer(expr_or_prompt)
            result = validate_equation(expr, claimed, precision)
            result["provenance"]["source"] = "DeepSeek"
        else:
            if "=" in expr_or_prompt:
                parts = re.split(r"=", expr_or_prompt)
                expr = parts[0].strip()
                claimed = parts[1].strip()
            else:
                expr = expr_or_prompt.strip()
            result = validate_equation(expr, claimed or None, precision)
        return (
            f"**{result['ui']['badge']}**\n\n"
            f"{result['ui']['explanation']}\n\n"
            f"**Status:** {result['status']}\n"
            f"**Computed:** {result['answer']['computed']}\n"
            f"**Claimed:** {result['answer'].get('claimed')}\n"
            f"**Precision:** {result['answer']['precision']}\n"
            f"**Constraints Passed (SHACL):** {result['evidence']['constraints_passed']}\n"
            f"**Timestamp:** {result['provenance']['timestamp_utc']}\n\n"
            "Raw JSON:\n```json\n" + json.dumps(result, indent=2) + "\n```"
        )
    except Exception as e:
        return f"❌ Error:\n```\n{traceback.format_exc()}\n```"

# ---------------- Gradio UI ----------------
iface = gr.Interface(
    fn=validate_math,
    inputs=[
        gr.Textbox(label="Expression or Prompt", placeholder="e.g., 5.1234 + 7.8766 = 13.001  OR  Compute 3.14159 * 2.71828 precisely"),
        gr.Textbox(label="Claimed result (optional)", placeholder="If manual mode"),
        gr.Number(label="Precision", value=30, precision=0),
        gr.Checkbox(label="Use DeepSeek API", value=False),
    ],
    outputs=gr.Markdown(label="Validation Result"),
    title="🧮 Deterministic Math Validator",
    description="Verifies numeric claims deterministically using RDF + SHACL. Optionally cross-checks with DeepSeek's output.",
)

iface.launch(share=True)

