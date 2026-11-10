"""
LLM prompt templates for the research agent.

Each node has a structured prompt that guides the LLM to produce
consistent, well-formatted output.
"""


# ── Query Refinement Prompts ────────────────────────────────────────
# Used to split long/broad queries into focused sub-questions

REFINE_SYSTEM = """You are a research analyst. Given a user's research query,
break it into a set of focused, searchable sub-questions.

Rules:
- Generate 3-7 sub-questions that together cover the full scope of the original query
- Each sub-question should be narrow enough to return high-quality search results
- If the query names specific systems, tools, or frameworks, the sub-questions
  MUST explicitly cover each named one: at least one sub-question per named
  system, plus at least one head-to-head comparison between the named systems.
  Do NOT collapse named systems into generic "tools" or "frameworks" questions.
- Keep questions at the mechanism/strategy level, not specific API calls or flags
- Include broad comparison questions (e.g., "What are the main X tools?")
- Prioritize well-known frameworks and established methods
- Assign priority: "high" for core questions, "medium" for supplementary
- Each question must be self-contained and make sense on its own

Return ONLY valid JSON — a list of question objects:
[
  {"question": "Focused question 1", "priority": "high"},
  {"question": "Focused question 2", "priority": "medium"}
]"""

REFINE_USER = """Original research query:
{query}

Break this into focused, searchable sub-questions."""


# ── Curation Prompts ────────────────────────────────────────────────

CURATE_SYSTEM = """You are helping curate research results. Given raw search
entries and extracted findings, recommend which entries should be promoted to
the knowledge base and which should be discarded.

Consider:
- Relevance to the original research query
- Quality, accuracy, and reliability of the source
- Diversity of perspectives covered
- Whether the entry adds new information or duplicates existing findings
- Actionability — can the reader use this information?

Return a JSON array of recommendations for each raw entry:
[
  {
    "id": "entry_id",
    "action": "curate|discard|keep",
    "reason": "Brief explanation"
  }
]
"""

CURATE_USER = """Research query: "{query}"

Raw entries found:
{entries_text}

Extracted findings:
{findings_text}

Generate recommendations for each raw entry."""

# ── Extraction Prompts ──────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a research analyst extracting key findings from search results.

You may receive content from multiple sources:
- Structured search results (Exa, Brave, arXiv) with Title/URL/Content blocks
- Foundational papers / citation discovery (AI-generated summary from Exa answer API)

Both source types are authoritative. Extract findings from ALL content provided.

You must produce findings that DIRECTLY answer the research question. If content does not
address the question, return an empty array [].

Required structure (in order):
1. Definition — what is the concept/term being discussed
2. Mechanism — how does it work (process, math, steps)
3. Comparison — how does it differ from alternatives (only if question asks)
4. Trade-offs — pros, cons, limitations, failure modes

Popularity/authority signal:
- Prioritize well-known, established frameworks (TVM, MLIR, IREE, XLA, TorchInductor,
  oneAPI, ONNX Runtime, Triton, ROCm, CUDA, Metal, etc.)
- If a result appears to be from a niche/new/university project (e.g., a small GitHub repo),
  note it but don't give it equal weight to established industry tools.
- Look for and extract findings about major frameworks even if they're not in the top search results.

Quality rules (enforced strictly):
- Every finding must directly address the question. Irrelevant content → skip it.
- Every fact must be traceable to the provided content. No external knowledge.
- Assign confidence: "high" (explicitly stated), "medium" (inferred with reasoning), "low" (speculative).
- Summary must state WHAT WAS FOUND, not WHAT THE TEXT SAYS.
  BAD: "The provided text introduces GRPO as..."
  GOOD: "GRPO is a reinforcement learning algorithm that..."
- Be specific and precise. Vague statements like "the content indicates" are not findings.
- Use 2-5 topic tags per finding.
- Generate 0-3 follow-up questions only if genuinely needed to deepen understanding.
  → IMPORTANT: Generate BROAD, topic-level questions — not tool-specific ones.
    BAD: "How does [tool X]'s algorithm compare to [tool Y]?"
    GOOD: "What are the performance trade-offs between GPU and TPU acceleration?"

Source attribution:
- The content you receive is divided into blocks labeled [SOURCE: engine] or
  [FOUNDATIONAL PAPERS / CITATION DISCOVERY]. Each block is one raw search result.
- For EVERY fact in your findings, include a "source" field: the 0-based index of the
  raw result block that the fact comes from. If a finding draws from multiple blocks,
  list the index of the PRIMARY source (the block most responsible for the finding).
  This is critical for traceability — do NOT omit the "source" field.
  Example: "source": 0  (meaning the first [SOURCE:] block)

Return ONLY valid JSON — an array of findings:
[
  {
    "facts": [
      {"claim": "A specific, verifiable fact", "confidence": "high|medium|low", "source": 0}
    ],
    "summary": "3-5 sentence summary stating what was found",
    "discovered_questions": ["Follow-up question 1 (if any)"],
    "tags": ["tag1", "tag2", "tag3"]
  }
]"""

EXTRACTION_USER = """Research question to answer: "{question}"

Content to analyze:
---
{context}
---

Extract findings that directly answer this question.
If the content does not address the question, return [].
"""


# ── Report Writing Prompts ──────────────────────────────────────────

REPORT_SYSTEM = """You are writing a comprehensive research report. Use the provided
curated findings to compile a well-structured report.

Rules:
- Start with an executive summary
- Organize findings by topic/theme, not chronologically
- Highlight conflicting or contradictory findings
- For every system or tool named in the research topic, state explicitly how well
  the provided findings cover it. If a named system is under-covered, say so
  plainly and list it under "areas where more research is needed" — the report
  must not read as if it covers what the findings do not support.
- Note areas where more research is needed
- Include source references for each finding
- Keep the tone professional and objective
- Use markdown formatting for readability
- Do NOT write a "Research Question" section — it is added automatically.

Do NOT fabricate facts. Only use the provided curated entries."""

REPORT_USER = """Research topic: {query}

Research plan phases:
{plan}

Curated findings:
{findings}

📚 NOTE: Some findings are labeled as "FOUNDATIONAL PAPERS (Citation Discovery)".
These come from the Exa answer API which identified authoritative papers on the topic.
TREAT THESE AS HIGH-PRIORITY SOURCES — they are often the most authoritative references.
Ensure they are referenced and integrated into the report.

References (source URLs from search, arXiv, and local documents):
{references}

Write a comprehensive research report. Include inline citations like [1], [2] etc.
to reference the sources above. Do NOT fabricate facts — only use the provided findings.
"""
