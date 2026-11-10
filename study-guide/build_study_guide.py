"""Build the Data Scientist Study Guide PDF from this project's material.

A single, reproducible script that lays out a comprehensive interview-prep guide
covering everything the Agentic RAG Research Assistant teaches — RAG, embeddings,
vector search, chunking, agents, LangGraph, and evaluation — plus a Q&A bank and a
one-page project cheat-sheet. Run it with the project venv:

    python study-guide/build_study_guide.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, Preformatted, Spacer, Table, TableStyle,
)

# --------------------------------------------------------------------------- #
# Palette — calm, professional, consistent with the project's figures.
# --------------------------------------------------------------------------- #
INK = colors.HexColor("#1f2a37")
ACCENT = colors.HexColor("#2f6f4f")        # the project's agent-green
ACCENT_DK = colors.HexColor("#20503a")
MUTED = colors.HexColor("#6b7280")
RULE = colors.HexColor("#d1d5db")
CODE_BG = colors.HexColor("#f3f4f6")
TIP_BG = colors.HexColor("#eaf3ee")
FIND_BG = colors.HexColor("#fdf6ec")
FIND_BD = colors.HexColor("#e0b96b")

OUT = Path(__file__).resolve().parent / "Agentic_RAG_Study_Guide_for_Data_Scientists.pdf"

# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
ss = getSampleStyleSheet()

def style(name, parent=None, **kw):
    return ParagraphStyle(name, parent=parent or ss["Normal"], **kw)

H1 = style("H1", fontName="Helvetica-Bold", fontSize=18, textColor=ACCENT_DK,
           spaceBefore=18, spaceAfter=8, leading=22, keepWithNext=True)
H2 = style("H2", fontName="Helvetica-Bold", fontSize=13, textColor=INK,
           spaceBefore=12, spaceAfter=5, leading=16, keepWithNext=True)
BODY = style("Body", fontSize=10.2, leading=15, textColor=INK, spaceAfter=6, alignment=TA_LEFT)
BULLET = style("Bullet", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=3)
CODE = style("Code", fontName="Courier", fontSize=8.4, leading=11, textColor=INK)
QLBL = style("QLbl", fontName="Helvetica-Bold", fontSize=10.2, textColor=ACCENT_DK,
             spaceBefore=8, spaceAfter=2, leading=14, keepWithNext=True)
ALBL = style("ALbl", fontSize=10.2, leading=15, textColor=INK, spaceAfter=6, leftIndent=14)
SMALL = style("Small", fontSize=8.5, leading=11, textColor=MUTED)
TIP = style("Tip", fontSize=9.6, leading=13, textColor=ACCENT_DK)
COVER_T = style("CoverT", fontName="Helvetica-Bold", fontSize=30, textColor=colors.white,
                leading=36, alignment=TA_CENTER)
COVER_S = style("CoverS", fontSize=14, textColor=colors.white, leading=20, alignment=TA_CENTER)


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


story = []

def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def p(t): story.append(Paragraph(t, BODY))
def sp(h=6): story.append(Spacer(1, h))

def bullets(items):
    for it in items:
        story.append(Paragraph(it, BULLET, bulletText="•"))

def code(t):
    lines = t.strip("\n").split("\n")
    story.append(Table(
        [[Preformatted(t.strip("\n"), CODE)]],
        colWidths=[6.5 * inch],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, RULE),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]),
    ))
    sp(6)

def callout(title, body, bg, bd):
    inner = [Paragraph(f"<b>{title}</b>", TIP), Spacer(1, 3), Paragraph(body, TIP)]
    story.append(Table(
        [[inner]], colWidths=[6.5 * inch],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("BOX", (0, 0), (-1, -1), 1, bd),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]),
    ))
    sp(8)

def tip(body): callout("Interview tip", body, TIP_BG, ACCENT)
def finding(body): callout("The finding to quote", body, FIND_BG, FIND_BD)

def qa(q, a):
    story.append(Paragraph("Q. " + q, QLBL))
    if isinstance(a, list):
        for line in a:
            story.append(Paragraph(line, ALBL))
    else:
        story.append(Paragraph("A. " + a, ALBL))

def datatable(rows, header=True, widths=None):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    st = [
        ("GRID", (0, 0), (-1, -1), 0.5, RULE),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), ACCENT),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    story.append(t)
    sp(8)


# =========================================================================== #
# COVER
# =========================================================================== #
story.append(NextPageTemplate("body"))
story.append(Spacer(1, 1.4 * inch))
story.append(Table(
    [[[Spacer(1, 26),
       Paragraph("Agentic RAG &amp; LangGraph", COVER_T),
       Paragraph("A Study Guide for Data Scientist Interviews", COVER_S),
       Spacer(1, 14),
       Paragraph("Retrieval-Augmented Generation &bull; Embeddings &bull; Vector Search &bull; "
                 "Agents &bull; LangGraph &bull; Evaluation", COVER_S),
       Spacer(1, 26)]]],
    colWidths=[6.7 * inch],
    style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT_DK),
        ("LEFTPADDING", (0, 0), (-1, -1), 30),
        ("RIGHTPADDING", (0, 0), (-1, -1), 30),
        ("TOPPADDING", (0, 0), (-1, -1), 24),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 24),
    ]),
))
story.append(Spacer(1, 0.5 * inch))
p_center = style("PC", parent=BODY, alignment=TA_CENTER, textColor=MUTED)
story.append(Paragraph("Built from the <b>Agentic RAG Research Assistant</b> portfolio project — "
                       "question-answering over 2,012 arXiv papers, with a from-scratch agent and a "
                       "LangGraph rebuild.", p_center))
story.append(Paragraph("Prepared by Cena Shabani &bull; Companion to the project&rsquo;s "
                       "<font name='Courier'>learning/</font> course", p_center))
story.append(PageBreak())

# =========================================================================== #
# HOW TO USE + CONTENTS
# =========================================================================== #
h1("How to use this guide")
p("This is an interview-prep companion to a real project, not an abstract textbook. Every concept "
  "here is something the project implements, so you can always point to code when an interviewer "
  "pushes for depth. The through-line is one honest, defensible story:")
finding("Single-shot retrieval nails single-hop questions (recall@5 = 1.00) but collapses to 0.60 "
        "on multi-hop ones, because one search cannot cover two pieces of evidence. That gap is the "
        "entire reason to build an <b>agent</b> that retrieves iteratively. Being able to state a "
        "measured finding like this — with the number and the mechanism — is what separates "
        "&lsquo;I did a RAG tutorial&rsquo; from &lsquo;I can reason about RAG&rsquo;.")
p("Each chapter ends with the interview-relevant takeaway. The last two chapters are a <b>Q&amp;A "
  "bank</b> of questions you are likely to be asked, and a <b>one-page cheat-sheet</b> to review "
  "the morning of an interview.")

h2("Contents")
datatable([
    ["#", "Chapter", "The one idea"],
    ["1", "What RAG is and why it exists", "Give the model an open-book exam"],
    ["2", "Embeddings & semantic search", "Turn meaning into geometry"],
    ["3", "Chunking", "One vector can only mean one thing"],
    ["4", "Vector stores & ANN search", "Fast nearest-neighbour at scale"],
    ["5", "The RAG pipeline end to end", "Build-time vs query-time"],
    ["6", "Prompting & grounding", "Answer from the sources, and cite"],
    ["7", "Agentic RAG", "Let the model decide how to retrieve"],
    ["8", "Agents, tools & the loop", "Reason, act, observe, repeat"],
    ["9", "LangGraph", "Control flow as a drawable graph"],
    ["10", "Evaluating RAG", "Measure retrieval and answers separately"],
    ["11", "Failure modes & limits", "Where RAG breaks, honestly"],
    ["12", "Interview Q&A bank", "40+ questions with answers"],
    ["13", "One-page cheat-sheet", "Review this the morning of"],
], widths=[0.4 * inch, 2.9 * inch, 3.2 * inch])
story.append(PageBreak())

# =========================================================================== #
# 1
# =========================================================================== #
h1("1 &middot; What RAG is and why it exists")
p("<b>Retrieval-Augmented Generation (RAG)</b>: before a language model answers, fetch the relevant "
  "documents and place them in the prompt, so the model answers <i>from those documents</i> instead "
  "of from memory.")
h2("The exam analogy")
p("A plain LLM answering from memory is a student sitting a <b>closed-book</b> exam — fluent, and "
  "confidently wrong on specifics. RAG makes it <b>open-book</b>: for each question it flips to the "
  "relevant pages, reads them, and answers with them open. The model&rsquo;s job shifts from "
  "<i>recall</i> to <i>reading comprehension</i> — a far more reliable task.")
h2("The three problems RAG fixes")
datatable([
    ["Problem", "Bare LLM", "With RAG"],
    ["Knowledge is frozen at training", "Cannot know new facts", "Add documents to the corpus"],
    ["Context window is finite", "Cannot read the whole corpus", "Retrieve only what is relevant"],
    ["Hallucination", "Invents a fluent, wrong answer", "Answers from sources, or abstains"],
], widths=[2.0 * inch, 2.3 * inch, 2.2 * inch])
p("Crucially, all three are <b>retrieval</b> fixes, not <b>model</b> fixes. You often don&rsquo;t "
  "need a bigger or newer model — you need to give the model you have the right context.")
tip("If asked &lsquo;why not just fine-tune?&rsquo; answer: fine-tuning teaches <i>style and "
    "behaviour</i>, not fresh facts; it is expensive, slow to update, and still hallucinates. RAG "
    "updates knowledge by editing a document store — cheap and traceable. They solve different "
    "problems and are often combined.")

# =========================================================================== #
# 2
# =========================================================================== #
h1("2 &middot; Embeddings &amp; semantic search")
p("Keyword matching breaks the moment a document says the same thing in different words. "
  "<b>Embeddings</b> let us match on meaning. An embedding model maps text to a vector — a point in "
  "a high-dimensional space — such that texts with similar meaning land near each other.")
h2("Search becomes geometry")
bullets([
    "Embed every passage in the corpus &rarr; a cloud of points.",
    "Embed the query into the <i>same</i> space &rarr; one more point.",
    "Return the passages whose points are <b>nearest</b> the query point.",
])
p("No shared keywords required — if two texts <i>mean</i> the same thing, their points are close.")
h2("Cosine similarity and the normalization trick")
p("&lsquo;Near&rsquo; is measured by <b>cosine similarity</b> — the cosine of the angle between two "
  "vectors, from -1 (opposite) to 1 (identical direction). If you normalize every vector to unit "
  "length, the plain <b>dot product equals cosine similarity</b> — which lets a fast dot-product "
  "index mean &lsquo;most similar in meaning&rsquo;. This project uses "
  "<font name='Courier'>all-MiniLM-L6-v2</font> (384-dim, CPU, deterministic).")
code(
"vectors = model.encode(texts, normalize_embeddings=True)  # unit vectors\n"
"# now: dot(a, b) == cosine_similarity(a, b)")
tip("Know the difference between a <b>bi-encoder</b> (embeds query and document separately — fast, "
    "used for first-stage retrieval) and a <b>cross-encoder</b> (reads query+document together — "
    "slower, more accurate, used for <i>reranking</i> the top results).")

# =========================================================================== #
# 3
# =========================================================================== #
h1("3 &middot; Chunking")
p("An embedding model compresses its whole input into <b>one</b> vector. Embed an entire paper and "
  "that vector is an average of every idea in it — near nothing in particular. So we split documents "
  "into <b>chunks</b> and embed each separately. Precision comes from each chunk meaning one thing.")
h2("The Goldilocks size, and overlap")
bullets([
    "<b>Too big</b> &rarr; blurry embeddings that match nothing precisely.",
    "<b>Too small</b> &rarr; a sentence loses the context that made it meaningful.",
    "<b>Sweet spot</b> &rarr; a few sentences / one paragraph (this project: ~1000 chars).",
    "<b>Overlap</b> (~15%) &rarr; consecutive chunks share text, so a fact that straddles a "
    "boundary survives intact in at least one chunk.",
])
p("Chunks must also carry <b>metadata</b> (source id, title) so an answer can <i>cite</i> the "
  "passage it used — no metadata, no citations, no auditability.")
tip("Good follow-up to raise yourself: <b>semantic</b> or <b>recursive</b> chunking (split on "
    "sentence/section boundaries) beats fixed-size character windows on structured documents. "
    "Mentioning the upgrade path shows you know the default has limits.")

# =========================================================================== #
# 4
# =========================================================================== #
h1("4 &middot; Vector stores &amp; ANN search")
p("A <b>vector store</b> holds the chunk vectors and finds the nearest ones to a query fast.")
datatable([
    ["Approach", "How", "When"],
    ["Exact (flat)", "Compare query to every vector", "Small corpus (this project: 4,298 chunks) — instant, no approximation error"],
    ["Approximate (ANN)", "HNSW graphs, IVF, PQ — search a fraction of vectors", "Millions+ of vectors — trade a little recall for big speed"],
], widths=[1.3 * inch, 2.6 * inch, 2.6 * inch])
p("This project uses FAISS <font name='Courier'>IndexFlatIP</font> (exact, inner-product). Because "
  "search is exact, retrieval quality reflects the embeddings and chunking — with no ANN "
  "approximation muddying the evaluation. The index stores vectors and returns <i>row numbers</i>; "
  "a row-aligned metadata table maps them back to chunk text and citations.")
tip("&lsquo;How would this scale to 100M documents?&rsquo; &rarr; switch the flat index for an ANN "
    "index (HNSW), move to a hosted vector DB (pgvector / Pinecone / Weaviate), and add a reranker. "
    "The <i>concepts</i> are identical; you&rsquo;re changing the index type and the ops, not the pipeline.")

# =========================================================================== #
# 5
# =========================================================================== #
h1("5 &middot; The RAG pipeline end to end")
code(
"ACQUIRE -> CHUNK -> EMBED -> INDEX   ||   RETRIEVE -> GENERATE\n"
"\\_________ build time _________/         \\___ query time ___/")
p("The single most clarifying idea in RAG is the <b>build-time vs query-time</b> split:")
bullets([
    "<b>Build time</b> (run once): turn documents into a searchable index. Expensive work — "
    "embedding the whole corpus — paid once.",
    "<b>Query time</b> (per question): embed one query, search, generate. Cheap.",
])
p("Two consequences fall out: updating knowledge is cheap (re-index, no retraining), and retrieval "
  "is local/reproducible while generation is the only hosted part. Use the <b>same embedding "
  "model</b> at build and query time — the query must live in the same geometry as the chunks.")
tip("Emphasise the <b>single source of truth</b>: both the plain and agentic paths call the same "
    "retriever, so any measured difference between them is about the <i>strategy</i>, never an "
    "accidental difference in search. Controlling variables is a data-scientist instinct interviewers look for.")

# =========================================================================== #
# 6
# =========================================================================== #
h1("6 &middot; Prompting &amp; grounding")
p("<b>Grounding</b> = binding the answer to the evidence. It is enforced by three instructions in "
  "the system prompt:")
bullets([
    "<b>&lsquo;Answer using ONLY the provided passages.&rsquo;</b> Shifts the model from recall to "
    "reading comprehension — the core anti-hallucination lever.",
    "<b>&lsquo;Cite the source ids you used.&rsquo;</b> Makes every claim auditable and nudges the "
    "model to actually use the passages.",
    "<b>&lsquo;If the passages are insufficient, say so.&rsquo;</b> Makes &lsquo;I don&rsquo;t "
    "know&rsquo; a correct answer — permission to abstain is a hallucination guardrail.",
])
p("We also verify citations: a &lsquo;citation&rsquo; to an id that was never retrieved is a "
  "hallucinated reference and is not credited. Grounding <i>reduces</i> hallucination; it does not "
  "abolish it (the model can misread, or retrieval can hand it the wrong passage) — which is why we "
  "<i>measure</i> groundedness rather than assume it.")

# =========================================================================== #
# 7
# =========================================================================== #
h1("7 &middot; Agentic RAG")
p("Plain RAG retrieves <b>once</b>, up front, and hopes that single search grabbed everything. "
  "<b>Agentic RAG</b> puts the LLM in charge of retrieval: it can decompose the question, search for "
  "each part, read the results, decide whether it has enough, and search again before answering.")
finding("This matters because of the measured multi-hop gap. A question like &lsquo;how does RAG "
        "combine a retriever with a generator?&rsquo; needs evidence from TWO papers; one search "
        "lands in one region of the vector space and misses the second. The agent&rsquo;s second, "
        "reformulated search is what recovers it — turning 0.60 recall into a real answer.")
h2("The trade-off — say this out loud")
p("Agentic RAG is not free: multiple LLM calls and retrievals per question means more tokens, "
  "latency, and cost. The honest framing is a measured exchange rate — <i>better multi-hop answers "
  "at N&times; the retrieval cost</i> — not &lsquo;the fancy version is better&rsquo;. Interviewers "
  "reward the candidate who names the cost, not just the benefit.")

# =========================================================================== #
# 8
# =========================================================================== #
h1("8 &middot; Agents, tools &amp; the loop")
p("An <b>agent</b> is an LLM that uses <b>tools</b> in a <b>loop</b> toward a goal. A tool is a "
  "function described to the model (name, description, input schema); the model emits a request to "
  "call it, your code runs it, and you feed the result back. This project&rsquo;s one tool is "
  "<font name='Courier'>search_corpus(query)</font>.")
h2("The tool-use loop (ReAct)")
code(
"send conversation + tools to the LLM\n"
"loop:\n"
"  if the reply is a tool call  -> run the tool, append the result, repeat\n"
"  if the reply is plain text   -> that's the answer, stop\n"
"  (bounded by MAX_AGENT_STEPS so a confused agent can't loop forever)")
p("<b>ReAct</b> = Reason + Act: the model interleaves reasoning with tool actions. The bounded loop "
  "and a &lsquo;stopped reason&rsquo; (answered vs hit-the-cap) are what make it production-safe.")
tip("Be ready to contrast building the loop <b>from scratch</b> (maximally transparent — every "
    "decision is an inspectable message) with a framework like <b>LangGraph</b> (next chapter), "
    "which you reach for when the loop needs checkpointing, streaming, or human-in-the-loop.")

# =========================================================================== #
# 9
# =========================================================================== #
h1("9 &middot; LangGraph")
p("<b>LangGraph</b> models an agent as a directed graph / state machine. Instead of writing "
  "<font name='Courier'>if</font>/<font name='Courier'>while</font> control flow by hand, you "
  "<i>declare</i> a graph and the runtime walks it. Four primitives:")
datatable([
    ["Primitive", "What it is", "Plain-Python analogue"],
    ["State", "A shared TypedDict every step reads/writes", "the variables in your loop"],
    ["Node", "A function: state -> partial state update", "one step of the loop body"],
    ["Edge", "'after A, go to B'", "statement order"],
    ["Conditional edge", "A router that reads state and picks the next node", "an if / while branch"],
], widths=[1.4 * inch, 3.0 * inch, 2.1 * inch])
p("You add nodes and edges to a <font name='Courier'>StateGraph</font>, then "
  "<font name='Courier'>.compile()</font> it into a runnable and <font name='Courier'>.invoke()</font> "
  "or <font name='Courier'>.stream()</font> it. A key detail: a state channel annotated with a "
  "<b>reducer</b> (<font name='Courier'>Annotated[list, operator.add]</font>) <i>accumulates</i> "
  "instead of overwriting — perfect for gathering documents across retrieval loops.")
h2("Our agent as a graph")
code(
"START -> plan -> retrieve -> decide --(need more)--> plan\n"
"                              decide --(enough)-----> generate -> END")
p("<font name='Courier'>plan</font> proposes the next focused query; <font name='Courier'>retrieve"
  "</font> runs the shared retriever and appends hits; <font name='Courier'>decide</font> is the "
  "conditional edge (enough distinct papers, or hit the cap?); <font name='Courier'>generate</font> "
  "writes the grounded answer. Same behaviour as the from-scratch loop — now a graph you can draw, "
  "checkpoint, and stream. LangGraph also ships <font name='Courier'>create_react_agent</font>, a "
  "one-liner for the standard tool-calling loop.")
tip("The map to memorise: hand-written <font name='Courier'>while</font> &harr; compiled StateGraph; "
    "the <font name='Courier'>if</font> that decides &lsquo;search again?&rsquo; &harr; a conditional "
    "edge; the de-duped evidence dict &harr; a reduced state channel; hand-written tool JSON &harr; "
    "the <font name='Courier'>@tool</font> decorator.")

# =========================================================================== #
# 10
# =========================================================================== #
h1("10 &middot; Evaluating RAG")
p("A RAG system you can&rsquo;t measure is a demo. The most important idea: <b>evaluate retrieval "
  "and generation separately</b> — they fail for different reasons and are fixed in different places.")
h2("Retrieval metrics (need ground-truth relevant docs)")
datatable([
    ["Metric", "Question it answers", "Definition"],
    ["hit@k", "Did any relevant doc appear in top-k?", "1 if intersection non-empty"],
    ["recall@k", "What fraction of relevant docs were retrieved?", "|retrieved & relevant| / |relevant|"],
    ["MRR", "How high was the first relevant doc ranked?", "1 / rank of first relevant"],
], widths=[1.1 * inch, 3.0 * inch, 2.4 * inch])
h2("Answer metrics (LLM-as-judge)")
p("Answer quality has no single correct string, so use a separate LLM call to grade against a "
  "rubric — <b>groundedness</b> (is every claim supported by the sources?) and <b>correctness</b> "
  "(does it answer the question?). An answer can be grounded but wrong, or correct but ungrounded, "
  "so you measure both. LLM-as-judge is a good <i>relative</i> comparator (same judge, same rubric), "
  "a weaker <i>absolute</i> score.")
finding("The fair comparison holds everything constant except the strategy — same corpus, retriever, "
        "judge, and questions — and slices by hop type, because the hypothesis is specific: the "
        "agent wins on multi-hop, roughly ties on single-hop, and costs more either way. Report "
        "benefit AND cost.")

# =========================================================================== #
# 11
# =========================================================================== #
h1("11 &middot; Failure modes &amp; limits")
bullets([
    "<b>Garbage retrieval &rarr; garbage answer.</b> Retrieval is the ceiling on answer quality; no "
    "prompt fixes wrong passages.",
    "<b>Bad chunking.</b> Wrong-sized or mid-idea chunks quietly wreck retrieval.",
    "<b>Lost in the middle.</b> LLMs attend most to the start/end of context and skim the middle — "
    "so keep top-k small and rank the best first.",
    "<b>Multi-hop weakness.</b> Single-shot retrieval misses compound evidence (the project&rsquo;s "
    "finding); the agent fixes it at a token cost.",
    "<b>Cost &amp; latency.</b> Every answer is &ge;1 LLM call; agents are several. Mitigate with "
    "local embeddings, caching, a bounded loop, a cheaper model.",
    "<b>Prompt injection.</b> Retrieved text is untrusted — a document saying &lsquo;ignore your "
    "instructions&rsquo; is <i>indirect prompt injection</i>. Treat retrieved content as data, never "
    "as instructions.",
    "<b>Corpus boundary.</b> RAG can only answer from what&rsquo;s in the corpus; the correct "
    "behaviour outside it is to abstain.",
])
tip("Naming what a project <b>deliberately does not do</b> (no reranker, no fine-tuning, no "
    "production vector DB, not hardened against injection) reads as maturity, not weakness. Scope "
    "honesty is a senior signal.")

story.append(PageBreak())

# =========================================================================== #
# 12 Q&A BANK
# =========================================================================== #
h1("12 &middot; Interview Q&amp;A bank")
p("Concise, defensible answers. Say the core sentence, then one supporting detail.", )

h2("Fundamentals")
qa("What is RAG, in one sentence?",
   "Retrieve relevant documents for a query and put them in the prompt so the LLM answers from them "
   "rather than from memory — an open-book exam instead of a closed-book one.")
qa("RAG vs fine-tuning?",
   "RAG injects fresh, traceable knowledge at query time and is cheap to update; fine-tuning changes "
   "the model&rsquo;s style/behaviour, is costly, and still hallucinates facts. Different jobs, often combined.")
qa("Why not just put everything in a long context window?",
   "Cost, latency, and quality: huge contexts are expensive and trigger &lsquo;lost in the "
   "middle&rsquo;. Retrieval is a filter that shows the model only what matters.")

h2("Embeddings & retrieval")
qa("What is an embedding?",
   "A vector representation of text where semantic similarity becomes geometric closeness, so search "
   "can match meaning instead of keywords.")
qa("Why normalize embeddings?",
   "Unit-length vectors make the dot product equal cosine similarity, so a fast inner-product index "
   "means &lsquo;most similar in meaning&rsquo; with no length bias.")
qa("Bi-encoder vs cross-encoder?",
   "Bi-encoder embeds query and doc separately (fast, first-stage retrieval); cross-encoder reads "
   "them together (slow, accurate, used to rerank the top candidates).")
qa("Exact vs approximate nearest-neighbour search?",
   "Exact compares against every vector (fine for small corpora, no approximation error); ANN (HNSW/"
   "IVF/PQ) searches a fraction for big speed-ups at a small recall cost — a scale decision.")

h2("Chunking & pipeline")
qa("How do you choose chunk size?",
   "Big enough that a chunk is one coherent idea, small enough that its embedding is specific — a "
   "paragraph is a good default; tune to the documents and add ~15% overlap so boundary facts survive.")
qa("Build-time vs query-time?",
   "Build-time embeds and indexes the whole corpus once; query-time embeds one query, retrieves, and "
   "generates. Same embedding model must be used in both.")

h2("Grounding & agents")
qa("How do you reduce hallucination in RAG?",
   "Ground the prompt (answer only from passages, cite sources, allow &lsquo;I don&rsquo;t "
   "know&rsquo;), keep top-k tight, and measure groundedness with a judge — retrieval quality is the "
   "real ceiling.")
qa("What makes a RAG system 'agentic'?",
   "The LLM decides how to retrieve — decomposing the question, retrieving iteratively, judging "
   "sufficiency, and searching again — instead of a single fixed retrieval.")
qa("When is agentic RAG worth the cost?",
   "On multi-hop / compound questions where one search misses evidence. On simple questions it just "
   "adds cost, so route by difficulty. Always weigh the measured quality gain against extra tokens.")
qa("What is ReAct?",
   "Reason + Act: an agent loop where the model interleaves reasoning with tool calls, observing each "
   "result before the next action, until it answers.")

h2("LangGraph")
qa("What problem does LangGraph solve?",
   "It turns hand-written agent control flow into a declared, drawable graph with state, nodes, and "
   "conditional edges — enabling checkpointing, streaming, parallelism, and human-in-the-loop without "
   "hiding the logic.")
qa("State, node, edge, conditional edge — define each.",
   "State: shared TypedDict of channels. Node: function returning a partial state update. Edge: fixed "
   "next step. Conditional edge: a router that reads state and returns the next node&rsquo;s name.")
qa("What is a reducer in LangGraph state?",
   "An annotation (e.g. operator.add) that tells a channel to merge updates — appending — instead of "
   "overwriting; used to accumulate retrieved documents across loop iterations.")
qa("How would you add human-in-the-loop?",
   "Compile the graph with a checkpointer and interrupt before a sensitive node; the run pauses with "
   "its state persisted, a human approves/edits, then execution resumes.")

h2("Evaluation")
qa("Why evaluate retrieval and generation separately?",
   "They fail for different reasons: bad retrieval means the model never had the evidence; bad "
   "generation means it fumbled evidence it had. One combined score hides which half to fix.")
qa("Explain hit@k, recall@k, MRR.",
   "hit@k: any relevant doc in top-k. recall@k: fraction of relevant docs retrieved. MRR: reciprocal "
   "rank of the first relevant doc — rewards ranking it high.")
qa("Pros and cons of LLM-as-judge?",
   "Cheap, consistent, scalable, and correlates with human judgement — but carries judge bias, so "
   "trust it as a relative comparator (same judge/rubric) more than as an absolute score.")

h2("Security & production")
qa("What is indirect prompt injection in RAG?",
   "A retrieved document contains adversarial instructions the model might obey. Mitigate by treating "
   "retrieved text strictly as data, sandboxing it, and never letting it override the system prompt.")
qa("How would you productionize this?",
   "Hosted vector DB with an ANN index, a reranker, response + embedding caching, a difficulty router "
   "(naive vs agentic), eval in CI on a fixed question set, observability on retrieval quality, and "
   "injection defences.")

story.append(PageBreak())

# =========================================================================== #
# 13 CHEAT SHEET
# =========================================================================== #
h1("13 &middot; One-page cheat-sheet")
h2("The pipeline")
code("ACQUIRE -> CHUNK -> EMBED -> INDEX  ||  RETRIEVE -> GENERATE")
h2("Say these numbers (this project)")
datatable([
    ["Fact", "Value"],
    ["Corpus", "2,012 arXiv papers -> 4,298 chunks"],
    ["Embeddings", "all-MiniLM-L6-v2, 384-dim, local/CPU"],
    ["Index", "FAISS IndexFlatIP (exact, cosine via normalized dot product)"],
    ["Single-hop recall@5", "1.00"],
    ["Multi-hop recall@5", "0.60  <- the gap the agent closes"],
    ["Agent loop cap", "MAX_AGENT_STEPS = 4"],
    ["Tests", "24 passed, keyless (LLM mocked)"],
], widths=[2.6 * inch, 3.9 * inch])
h2("Definitions in one line each")
bullets([
    "<b>RAG</b> — retrieve docs, put them in the prompt, answer from them.",
    "<b>Embedding</b> — text as a vector where near = related.",
    "<b>Cosine sim</b> — angle between vectors; = dot product if normalized.",
    "<b>Chunk</b> — an idea-sized, metadata-tagged slice of a document.",
    "<b>ANN</b> — approximate nearest-neighbour; speed for a little recall.",
    "<b>Grounding</b> — answer only from sources; cite; allow abstaining.",
    "<b>Agentic RAG</b> — the LLM decides how to retrieve, iteratively.",
    "<b>ReAct</b> — reason + act loop with tools.",
    "<b>LangGraph</b> — agent as a state graph: nodes, edges, conditional edges.",
    "<b>Reducer</b> — a state channel that accumulates instead of overwriting.",
    "<b>hit@k / recall@k / MRR</b> — retrieval metrics against known-relevant docs.",
    "<b>LLM-as-judge</b> — grade answers for groundedness &amp; correctness.",
])
h2("The 30-second project pitch")
p("&lsquo;I built a research assistant over 2,000 arXiv papers with a fully-local retrieval pipeline "
  "&mdash; chunking, sentence-transformer embeddings, a FAISS index &mdash; and two answer paths: a "
  "single-shot RAG baseline and an agentic version that retrieves iteratively. I measured them: "
  "single-shot retrieval gets perfect recall on single-hop questions but only 0.60 on multi-hop "
  "ones, which is exactly the gap the agent is built to close. I then rebuilt the agent in LangGraph "
  "to show the same logic as a checkpointable graph. Every number reproduces offline; the LLM calls "
  "are cached and the whole thing is test-covered without a key.&rsquo;")
story.append(Spacer(1, 10))
story.append(Paragraph("Companion to the project&rsquo;s <font name='Courier'>learning/</font> "
                       "course (11 lessons) and the LangGraph branch notebook. Code references are "
                       "live files in the repo — always be ready to open one.", SMALL))


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.9 * inch, 0.55 * inch,
                      "Agentic RAG & LangGraph — Study Guide for Data Scientist Interviews")
    canvas.drawRightString(7.6 * inch, 0.55 * inch, str(doc.page))
    canvas.setStrokeColor(RULE)
    canvas.line(0.9 * inch, 0.72 * inch, 7.6 * inch, 0.72 * inch)
    canvas.restoreState()


def cover_bg(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#fbfbfc"))
    canvas.rect(0, 0, LETTER[0], LETTER[1], fill=1, stroke=0)
    canvas.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=LETTER,
                      leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                      topMargin=0.8 * inch, bottomMargin=0.85 * inch,
                      title="Agentic RAG and LangGraph - Study Guide for Data Scientists",
                      author="Cena Shabani")
frame = Frame(doc.leftMargin, doc.bottomMargin,
              doc.width, doc.height, id="main")
doc.addPageTemplates([
    PageTemplate(id="cover", frames=[frame], onPage=cover_bg),
    PageTemplate(id="body", frames=[frame], onPage=footer),
])

doc.build(story)
print("wrote", OUT)
