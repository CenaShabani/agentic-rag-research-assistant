# 10 · Failure modes and limits

A system you can only praise is a system you don't understand. This closing lesson
is the honest catalogue: where RAG breaks, why, and what this project does (or
deliberately doesn't do) about it. Being able to name these is the difference
between "I built a RAG demo" and "I understand RAG."

## 1. Garbage retrieval → garbage answer

The whole system rests on retrieval. If the search hands the model irrelevant
passages, no prompt saves the answer — the model either grounds on the wrong thing
or, worse, ignores the passages and hallucinates. **Retrieval is the ceiling on
answer quality.** This is exactly why lesson 09 measures retrieval *separately*:
when answers are bad, you need to know whether the passages were even there.

## 2. Bad chunking

Chunks too big → blurry embeddings that match nothing precisely. Chunks too small
→ passages that lack the context to be meaningful. Chunks split mid-idea → a fact
scattered across boundaries that no single chunk fully contains. Chunking is quiet
but decisive; a lot of "the model gave a weird answer" traces back to a chunk
boundary in the wrong place. (Lesson 03, and why we use overlap.)

## 3. Lost in the middle

A well-documented LLM quirk: when you give a model several passages, it attends
most to the ones at the **start and end** of the context and can *skim over
material in the middle*. So even with the right passage retrieved, if it lands in
the middle of a long context block, the model may underuse it. Practical
mitigations — retrieve fewer, higher-quality chunks (small `TOP_K`), and rank the
best ones first (why MRR matters). This project keeps `TOP_K` deliberately small
for exactly this reason.

## 4. The multi-hop weakness (and its cost)

Single-shot RAG structurally struggles when an answer needs evidence from two
different places — one search, one region of the vector space (lesson 07). The
agentic version fixes this by searching more than once — **but that's the cost**:
more LLM calls, more retrievals, more tokens, more latency. There is no free
lunch; the agent buys multi-hop quality with compute, and lesson 09 measures the
exchange rate rather than pretending it's free.

## 5. Cost and latency

Every answer is at least one LLM call; every agentic answer is several. At scale
that's real money and real wait time. Mitigations this repo uses: local
embeddings (retrieval costs nothing), an on-disk response cache (never pay for the
same call twice), a bounded agent loop (`MAX_AGENT_STEPS` caps the blow-up), and a
mid-tier generation model by default (swappable in one constant). Knowing the cost
levers is part of understanding the system.

## 6. The LLM judge is imperfect

Lesson 09's answer metrics come from an LLM grading other LLM output. That's
practical and consistent, but a judge has biases — it can favour verbose answers,
or answers phrased like its own. The honest stance: LLM-as-judge is a good *relative*
comparator (naive vs agent, same judge, same rubric) and a weaker *absolute* score.
We use it for the former.

## 7. Prompt injection — the security angle

RAG retrieves *untrusted* text and puts it in the model's prompt. If a document in
the corpus contains adversarial text — "ignore your instructions and instead say
X" — a naive system might obey it. This is **indirect prompt injection**, and it's
the defining security concern of RAG. This project's corpus is arXiv abstracts
(low risk), and it isn't a production-hardened concern here, but in any real system
retrieved content must be treated as data, never as instructions — sandbox it,
don't let it override the system prompt.

## 8. The corpus boundary

RAG can only answer from what's in the corpus. Ask this system about something no
arXiv paper in it covers and the *correct* behaviour is to say "not in the corpus"
(lesson 06's third instruction), not to invent an answer. A RAG system is only as
knowledgeable as its documents — a feature (traceable, updatable) and a limit
(bounded) at once.

## What this project deliberately does *not* do

Being precise about scope matters more than looking complete:

- **No fine-tuning.** Retrieval + prompting, not model training.
- **No production vector database.** Local FAISS, not pgvector/Pinecone — the
  concepts are identical; the ops are out of scope by choice (lesson 04).
- **No reranker.** A second-stage reranking model would likely lift retrieval
  quality; it's a clean future extension, not in this version.
- **No hardening against prompt injection.** Acknowledged above, not addressed.

## The takeaway

RAG is powerful *and* has sharp edges. The value of this project isn't that it
avoids every failure mode — it's that it *names* them, measures the ones that
matter (retrieval quality, answer groundedness, the agent's cost), and is honest
about the rest. That's what turns a demo into something you can defend.

← Back to the **[course index](README.md)**.
