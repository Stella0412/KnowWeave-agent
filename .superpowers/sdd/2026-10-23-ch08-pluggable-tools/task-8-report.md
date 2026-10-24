# Task 8 implementation evidence

Base 949197d, head 9af1e78. Draft confirmation, deterministic ticket identity and idempotent writer implemented. Agent old interception deliberately remains until Task 9 wiring as required by plan.

RED: tests/test_ch08_ticket_confirmation.py and tests/test_ch08_ticket_idempotency.py failed collection because ticket_confirmation module was missing.
GREEN: same files 6 passed. Combined with tests/test_ch08_agent_tools.py and tests/test_ch08_executor.py: 36 passed in 8.13s.
Command used existing locked environment: C:/Users/20143/.codex/worktrees/ch08-tools/knowweave/.venv/Scripts/python.exe -m pytest -q tests/test_ch08_ticket_confirmation.py tests/test_ch08_ticket_idempotency.py tests/test_ch08_agent_tools.py tests/test_ch08_executor.py (cwd D:/knowweave).

Limitations: MySQL concurrency not verified; cancellation and interrupt tests use patched interrupt; independent review not yet performed. Task 7 live model evaluation blocked on Connection error. Main checkout dependencies unavailable due mirror DNS; existing interpreter requires escalation for native extension loading.

## Review fix report (2026-10-24, base 9af1e78)

Implemented P1: protected transaction now includes session.add, Conversation lookup (which can autoflush), status mutation and commit. IntegrityError rolls back before a fresh session reads the winning ticket. Identical content returns its stable number; changed content raises TicketDraftConflict; absent winning ticket re-raises the original IntegrityError, preserving its DBAPI cause.

Implemented P2: file-backed SQLite regression deterministically commits a competing Ticket from an independent real session after the writer's absent lookup, before its autoflush. Parameterized identical and changed contents both traverse an actual unique constraint violation. Existing gather test remains a same-process serialization test, not evidence of a DB collision. WeakValueDictionary retains per-ticket locks only while active callers hold them, with a GC lifecycle regression.

Confirmation coverage now additionally runs an actual compiled LangGraph with InMemorySaver, interrupt and Command resume through ToolEngine. Confirm writes once; cancel and mismatched ID write zero times and emit permission_denied. Initial interrupt emits neither writes nor terminal audit; each resume emits one audit. Draft/context identity mismatch fails before interrupt; model arguments cannot supply trusted request evidence; blank description/call ID and nonpositive request ID are rejected. No Task 9 production wiring was changed.

TDD RED command: `& C:/Users/20143/.codex/worktrees/ch08-tools/knowweave/.venv/Scripts/python.exe -m pytest -q tests/test_ch08_ticket_idempotency.py` (cwd D:/knowweave, escalated locked interpreter). Output: `2 failed, 3 passed in 4.57s`. Both new collision cases failed with `sqlalchemy.exc.IntegrityError (raised as a result of Query-invoked autoflush) UNIQUE constraint failed: tickets.ticket_no`, proving the original unprotected flush defect.

GREEN command: same interpreter, `-m pytest -q tests/test_ch08_ticket_confirmation.py tests/test_ch08_ticket_idempotency.py`. Output: `16 passed in 2.51s`. Added original-error and lock-lifecycle regressions afterward. Expanded confirmation/idempotency/executor/agent-tools/authorization run: `1 failed, 80 passed in 9.41s`; sole failure was premature weakref reclamation assertion while prior exception tracebacks remained reachable. Added explicit gc.collect to test eventual weakref cleanup.

Context7 checked SQLAlchemy query autoflush / IntegrityError rollback and LangGraph interrupt replay / Command resume docs. Relevant sources: https://docs.sqlalchemy.org/en/20/orm/session_transaction.html and https://docs.langchain.com/oss/python/langgraph/interrupts . Recovery uses rollback plus a fresh read transaction; confirmation has no pre-interrupt side effects.

Files changed: app/tools/ticket_confirmation.py; tests/test_ch08_ticket_confirmation.py; tests/test_ch08_ticket_idempotency.py; this report. Self-review: read implementation diff, verified all flush-capable operations are protected, recovery retains the original error, concurrent lock users retain strong references, no Task 9 wiring edits. git diff --check passes (only repository CRLF conversion notices).

Remaining mandatory verification: Task 11 must exercise live MySQL concurrency with independent sessions. SQLite collision coverage here does not fulfill that deployment-backend requirement. Backend HTTP resume decision verification remains Task 9; test Command inputs represent its trusted output.

Final GREEN full-suite command: `& C:/Users/20143/.codex/worktrees/ch08-tools/knowweave/.venv/Scripts/python.exe -m pytest -q` (cwd D:/knowweave, escalated). Output: `699 passed in 61.68s (0:01:01)`; no warnings or failures. This includes the 18 confirmation/idempotency tests and all adjacent authorization/executor regressions.
