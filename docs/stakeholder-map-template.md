# Stakeholder Map Template
Use this to track key stakeholders (users, developers, product managers) and their needs.

---

# ADR 0001: Stakeholder Map & Use Case Definition

## 1. Project Context & Objective

- **Project Name:** Enterprise Knowledge Assistant (RAG) — Basecamp Employee Handbook
- **Dataset:** 37signals / Basecamp Employee Handbook (Markdown corpus covering benefits, PTO, insurance, retirement, sabbatical, titles, and HR policies)
- **Objective:** Build a RAG agent that grounds internal HR Q&A in the verified Basecamp handbook, replacing fragmented manual searches and eliminating policy hallucinations.

---

## 2. Stakeholder Map

| Stakeholder | Role | Primary Need | Key Concern |
|---|---|---|---|
| HR Specialists | Primary User | Instant, cited answers to recurring policy questions | Accuracy — no unverified policy generation |
| Customer Support Ops | Primary User | Fast lookup of employee-facing policies during live interactions | Speed — sub-30s resolution |
| Head of People Operations | Executive Sponsor | Reduced query resolution time across the department | Compliance risk from hallucinated answers |
| Director of IT Support | Executive Sponsor | Secure, auditable system with no data leakage | Data privacy and access control |
| New Employees (Onboarding) | Secondary User | Self-service answers to handbook questions without bothering HR | Completeness — all handbook sections covered |
| Capstone Developer | Builder | Clean, evaluable RAG pipeline with measurable KPIs | Faithfulness and latency benchmarks met |

---

## 3. Core Stakeholder Profiles

### ① Primary User — HR Specialist & Customer Support Ops

- **Daily Pain Point:** Sifting through scattered handbook sections, PDF exports, and Slack threads to answer recurring questions about benefits, PTO, sabbatical eligibility, and insurance (avg. 8–12 min per search).
- **Representative Questions from Dataset:**
  - *"How many vacation days do I get and do they roll over?"*
  - *"When does my health insurance start if I'm a new hire?"*
  - *"What is the 401K match policy?"*
  - *"Am I eligible for a sabbatical after 2 years?"*
- **Target Interaction:** Type a natural-language question into a chat UI → receive an immediate answer with inline citations pointing to the exact handbook section.

### ② Executive Sponsors — Head of People Ops & Director of IT Support

- **Core Concerns:**
  - Zero hallucinated policy statements (compliance liability)
  - No leakage of employee PII or confidential compensation data
  - Auditable answer provenance (every answer traceable to a source chunk)
- **Success Gate:** System must refuse to answer when the handbook does not contain the information, rather than generating an ungrounded response.

### ③ Before (As-Is Workflow)

1. Employee or HR rep receives a policy question via Slack or email.
2. Manually keyword-searches Basecamp, Google Drive, or a downloaded PDF copy.
3. Cross-references multiple handbook sections — risks using an outdated version.
4. Composes a reply from memory or copy-paste — no citation, no audit trail.
5. Average resolution time: **8–12 minutes**.

### ④ After (To-Be Workflow)

1. User types question into the RAG chat interface.
2. Query is embedded → top-k handbook chunks retrieved from ChromaDB.
3. GPT-4 generates a grounded answer using only retrieved context.
4. Response is returned with inline source citations (section title + chunk text).
5. Average resolution time: **< 30 seconds**.

---

## 4. Top 3 Success Metrics

| # | KPI | Target | Rationale |
|---|-----|--------|-----------|
| 1 | **Faithfulness Score** | ≥ 95% adherence to retrieved handbook chunks | Prevents compliance risk from hallucinated policy statements |
| 2 | **Time to Answer** | Lookup time reduced from ~10 min → < 30 sec | Core productivity gain for HR and support teams |
| 3 | **Adoption Rate** | > 70% Monthly Active Users (MAU) in target dept within 60 days | Validates that the tool solves a real, recurring workflow pain |

---

## 5. Out of Scope

- Answering questions not covered by the Basecamp handbook corpus (system must decline gracefully)
- Real-time handbook updates / live sync with source-of-truth CMS (post-MVP)
- Multi-tenant access control per employee role (post-MVP)
