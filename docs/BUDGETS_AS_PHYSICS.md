# Budgets as Physics for LLM Agents: contracts, containment, and delegated work

> A short essay on why “chatbot loops” don’t scale into unattended agent systems — and why hard budgets + containment + typed outcomes become operational primitives.

## 0. The shift: from interaction to delegation

Most discourse about LLM “agents” still implicitly assumes a conversational frame: a user watches the model step-by-step, corrects it, and eventually stops it.

That frame breaks the moment you delegate.

Delegation means: you hand the system a task that can continue while you’re not watching, potentially with tool access, running over time, consuming resources, and producing side effects. At that point the LLM system is no longer “a chat” in any operational sense. It starts behaving like an economic actor: it spends tokens, time, and money on its own trajectory.

This shift changes the control problem. Prompts are persuasive. Delegation requires governance.

## 1. An agent is a process with a budget, not a prompt with a personality

If an “agent” can:

- run without constant human attention,
- call tools, write files, make network requests,
- persist state across steps,
- incur unbounded compute cost unless stopped,

then what you are operating is closer to a distributed system component (or a worker process) than a chatbot.

The core operational question becomes:

**What boundary conditions make delegation safe?**

Not in the moral sense. In the systems sense: safe to run without silently escalating cost, touching the wrong surface, or leaving ambiguous failure modes.

## 1.5 The chatbot loop trap (interface mismatch → engineering debt)

A common “agent” architecture today is a chatbot in a loop: prompt → tool call → append transcript → prompt again.

It works until it doesn’t.

The failure isn’t just cost (though cost compounds quickly). The deeper failure is an **interface mismatch**:

- The system tries to represent durable state as *textual history*.
- But the model is not actually “thinking in the transcript”; the transcript is an *externalized environment* it must repeatedly re-parse.
- Tool schemas, traces, and long histories become the substrate of cognition by accident.

That mismatch creates predictable engineering debt:

- **Model changes break workflows.** A prompt loop that “works” on one model can degrade on another because you were relying on implicit behavioral quirks (how it attends to long context, how it obeys formatting, how it prioritizes instructions).
- **Optimization becomes bottom-up patching.** You end up tuning prompts, compressing context, and hunting for smaller/cheaper models that still pass the brittle loop — instead of improving the delegation interface.
- **Cost management becomes retrospective or soft.** Dashboards tell you what happened after the run; “be concise” tries to negotiate with a process that has no enforceable boundary conditions.

The alternative is to treat the prompt as only one piece of the agent’s environment, not the agent’s memory.

As soon as you let the agent move some of its cognition into **symbolic state** (code variables, structured artifacts, explicit logs) and you bound its behavior with **hard budgets + typed terminal states + containment**, you get a scaling story that improves *with* model capability:

- Smarter models can use richer state representations (summaries, variables, intermediate artifacts) instead of replaying history.
- Your control plane (budgets/sandbox/outcomes) doesn’t need to be reinvented every time a new model behaves differently.

In other words: stop searching for prompts that survive model churn. Build boundary conditions that survive it.

A neutral way to frame this is as two different scaling strategies:

- **Scaling by model choice:** you keep searching for a smaller/cheaper model that still performs acceptably inside a brittle prompt-loop. This can work locally, but it tends to break when the task, tools, or model family changes.
- **Scaling by interface:** you move cognition into a richer environment (symbolic state, artifacts, explicit protocols), and you add hard boundary conditions (budgets, containment, typed outcomes). As models improve, they can use that substrate more effectively without requiring a full re-engineering of the workflow.

**Provisional term:** *interface scaling* — improving the agent’s substrate (symbolic state + protocols + hard boundaries) so capability gains from better models translate into reliability gains without rewriting the whole loop.

This term is only useful if it bites reality. A few ways it could be tested:
- **Model churn robustness:** swapping model families should change quality/latency, but not break the control plane (budgets/outcomes/cancel).
- **Tool growth robustness:** adding tools should not linearly explode context cost if tools are disclosed progressively and state is kept outside transcripts.
- **Graceful degradation:** longer tasks should increasingly end in typed partial outcomes (`budget_exceeded`/`timeout`) rather than silent loops.

## 2. Budgets as physics (not policy)

### A recurring failure mode (a neutral vignette)

A pattern that shows up in practice:

- You get something working as a “chatbot loop”: prompt → tool → transcript → prompt.
- It feels stable—until you swap models, add a tool, or run tasks longer than a few minutes.
- Then the loop degrades: costs creep up, behaviors shift, and the system becomes a patchwork of prompt fixes and context compression.

The core issue isn’t that models are “bad”. It’s that the interface is doing too much: the transcript is simultaneously memory, state store, audit log, and control plane.

Budgets-as-physics is one way to re-separate concerns: the control plane becomes enforceable boundary conditions rather than emergent behavior.

In most stacks today, “budgets” are soft:

- warnings,
- best-effort throttles,
- “please be concise”,
- spending dashboards that tell you what happened after the fact.

Those are policy. Policies are negotiable.

Delegation needs physics.

**Budgets-as-physics** means: the system must be able to hard-stop when it hits a limit, in a way that is:

- deterministic (not “hopefully the model stops”),
- machine-readable (so downstream can handle it),
- treated as a normal terminal state (not a crash).

Three budgets matter because they capture almost all real-world failure modes:

- **tokens** (bounded output / bounded state growth),
- **time** (bounded wall-clock exposure and latency),
- **money** (bounded economic downside).

When you can’t enforce hard budgets, you haven’t actually delegated. You’ve merely opened a channel through which resource consumption can grow unbounded.

This is why budgets belong in the “safety” bucket as much as in the “cost control” bucket: they bound agency in the most universal units we have.

*(Related: long context is not the same as effective reasoning over long context. Provider context window numbers are often marketing ceilings, not stable reasoning floors. See “Evaluating Long Context (Reasoning) Ability”: https://nrehiew.github.io/blog/long_context/ )*

## 3. Containment is the real safety boundary

*(Epistemic status: we point at an important direction here, but we’re not claiming novelty or comprehensive security expertise. Think of this as a signpost, not a spec.)*

If delegation is about governance, then containment is about capability.

A large fraction of “agent risk” comes from the mismatch between:

- a non-deterministic planner (the model),
- and an extremely deterministic capability surface (filesystem, network, credentials).

Prompts are porous. Sandboxes are enforceable.

In practice, an “ideal sandbox” is one that lets the agent do **symbolic work cheaply** (code/variables/artifacts as state) while being **safe by default** (least privilege, explicit capabilities, default-deny).

There is a lot of active work here (containers, seccomp, microVMs, capability systems), and it’s easy to overclaim. A useful stance is to treat sandboxing as an *open engineering frontier* rather than a solved checkbox.

Once a system can act on the world, safety is not mainly “the model behaves nicely.” It’s whether the process is **boxed**: least privilege, explicit capabilities, default-deny.

The minimal containment questions look like security engineering questions:

- Can this run touch the network? If yes, which domains?
- Can it read the filesystem? If yes, which paths?
- Can it execute code? If yes, under what syscall / resource restrictions?
- Can it access secrets? If yes, how do we prevent them from being serialized into an execution environment?

Containment is not solved by vibes. It is solved by guardrails that the model cannot argue with.

## 4. Typed terminal states are epistemics

A big part of why agent systems fail in practice is not that they fail—it’s that they fail ambiguously.

“Something went wrong” is poisonous in delegation, because ambiguity propagates. You can’t build robust downstream behavior on top of ambiguous outcomes.

Delegation needs a small set of **typed terminal states** such as:

- `success`
- `budget_exceeded` (partial result may exist)
- `timeout`
- `cancelled`
- `provider_error`
- `tool_error`
- `verification_failed`

This is an epistemic hygiene move: it forces you to decide what “done” means, and it makes “stop” a first-class concept.

Importantly: **`budget_exceeded` should not be treated like an exception.** In delegated systems, budget hits are a normal outcome. Treat them like a normal outcome, and your workflows become predictable.

## 5. Job protocols beat chat loops

A minimal “agent contract” can be written down as data, not vibes. For example:

```yaml
job:
  task: "Research X and produce a report"
  budgets:
    tokens_out: 1200
    wall_seconds: 120
    cost_usd: 0.25
  capabilities:
    network:
      enabled: true
      allow_domains: ["api.github.com", "arxiv.org"]
    filesystem:
      read_roots: ["./data"]
      write_roots: ["./artifacts"]
    code_execution:
      enabled: true
      isolation: "container"   # or "subprocess" / "none"
  outcomes:
    terminal: ["success", "budget_exceeded", "timeout", "cancelled", "tool_error"]
```

You can argue about the schema, but the point is: delegation becomes governable when these fields exist.

A “chat loop” is optimized for continuous interaction, not delegation.

Delegation wants a job protocol:

- `submit` → returns immediately
- `status` → observable progress
- `cancel` → revocation is possible
- (optionally) `partial` → salvage intermediate artifacts

This turns the agent from a chat partner into an operator-friendly process.

It also reduces the temptation to store state in transcripts, which is one of the main drivers of token explosion: turn count × tool schemas × verbose traces.

## 6. Where enzu fits (lightly, but clearly)

One concrete implementation of the “budgets-as-physics + typed outcomes” layer is **enzu** (OSS): a Python toolkit for budgeted LLM runs with hard caps (tokens/time/$) and explicit outcome states (e.g., `budget_exceeded`, `timeout`). It also supports job-style delegation (submit/status/cancel) for long runs.

If you only remember one thing: **enzu treats budgets as a first-class contract and “budget hit” as a typed outcome** rather than an exception.

Demos (for inspection, not marketing):

- Hard stop: https://github.com/teilomillet/enzu/blob/main/examples/budget_hardstop_demo.py  
- Typed outcomes: https://github.com/teilomillet/enzu/blob/main/examples/typed_outcomes_demo.py  
- Job delegation: https://github.com/teilomillet/enzu/blob/main/examples/job_delegation_demo.py  

The point of mentioning it here is not “use this.” It’s that the philosophy can be implemented: budgets can be treated as physics and “stop” can become a typed state.

## 7. What is not solved (and should be treated as open)

A few things are easy to overclaim. Don’t.

- **Containers are not perfect security.** They are a pragmatic containment tool, not a proof. Kernel escapes exist; supply chain risk exists; side channels exist.
- **Budgeting doesn’t align goals.** It bounds damage. It is a safety primitive, not a value primitive.
- **Typed outcomes don’t guarantee truth.** They make failure legible; they don’t certify correctness.

So the open problem is not “add a sandbox and declare victory.” The open problem is: what minimal set of primitives makes delegation safe enough to be normal?

My current candidate set:

1) hard budgets (tokens/time/$)
2) typed terminal states
3) least-privilege containment (capabilities, default-deny)
4) job protocol (submit/status/cancel)
5) audit trail (what happened, what was touched, what was spent)

## 8. Falsifiable predictions (so this isn’t just vibes)

1) **Agent reliability winter**: systems that rely on prompt discipline without hard budgets will hit a wall as soon as they scale to unattended runs. The dominant failure will be runaway cost and ambiguous stop conditions, not “model quality.”
2) **Capability tokens become standard**: tool access will evolve toward explicit, scoped capabilities (time-limited, path-limited, domain-limited), because “the model shouldn’t do X” will not hold under pressure.
3) **Job protocols will outcompete chat loops** in serious deployments, because cancellation + observability + typed outcomes matter more than conversational continuity.

## 9. Invitation to critique (especially security-minded)

If you’re building or reviewing agent systems, the useful questions are not “does it feel aligned?” but:

- What are your hard boundaries (tokens/time/$)?
- What is your containment model (FS/network/exec/secrets)?
- What are your terminal states and how does downstream handle them?
- Can you cancel work deterministically?
- Can you audit what happened?

If you think budgets-as-physics is the wrong primitive, what would you replace it with?  
If you think containers are the wrong containment layer, what would you use (microVMs, capability OS, something else)?  
If you think typed outcomes are incomplete, what is the minimal outcome algebra you’d want?

---

## Closing: what “agent ops” is converging toward

If LLMs remain primarily interactive, we can keep arguing about prompts. But if LLMs become *delegated workers*—processes that run while you’re not watching—then a different stack becomes necessary.

The claim of this post is narrow and (hopefully) testable:

- **Budgets** (tokens/time/$) should be treated as enforceable boundary conditions.
- **Containment** should be treated as a capability system, not a vibe.
- **Typed terminal states** should be treated as epistemic hygiene.
- **Job protocols** (submit/status/cancel) should be treated as the default interface for long-horizon work.

This doesn’t solve alignment. It doesn’t solve security. But it changes the shape of failure: from silent drift and unbounded spend to bounded, legible outcomes you can route and audit.

If you build agent systems: try writing your agent contract down as data. If you do security: treat agent sandboxes as a new, high-leverage target—and tell us what we’re missing.
