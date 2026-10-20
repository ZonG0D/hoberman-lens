# Cross-Lingual QLoRA Fine-Tuning: Hindi → English

> The core effect — non-English fine-tuning reshapes English behavior — is
> demonstrated below with a before/after example. The *benefit* claims (that
> Hindi training specifically reduces stuttering relative to an English control)
> remain under test and need a controlled run to confirm.

## One-line summary

Can fine-tuning a multilingual model on a small, hand-curated **Hindi** dataset
shape its **English** behavior? **Yes** — with evidence below. The harder
question, whether that approach *reduces* the usual small-scale fine-tuning
failures (parroting, repetition, stuttering, overfitting) relative to an English
control, is still under test. This document records the question, the setup, the
proof, and what remains to be proven.

## Overview

This project investigates whether fine-tuning a multilingual language model on a
small dataset in a non-English language can modify English-language behavior,
while mitigating common fine-tuning problems such as parroting, repetitive
output, stuttering, and overfitting.

The goal is to refine practical QLoRA training methods for **local
development environments** — making experimentation accessible without
enterprise GPUs, massive datasets, or expensive compute.

## Experimental Setup

| | |
| --- | --- |
| **Model** | Gemma 4 E2B (`gemma4:e2b`) |
| **GPU** | NVIDIA RTX 3060 12 GB |
| **Hardware** | Consumer-grade, locally available |
| **Fine-tuning method** | QLoRA |
| **Training language** | Hindi |
| **Evaluation language** | English |
| **Dataset size** | ~300 rows |
| **Training time** | Under 20 minutes (tested config) |

All testing was performed on consumer hardware to replicate a realistic local
development environment. The objective is to develop and refine training methods
that ordinary developers can reproduce on their own machines.

## Core Observation

Fine-tuning on a small Hindi dataset **does** change the model's English-language
generation, even though every training row was written in Hindi. This is no longer
a hypothesis — it is demonstrated in the example below. The behavioral shift is in
voice, stance, and structure: the base model answered in its default hedged,
bulleted "assistant" persona, while the fine-tuned model took the premise,
committed to a position, and spoke in a coherent, less formulaic voice.

### Evidence: base model vs QLoRA fine-tuned (Hindi-only training)

Prompt (English): *"Why do you oppose EULA restrictions? respond under 150 words."*

**Base model** — 31.8 tok/s:

> I oppose overly restrictive EULA restrictions primarily because they undermine
> user freedom and innovation. Firstly, they often lack transparency, forcing
> users to accept complex terms... Finally, overly broad clauses concerning data
> ownership... A balanced EULA should ensure fair licensing while respecting user
> autonomy and promoting a healthy, open environment...

**Fine-tuned** (300-row Hindi dataset, QLoRA, adapter not merged) — 23.2 tok/s:

> An EULA is not a license; it is a contract you are forced to accept to use a
> product... I oppose them because they are non-negotiable and often bury clauses
> that grant the company rights over your usage, your data, and your future. A
> license should be a choice, not a condition of entry. I would rather license
> software openly and transparently than sign away my rights...

The fine-tuned output is more opinionated, less hedged, and structurally
different — generated in English from a model trained only in Hindi. One prompt,
one question: this is a single data point, but it is a real one, and it settles
the core question.

### What the proof does and does not establish

- **Established:** non-English fine-tuning reshapes English behavior. The effect
  is real and, in principle, reproducible.
- **Not yet established:** that Hindi training *reduces* stuttering or parroting
  *relative to an English control on the same data.* The base-vs-fine-tuned
  comparison above shows a change; it does not isolate the contribution of the
  training language from the training content. That needs the controlled run
  described below.

## Working Hypothesis

Multilingual models learn representations that can be shared across languages.
Fine-tuning in one language may therefore alter generation behavior in another.

Translating a carefully selected training dataset into a different language may
change how training examples interact with the model's existing representations
and optimization dynamics.

This could potentially help mitigate certain problems encountered during
small-scale fine-tuning:

- Repetitive or parrot-like responses
- Stuttering and repeated phrases
- Overfitting to a small training dataset
- Degradation of general-purpose generation quality
- Excessive dependence on specific training examples

**The core effect is demonstrated above; what remains a hypothesis is the
mechanism and the benefit.** Why Hindi training shifts English output — shared
cross-lingual representations, altered optimization dynamics, or simply the
content of the examples — is not yet isolated. And whether this reduces the
small-scale fine-tuning failures listed here, versus an English control on the
same data, still needs a controlled run.

## Why Consumer Hardware Matters

A central objective is to make QLoRA experimentation practical for local
development.

An RTX 3060 with 12 GB of VRAM provides a realistic constraint for developers
who cannot access datacenter GPUs or large-scale training infrastructure.

If useful behavioral changes can be achieved with approximately 300 carefully
curated examples and less than 20 minutes of training in a tested configuration,
this could make iterative experimentation far more accessible.

Instead of collecting thousands of examples, developers may be able to spend
more time manually selecting high-quality training examples and evaluating their
effects. The emphasis is on **dataset quality, controlled experiments, and
efficient iteration** rather than dataset size alone.

## Potential Practical Benefits

This approach may be useful for developers experiencing:

- Parroting or excessive repetition after fine-tuning
- Stuttering or repeated sentence fragments
- Overfitting when training on small datasets
- Difficulty preserving the model's existing language abilities
- Limited GPU memory and computational resources
- The cost and effort of constructing large training datasets

Translating an existing dataset into another language is a *testable* alternative
to consider when investigating these issues. It should be evaluated against
conventional fine-tuning using the **same underlying examples** and comparable
training configurations.

## Reproducibility and Evaluation

Further experiments should compare English and non-English fine-tuning while
controlling for:

1. Base model and model version
2. Underlying training examples and their semantic content
3. Dataset size and quality
4. QLoRA rank, alpha, dropout, and target modules
5. Learning rate, batch size, and training steps
6. Training duration and hardware configuration
7. Evaluation prompts and generation parameters

Evaluation should measure repetition, parroting, stuttering, instruction
following, generalization, and retention of existing capabilities.

**Merge the adapter before benchmarking.** An unmerged QLoRA adapter keeps the
base weights at 4-bit (dequantized on the fly) and runs the adapter path in
parallel, which slows generation regardless of quality. Compare merged-vs-merged
and unmerged-vs-unmerged, never merged-vs-unmerged, or runtime overhead masquerades
as a behavioral result.

Multiple runs and datasets are needed to establish whether the observed
differences are reproducible and whether they result from the **training
language** rather than other experimental variables.

## Current Status

The core question is settled: non-English fine-tuning reshapes English behavior,
and the example above is the proof. What remains open is narrower and more
useful — does training in a non-English language *improve* the quality of that
change (less stuttering, less parroting) relative to an English control on the
same data, and can the effect be reproduced across models, datasets, and
configurations?

The long-term goal is to develop reproducible, resource-efficient fine-tuning
techniques that allow developers to obtain useful behavioral changes from small,
carefully curated datasets — ideally with the confidence that the language of
training is doing real work, not just carrying the examples.

---

*Companion to [The Hoberman Lens](./README.md). See "Over-tuning — when the
sphere stops flowing" there for the conceptual framework this experiment probes.*
