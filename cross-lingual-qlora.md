# Cross-Lingual QLoRA Fine-Tuning: Hindi → English

> Preliminary experimental observation. These are hypotheses to investigate,
> not guaranteed benefits of training in a non-English language.

## One-line summary

Can fine-tuning a multilingual model on a small, hand-curated **Hindi** dataset
shape its **English** behavior — while *reducing* the usual small-scale
fine-tuning failures (parroting, repetition, stuttering, overfitting)? Early
results suggest maybe. This document is an honest record of that question, the
setup, and what still needs to be proven.

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

Initial experiments suggest that fine-tuning with a small Hindi dataset can
influence English-language generation, despite the training data being written in
a different language.

Compared with the English fine-tuning approaches tested, the Hindi-based
approach appeared to produce desirable behavioral changes with **less
parroting, repetition, and stuttering**.

This suggests that cross-lingual transfer may offer an alternative approach to
shaping model behavior without requiring large English-language training
datasets.

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

**These are hypotheses to investigate, not guaranteed benefits.**

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

Multiple runs and datasets are needed to establish whether the observed
differences are reproducible and whether they result from the **training
language** rather than other experimental variables.

## Current Status

Preliminary experimental observation.

The current results motivate further investigation into cross-lingual QLoRA as a
practical method for local model experimentation. The long-term goal is to
develop reproducible, resource-efficient fine-tuning techniques that allow
developers to obtain useful behavioral changes from small, carefully curated
datasets.

---

*Companion to [The Hoberman Lens](./README.md). See "Over-tuning — when the
sphere stops flowing" there for the conceptual framework this experiment probes.*
