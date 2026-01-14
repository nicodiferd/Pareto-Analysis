# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Pareto analysis project for Flora (an AI project manager/scrum master by Bloomfilter AI). The goal is to analyze user chat data to understand:
- Common prompt types users send to Flora
- Recurring insights across users and prompts
- Distribution of AI models used
- Main topics of consideration
- Individual user behavior patterns and personas

## Project Structure

```
app.py              - Streamlit dashboard with two main pages:
                      1. Flora Pareto Analysis (overall metrics)
                      2. Top 4 User Deep Dive (individual user analysis)
data/
  flora-chats-*.csv         - Raw and cleaned CSV data
  flora-chats-analyzed.csv  - Data with intent categorization
  top4_user_analysis.json   - Exported user profile data for dashboard
  *.png                     - Generated Pareto charts and visualizations
notebooks/
  cleaning.ipynb    - Data loading and cleaning
  analysis.ipynb    - Overall Pareto analysis and visualizations
  quatro_users.ipynb - Deep dive analysis on top 4 users (94% of messages)
```

## Data Format

The main data file (`flora-chats-01-09-26.csv`) contains columns:
- `id`, `timestamp`, `name`, `userId`, `sessionId`
- `release`, `version`, `environment`, `tags`
- `input`, `output` - Chat content (markdown formatted within CSV)

Note: The input/output columns contain markdown text which requires careful parsing when extracted from CSV format. Use Python's `csv` module with `quotechar='"', doublequote=True` for accurate parsing.

## Intent Categorization

The analysis uses a pattern-based intent categorization system that extracts ACTION + TARGET from instructions:

| Category | Pattern | Example |
|----------|---------|---------|
| Executive Reporting | `provide.*summary/report` | "provide Executive Summary" |
| Data Analysis | `analyze.*data` | "Analyze the following data..." |
| Sprint Retrospective | `sprint.*analysis/review` | "sprint retro analysis" |
| Metrics Query | `velocity/throughput/metrics` | "what is the velocity" |
| Information Request | `^tell me/what is` | "tell me about this initiative" |

This avoids naive keyword matching where "sprint" anywhere = "Sprint Planning".

## Development

This project uses uv for dependency management.

**Run notebooks:**
```bash
uv run jupyter notebook notebooks/
```

**Run Streamlit dashboard:**
```bash
uv run streamlit run app.py
```

**Workflow:**
1. Run `cleaning.ipynb` to generate `flora-chats-cleaned.csv`
2. Run `analysis.ipynb` for detailed analysis and charts (generates `flora-chats-analyzed.csv`)
3. Run `quatro_users.ipynb` to generate user profiles and charts (generates `top4_user_analysis.json`)
4. Run the Streamlit app for interactive presentation

## Top 4 User Analysis

The top 4 users account for 94% of all Flora messages. The `quatro_users.ipynb` notebook provides:

| User | Persona | Prompts | Key Characteristic |
|------|---------|---------|-------------------|
| User 1 | Power User | 33 | Diverse queries across 6 teams, sophisticated analysis |
| User 2 | Reporter | 27 | Heavy executive reporting with templated queries |
| User 3 | Sprint Prepper | 17 | Focused on sprint prep and retrospectives |
| User 4 | Explorer | 10 | New user learning through conversational iteration |

**Key Contrast (User 1 vs User 4):**
- User 1: Single-shot queries (81%), knows exactly what to ask
- User 4: Conversational (75% multi-turn), learns through iteration

## Dashboard Pages

The Streamlit app (`app.py`) has two main pages accessible via sidebar:

1. **Flora Pareto Analysis** - Overall metrics including:
   - Request type distribution (Pareto)
   - User activity distribution
   - First prompt analysis
   - Engagement depth
   - Model distribution
   - Category validation

2. **Top 4 User Deep Dive** - Individual analysis including:
   - Comparison overview with charts
   - Individual user profiles with Pareto charts
   - Combined User 1 + User 4 Pareto visualization
   - Full prompt history tables
