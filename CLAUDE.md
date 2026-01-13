# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Pareto analysis project for Flora (an AI project manager/scrum master by Bloomfilter AI). The goal is to analyze user chat data to understand:
- Common prompt types users send to Flora
- Recurring insights across users and prompts
- Distribution of AI models used
- Main topics of consideration

## Project Structure

```
app.py          - Streamlit dashboard for presenting analysis findings
data/           - Raw and cleaned CSV data
notebooks/
  cleaning.ipynb  - Data loading and cleaning
  analysis.ipynb  - Pareto analysis and visualizations
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
2. Run `analysis.ipynb` for detailed analysis and charts
3. Run the Streamlit app for interactive presentation
