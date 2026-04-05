# KPI Decision Support Tool

An AI-powered KPI analysis tool that transforms fragmented CSV data into structured insights and actionable decisions.

## What it does

This app allows you to:

- Upload multiple KPI files (revenue, churn, active customers)
- Combine and normalize data from different sources
- Map different column structures dynamically
- Generate monthly KPI metrics
- Merge KPIs into a unified table
- Ask questions in natural language
- Get data-driven insights and risk analysis

## Key Features

- Multi-file ingestion
- Dynamic column mapping
- Automated KPI generation
- Unified KPI view
- Query-based analysis
- AI-assisted insights

## Tech Stack

- Python
- Streamlit
- Pandas
- OpenAI API

## Project Structure

```
kpi-app/
│
├── kpi-app.py
├── requirements.txt
├── README.md
│
├── revenue.csv
├── churn.csv
├── activecustomers_jan.csv
├── activecustomers_feb.csv
├── activecustomers_mar.csv
│
└── .streamlit/
    └── secrets.toml (local only)
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the following file:

```
.streamlit/secrets.toml
```

Add your API key:

```toml
OPENAI_API_KEY = "your-api-key"
```

## Run the app

```bash
streamlit run kpi-app.py
```

## API Key Requirement

To use AI-powered insights, you need to provide your own OpenAI API key.

You can generate a key here:
https://platform.openai.com/api-keys

Without an API key, the app will still work for KPI calculations and basic analysis, but AI-based responses will be disabled.

## Deployment

The app can be deployed using Streamlit Community Cloud.

Make sure to:
- Add OPENAI_API_KEY in Streamlit Cloud secrets
- Keep secrets.toml out of version control

## Example Questions

- revenue trend
- churn in february
- active customers january
- are we going down
- what should we do

## Version

v1.0