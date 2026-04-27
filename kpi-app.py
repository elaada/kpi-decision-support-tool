import streamlit as st
import pandas as pd
import io
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


st.set_page_config(page_title="KPI Decision Support Tool V2", layout="wide")

st.title("KPI Decision Support Tool V2")
st.write("Upload fragmented KPI files by category and build a unified KPI view.")

# -----------------------------
# SESSION MEMORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "column_mappings" not in st.session_state:
    st.session_state["column_mappings"] = {}

if "raw_dataframes" not in st.session_state:
    st.session_state["raw_dataframes"] = {}

if "mapping_ready" not in st.session_state:
    st.session_state["mapping_ready"] = False

if "final_kpi_df" not in st.session_state:
    st.session_state["final_kpi_df"] = None

if "metrics" not in st.session_state:
    st.session_state["metrics"] = {}

# -----------------------------
# HELPERS
# -----------------------------
def read_uploaded_csvs(files):
    dfs = []

    for file in files:
        try:
            raw_text = file.getvalue().decode("utf-8")

            cleaned_lines = []
            for line in raw_text.splitlines():
                cleaned_line = line.strip()
                cleaned_line = cleaned_line.strip(";")
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)

            cleaned_text = "\n".join(cleaned_lines)

            try:
                df = pd.read_csv(
                    io.StringIO(cleaned_text),
                    sep=None,
                    engine="python"
                )
            except Exception:
                df = pd.read_csv(
                    io.StringIO(cleaned_text),
                    sep=";",
                    engine="python"
                )

            if len(df.columns) == 1:
                df = pd.read_csv(
                    io.StringIO(cleaned_text),
                    sep=";",
                    engine="python"
                )

            df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", na=False)]

            df.columns = (
                df.columns.astype(str)
                .str.strip()
                .str.replace(";", "", regex=False)
            )

            df = df.loc[:, df.columns != ""]

            df["source_file"] = file.name
            dfs.append(df)

        except Exception as e:
            st.error(f"Could not read file {file.name}: {e}")

    return dfs


def combine_files(files):
    dfs = read_uploaded_csvs(files)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


# -----------------------------
# COLUMN MAPPING
# -----------------------------
SCHEMA_BY_TYPE = {
    "revenue": ["date", "invoice_amount"],
    "churn": ["date", "customer_id"],
    "active": ["activity_date", "customer_id"]
}


def suggest_mapping(columns, required_fields):
    keyword_map = {
        "date": [
            "date", "month", "period", "tarih", "ay",
            "report date", "invoice date", "billing date"
        ],
        "activity_date": [
            "activity_date", "activity date", "usage date",
            "event date", "activity", "usage", "date"
        ],
        "invoice_amount": [
            "invoice_amount", "invoice amount", "revenue", "amount",
            "sales", "ciro", "gelir", "net revenue", "mrr"
        ],
        "customer_id": [
            "customer_id", "customer id", "client_id", "client id",
            "account_id", "account id", "user_id", "user id",
            "customer", "client", "account", "user"
        ]
    }

    suggestions = {}
    lower_columns = {col: str(col).lower().strip() for col in columns}

    for field in required_fields:
        suggestions[field] = None
        for original_col, lower_col in lower_columns.items():
            if any(keyword in lower_col for keyword in keyword_map.get(field, [])):
                suggestions[field] = original_col
                break

    return suggestions


def render_mapping_ui(label, df, required_fields):
    st.write(f"### {label} column mapping")
    st.dataframe(df.head(), use_container_width=True)

    suggestions = suggest_mapping(df.columns.tolist(), required_fields)
    options = ["-- Select --"] + df.columns.tolist()

    mapping = {}
    saved_mapping = st.session_state["column_mappings"].get(label, {})

    for field in required_fields:
        suggested_value = suggestions.get(field)
        default_index = options.index(suggested_value) if suggested_value in options else 0

        if saved_mapping.get(field) in options:
            default_index = options.index(saved_mapping[field])

        mapping[field] = st.selectbox(
            f"Map '{field}'",
            options=options,
            index=default_index,
            key=f"{label}_{field}"
        )

    return mapping


def validate_mapping(mapping):
    selected = [v for v in mapping.values() if v != "-- Select --"]

    if len(selected) != len(mapping):
        return False, "Please map all required fields."

    if len(selected) != len(set(selected)):
        return False, "Duplicate column mapping detected."

    return True, "OK"


def apply_mapping(df, mapping):
    rename_dict = {source: target for target, source in mapping.items()}
    return df.rename(columns=rename_dict).copy()


# -----------------------------
# KPI BUILDERS
# -----------------------------
def build_revenue_kpi(revenue_raw_df):
    required_cols = ["date", "invoice_amount"]

    for col in required_cols:
        if col not in revenue_raw_df.columns:
            st.error(f"Revenue files must contain '{col}' column.")
            return None

    df = revenue_raw_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["invoice_amount"] = pd.to_numeric(df["invoice_amount"], errors="coerce")
    df = df.dropna(subset=["date", "invoice_amount"])

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    revenue_kpi_df = (
        df.groupby("month", as_index=False)["invoice_amount"]
        .sum()
        .rename(columns={"invoice_amount": "revenue"})
        .sort_values("month")
    )

    return revenue_kpi_df


def build_churn_kpi(churn_raw_df):
    required_cols = ["date", "customer_id"]

    for col in required_cols:
        if col not in churn_raw_df.columns:
            st.error(f"Churn files must contain '{col}' column.")
            return None

    df = churn_raw_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["customer_id"] = df["customer_id"].astype(str)
    df = df.dropna(subset=["date", "customer_id"])

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly_customers = (
        df.groupby("month")["customer_id"]
        .apply(set)
        .reset_index()
        .sort_values("month")
    )

    churn_data = []

    for i in range(1, len(monthly_customers)):
        current_month = monthly_customers.iloc[i]["month"]
        prev_customers = monthly_customers.iloc[i - 1]["customer_id"]
        current_customers = monthly_customers.iloc[i]["customer_id"]

        churned = prev_customers - current_customers
        churn_rate = len(churned) / len(prev_customers) if len(prev_customers) > 0 else 0

        churn_data.append({
            "month": current_month,
            "churn": churn_rate,
            "churned_customers": len(churned),
            "previous_customers": len(prev_customers)
        })

    churn_kpi_df = pd.DataFrame(churn_data)
    return churn_kpi_df


def build_active_customers_kpi(active_customers_raw_df):
    required_cols = ["customer_id", "activity_date"]

    for col in required_cols:
        if col not in active_customers_raw_df.columns:
            st.error(f"Active customer files must contain '{col}' column.")
            return None

    df = active_customers_raw_df.copy()
    df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
    df["customer_id"] = df["customer_id"].astype(str)
    df = df.dropna(subset=["activity_date", "customer_id"])

    df["month"] = df["activity_date"].dt.to_period("M").dt.to_timestamp()

    active_customers_kpi_df = (
        df.groupby("month")["customer_id"]
        .nunique()
        .reset_index(name="active_customers")
        .sort_values("month")
    )

    return active_customers_kpi_df


def merge_kpi_tables(revenue_kpi_df=None, churn_kpi_df=None, active_customers_kpi_df=None):
    final_df = None

    if revenue_kpi_df is not None:
        final_df = revenue_kpi_df.copy()

    if churn_kpi_df is not None:
        if final_df is None:
            final_df = churn_kpi_df.copy()
        else:
            final_df = final_df.merge(churn_kpi_df, on="month", how="outer")

    if active_customers_kpi_df is not None:
        if final_df is None:
            final_df = active_customers_kpi_df.copy()
        else:
            final_df = final_df.merge(active_customers_kpi_df, on="month", how="outer")

    if final_df is not None:
        final_df = final_df.sort_values("month")

    return final_df


def compute_metrics(final_kpi_df):
    metrics = {}

    if final_kpi_df is None or final_kpi_df.empty:
        return metrics

    df = final_kpi_df.copy().sort_values("month")

    # Revenue metrics
    if "revenue" in df.columns and df["revenue"].notna().sum() >= 2:
        revenue_df = df.dropna(subset=["revenue"]).sort_values("month")
        first_revenue = revenue_df["revenue"].iloc[0]
        last_revenue = revenue_df["revenue"].iloc[-1]
        previous_revenue = revenue_df["revenue"].iloc[-2]

        metrics["average_revenue"] = revenue_df["revenue"].mean()
        metrics["overall_revenue_change"] = last_revenue - first_revenue
        metrics["overall_revenue_change_percentage"] = (last_revenue / first_revenue) - 1 if first_revenue != 0 else None
        metrics["last_revenue_change_percentage"] = (last_revenue / previous_revenue) - 1 if previous_revenue != 0 else None
        metrics["revenue_down_alert"] = (last_revenue - first_revenue) < 0
        metrics["sharp_last_drop_alert"] = (
            metrics["last_revenue_change_percentage"] is not None
            and metrics["last_revenue_change_percentage"] < -0.05
        )
    elif "revenue" in df.columns and df["revenue"].notna().sum() == 1:
        revenue_df = df.dropna(subset=["revenue"]).sort_values("month")
        metrics["average_revenue"] = revenue_df["revenue"].mean()

    # Churn metrics
    if "churn" in df.columns and df["churn"].notna().sum() >= 2:
        churn_df = df.dropna(subset=["churn"]).sort_values("month")
        average_churn = churn_df["churn"].mean()
        last_churn = churn_df["churn"].iloc[-1]
        previous_churn = churn_df["churn"].iloc[-2]

        metrics["average_churn"] = average_churn
        metrics["last_churn"] = last_churn
        metrics["last_churn_change"] = last_churn - previous_churn
        metrics["high_churn_alert"] = average_churn > 0.05
        metrics["high_last_churn_alert"] = last_churn > 0.05
        metrics["spike_churn_alert"] = (last_churn - previous_churn) > 0.02
    elif "churn" in df.columns and df["churn"].notna().sum() == 1:
        churn_df = df.dropna(subset=["churn"]).sort_values("month")
        average_churn = churn_df["churn"].mean()
        last_churn = churn_df["churn"].iloc[-1]
        metrics["average_churn"] = average_churn
        metrics["last_churn"] = last_churn
        metrics["high_churn_alert"] = average_churn > 0.05
        metrics["high_last_churn_alert"] = last_churn > 0.05

    # Active customer metrics
    if "active_customers" in df.columns and df["active_customers"].notna().sum() >= 2:
        active_df = df.dropna(subset=["active_customers"]).sort_values("month")
        average_active_customers = active_df["active_customers"].mean()
        last_active_customers = active_df["active_customers"].iloc[-1]
        previous_active_customers = active_df["active_customers"].iloc[-2]

        metrics["average_active_customers"] = average_active_customers
        metrics["last_active_customers"] = last_active_customers
        metrics["active_customers_change_percentage"] = (
            (last_active_customers / previous_active_customers) - 1
            if previous_active_customers != 0
            else None
        )
        metrics["active_customers_down_alert"] = (
            metrics["active_customers_change_percentage"] is not None
            and metrics["active_customers_change_percentage"] < -0.05
        )
    elif "active_customers" in df.columns and df["active_customers"].notna().sum() == 1:
        active_df = df.dropna(subset=["active_customers"]).sort_values("month")
        metrics["average_active_customers"] = active_df["active_customers"].mean()
        metrics["last_active_customers"] = active_df["active_customers"].iloc[-1]

    return metrics

def ask_ai(question, final_kpi_df, metrics):

    context = {
        "metrics": metrics,
        "recent_data": final_kpi_df.tail(6).to_dict(),
        "columns": list(final_kpi_df.columns)
    }

    prompt = f"""

You are a senior product and business analyst.

Analyze KPI trends carefully.

Context:
{context}

Question:
{question}

Rules:
- Only use the provided data
- Do NOT make generic assumptions
- Be specific and data-driven

Answer format:
1. Insight (1-2 sentences)
2. Reason
3. Risk level (low/medium/high)


Instructions:
- Give a clear insight
- Explain reasoning
- Highlight risks if any
- Be concise but smart

"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI error: {e}"

def extract_month_from_question(question, available_months):
    question = question.lower().strip()

    month_name_map = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12
    }

    for word, month_num in month_name_map.items():
        if word in question:
            matches = [m for m in available_months if pd.Timestamp(m).month == month_num]
            if matches:
                return pd.Timestamp(matches[-1])

    for token in question.replace("/", "-").split():
        try:
            parsed = pd.to_datetime(token, format="%Y-%m", errors="raise")
            matches = [
                m for m in available_months
                if pd.Timestamp(m).year == parsed.year and pd.Timestamp(m).month == parsed.month
            ]
            if matches:
                return pd.Timestamp(matches[0])
        except Exception:
            pass

    return None


def get_last_two_months(final_kpi_df):
    months = sorted(final_kpi_df["month"].dropna().unique())
    if len(months) >= 2:
        return pd.Timestamp(months[-2]), pd.Timestamp(months[-1])
    return None, None

def detect_metric_from_question(question):
    question = question.lower().strip()

    if "revenue" in question or "sales" in question:
        return "revenue"

    if "churn" in question:
        return "churn"

    if (
        "active customers" in question
        or "active customer" in question
        or "active users" in question
        or "users" in question
        or "customers" in question
        or "usage" in question
        or "active" in question
    ):
        return "active_customers"

    return None


def get_metric_value_for_month(final_kpi_df, metric_name, question):
    if final_kpi_df is None or final_kpi_df.empty:
        return None, None

    if "month" not in final_kpi_df.columns or metric_name not in final_kpi_df.columns:
        return None, None

    available_months = final_kpi_df["month"].dropna().unique()
    target_month = extract_month_from_question(question, available_months)

    if target_month is None:
        return None, None

    row = final_kpi_df[final_kpi_df["month"] == target_month]

    if row.empty:
        return target_month, None

    value = row[metric_name].iloc[0]

    if pd.isna(value):
        return target_month, None

    return target_month, value


def compare_metric_last_two_months(final_kpi_df, metric_name):
    if final_kpi_df is None or final_kpi_df.empty:
        return None

    if "month" not in final_kpi_df.columns or metric_name not in final_kpi_df.columns:
        return None

    df = final_kpi_df[["month", metric_name]].dropna().sort_values("month")

    if len(df) < 2:
        return None

    prev_row = df.iloc[-2]
    last_row = df.iloc[-1]

    prev_month = pd.Timestamp(prev_row["month"])
    last_month = pd.Timestamp(last_row["month"])
    prev_val = prev_row[metric_name]
    last_val = last_row[metric_name]

    change = last_val - prev_val

    pct_change = None
    if prev_val != 0:
        pct_change = (last_val / prev_val) - 1

    return {
        "prev_month": prev_month,
        "last_month": last_month,
        "prev_val": prev_val,
        "last_val": last_val,
        "change": change,
        "pct_change": pct_change
    }


def format_metric_value(metric_name, value):
    if metric_name == "churn":
        return f"{value:.2%}"

    if metric_name == "revenue":
        return f"{value:,.0f}"

    if metric_name == "active_customers":
        return f"{value:,.0f}"

    return str(value)


def format_metric_label(metric_name):
    labels = {
        "revenue": "Revenue",
        "churn": "Churn",
        "active_customers": "Active customers"
    }
    return labels.get(metric_name, metric_name)

def answer_question(question: str, final_kpi_df, metrics):
    question = question.lower().strip()
    response = ""
    
   
    has_revenue = "average_revenue" in metrics
    has_churn = "average_churn" in metrics
    has_active = "average_active_customers" in metrics
    
  
    metric_name = detect_metric_from_question(question)

  
    if metric_name:
        target_month, metric_value = get_metric_value_for_month(final_kpi_df, metric_name, question)
        if target_month is not None:
            label = format_metric_label(metric_name)
            if metric_value is not None:
                return f"📅 **{label} for {target_month.strftime('%B %Y')}**: {format_metric_value(metric_name, metric_value)}"
            else:
                return f"ℹ️ I found the date **{target_month.strftime('%B %Y')}**, but there is no data for **{label}** in that period."

  
    if any(k in question for k in ["summary", "overall", "general", "going on", "risk", "alert", "emergency"]):
        is_risk_query = any(k in question for k in ["risk", "alert", "emergency"])
        
        if is_risk_query:
            response = "⚠️ **Risk Assessment**\n\n"
            alerts = []
            if metrics.get("high_churn_alert"): alerts.append("- **High Churn**: Average churn is above 5%.")
            if metrics.get("revenue_down_alert"): alerts.append("- **Revenue Drop**: Overall revenue trend is negative.")
            if metrics.get("spike_churn_alert"): alerts.append("- **Churn Spike**: Significant increase in churn last month.")
            if metrics.get("active_customers_down_alert"): alerts.append("- **User Loss**: Active customers dropped by >5% recently.")
            
            response += "\n".join(alerts) if alerts else "✅ No major alerts detected in the current data."
            return response
        else:
           
            response = "📊 **KPI Executive Summary**\n\n"
            if has_revenue:
                direction = "📈 increasing" if metrics.get("overall_revenue_change", 0) > 0 else "📉 decreasing"
                response += f"- **Revenue:** The trend is {direction}.\n"
            if has_churn:
                status = "🔴 high" if metrics.get("high_churn_alert") else "🟢 stable"
                response += f"- **Churn:** Currently {status} (Avg: {metrics['average_churn']:.2%}).\n"
            if has_active:
                response += f"- **Users:** {metrics['last_active_customers']:,.0f} active customers last period.\n"
            
            return response

    if metric_name and any(k in question for k in ["trend", "compare", "performance", "vs", "mom", "month on month", "loosing", "losing", "dropping", "growing", "growth"]):
        comparison = compare_metric_last_two_months(final_kpi_df, metric_name)
        if comparison:
            label = format_metric_label(metric_name)
            change_val = format_metric_value(metric_name, comparison['change'])
            pct_text = f" ({comparison['pct_change']:.2%})" if comparison['pct_change'] is not None else ""
            
            return (f"🔄 **{label} Comparison**\n\n"
                    f"- {comparison['prev_month'].strftime('%B %Y')}: {format_metric_value(metric_name, comparison['prev_val'])}\n"
                    f"- {comparison['last_month'].strftime('%B %Y')}: {format_metric_value(metric_name, comparison['last_val'])}\n"
                    f"- **Change:** {change_val}{pct_text}")


    if metric_name == "churn" and has_churn:
        return f"📉 **Churn Stats:** Average is {metrics['average_churn']:.2%}, with the last period at {metrics['last_churn']:.2%}."
    
    if metric_name == "revenue" and has_revenue:
        return f"💰 **Revenue Stats:** Average monthly revenue is {metrics['average_revenue']:,.0f}."
    
    if metric_name == "active_customers" and has_active:
        return f"👥 **User Stats:** Average active customers: {metrics['average_active_customers']:,.0f}."

   
    return ask_ai(question, final_kpi_df, metrics)


# -----------------------------
# UPLOAD SECTION
# -----------------------------
st.subheader("Upload KPI files")

revenue_files = st.file_uploader(
    "Upload revenue files (max 12)",
    type=["csv"],
    accept_multiple_files=True,
    key="revenue_files"
)

churn_files = st.file_uploader(
    "Upload churn files (max 12)",
    type=["csv"],
    accept_multiple_files=True,
    key="churn_files"
)

active_customer_files = st.file_uploader(
    "Upload active customer files (max 12)",
    type=["csv"],
    accept_multiple_files=True,
    key="active_customer_files"
)

# -----------------------------
# VALIDATION
# -----------------------------
revenue_count = len(revenue_files) if revenue_files else 0
churn_count = len(churn_files) if churn_files else 0
active_customer_count = len(active_customer_files) if active_customer_files else 0

col1, col2, col3 = st.columns(3)
col1.metric("Revenue files", revenue_count)
col2.metric("Churn files", churn_count)
col3.metric("Active customer files", active_customer_count)

validation_ok = True

if revenue_count > 12:
    st.error("You can upload maximum 12 revenue files.")
    validation_ok = False

if churn_count > 12:
    st.error("You can upload maximum 12 churn files.")
    validation_ok = False

if active_customer_count > 12:
    st.error("You can upload maximum 12 active customer files.")
    validation_ok = False

# -----------------------------
# PREPARE FILES
# -----------------------------
st.divider()

if st.button("Prepare Files"):
    if not validation_ok:
        st.stop()

    if revenue_count == 0 and churn_count == 0 and active_customer_count == 0:
        st.warning("Please upload at least one file.")
        st.stop()

    st.session_state["raw_dataframes"] = {}
    st.session_state["final_kpi_df"] = None
    st.session_state["metrics"] = {}
    st.session_state["messages"] = []

    if revenue_count > 0:
        revenue_raw_df = combine_files(revenue_files)
        if revenue_raw_df is not None:
            st.session_state["raw_dataframes"]["revenue"] = revenue_raw_df

    if churn_count > 0:
        churn_raw_df = combine_files(churn_files)
        if churn_raw_df is not None:
            st.session_state["raw_dataframes"]["churn"] = churn_raw_df

    if active_customer_count > 0:
        active_customers_raw_df = combine_files(active_customer_files)
        if active_customers_raw_df is not None:
            st.session_state["raw_dataframes"]["active"] = active_customers_raw_df

    st.session_state["mapping_ready"] = True

# -----------------------------
# MAPPING UI
# -----------------------------
if st.session_state["mapping_ready"]:
    st.subheader("Column Mapping")

    if "revenue" in st.session_state["raw_dataframes"]:
        revenue_raw_df = st.session_state["raw_dataframes"]["revenue"]
        revenue_mapping = render_mapping_ui(
            "Revenue",
            revenue_raw_df,
            SCHEMA_BY_TYPE["revenue"]
        )
        st.session_state["column_mappings"]["revenue"] = revenue_mapping

    if "churn" in st.session_state["raw_dataframes"]:
        churn_raw_df = st.session_state["raw_dataframes"]["churn"]
        churn_mapping = render_mapping_ui(
            "Churn",
            churn_raw_df,
            SCHEMA_BY_TYPE["churn"]
        )
        st.session_state["column_mappings"]["churn"] = churn_mapping

    if "active" in st.session_state["raw_dataframes"]:
        active_raw_df = st.session_state["raw_dataframes"]["active"]
        active_mapping = render_mapping_ui(
            "Active Customers",
            active_raw_df,
            SCHEMA_BY_TYPE["active"]
        )
        st.session_state["column_mappings"]["active"] = active_mapping

# -----------------------------
# BUILD KPI VIEW
# -----------------------------
if st.session_state["mapping_ready"] and st.button("Build KPI View"):
    revenue_kpi_df = None
    churn_kpi_df = None
    active_customers_kpi_df = None

    # Revenue
    if "revenue" in st.session_state["raw_dataframes"]:
        revenue_raw_df = st.session_state["raw_dataframes"]["revenue"].copy()
        revenue_mapping = st.session_state["column_mappings"].get("revenue", {})

        valid, msg = validate_mapping(revenue_mapping)
        if not valid:
            st.warning(f"Revenue mapping error: {msg}")
            st.stop()

        revenue_raw_df = apply_mapping(revenue_raw_df, revenue_mapping)

        st.subheader("Revenue pipeline")
        st.write("Combined revenue raw data")
        st.dataframe(revenue_raw_df, use_container_width=True)

        revenue_kpi_df = build_revenue_kpi(revenue_raw_df)

        if revenue_kpi_df is not None:
            st.success("Revenue KPI table created")
            st.dataframe(revenue_kpi_df, use_container_width=True)

    else:
        st.info("No revenue files uploaded.")

    # Churn
    if "churn" in st.session_state["raw_dataframes"]:
        churn_raw_df = st.session_state["raw_dataframes"]["churn"].copy()
        churn_mapping = st.session_state["column_mappings"].get("churn", {})

        valid, msg = validate_mapping(churn_mapping)
        if not valid:
            st.warning(f"Churn mapping error: {msg}")
            st.stop()

        churn_raw_df = apply_mapping(churn_raw_df, churn_mapping)

        st.subheader("Churn pipeline")
        st.write("Combined churn raw data")
        st.dataframe(churn_raw_df, use_container_width=True)

        churn_kpi_df = build_churn_kpi(churn_raw_df)

        if churn_kpi_df is not None:
            st.success("Churn KPI table created")
            st.dataframe(churn_kpi_df, use_container_width=True)

    else:
        st.info("No churn files uploaded.")

    # Active
    if "active" in st.session_state["raw_dataframes"]:
        active_raw_df = st.session_state["raw_dataframes"]["active"].copy()
        active_mapping = st.session_state["column_mappings"].get("active", {})

        valid, msg = validate_mapping(active_mapping)
        if not valid:
            st.warning(f"Active customers mapping error: {msg}")
            st.stop()

        active_raw_df = apply_mapping(active_raw_df, active_mapping)

        st.subheader("Active customers pipeline")
        st.write("Combined active customer raw data")
        st.dataframe(active_raw_df, use_container_width=True)

        active_customers_kpi_df = build_active_customers_kpi(active_raw_df)

        if active_customers_kpi_df is not None:
            st.success("Active customers KPI table created")
            st.dataframe(active_customers_kpi_df, use_container_width=True)

    else:
        st.info("No active customer files uploaded.")

    final_kpi_df = merge_kpi_tables(
        revenue_kpi_df=revenue_kpi_df,
        churn_kpi_df=churn_kpi_df,
        active_customers_kpi_df=active_customers_kpi_df
    )

    if final_kpi_df is not None:
        st.subheader("Merged KPI View")
        st.dataframe(final_kpi_df, use_container_width=True)

        metrics = compute_metrics(final_kpi_df)
        st.session_state["final_kpi_df"] = final_kpi_df
        st.session_state["metrics"] = metrics

# -----------------------------
# RESTORE BUILT DATA FROM SESSION
# -----------------------------
final_kpi_df = st.session_state.get("final_kpi_df")
metrics = st.session_state.get("metrics", {})

# -----------------------------
# KPI CARDS
# -----------------------------
if final_kpi_df is not None and metrics:
    st.divider()
    st.subheader("Current KPI Snapshot")

    col1, col2, col3 = st.columns(3)

    if "average_revenue" in metrics:
        col1.metric("Average Revenue", f"{metrics['average_revenue']:,.0f}")
    else:
        col1.metric("Average Revenue", "N/A")

    if "average_churn" in metrics:
        col2.metric("Average Churn", f"{metrics['average_churn']:.2%}")
    else:
        col2.metric("Average Churn", "N/A")

    if "last_active_customers" in metrics:
        col3.metric("Last Active Customers", f"{metrics['last_active_customers']:,.0f}")
    else:
        col3.metric("Last Active Customers", "N/A")

# -----------------------------
# CHAT SECTION
# -----------------------------
if final_kpi_df is not None:
    st.divider()
    st.subheader("Ask about the KPIs")

    st.info("AI insights require a valid OpenAI API key with available credits.")

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Type your question",
            placeholder="Examples: general, risk, churn, revenue trend, active customers"
        )
        submitted = st.form_submit_button("Send")

    if submitted and question:
        st.session_state["messages"].append(("user", question))
        response = answer_question(question, final_kpi_df, metrics)
        st.session_state["messages"].append(("bot", response))

    for role, msg in st.session_state["messages"]:
        if role == "user":
            st.markdown(f"**🧑‍💼 You:** {msg}")
        else:
            st.markdown(f"**🤖 Assistant:**\n\n{msg}")