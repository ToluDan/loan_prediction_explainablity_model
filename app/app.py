"""
Cost-sensitive, explainable credit default prediction — demonstration app.

My Final-year dissertation, BSc Data Science, Miva Open University.
Model: CatBoost, calibrated with Platt scaling, decision threshold derived from
the cost matrix (Elkan, 2001).

Run with:  streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Default Risk", page_icon="•", layout="wide")


# ----------------------------------------------------------------- loading
@st.cache_resource
def load_bundle():
    return joblib.load("model_bundle.joblib")


@st.cache_resource
def load_explainer(_raw_model):
    import shap
    return shap.TreeExplainer(_raw_model)


B = load_bundle()
P_STAR = B["p_star"]
FEATURES = B["feature_order"]
CAT = B["cat_cols"]
NUM = B["num_cols"]
LEVELS = B["cat_levels"]
STATS = B["num_stats"]


def default_of(col, lo=None, hi=None):
    """Starting value for a numeric field, clamped to the widget's bounds."""
    v = None
    if st.session_state.get("example"):
        v = st.session_state["example"]["values"].get(col)
    v = float(STATS[col]["median"]) if v is None else float(v)
    if lo is not None:
        v = max(v, lo)
    if hi is not None:
        v = min(v, hi)
    return v


# ----------------------------------------------------------------- customer lookup
LOOKUP_FILE = "customer_lookup.csv"


@st.cache_data
def load_lookup():
    """Engineered features for test-set customers, keyed by customer ID."""
    try:
        df = pd.read_csv(LOOKUP_FILE, dtype={"customerid": str})
    except FileNotFoundError:
        return None
    df["customerid"] = df["customerid"].str.strip()
    return df.set_index("customerid")


def lookup_customer(cid):
    """Return the customer's stored record in the same shape as a preset example."""
    df = load_lookup()
    if df is None or cid not in df.index:
        return None
    rec = df.loc[cid]
    if isinstance(rec, pd.DataFrame):          # duplicate IDs: take the latest row
        rec = rec.iloc[-1]
    values = {k: (None if pd.isna(v) else v) for k, v in rec.items()}
    actual = values.pop("actual", None)
    return {"label": f"Customer {cid}", "values": values,
            "actual": actual, "source": "lookup"}


def history_locked():
    """Repayment history is read-only when it came from the lender's records."""
    ex = st.session_state.get("example")
    return bool(ex and ex.get("source") == "lookup")


def cat_default(col):
    if st.session_state.get("example"):
        v = st.session_state["example"]["values"].get(col)
        if v in LEVELS[col]:
            return LEVELS[col].index(v)
    return 0


def rate_default():
    """Starting interest rate (%): the loaded example's own margin, else the median."""
    if st.session_state.get("example"):
        v = st.session_state["example"]["values"].get("interest_margin")
        if v is not None:
            return float(v) * 100
    return float(B["m_median"]) * 100


# ----------------------------------------------------------------- header
st.title("Credit default risk assessment")
st.caption(
    f"{B['model_name']} · calibrated probabilities · "
    f"cost-derived decision threshold p* = {P_STAR:.4f}"
)

with st.expander("How this works"):
    st.markdown(
        f"""
The model estimates the probability that a loan will **not be settled on time**.

Rather than the conventional 0.5 cut-off, the decision uses a threshold derived
from the relative cost of the two possible errors (Elkan, 2001):

$$p^* = \\frac{{m}}{{m + \\text{{LGD}}}} = \\frac{{{B['m_median']:.4f}}}{{{B['m_median']:.4f} + {B['lgd']:.2f}}} = {P_STAR:.4f}$$

where *m* is the interest margin on the loan and LGD the proportion of principal
lost when a loan goes bad. Approving a loan that defaults risks the principal;
rejecting one that would have repaid loses only the interest. Because the first
error costs roughly {B['lgd'] / B['m_median']:.1f} times the second, the
threshold sits well below 0.5.

The third output lists the features that moved this particular prediction
furthest from the model's average, computed with SHAP (Lundberg & Lee, 2017).
        """
    )

# ----------------------------------------------------------------- lookup UI
st.sidebar.header("Existing customer")
if load_lookup() is None:
    st.sidebar.caption(f"Customer lookup unavailable: `{LOOKUP_FILE}` not found.")
else:
    cid = st.sidebar.text_input("Customer ID", key="cid",
                                placeholder="Paste a customer ID from the test set")
    b1, b2 = st.sidebar.columns(2)
    go = b1.button("Look up", use_container_width=True)
    rnd = b2.button("Random", use_container_width=True,
                    help="Load a random customer from the test set")
    if rnd:
        found = lookup_customer(str(load_lookup().sample(1).index[0]))
        st.session_state["example"] = found
        st.rerun()
    if go:
        clean = cid.strip().strip('"').strip("'")
        if not clean:
            st.sidebar.warning("Enter a customer ID first.")
        else:
            found = lookup_customer(clean)
            if found:
                st.session_state["example"] = found
                st.rerun()
            else:
                st.sidebar.error(f"No customer with ID `{clean}` in the records.")
    st.sidebar.caption(
        "Repayment history is computed from the lender's loan records, "
        "so it is filled in and locked when a customer is looked up."
    )
st.sidebar.divider()

# ----------------------------------------------------------------- examples
st.sidebar.header("Example borrowers")
st.sidebar.caption("Load a real case from the test set to populate the form.")
for i, ex in enumerate(B["examples"]):
    if st.sidebar.button(ex["label"], key=f"ex{i}", use_container_width=True):
        st.session_state["example"] = ex
        st.rerun()

if st.session_state.get("example"):
    ex = st.session_state["example"]
    outcome = ("" if ex.get("actual") is None else
               f"\n\nActual outcome: {'defaulted' if int(ex['actual']) else 'repaid'}")
    st.sidebar.success(f"Loaded: {ex['label']}{outcome}")
    if st.sidebar.button("Clear", use_container_width=True):
        del st.session_state["example"]
        st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    f"Test-set ROC-AUC {B['test_auc']:.3f}\n\n"
    "Demonstration only. Not for real lending decisions."
)

# ----------------------------------------------------------------- loan details
# Kept outside the form so total due recalculates live as the inputs change
# (widgets inside st.form only update the page on submit).
with st.container(border=True):
    st.subheader("Loan details")
    c1, c2, c3 = st.columns(3)
    loanamount = c1.number_input(
        "Loan amount (₦)", min_value=1000.0, max_value=200000.0,
        value=default_of("loanamount", 1000.0, 200000.0), step=1000.0)
    rate_pct = c2.number_input(
        "Interest rate (%)", min_value=0.0, max_value=100.0,
        value=rate_default(), step=0.5,
        help="Flat interest charged on the principal for the whole term. "
             "Total due is calculated from this.")
    termdays = c3.number_input(
        "Term (days)", min_value=1.0, max_value=365.0,
        value=default_of("termdays", 1.0, 365.0), step=1.0)

    margin = rate_pct / 100
    totaldue = loanamount * (1 + margin)
    st.caption(
        f"Total due: **₦{totaldue:,.2f}** · "
        f"interest margin **{margin:.4f}** ({margin*100:.2f}%)"
    )

# ----------------------------------------------------------------- the form
with st.form("application"):
    locked = history_locked()
    st.subheader("Repayment history")
    if locked:
        st.caption("From the lender's loan records for this customer (read-only).")
    else:
        st.caption("For a real application these come from the lender's records; "
                   "here you can edit them to explore scenarios.")
    h1, h2, h3, h4 = st.columns(4)
    prev_loan_count = h1.number_input(
        "Prior loans", min_value=0.0, max_value=50.0,
        value=default_of("prev_loan_count", 0.0, 50.0), step=1.0,
        disabled=locked)
    prev_late_rate = h2.number_input(
        "Share repaid late", min_value=0.0, max_value=1.0,
        value=default_of("prev_late_rate", 0.0, 1.0), step=0.01,
        disabled=locked,
        help="Proportion of prior loans whose first repayment fell after the due date.")
    prev_days_late_mean = h3.number_input(
        "Mean days late", min_value=-60.0, max_value=400.0,
        value=default_of("prev_days_late_mean", -60.0, 400.0), step=1.0,
        disabled=locked,
        help="Negative means repaid early on average.")
    days_since_last_loan = h4.number_input(
        "Days since last loan", min_value=0.0, max_value=1000.0,
        value=default_of("days_since_last_loan", 0.0, 1000.0), step=1.0,
        disabled=locked)

    h5, h6, h7 = st.columns(3)
    prev_amount_mean = h5.number_input(
        "Mean prior amount (₦)", min_value=0.0, max_value=200000.0,
        value=default_of("prev_amount_mean", 0.0, 200000.0), step=1000.0,
        disabled=locked)
    prev_amount_max = h6.number_input(
        "Largest prior amount (₦)", min_value=0.0, max_value=200000.0,
        value=default_of("prev_amount_max", 0.0, 200000.0), step=1000.0,
        disabled=locked)
    prev_termdays_mean = h7.number_input(
        "Mean prior term (days)", min_value=0.0, max_value=365.0,
        value=default_of("prev_termdays_mean", 0.0, 365.0), step=1.0,
        disabled=locked)

    st.subheader("Borrower")
    d1, d2, d3 = st.columns(3)
    age = d1.number_input(
        "Age at application", min_value=18.0, max_value=80.0,
        value=default_of("age_at_approval", 18.0, 80.0), step=1.0)
    bank_account_type = d2.selectbox(
        "Bank account type", LEVELS["bank_account_type"],
        index=cat_default("bank_account_type"))
    region = d3.selectbox(
        "Region", LEVELS["region"], index=cat_default("region"))

    d4, d5, d6 = st.columns(3)
    bank_name_clients = d4.selectbox(
        "Bank", LEVELS["bank_name_clients"],
        index=cat_default("bank_name_clients"))
    employment_status_clients = d5.selectbox(
        "Employment status", LEVELS["employment_status_clients"],
        index=cat_default("employment_status_clients"))
    level_of_education_clients = d6.selectbox(
        "Education", LEVELS["level_of_education_clients"],
        index=cat_default("level_of_education_clients"))

    was_referred = st.checkbox(
        "Referred by an existing customer",
        value=bool(default_of("was_referred")))

    submitted = st.form_submit_button("Assess application", type="primary",
                                      use_container_width=True)

# ----------------------------------------------------------------- predict
if submitted:
    values = {
        "loanamount": loanamount,
        "termdays": termdays,
        "interest_margin": margin,
        "age_at_approval": age,
        "prev_loan_count": prev_loan_count,
        "prev_late_rate": prev_late_rate,
        "prev_days_late_mean": prev_days_late_mean,
        "prev_amount_mean": prev_amount_mean,
        "prev_amount_max": prev_amount_max,
        "prev_termdays_mean": prev_termdays_mean,
        "days_since_last_loan": days_since_last_loan,
        "was_referred": int(was_referred),
        # derived indicators, matching Section 3.5.1 and 3.6.1
        "has_history": int(prev_loan_count > 0),
        "has_demographics": int(bank_account_type != "Unknown"),
        "bank_account_type": bank_account_type,
        "bank_name_clients": bank_name_clients,
        "employment_status_clients": employment_status_clients,
        "level_of_education_clients": level_of_education_clients,
        "region": region,
    }

    row = pd.DataFrame([[values[c] for c in FEATURES]], columns=FEATURES)
    for c in CAT:
        row[c] = row[c].astype(str)
    for c in NUM:
        row[c] = row[c].astype(float)

    p = float(B["model"].predict_proba(row)[:, 1][0])
    reject = p >= P_STAR

    st.divider()
    r1, r2 = st.columns([1, 1])

    with r1:
        st.metric("Probability of late settlement", f"{p:.1%}")
        st.progress(min(p / 0.6, 1.0))
        st.caption(f"Decision threshold p\\* = {P_STAR:.1%}")

    with r2:
        if reject:
            st.error(f"### Reject\n\nEstimated risk {p:.1%} is at or above the "
                     f"threshold of {P_STAR:.1%}.")
        else:
            st.success(f"### Approve\n\nEstimated risk {p:.1%} is below the "
                       f"threshold of {P_STAR:.1%}.")

    # ------------------------------------------------------------ SHAP
    st.subheader("Why this decision")
    try:
        explainer = load_explainer(B["raw_model"])
        sv = explainer.shap_values(row)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        sv = sv[0]

        contrib = (pd.DataFrame({"Feature": FEATURES,
                                 "Contribution": sv,
                                 "Value": [values[c] for c in FEATURES]})
                   .assign(abs_c=lambda d: d["Contribution"].abs())
                   .sort_values("abs_c", ascending=False)
                   .head(3)
                   .drop(columns="abs_c")
                   .reset_index(drop=True))

        for _, r in contrib.iterrows():
            direction = "increases" if r["Contribution"] > 0 else "reduces"
            val = (f"{r['Value']:.4g}" if isinstance(r["Value"], float)
                   else r["Value"])
            st.markdown(
                f"**{r['Feature']}** = `{val}` — {direction} estimated risk "
                f"({r['Contribution']:+.3f} in log-odds)"
            )

        with st.expander("All feature contributions"):
            full = (pd.DataFrame({"Feature": FEATURES,
                                  "Contribution": sv,
                                  "Value": [values[c] for c in FEATURES]})
                    .assign(abs_c=lambda d: d["Contribution"].abs())
                    .sort_values("abs_c", ascending=False)
                    .drop(columns="abs_c"))
            st.dataframe(full, use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"Explanation unavailable: {e}")

    st.caption(
        "Contributions are SHAP values in log-odds units. They describe how the "
        "model uses each feature for this application; they do not establish a "
        "causal relationship with repayment behaviour."
    )