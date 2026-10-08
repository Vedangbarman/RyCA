import inspect
import json
import math
import os
import re

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000")
# Where the company profile is stored. Point your backend at the same file.
PROFILE_PATH = os.getenv("PROFILE_PATH", "profile.json")
CHAT_TIMEOUT = 1800        # a chat message waits behind any running notification batch
DEFAULT_NAME = "gatorade"
STARTERS = [
    "What is the Upper Layer in NBFCs?",
    "Which master directions apply to all NBFCs?",
    "What are the large exposure limits for IDF-NBFCs?",
]
CHAT_VIEW, NOTES_VIEW, COMPANY_VIEW = "Chat", "Notifications", "Company details"

st.set_page_config(page_title="RBI Compliance Assistant", layout="wide")

# full-width buttons: the argument was renamed between Streamlit versions
STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.button).parameters
           else {"use_container_width": True})

ss = st.session_state
ss.setdefault("chats", {})       # chat name -> [{"role", "content"}]
ss.setdefault("active", None)    # name of the open chat
ss.setdefault("naming", True)    # True = show the "name your chat" form
ss.setdefault("loaded", False)   # saved chats fetched from the backend yet?
ss.setdefault("pending", None)   # question from a starter button


# ---------- chat state ----------

def safe_name(name):
    """Same rule as the backend. The chat name becomes a folder name."""
    return re.sub(r"[^\w\- ]", "", name).strip()


def load_saved_chats():
    try:
        saved = requests.get(f"{API}/chats", timeout=10).json()["chats"]
    except (requests.RequestException, KeyError, ValueError):
        return                   # backend not up yet, try again on the next rerun
    ss.chats = {c["name"]: c["messages"] for c in saved}
    ss.active = saved[0]["name"] if saved else None
    ss.naming = not saved
    ss.loaded = True


def create_chat(name=""):
    base = safe_name(name) or DEFAULT_NAME
    taken = {c.lower() for c in ss.chats}
    name, n = base, 2
    while name.lower() in taken:     # same name = same folder, so number it
        name = f"{base} {n}"
        n += 1
    ss.chats[name] = []
    ss.active = name
    ss.naming = False


def open_chat(name):
    ss.active = name
    ss.naming = False


def start_naming():
    ss.naming = True


def use_starter(question):
    ss.pending = question


def ask(chat_name, question):
    r = requests.post(f"{API}/chat",
                      json={"session_id": chat_name, "question": question},
                      timeout=CHAT_TIMEOUT)
    r.raise_for_status()
    return r.json()["answer"]


def fetch_notifications():
    """All processed notifications, or None if the backend can't be reached."""
    try:
        return requests.get(f"{API}/notifications", timeout=10).json()["rows"]
    except (requests.RequestException, KeyError, ValueError):
        return None


if not ss.loaded:
    load_saved_chats()


# ---------- sidebar ----------

@st.fragment(run_every=5)       # counts update on their own, whichever view is open
def sidebar_stats():
    rows = fetch_notifications()
    if rows is None:
        st.caption("Backend offline. Notification counts will appear when it is running.")
        return
    applicable = [r for r in rows if r.get("is_applicable")]
    mismatched = [r for r in applicable if r.get("is_master_direction_matching") is False]

    st.caption("Notifications")
    a, b = st.columns(2)
    a.metric("Checked", len(rows))
    b.metric("Applicable", len(applicable))
    c, d = st.columns(2)
    c.metric("Not applicable", len(rows) - len(applicable))
    d.metric("Mismatch", len(mismatched))


with st.sidebar:
    st.title("RBI Compliance")
    st.markdown(
        "Watches new RBI notifications and checks each one against your company profile "
        "and the master directions. For every notification that applies to you, it lists "
        "what changed, what it means and what to do, and flags notifications that cite the "
        "wrong master direction. The chat answers questions from the master directions."
    )
    sidebar_stats()
    st.divider()
    view = st.radio("View", [CHAT_VIEW, NOTES_VIEW, COMPANY_VIEW], label_visibility="collapsed")

    if view == CHAT_VIEW:
        st.button("New chat", on_click=start_naming, **STRETCH)
        if ss.chats:
            st.caption("Your chats")
        for name in ss.chats:
            st.button(name, key=f"chat_{name}",
                      on_click=open_chat, args=(name,),
                      type="primary" if name == ss.active and not ss.naming else "secondary",
                      **STRETCH)


# ---------- chat view ----------

def name_form():
    st.header("Name your chat")
    st.caption("Chats are saved under this name, so you can come back to them later.")
    with st.form("new_chat", clear_on_submit=True):
        name = st.text_input("Chat name", max_chars=40)
        if st.form_submit_button("Start chat", type="primary"):
            if safe_name(name):
                create_chat(name)
                st.rerun()
            else:
                st.error("Enter a name using letters or numbers.")


def chat_view():
    question = st.chat_input("Ask about the RBI master directions") or ss.pending
    ss.pending = None

    # Typed a message without naming the chat -> easter egg: it becomes "gatorade".
    created = False
    if question and (ss.naming or ss.active is None):
        create_chat()
        created = True

    msgs = ss.chats.get(ss.active)
    if ss.naming or msgs is None:
        name_form()
        return

    st.header(ss.active)
    for m in msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if not msgs and not question:
        st.caption("Ask a question below, or start with one of these.")
        for q in STARTERS:
            st.button(q, key=f"starter_{q}", on_click=use_starter, args=(q,))

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Waiting for the model. New notifications are processed first."):
                try:
                    answer = ask(ss.active, question)
                except requests.RequestException as e:
                    st.error(f"Could not get an answer: {e}")
                    return
            st.markdown(answer)
        msgs += [{"role": "user", "content": question},
                 {"role": "assistant", "content": answer}]
        if created:
            st.rerun()          # refresh the sidebar so the new chat shows up


# ---------- notifications view ----------

@st.fragment(run_every=5)       # only this panel refreshes, the rest of the page stays put
def notifications_panel():
    st.header("Notifications")
    rows = fetch_notifications()
    if rows is None:
        st.error("Can't reach the backend. Start it with `python one_ring.py`.")
        return

    applicable = [r for r in rows if r.get("is_applicable")]
    if not applicable:
        st.info("Nothing applicable yet. New notifications show up here as they are checked."
                if not rows else
                "None of the notifications checked so far apply to your profile.")
        return

    # newest first; the #number is the position among applicable ones, so it never changes
    options = {f"#{i + 1}  {r.get('title', 'Untitled')}": r
               for i, r in reversed(list(enumerate(applicable)))}
    r = options[st.selectbox("Notification", list(options))]

    with st.container(border=True):
        st.subheader(r.get("title", "Untitled"))
        if r.get("link"):
            st.markdown(f"[Open the original notification]({r['link']})")
        d1, d2, d3 = st.columns(3)
        for col, label, key in [(d1, "Published", "pubdate"),
                                (d2, "Takes effect", "effective_date"),
                                (d3, "Action needed by", "action_date")]:
            col.caption(label)
            col.write(r.get(key, "-"))

    if r.get("is_master_direction_matching") is False:
        st.warning(f"**Master direction mismatch.** {r.get('reason') or ''}  \n"
                   f"Correct one: {r.get('correct_master_dir') or '-'}")

    tab_diff, tab_meaning, tab_actions = st.tabs(["What changed", "What it means", "Actions required"])
    with tab_diff:
        st.markdown(r.get("diff", "-"))
    with tab_meaning:
        st.markdown(r.get("explanation", "-"))
    with tab_actions:
        st.markdown(r.get("actions_required", "-"))


# ---------- company details view ----------

CATEGORIES = ["NBFC-ICC", "NBFC-MFI", "NBFC-Factor", "NBFC-IFC", "NBFC-IDF",
              "NBFC-AA", "NBFC-P2P", "NBFC-CIC", "NBFC-MGC"]
LAYERS = ["BASE", "MIDDLE", "UPPER", "TOP"]
LEGACY = ["NBFC-D", "NBFC-ND", "NBFC-ND-SI"]
COR_STATUS = ["active", "suspended", "cancelled", "surrendered"]
COMPANY_TYPES = ["public_limited", "private_limited", "other"]
FDI_ROUTES = ["automatic", "government", "not_applicable"]
CORE_SYSTEMS = ["in_house", "vendor", "hybrid"]
SECURITISATION_ROLES = ["originator", "investor", "servicer"]
CUSTOMER_TYPES = ["MSME", "salaried", "self_employed", "retail", "corporate", "agriculture"]
COMMITTEES = ["audit", "nomination_and_remuneration", "risk_management", "it_strategy",
              "asset_liability_management", "credit_approval",
              "stakeholders_relationship", "corporate_social_responsibility"]
OUTSOURCED = ["collections", "kyc_verification", "customer_support", "cloud_hosting",
              "loan_origination", "credit_underwriting", "it_operations"]
FUNDING_TYPES = ["bank_term_loans", "ncd_listed", "ncd_unlisted_private_placement",
                 "commercial_paper", "securitisation_direct_assignment", "subordinated_debt",
                 "refinance_sidbi_nabard", "ecb", "public_deposits", "other"]
PARTNER_TYPES = ["scheduled_commercial_bank", "small_finance_bank", "other_nbfc", "other"]
STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh",
    "Lakshadweep", "Puducherry",
]

ACTIVITY_FLAGS = {
    "gold_loans": ("Gold loans", "Lends against gold jewellery or ornaments."),
    "microfinance": ("Microfinance", "Gives small loans to low-income households."),
    "housing_finance": ("Housing finance", "Gives home loans or other housing finance."),
    "factoring": ("Factoring", "Buys receivables or runs a factoring business."),
    "infrastructure_financing": ("Infrastructure financing", "Finances infrastructure projects."),
    "loans_against_securities": ("Loans against securities",
                                 "Lends against shares, mutual funds or other securities."),
    "digital_lending": ("Digital lending",
                        "Gives loans through apps or online platforms, on your own or through partners."),
    "co_lending": ("Co-lending", "Lends jointly with a bank or another regulated lender."),
    "securitisation_or_assignment": ("Securitisation or assignment",
                                     "Sells or buys loan pools through securitisation or direct assignment."),
    "insurance_distribution": ("Insurance distribution",
                               "Sells insurance products as a corporate agent or broker."),
    "peer_to_peer": ("Peer-to-peer lending", "Runs a P2P lending platform."),
    "account_aggregator": ("Account aggregator", "Works as an RBI-licensed account aggregator."),
    "forex_or_cross_border": ("Forex or cross-border", "Has foreign exchange or cross-border business."),
}

KEY_OFFICERS = {
    "cfo": ("Chief Financial Officer", "A CFO is appointed."),
    "company_secretary": ("Company Secretary", "A company secretary is appointed."),
    "chief_compliance_officer": ("Chief Compliance Officer", "A compliance head is appointed."),
    "chief_risk_officer": ("Chief Risk Officer", "A risk head is appointed."),
    "ciso": ("Chief Information Security Officer", "A CISO is appointed."),
    "principal_officer_pmla": ("Principal Officer (PMLA)",
                               "Principal Officer appointed under the anti-money-laundering rules."),
    "designated_director_pmla": ("Designated Director (PMLA)",
                                 "Designated Director appointed under the anti-money-laundering rules."),
}

PRODUCT_COLS = ["product_id", "product", "share_of_aum_percent", "secured", "collateral_type",
                "average_ticket_size_inr", "customer_segment", "channels",
                "digitally_originated", "co_lent"]


def load_profile():
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_profile(profile):
    tmp = PROFILE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, PROFILE_PATH)     # never leaves a half-written file


def g(d, path, default=None):
    """Read a nested value like g(p, 'identity.legal_name')."""
    for k in path.split("."):
        if not isinstance(d, dict) or d.get(k) is None:
            return default
        d = d[k]
    return d


def deep_merge(old, new):
    """Overwrite what the form edits, keep any other keys already in the file."""
    out = json.loads(json.dumps(old))
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# --- input helpers. Each returns None for "unknown" so the agent can tell it apart from "no".

def txt(label, val, help):
    return st.text_input(label, value=val or "", help=help).strip() or None


def lines(label, val, help):
    raw = st.text_area(label, value="\n".join(val or []), help=help + " One per line.", height=100)
    return [x.strip() for x in raw.splitlines() if x.strip()]


def date_txt(label, val, help, errors):
    raw = st.text_input(label, value=val or "", help=help + " Format: YYYY-MM-DD.",
                        placeholder="YYYY-MM-DD").strip()
    if not raw:
        return None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        errors.append(f"{label}: use the format YYYY-MM-DD.")
        return val
    return raw


def num(label, val, help, errors, integer=False):
    raw = st.text_input(label, value="" if val is None else str(val), help=help)
    raw = raw.strip().replace(",", "")
    if not raw:
        return None
    try:
        n = float(raw)
        if not math.isfinite(n) or (integer and not n.is_integer()):
            raise ValueError
    except ValueError:
        errors.append(f"{label}: enter {'a whole number' if integer else 'a number'} or leave it blank.")
        return val
    return int(n) if integer or n.is_integer() else n


def tri(label, val, help):
    sel = st.selectbox(label, ["Unknown", "Yes", "No"],
                       index=0 if val is None else (1 if val else 2), help=help)
    return {"Unknown": None, "Yes": True, "No": False}[sel]


def pick(label, options, val, help):
    opts = ["Unknown"] + list(options)
    if val and val not in opts:
        opts.append(val)             # keep a value that came from the file
    sel = st.selectbox(label, opts, index=opts.index(val) if val in opts else 0, help=help)
    return None if sel == "Unknown" else sel


def multi(label, options, val, help):
    val = val or []
    opts = list(options) + [v for v in val if v not in options]
    return st.multiselect(label, opts, default=val, help=help)


def table(rows, columns, config):
    df = pd.DataFrame(rows, columns=columns)
    edited = st.data_editor(df, column_config=config, num_rows="dynamic", hide_index=True)
    out = []
    for rec in edited.to_dict("records"):
        rec = {k: (None if pd.isna(v) else v) for k, v in rec.items()}
        if any(v not in (None, "", False) for v in rec.values()):
            out.append(rec)
    return out


def split_csv(s):
    return [x.strip() for x in str(s or "").split(",") if x.strip()]


def company_view():
    p = load_profile()
    errors, warnings = [], []

    st.header("Company details")
    st.markdown(
        "The agent checks every new notification against these details to decide whether it "
        "applies to you. The more complete they are, the fewer notifications it has to guess about."
    )
    st.caption(f"Saved to `{PROFILE_PATH}`. Where you are not sure, pick **Unknown** or leave the box "
               "empty. The agent treats that as \"not sure\", which is different from \"No\".")

    with st.expander("Import or export as JSON"):
        st.download_button("Download current profile", data=json.dumps(p, indent=2, ensure_ascii=False),
                           file_name="profile.json", mime="application/json")
        up = st.file_uploader("Import a profile JSON (replaces the saved profile)", type="json")
        if up is not None:
            sig = f"{up.name}:{up.size}"
            if ss.get("imported_sig") != sig:
                try:
                    data = json.load(up)
                    if not isinstance(data, dict):
                        raise ValueError("The file must contain a JSON object.")
                    save_profile(data)
                    ss.imported_sig = sig
                    ok = True
                except (ValueError, OSError) as e:
                    ok = False
                    st.error(f"Could not import: {e}")
                if ok:
                    st.rerun()

    new = {}
    with st.form("company_form"):
        tabs = st.tabs(["Identity", "Classification", "Size", "Business", "Funding",
                        "Digital lending", "Network", "Ownership", "Governance",
                        "IT and outsourcing", "Supervisory", "Other"])

        # ----- Identity -----
        with tabs[0]:
            st.subheader("Profile")
            a, b = st.columns(2)
            with a:
                profile_id = txt("Profile ID", g(p, "profile_meta.profile_id"),
                                 "A short label for this profile, for example NBFC-DEMO-0001.")
                as_of = date_txt("Figures as of", g(p, "profile_meta.as_of_date"),
                                 "Date the financial figures are taken from, usually the last balance sheet date.",
                                 errors)
                fy = txt("Financial year", g(p, "profile_meta.financial_year"), "For example FY2025-26.")
            with b:
                source = txt("Data source", g(p, "profile_meta.data_source"),
                             "Where these details come from, for example audited financials and the RBI "
                             "Certificate of Registration.")
                verified = date_txt("Last verified on", g(p, "profile_meta.last_verified_on"),
                                    "Date someone last checked these details against the source.", errors)
                demo = st.checkbox("This is demo data", value=bool(g(p, "profile_meta.is_demo_data", False)),
                                   help="Tick this if the company is made up for testing.")
            new["profile_meta"] = {"profile_id": profile_id, "as_of_date": as_of, "financial_year": fy,
                                   "data_source": source, "last_verified_on": verified,
                                   "is_demo_data": demo}

            st.subheader("Company")
            a, b = st.columns(2)
            with a:
                legal = txt("Legal name", g(p, "identity.legal_name"), "Registered name of the company.")
                cor_no = txt("RBI Certificate of Registration no.",
                             g(p, "identity.rbi_certificate_of_registration_no"),
                             "Number on your RBI Certificate of Registration, for example N-14.00000.")
                reg_date = date_txt("RBI registration date", g(p, "identity.rbi_registration_date"),
                                    "Date the RBI registered you as an NBFC.", errors)
                state = pick("Registered office state", STATES, g(p, "identity.registered_office_state"),
                             "State where the registered office is located.")
            with b:
                cor_status = pick("Certificate status", COR_STATUS, g(p, "identity.cor_status"),
                                  "Current status of your Certificate of Registration.")
                ctype = pick("Company type", COMPANY_TYPES, g(p, "identity.company_type"),
                             "How the company is incorporated.")
                listed_eq = tri("Equity shares listed", g(p, "identity.is_listed_equity"),
                                "Whether your equity shares trade on a stock exchange.")
            new["identity"] = {"legal_name": legal, "rbi_certificate_of_registration_no": cor_no,
                               "rbi_registration_date": reg_date, "cor_status": cor_status,
                               "company_type": ctype, "is_listed_equity": listed_eq,
                               "registered_office_state": state}

        # ----- Classification -----
        with tabs[1]:
            st.caption("This decides which RBI directions cover you, so it matters most.")
            a, b = st.columns(2)
            with a:
                cat = pick("NBFC category", CATEGORIES, g(p, "regulatory_classification.nbfc_category"),
                           "Your main category as stated on the Certificate of Registration.")
                sec = multi("Other categories", CATEGORIES,
                            g(p, "regulatory_classification.secondary_categories"),
                            "Any additional categories you are registered under.")
                legacy = pick("Older classification", LEGACY,
                              g(p, "regulatory_classification.legacy_classification"),
                              "The pre-2022 label. Older circulars still use it, for example NBFC-ND-SI.")
                layer = pick("Layer", LAYERS, g(p, "regulatory_classification.layer"),
                             "Layer under RBI's scale-based regulation. For example, a non-deposit-taking "
                             "ICC with assets of Rs 1,000 crore or more is in the Middle Layer.")
                exemptions = lines("Exemptions claimed", g(p, "regulatory_classification.exemptions_claimed"),
                                   "Any RBI exemptions you rely on.")
            with b:
                dep = tri("Takes public deposits", g(p, "regulatory_classification.is_deposit_taking"),
                          "Whether you accept public deposits.")
                sysimp = tri("Systemically important", g(p, "regulatory_classification.is_systemically_important"),
                             "Whether RBI treats you as systemically important based on asset size.")
                hfc = tri("Housing finance company", g(p, "regulatory_classification.is_housing_finance_company"),
                          "Whether you are registered as a housing finance company.")
                cic = tri("Core investment company", g(p, "regulatory_classification.is_core_investment_company"),
                          "Whether you are a core investment company.")
                govt = tri("Government owned", g(p, "regulatory_classification.is_government_owned"),
                           "Whether the government owns you. Some circulars exclude these.")
                pubfunds = tri("Accesses public funds", g(p, "regulatory_classification.accesses_public_funds"),
                               "Whether you raise money from the public, for example through debentures.")
                cust = tri("Has customer interface", g(p, "regulatory_classification.has_customer_interface"),
                           "Whether you deal directly with customers.")
            new["regulatory_classification"] = {
                "nbfc_category": cat, "secondary_categories": sec, "legacy_classification": legacy,
                "layer": layer, "is_deposit_taking": dep, "is_systemically_important": sysimp,
                "is_housing_finance_company": hfc, "is_core_investment_company": cic,
                "is_government_owned": govt, "accesses_public_funds": pubfunds,
                "has_customer_interface": cust, "exemptions_claimed": exemptions}

        # ----- Size -----
        with tabs[2]:
            st.caption("All amounts in Rs crore. Many circulars apply only above a size threshold.")
            a, b = st.columns(2)
            with a:
                ta = num("Total assets", g(p, "size_and_financials.total_assets"),
                         "Total assets on the balance sheet.", errors)
                aum = num("Assets under management", g(p, "size_and_financials.assets_under_management"),
                          "On-book assets plus assets you manage off the books.", errors)
                obs = num("Off-balance-sheet exposure", g(p, "size_and_financials.off_balance_sheet_exposure"),
                          "Exposure that does not appear on the balance sheet.", errors)
            with b:
                grp = num("Group NBFC assets (total)", g(p, "size_and_financials.group_aggregate_nbfc_assets"),
                          "Combined assets of all NBFCs in your group, including this one.", errors)
                nof = num("Net owned fund", g(p, "size_and_financials.net_owned_fund"),
                          "Net owned fund as defined by RBI.", errors)
                borr = num("Total borrowings", g(p, "size_and_financials.total_borrowings"),
                           "All borrowings outstanding.", errors)
            new["size_and_financials"] = {
                "currency": "INR", "unit": "crore", "total_assets": ta, "assets_under_management": aum,
                "off_balance_sheet_exposure": obs, "group_aggregate_nbfc_assets": grp,
                "net_owned_fund": nof, "total_borrowings": borr}

        # ----- Business -----
        with tabs[3]:
            primary = txt("Main business", g(p, "business_profile.primary_activity"),
                          "One line on what you mainly do, for example secured MSME and vehicle lending.")
            st.markdown("**Lending products**")
            st.caption("One row per product. Add rows with the + at the bottom of the table. "
                       "Shares of assets under management should add up to 100. "
                       "Channels is a comma-separated list, for example branch, digital.")
            prod_rows = []
            for r in g(p, "business_profile.lending_products", []):
                r = dict(r)
                r["channels"] = ", ".join(r.get("channels") or [])
                prod_rows.append(r)
            products = table(prod_rows, PRODUCT_COLS, {
                "product_id": st.column_config.TextColumn("ID", help="Short unique label, for example gold_loan."),
                "product": st.column_config.TextColumn("Product"),
                "share_of_aum_percent": st.column_config.NumberColumn("% of AUM", min_value=0, max_value=100),
                "secured": st.column_config.CheckboxColumn("Secured"),
                "collateral_type": st.column_config.TextColumn("Collateral"),
                "average_ticket_size_inr": st.column_config.NumberColumn("Avg ticket (Rs)", min_value=0),
                "customer_segment": st.column_config.TextColumn("Segment"),
                "channels": st.column_config.TextColumn("Channels"),
                "digitally_originated": st.column_config.CheckboxColumn(
                    "Digital", help="Loans are originated through an app or online."),
                "co_lent": st.column_config.CheckboxColumn(
                    "Co-lent", help="Loans are made jointly with another lender."),
            })
            for r in products:
                r["channels"] = split_csv(r.get("channels"))
                for k in ("secured", "digitally_originated", "co_lent"):
                    r[k] = bool(r.get(k))
            share = sum(r.get("share_of_aum_percent") or 0 for r in products)
            if products and abs(share - 100) > 0.5:
                warnings.append(f"Product shares add up to {share:g}%, not 100%.")

            st.markdown("**Activities**")
            st.caption("Whether you do each of these. Many circulars apply only to firms with a given activity.")
            flags = {}
            cols = st.columns(2)
            for i, (key, (label, help_)) in enumerate(ACTIVITY_FLAGS.items()):
                with cols[i % 2]:
                    flags[key] = tri(label, g(p, f"business_profile.activity_flags.{key}"), help_)
            roles = multi("Securitisation role", SECURITISATION_ROLES,
                          g(p, "business_profile.securitisation_role"),
                          "Which side of securitisation or assignment deals you are on.")
            new["business_profile"] = {"primary_activity": primary, "lending_products": products,
                                       "activity_flags": flags, "securitisation_role": roles}

        # ----- Funding -----
        with tabs[4]:
            a, b = st.columns(2)
            with a:
                dep_out = num("Public deposits outstanding (Rs crore)", g(p, "funding.public_deposits_outstanding"),
                              "Public deposits you currently hold. Enter 0 if none.", errors)
                listed = tri("Listed debt securities", g(p, "funding.listed_debt_securities"),
                             "Whether you have debentures or bonds listed on an exchange. "
                             "This is what brings SEBI listing rules in.")
            with b:
                ecb = tri("Foreign borrowing (ECB)", g(p, "funding.has_foreign_borrowing_ecb"),
                          "Whether you have external commercial borrowings.")
            st.markdown("**Sources of funds**")
            st.caption("Share of total borrowings from each source. Should add up to 100.")
            sources = table(g(p, "funding.sources", []), ["type", "share_percent"], {
                "type": st.column_config.SelectboxColumn("Source", options=FUNDING_TYPES),
                "share_percent": st.column_config.NumberColumn("% of borrowings", min_value=0, max_value=100),
            })
            fshare = sum(r.get("share_percent") or 0 for r in sources)
            if sources and abs(fshare - 100) > 0.5:
                warnings.append(f"Funding shares add up to {fshare:g}%, not 100%.")
            new["funding"] = {"public_deposits_outstanding": dep_out, "listed_debt_securities": listed,
                              "has_foreign_borrowing_ecb": ecb, "sources": sources}

        # ----- Digital lending -----
        with tabs[5]:
            st.caption("Covers the RBI Digital Lending Directions: apps, partners and guarantees.")
            a, b = st.columns(2)
            with a:
                offers = tri("Offers digital loans", g(p, "digital_lending.offers_digital_loans"),
                             "Whether you give loans through digital channels.")
                uses_lsp = tri("Uses lending service providers", g(p, "digital_lending.uses_lending_service_providers"),
                               "Whether third parties help you source, underwrite or collect on loans.")
                lsp_n = num("Number of lending service providers", g(p, "digital_lending.lsp_count"),
                            "How many such partners you work with.", errors, integer=True)
            with b:
                dlg = tri("Uses default loss guarantee", g(p, "digital_lending.uses_default_loss_guarantee"),
                          "Whether a partner guarantees you against loan losses.")
                co_active = tri("Co-lending active", g(p, "digital_lending.co_lending.active"),
                                "Whether you currently co-lend with another lender.")
            st.markdown("**Your lending apps**")
            st.caption("Platforms is a comma-separated list, for example android, ios.")
            app_rows = [{"name": a_.get("name"), "platforms": ", ".join(a_.get("platforms") or [])}
                        for a_ in g(p, "digital_lending.own_lending_apps", [])]
            apps = table(app_rows, ["name", "platforms"], {
                "name": st.column_config.TextColumn("App name"),
                "platforms": st.column_config.TextColumn("Platforms")})
            for r in apps:
                r["platforms"] = split_csv(r.get("platforms"))
            st.markdown("**Co-lending partners**")
            partners = table(g(p, "digital_lending.co_lending.partners", []), ["partner_type", "count"], {
                "partner_type": st.column_config.SelectboxColumn("Partner type", options=PARTNER_TYPES),
                "count": st.column_config.NumberColumn("How many", min_value=0, step=1)})
            new["digital_lending"] = {
                "offers_digital_loans": offers, "own_lending_apps": apps,
                "uses_lending_service_providers": uses_lsp, "lsp_count": lsp_n,
                "uses_default_loss_guarantee": dlg,
                "co_lending": {"active": co_active, "partners": partners}}

        # ----- Network -----
        with tabs[6]:
            a, b = st.columns(2)
            with a:
                ctypes = multi("Customer types", CUSTOMER_TYPES, g(p, "customer_and_network.customer_type"),
                               "Kinds of customers you lend to.")
                branches = num("Number of branches", g(p, "customer_and_network.number_of_branches"),
                               "Total branches.", errors, integer=True)
                states = multi("States of operation", STATES, g(p, "customer_and_network.states_of_operation"),
                               "States and union territories where you operate.")
            with b:
                recovery = tri("Uses recovery agents", g(p, "customer_and_network.uses_recovery_agents"),
                               "Whether you use agents to recover overdue loans.")
                ombud = tri("Internal ombudsman appointed",
                            g(p, "customer_and_network.internal_ombudsman_appointed"),
                            "Whether you have appointed an internal ombudsman for complaints.")
                nodal = tri("Principal nodal officer appointed",
                            g(p, "customer_and_network.principal_nodal_officer_appointed"),
                            "Whether a principal nodal officer handles grievances.")
            new["customer_and_network"] = {
                "customer_type": ctypes, "number_of_branches": branches, "states_of_operation": states,
                "uses_recovery_agents": recovery, "internal_ombudsman_appointed": ombud,
                "principal_nodal_officer_appointed": nodal}

        # ----- Ownership -----
        with tabs[7]:
            a, b = st.columns(2)
            with a:
                promo = num("Promoter holding (%)", g(p, "ownership_and_group.promoter_holding_percent"),
                            "Share of the company held by promoters.", errors)
                foreign = num("Foreign holding (%)", g(p, "ownership_and_group.foreign_holding_percent"),
                              "Share of the company held by foreign investors.", errors)
                fdi = pick("FDI route", FDI_ROUTES, g(p, "ownership_and_group.fdi_route"),
                           "Route used for foreign investment.")
            with b:
                parent_reg = tri("Parent regulated by RBI", g(p, "ownership_and_group.parent_is_rbi_regulated"),
                                 "Whether the parent company is itself regulated by RBI.")
                bank_sub = tri("Subsidiary of a bank", g(p, "ownership_and_group.is_subsidiary_of_bank"),
                               "Whether a bank owns you.")
                group_nbfcs = lines("Other NBFCs in the group", g(p, "ownership_and_group.other_group_nbfcs"),
                                    "Names of other NBFCs under the same group.")
            new["ownership_and_group"] = {
                "promoter_holding_percent": promo, "foreign_holding_percent": foreign, "fdi_route": fdi,
                "parent_is_rbi_regulated": parent_reg, "is_subsidiary_of_bank": bank_sub,
                "other_group_nbfcs": group_nbfcs}

        # ----- Governance -----
        with tabs[8]:
            committees = multi("Board committees", COMMITTEES, g(p, "governance_and_organisation.committees"),
                               "Committees your board has set up.")
            st.markdown("**Key officers**")
            officers = {}
            cols = st.columns(2)
            for i, (key, (label, help_)) in enumerate(KEY_OFFICERS.items()):
                with cols[i % 2]:
                    officers[key] = tri(label, g(p, f"governance_and_organisation.key_officers.{key}"), help_)
            new["governance_and_organisation"] = {"committees": committees, "key_officers": officers}

        # ----- IT and outsourcing -----
        with tabs[9]:
            a, b = st.columns(2)
            with a:
                core = pick("Core lending system", CORE_SYSTEMS, g(p, "it_and_outsourcing.core_lending_system"),
                            "Whether the system is built in-house, bought from a vendor, or both.")
                cloud = tri("Uses cloud hosting", g(p, "it_and_outsourcing.uses_cloud_hosting"),
                            "Whether any systems run on the cloud.")
                outsourced = multi("Outsourced functions", OUTSOURCED, g(p, "it_and_outsourcing.outsourced_functions"),
                                   "Work you hand to outside firms.")
            with b:
                out_pol = tri("Board-approved outsourcing policy",
                              g(p, "it_and_outsourcing.board_approved_outsourcing_policy"),
                              "Whether your board has approved an outsourcing policy.")
                it_pol = tri("Board-approved IT policy", g(p, "it_and_outsourcing.board_approved_it_policy"),
                             "Whether your board has approved an IT policy.")
            new["it_and_outsourcing"] = {
                "core_lending_system": core, "uses_cloud_hosting": cloud, "outsourced_functions": outsourced,
                "board_approved_outsourcing_policy": out_pol, "board_approved_it_policy": it_pol}

        # ----- Supervisory -----
        with tabs[10]:
            penalties = lines("RBI penalties in the last 3 years", g(p, "supervisory_history.rbi_penalties_last_3_years"),
                              "Short description of each penalty.")
            restrictions = lines("RBI business restrictions", g(p, "supervisory_history.rbi_business_restrictions"),
                                 "Any restrictions RBI has placed on your business.")
            pending = tri("Pending inspection findings", g(p, "supervisory_history.pending_inspection_findings"),
                          "Whether findings from an RBI inspection are still open.")
            new["supervisory_history"] = {"rbi_penalties_last_3_years": penalties,
                                          "rbi_business_restrictions": restrictions,
                                          "pending_inspection_findings": pending}

        # ----- Other -----
        with tabs[11]:
            st.caption("Anything the other tabs do not cover goes here. It is saved with the profile as it is.")
            notes = st.text_area("Other notes", value=g(p, "other_details.notes", ""), height=140,
                                 help="Free text about your business that could affect which rules apply.")
            st.markdown("**Extra fields**")
            st.caption("Add your own name and value pairs, with an optional description.")
            extras = table(g(p, "other_details.extra_fields", []), ["field", "value", "description"], {
                "field": st.column_config.TextColumn("Field"),
                "value": st.column_config.TextColumn("Value"),
                "description": st.column_config.TextColumn("Description")})
            new["other_details"] = {"notes": notes.strip() or None, "extra_fields": extras}

        submitted = st.form_submit_button("Save company details", type="primary")

    if submitted:
        if errors:
            st.error("Not saved. Please fix:\n\n" + "\n".join(f"- {e}" for e in errors))
            return
        try:
            save_profile(deep_merge(p, new))
        except OSError as e:
            st.error(f"Could not save to `{PROFILE_PATH}`: {e}")
            return
        st.success("Company details saved.")
        for w in warnings:
            st.warning(w)


if view == CHAT_VIEW:
    chat_view()
elif view == NOTES_VIEW:
    notifications_panel()
else:
    company_view()