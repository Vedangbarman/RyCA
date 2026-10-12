import inspect
import json
import math
import os
import re

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000")
# Where the company profile and the agent settings are stored. Point your backend at the same files.
PROFILE_PATH = os.getenv("PROFILE_PATH", "profile.json")
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")     # config.json in the root of the project
CHAT_TIMEOUT = 1800        # a chat message waits behind any running notification batch
DEFAULT_NAME = "gatorade"
STARTERS = [
    "What is the Upper Layer in NBFCs?",
    "Which master directions apply to all NBFCs?",
    "What are the large exposure limits for IDF-NBFCs?",
]
CHAT_VIEW, NOTES_VIEW, COMPANY_VIEW, CONFIG_VIEW = (
    "Chat", "Notifications", "Company details", "Agent configuration")

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


# ---------- shared file helpers ----------

def write_json(path, data, indent=2):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)     # never leaves a half-written file


def deep_merge(old, new):
    """Overwrite what the form edits, keep any other keys already in the file."""
    out = json.loads(json.dumps(old))
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


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
    view = st.radio("View", [CHAT_VIEW, NOTES_VIEW, COMPANY_VIEW, CONFIG_VIEW],
                    label_visibility="collapsed")

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
ENTITY_TYPES = ["NBFC", "HFC", "ARC", "other"]
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
FUNDING_TYPES = ["bank_term_loans", "listed_ncd", "unlisted_ncd", "commercial_paper",
                 "securitisation_direct_assignment", "subordinated_debt",
                 "refinance_sidbi_nabard", "ecb", "public_deposits", "other"]
PARTNER_TYPES = ["scheduled_commercial_bank", "small_finance_bank", "other_nbfc", "other"]
PRODUCT_TYPES = ["loan_against_property", "vehicle_finance", "personal_loan_unsecured", "gold_loan",
                 "home_loan", "business_loan_unsecured", "microfinance_loan",
                 "loan_against_securities", "education_loan", "supply_chain_finance",
                 "factoring", "infrastructure_loan", "other"]
COLLATERAL_TYPES = ["immovable_property", "vehicle", "gold_jewellery", "securities",
                    "receivables", "plant_and_machinery", "other"]
EXPORT_IMPORT = ["none", "exports", "imports", "both"]
AD_CATEGORIES = ["none", "AD-I", "AD-II", "AD-III"]
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

PRODUCT_COLS = ["product_type", "share_of_aum_percent", "secured", "collateral_type",
                "customer_segment", "digitally_originated", "co_lent"]

# The profile keeps a few facts in two places: the short sections the agent matches on first
# (match_keys, gates, relevance) and the detailed sections. Each fact is asked for once in the
# form and saved to both places, so the two never disagree.
MIRRORS = {
    "identity.legal_name": ["match_keys.legal_name"],
    "identity.rbi_certificate_of_registration_no": ["match_keys.rbi_certificate_of_registration_no"],
    "identity.cor_status": ["match_keys.cor_status"],
    "regulatory_classification.nbfc_category": ["gates.nbfc_category"],
    "regulatory_classification.secondary_categories": ["gates.secondary_categories"],
    "regulatory_classification.layer": ["gates.layer"],
    "regulatory_classification.is_deposit_taking": ["gates.is_deposit_taking"],
    "regulatory_classification.accesses_public_funds": ["gates.accesses_public_funds"],
    "regulatory_classification.has_customer_interface": ["gates.has_customer_interface"],
    "regulatory_classification.is_government_owned": ["gates.is_government_owned"],
    "regulatory_classification.exemptions_claimed": ["gates.exemptions_claimed"],
    "size_and_financials.total_assets": ["gates.total_assets_crore"],
    "size_and_financials.group_aggregate_nbfc_assets": ["gates.group_aggregate_nbfc_assets_crore"],
    "ownership_and_group.is_subsidiary_of_bank": ["gates.is_subsidiary_of_bank"],
    "ownership_and_group.foreign_holding_percent": ["relevance.cross_border.foreign_holding_percent"],
    "funding.has_foreign_borrowing_ecb": ["relevance.cross_border.has_foreign_borrowing_ecb"],
    "business_profile.lending_products": ["relevance.lending_products"],
    "business_profile.securitisation_role": ["relevance.other_activities.securitisation_roles"],
    "business_profile.activity_flags.insurance_distribution":
        ["relevance.other_activities.insurance_distribution"],
    "customer_and_network.uses_recovery_agents": ["relevance.other_activities.uses_recovery_agents"],
    "it_and_outsourcing.outsourced_functions": ["relevance.other_activities.outsourced_functions"],
    "digital_lending.offers_digital_loans": ["relevance.digital_lending.offers_digital_loans"],
    "digital_lending.uses_lending_service_providers":
        ["relevance.digital_lending.uses_lending_service_providers"],
    "digital_lending.uses_default_loss_guarantee":
        ["relevance.digital_lending.uses_default_loss_guarantee"],
}
MISSING = object()


def load_profile():
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_profile(profile):
    write_json(PROFILE_PATH, profile)


def g(d, path, default=None):
    """Read a nested value like g(p, 'identity.legal_name')."""
    for k in path.split("."):
        if not isinstance(d, dict) or d.get(k) is None:
            return default
        d = d[k]
    return d


def rd(p, path, default=None):
    """Read a profile value. If it is empty, use the same fact from its other place in the file."""
    for q in (path, *MIRRORS.get(path, ())):
        v = g(p, q)
        if v is not None and v != [] and v != "":
            return v
    return default


def put(d, path, value):
    *parents, last = path.split(".")
    for k in parents:
        d = d.setdefault(k, {})
    d[last] = value


def peek(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return MISSING
        d = d[k]
    return d


def sync_mirrors(new):
    for src, dsts in MIRRORS.items():
        v = peek(new, src)
        if v is not MISSING:
            for dst in dsts:
                put(new, dst, v)


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


def with_existing(options, rows, col):
    """Dropdown options for a table column, plus any value already in the file."""
    extra = [r.get(col) for r in rows if r.get(col) and r.get(col) not in options]
    return list(options) + list(dict.fromkeys(extra))


def table(rows, columns, config):
    df = pd.DataFrame(rows, columns=columns)
    edited = st.data_editor(df, column_config=config, num_rows="dynamic", hide_index=True)
    out = []
    for rec in edited.to_dict("records"):
        rec = {k: (None if pd.isna(v) else v) for k, v in rec.items()}
        rec = {k: (int(v) if isinstance(v, float) and v.is_integer() else v) for k, v in rec.items()}
        if any(v not in (None, "", False) for v in rec.values()):
            out.append(rec)
    return out


def split_csv(s):
    return [x.strip() for x in str(s or "").split(",") if x.strip()]


def unique(rows, col):
    return list(dict.fromkeys(r[col] for r in rows if r.get(col)))


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
        (t_id, t_cls, t_size, t_biz, t_fund, t_dig, t_net, t_own, t_gov, t_it,
         t_pay, t_sup, t_other) = st.tabs([
            "Identity", "Classification", "Size", "Business", "Funding", "Digital lending",
            "Network", "Ownership", "Governance", "IT and outsourcing",
            "Payments and cross-border", "Supervisory", "Other"])

        # ----- Identity -----
        with t_id:
            st.subheader("Profile")
            schema = rd(p, "profile_meta.schema_version")
            if schema:
                st.caption(f"Profile format version {schema}.")
            a, b = st.columns(2)
            with a:
                profile_id = txt("Profile ID", rd(p, "profile_meta.profile_id"),
                                 "A short label for this profile, for example NBFC-DEMO-0001.")
                as_of = date_txt("Figures as of", rd(p, "profile_meta.as_of_date"),
                                 "Date the financial figures are taken from, usually the last balance sheet date.",
                                 errors)
                fy = txt("Financial year", rd(p, "profile_meta.financial_year"), "For example FY2025-26.")
            with b:
                source = txt("Data source", rd(p, "profile_meta.data_source"),
                             "Where these details come from, for example audited financials and the RBI "
                             "Certificate of Registration.")
                verified = date_txt("Last verified on", rd(p, "profile_meta.last_verified_on"),
                                    "Date someone last checked these details against the source.", errors)
                demo = st.checkbox("This is demo data", value=bool(rd(p, "profile_meta.is_demo_data", False)),
                                   help="Tick this if the company is made up for testing.")
            new["profile_meta"] = {"profile_id": profile_id, "as_of_date": as_of, "financial_year": fy,
                                   "data_source": source, "last_verified_on": verified,
                                   "is_demo_data": demo}

            st.subheader("Company")
            a, b = st.columns(2)
            with a:
                legal = txt("Legal name", rd(p, "identity.legal_name"), "Registered name of the company.")
                cor_no = txt("RBI Certificate of Registration no.",
                             rd(p, "identity.rbi_certificate_of_registration_no"),
                             "Number on your RBI Certificate of Registration, for example N-14.00000.")
                reg_date = date_txt("RBI registration date", rd(p, "identity.rbi_registration_date"),
                                    "Date the RBI registered you as an NBFC.", errors)
                state = pick("Registered office state", STATES, rd(p, "identity.registered_office_state"),
                             "State where the registered office is located.")
            with b:
                cor_status = pick("Certificate status", COR_STATUS, rd(p, "identity.cor_status"),
                                  "Current status of your Certificate of Registration.")
                ctype = pick("Company type", COMPANY_TYPES, rd(p, "identity.company_type"),
                             "How the company is incorporated.")
                listed_eq = tri("Equity shares listed", rd(p, "identity.is_listed_equity"),
                                "Whether your equity shares trade on a stock exchange.")
            new["identity"] = {"legal_name": legal, "rbi_certificate_of_registration_no": cor_no,
                               "rbi_registration_date": reg_date, "cor_status": cor_status,
                               "company_type": ctype, "is_listed_equity": listed_eq,
                               "registered_office_state": state}

        # ----- Classification -----
        with t_cls:
            st.caption("This decides which RBI directions cover you, so it matters most. "
                       "These answers are saved in both the classification and the gates sections "
                       "of the profile.")
            a, b = st.columns(2)
            with a:
                entity = pick("Entity type", ENTITY_TYPES, rd(p, "gates.entity_type"),
                              "The kind of regulated entity. NBFC for a non-banking financial company.")
                cat = pick("NBFC category", CATEGORIES, rd(p, "regulatory_classification.nbfc_category"),
                           "Your main category as stated on the Certificate of Registration.")
                sec = multi("Other categories", CATEGORIES,
                            rd(p, "regulatory_classification.secondary_categories"),
                            "Any additional categories you are registered under.")
                legacy = pick("Older classification", LEGACY,
                              rd(p, "regulatory_classification.legacy_classification"),
                              "The pre-2022 label. Older circulars still use it, for example NBFC-ND-SI.")
                layer = pick("Layer", LAYERS, rd(p, "regulatory_classification.layer"),
                             "Layer under RBI's scale-based regulation. For example, a non-deposit-taking "
                             "ICC with assets of Rs 1,000 crore or more is in the Middle Layer.")
                exemptions = lines("Exemptions claimed", rd(p, "regulatory_classification.exemptions_claimed"),
                                   "Any RBI exemptions you rely on.")
            with b:
                dep = tri("Takes public deposits", rd(p, "regulatory_classification.is_deposit_taking"),
                          "Whether you accept public deposits.")
                sysimp = tri("Systemically important", rd(p, "regulatory_classification.is_systemically_important"),
                             "Whether RBI treats you as systemically important based on asset size.")
                hfc = tri("Housing finance company", rd(p, "regulatory_classification.is_housing_finance_company"),
                          "Whether you are registered as a housing finance company.")
                cic = tri("Core investment company", rd(p, "regulatory_classification.is_core_investment_company"),
                          "Whether you are a core investment company.")
                govt = tri("Government owned", rd(p, "regulatory_classification.is_government_owned"),
                           "Whether the government owns you. Some circulars exclude these.")
                pubfunds = tri("Accesses public funds", rd(p, "regulatory_classification.accesses_public_funds"),
                               "Whether you raise money from the public, for example through debentures.")
                cust = tri("Has customer interface", rd(p, "regulatory_classification.has_customer_interface"),
                           "Whether you deal directly with customers.")
            new["regulatory_classification"] = {
                "nbfc_category": cat, "secondary_categories": sec, "legacy_classification": legacy,
                "layer": layer, "is_deposit_taking": dep, "is_systemically_important": sysimp,
                "is_housing_finance_company": hfc, "is_core_investment_company": cic,
                "is_government_owned": govt, "accesses_public_funds": pubfunds,
                "has_customer_interface": cust, "exemptions_claimed": exemptions}
            put(new, "gates.entity_type", entity)

        # ----- Size -----
        with t_size:
            st.caption("All amounts in Rs crore. Many circulars apply only above a size threshold.")
            a, b = st.columns(2)
            with a:
                ta = num("Total assets", rd(p, "size_and_financials.total_assets"),
                         "Total assets on the balance sheet.", errors)
                aum = num("Assets under management", rd(p, "size_and_financials.assets_under_management"),
                          "On-book assets plus assets you manage off the books.", errors)
                obs = num("Off-balance-sheet exposure", rd(p, "size_and_financials.off_balance_sheet_exposure"),
                          "Exposure that does not appear on the balance sheet.", errors)
            with b:
                grp = num("Group NBFC assets (total)", rd(p, "size_and_financials.group_aggregate_nbfc_assets"),
                          "Combined assets of all NBFCs in your group, including this one.", errors)
                nof = num("Net owned fund", rd(p, "size_and_financials.net_owned_fund"),
                          "Net owned fund as defined by RBI.", errors)
                borr = num("Total borrowings", rd(p, "size_and_financials.total_borrowings"),
                           "All borrowings outstanding.", errors)
            new["size_and_financials"] = {
                "currency": "INR", "unit": "crore", "total_assets": ta, "assets_under_management": aum,
                "off_balance_sheet_exposure": obs, "group_aggregate_nbfc_assets": grp,
                "net_owned_fund": nof, "total_borrowings": borr}

        # ----- Business -----
        with t_biz:
            primary = txt("Main business", rd(p, "business_profile.primary_activity"),
                          "One line on what you mainly do, for example secured MSME and vehicle lending.")
            st.markdown("**Lending products**")
            st.caption("One row per product. Add rows with the + at the bottom of the table. "
                       "Shares of assets under management should add up to 100.")
            prod_rows = []
            for r in rd(p, "business_profile.lending_products", []):
                r = dict(r)
                r.setdefault("product_type", r.get("product"))     # older files call it "product"
                prod_rows.append(r)
            products = table(prod_rows, PRODUCT_COLS, {
                "product_type": st.column_config.SelectboxColumn(
                    "Product", options=with_existing(PRODUCT_TYPES, prod_rows, "product_type")),
                "share_of_aum_percent": st.column_config.NumberColumn("% of AUM", min_value=0, max_value=100),
                "secured": st.column_config.CheckboxColumn("Secured"),
                "collateral_type": st.column_config.SelectboxColumn(
                    "Collateral", options=with_existing(COLLATERAL_TYPES, prod_rows, "collateral_type")),
                "customer_segment": st.column_config.SelectboxColumn(
                    "Segment", options=with_existing(CUSTOMER_TYPES, prod_rows, "customer_segment")),
                "digitally_originated": st.column_config.CheckboxColumn(
                    "Digital", help="Loans are originated through an app or online."),
                "co_lent": st.column_config.CheckboxColumn(
                    "Co-lent", help="Loans are made jointly with another lender."),
            })
            for r in products:
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
                    flags[key] = tri(label, rd(p, f"business_profile.activity_flags.{key}"), help_)
            roles = multi("Securitisation role", SECURITISATION_ROLES,
                          rd(p, "business_profile.securitisation_role"),
                          "Which side of securitisation or assignment deals you are on.")
            new["business_profile"] = {"primary_activity": primary, "lending_products": products,
                                       "activity_flags": flags, "securitisation_role": roles}

        # ----- Funding -----
        with t_fund:
            a, b = st.columns(2)
            with a:
                dep_out = num("Public deposits outstanding (Rs crore)", rd(p, "funding.public_deposits_outstanding"),
                              "Public deposits you currently hold. Enter 0 if none.", errors)
                listed = tri("Listed debt securities", rd(p, "funding.listed_debt_securities"),
                             "Whether you have debentures or bonds listed on an exchange. "
                             "This is what brings SEBI listing rules in.")
            with b:
                ecb = tri("Foreign borrowing (ECB)", rd(p, "funding.has_foreign_borrowing_ecb"),
                          "Whether you have external commercial borrowings.")
            st.markdown("**Sources of funds**")
            st.caption("Pick every source you borrow from. The share of total borrowings is optional, "
                       "but if you fill it in it should add up to 100.")
            src_rows = rd(p, "funding.sources") or [
                {"type": t} for t in rd(p, "relevance.funding_instruments", [])]
            sources = table(src_rows, ["type", "share_percent"], {
                "type": st.column_config.SelectboxColumn(
                    "Source", options=with_existing(FUNDING_TYPES, src_rows, "type")),
                "share_percent": st.column_config.NumberColumn("% of borrowings", min_value=0, max_value=100),
            })
            shares = [r["share_percent"] for r in sources if r.get("share_percent") is not None]
            if shares and abs(sum(shares) - 100) > 0.5:
                warnings.append(f"Funding shares add up to {sum(shares):g}%, not 100%.")
            new["funding"] = {"public_deposits_outstanding": dep_out, "listed_debt_securities": listed,
                              "has_foreign_borrowing_ecb": ecb, "sources": sources}
            put(new, "relevance.funding_instruments", unique(sources, "type"))

        # ----- Digital lending -----
        with t_dig:
            st.caption("Covers the RBI Digital Lending Directions: apps, partners and guarantees.")
            a, b = st.columns(2)
            with a:
                offers = tri("Offers digital loans", rd(p, "digital_lending.offers_digital_loans"),
                             "Whether you give loans through digital channels.")
                has_app = tri("Has its own lending app", rd(p, "relevance.digital_lending.has_own_lending_app"),
                              "Whether you run a lending app under your own name.")
                uses_lsp = tri("Uses lending service providers", rd(p, "digital_lending.uses_lending_service_providers"),
                               "Whether third parties help you source, underwrite or collect on loans.")
                lsp_n = num("Number of lending service providers", rd(p, "digital_lending.lsp_count"),
                            "How many such partners you work with.", errors, integer=True)
            with b:
                dlg = tri("Uses default loss guarantee", rd(p, "digital_lending.uses_default_loss_guarantee"),
                          "Whether a partner guarantees you against loan losses.")
                co_active = tri("Co-lending active", rd(p, "digital_lending.co_lending.active"),
                                "Whether you currently co-lend with another lender.")
            st.markdown("**Your lending apps**")
            st.caption("Platforms is a comma-separated list, for example android, ios.")
            app_rows = [{"name": a_.get("name"), "platforms": ", ".join(a_.get("platforms") or [])}
                        for a_ in rd(p, "digital_lending.own_lending_apps", [])]
            apps = table(app_rows, ["name", "platforms"], {
                "name": st.column_config.TextColumn("App name"),
                "platforms": st.column_config.TextColumn("Platforms")})
            for r in apps:
                r["platforms"] = split_csv(r.get("platforms"))
            st.markdown("**Co-lending partners**")
            partner_rows = rd(p, "digital_lending.co_lending.partners") or [
                {"partner_type": t} for t in rd(p, "relevance.digital_lending.co_lending_partner_types", [])]
            partners = table(partner_rows, ["partner_type", "count"], {
                "partner_type": st.column_config.SelectboxColumn(
                    "Partner type", options=with_existing(PARTNER_TYPES, partner_rows, "partner_type")),
                "count": st.column_config.NumberColumn("How many", min_value=0, step=1)})
            if apps and has_app is False:
                warnings.append("You listed lending apps but answered No to \"Has its own lending app\".")
            if partners and co_active is False:
                warnings.append("You listed co-lending partners but answered No to \"Co-lending active\".")
            new["digital_lending"] = {
                "offers_digital_loans": offers, "own_lending_apps": apps,
                "uses_lending_service_providers": uses_lsp, "lsp_count": lsp_n,
                "uses_default_loss_guarantee": dlg,
                "co_lending": {"active": co_active, "partners": partners}}
            put(new, "relevance.digital_lending.has_own_lending_app", has_app)
            put(new, "relevance.digital_lending.co_lending_partner_types", unique(partners, "partner_type"))

        # ----- Network -----
        with t_net:
            a, b = st.columns(2)
            with a:
                ctypes = multi("Customer types", CUSTOMER_TYPES, rd(p, "customer_and_network.customer_type"),
                               "Kinds of customers you lend to.")
                branches = num("Number of branches", rd(p, "customer_and_network.number_of_branches"),
                               "Total branches.", errors, integer=True)
                states = multi("States of operation", STATES, rd(p, "customer_and_network.states_of_operation"),
                               "States and union territories where you operate.")
            with b:
                recovery = tri("Uses recovery agents", rd(p, "customer_and_network.uses_recovery_agents"),
                               "Whether you use agents to recover overdue loans.")
                ombud = tri("Internal ombudsman appointed",
                            rd(p, "customer_and_network.internal_ombudsman_appointed"),
                            "Whether you have appointed an internal ombudsman for complaints.")
                nodal = tri("Principal nodal officer appointed",
                            rd(p, "customer_and_network.principal_nodal_officer_appointed"),
                            "Whether a principal nodal officer handles grievances.")
            new["customer_and_network"] = {
                "customer_type": ctypes, "number_of_branches": branches, "states_of_operation": states,
                "uses_recovery_agents": recovery, "internal_ombudsman_appointed": ombud,
                "principal_nodal_officer_appointed": nodal}

        # ----- Ownership -----
        with t_own:
            a, b = st.columns(2)
            with a:
                promo = num("Promoter holding (%)", rd(p, "ownership_and_group.promoter_holding_percent"),
                            "Share of the company held by promoters.", errors)
                foreign = num("Foreign holding (%)", rd(p, "ownership_and_group.foreign_holding_percent"),
                              "Share of the company held by foreign investors.", errors)
                fdi = pick("FDI route", FDI_ROUTES, rd(p, "ownership_and_group.fdi_route"),
                           "Route used for foreign investment.")
            with b:
                parent_reg = tri("Parent regulated by RBI", rd(p, "ownership_and_group.parent_is_rbi_regulated"),
                                 "Whether the parent company is itself regulated by RBI.")
                bank_sub = tri("Subsidiary of a bank", rd(p, "ownership_and_group.is_subsidiary_of_bank"),
                               "Whether a bank owns you.")
                group_nbfcs = lines("Other NBFCs in the group", rd(p, "ownership_and_group.other_group_nbfcs"),
                                    "Names of other NBFCs under the same group.")
            new["ownership_and_group"] = {
                "promoter_holding_percent": promo, "foreign_holding_percent": foreign, "fdi_route": fdi,
                "parent_is_rbi_regulated": parent_reg, "is_subsidiary_of_bank": bank_sub,
                "other_group_nbfcs": group_nbfcs}

        # ----- Governance -----
        with t_gov:
            committees = multi("Board committees", COMMITTEES, rd(p, "governance_and_organisation.committees"),
                               "Committees your board has set up.")
            st.markdown("**Key officers**")
            officers = {}
            cols = st.columns(2)
            for i, (key, (label, help_)) in enumerate(KEY_OFFICERS.items()):
                with cols[i % 2]:
                    officers[key] = tri(label, rd(p, f"governance_and_organisation.key_officers.{key}"), help_)
            new["governance_and_organisation"] = {"committees": committees, "key_officers": officers}

        # ----- IT and outsourcing -----
        with t_it:
            a, b = st.columns(2)
            with a:
                core = pick("Core lending system", CORE_SYSTEMS, rd(p, "it_and_outsourcing.core_lending_system"),
                            "Whether the system is built in-house, bought from a vendor, or both.")
                cloud = tri("Uses cloud hosting", rd(p, "it_and_outsourcing.uses_cloud_hosting"),
                            "Whether any systems run on the cloud.")
                outsourced = multi("Outsourced functions", OUTSOURCED, rd(p, "it_and_outsourcing.outsourced_functions"),
                                   "Work you hand to outside firms.")
            with b:
                out_pol = tri("Board-approved outsourcing policy",
                              rd(p, "it_and_outsourcing.board_approved_outsourcing_policy"),
                              "Whether your board has approved an outsourcing policy.")
                it_pol = tri("Board-approved IT policy", rd(p, "it_and_outsourcing.board_approved_it_policy"),
                             "Whether your board has approved an IT policy.")
            new["it_and_outsourcing"] = {
                "core_lending_system": core, "uses_cloud_hosting": cloud, "outsourced_functions": outsourced,
                "board_approved_outsourcing_policy": out_pol, "board_approved_it_policy": it_pol}

        # ----- Payments and cross-border -----
        with t_pay:
            st.caption("Foreign borrowing (ECB) is on the Funding tab and foreign holding is on the "
                       "Ownership tab.")
            a, b = st.columns(2)
            with a:
                pay_auth = lines("Payment system authorisations",
                                 rd(p, "relevance.payments_and_cards.payment_authorisations"),
                                 "Authorisations RBI has given you to run payment services, for example "
                                 "prepaid payment instruments. Leave empty if you have none.")
                cards = tri("Issues credit cards", rd(p, "relevance.payments_and_cards.issues_credit_cards"),
                            "Whether you issue credit cards.")
            with b:
                trade = pick("Exports or imports", EXPORT_IMPORT,
                             rd(p, "relevance.cross_border.exports_or_imports"),
                             "Whether the company has export or import business. Pick none if it has neither.")
                overseas = tri("Overseas investment made", rd(p, "relevance.cross_border.overseas_investment_made"),
                               "Whether you have invested in entities outside India.")
                ad_cat = pick("Authorised dealer category", AD_CATEGORIES,
                              rd(p, "relevance.cross_border.ad_category"),
                              "Your authorised dealer category for foreign exchange. Pick none if you do not hold one.")
            put(new, "relevance.payments_and_cards.payment_authorisations", pay_auth)
            put(new, "relevance.payments_and_cards.issues_credit_cards", cards)
            put(new, "relevance.cross_border.exports_or_imports", trade)
            put(new, "relevance.cross_border.overseas_investment_made", overseas)
            put(new, "relevance.cross_border.ad_category", ad_cat)

        # ----- Supervisory -----
        with t_sup:
            penalties = lines("RBI penalties in the last 3 years", rd(p, "supervisory_history.rbi_penalties_last_3_years"),
                              "Short description of each penalty.")
            restrictions = lines("RBI business restrictions", rd(p, "supervisory_history.rbi_business_restrictions"),
                                 "Any restrictions RBI has placed on your business.")
            pending = tri("Pending inspection findings", rd(p, "supervisory_history.pending_inspection_findings"),
                          "Whether findings from an RBI inspection are still open.")
            new["supervisory_history"] = {"rbi_penalties_last_3_years": penalties,
                                          "rbi_business_restrictions": restrictions,
                                          "pending_inspection_findings": pending}

        # ----- Other -----
        with t_other:
            st.caption("Anything the other tabs do not cover goes here. It is saved with the profile as it is.")
            notes = st.text_area("Other notes", value=rd(p, "other_details.notes", ""), height=140,
                                 help="Free text about your business that could affect which rules apply.")
            st.markdown("**Extra fields**")
            st.caption("Add your own name and value pairs, with an optional description.")
            extras = table(rd(p, "other_details.extra_fields", []), ["field", "value", "description"], {
                "field": st.column_config.TextColumn("Field"),
                "value": st.column_config.TextColumn("Value"),
                "description": st.column_config.TextColumn("Description")})
            new["other_details"] = {"notes": notes.strip() or None, "extra_fields": extras}

        submitted = st.form_submit_button("Save company details", type="primary")

    if submitted:
        if errors:
            st.error("Not saved. Please fix:\n\n" + "\n".join(f"- {e}" for e in errors))
            return
        sync_mirrors(new)       # copy each shared answer into its second place in the profile
        try:
            save_profile(deep_merge(p, new))
        except OSError as e:
            st.error(f"Could not save to `{PROFILE_PATH}`: {e}")
            return
        st.success("Company details saved.")
        for w in warnings:
            st.warning(w)


# ---------- agent configuration view ----------

HOURS = list(range(24))

# key, label, whole number?, lowest, highest, step, default, help
HYPER_PARAMS = [
    ("temperature", "Temperature", False, 0.0, 2.0, 0.05, 0.7,
     "Higher gives more varied answers. Lower gives more focused and repeatable ones."),
    ("top_p", "Top-p", False, 0.0, 1.0, 0.01, 0.95,
     "Only the most likely words that together make up this share of the probability are considered."),
    ("top_k", "Top-k", True, 0, 1000, 1, 64,
     "Only this many of the most likely words are considered at each step. 0 turns it off."),
    ("min_p", "Min-p", False, 0.0, 1.0, 0.01, 0.05,
     "Drops words that are much less likely than the best one. 0 turns it off."),
    ("max_tokens", "Max answer length (tokens)", True, 1, 32768, 64, 1024,
     "The longest answer the model may write."),
    ("repeat_penalty", "Repeat penalty", False, 0.5, 2.0, 0.05, 1.1,
     "Above 1 discourages repeating the same words. 1 turns it off."),
    ("frequency_penalty", "Frequency penalty", False, -2.0, 2.0, 0.1, 0.0,
     "Positive values discourage words that have already appeared often."),
    ("presence_penalty", "Presence penalty", False, -2.0, 2.0, 0.1, 0.0,
     "Positive values push the model toward words and topics it has not used yet."),
    ("seed", "Seed", True, -1, 2147483647, 1, 42,
     "The same seed and settings give repeatable answers. Some backends treat -1 as a random seed."),
]
# Settings this page handles itself. Model_Details is shown read-only and data_check (the watcher's
# reference dates and article link) is not shown at all. Both are listed here so they never turn up
# under "Other settings", and neither is ever written by this page, so the file keeps its values.
KNOWN_SETTINGS = {
    "scraper_settings": {"time_scraper", "run_count"},
    "data_check": {"ref_date", "delta_ref_date", "ref_article"},
    "Model_Details": {"Embedding_Model", "Main_Model"},
    "Hyper_Param": {h[0] for h in HYPER_PARAMS},
}


def load_config():
    """(config, problem). A missing file is fine. A file that can't be read is a problem, and
    saving over it would wipe the settings this page does not show."""
    if not os.path.exists(CONFIG_PATH):
        return {}, None
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        return None, str(e)
    if not isinstance(data, dict):
        return None, "The file must contain a JSON object."
    return data, None


def save_config(cfg):
    write_json(CONFIG_PATH, cfg, indent=4)


def is_path_setting(key, value):
    """Settings that point to a file or folder are never shown on this page."""
    if str(key).lower().endswith(("path", "_file", "_dir", "_folder")):
        return True
    return isinstance(value, str) and bool(re.match(r"^([A-Za-z]:[\\/]|[\\/]|\.{1,2}[\\/]|~)", value))


def section(cfg, name):
    body = cfg.get(name)
    return body if isinstance(body, dict) else {}


def as_num(v, default, integer=False):
    try:
        n = float(v)
        if not math.isfinite(n):
            raise ValueError
    except (TypeError, ValueError):
        return default
    return int(n) if integer else n


def hour_label(h):
    return f"{h % 12 or 12}:00 {'AM' if h < 12 else 'PM'}"


def setting_number(label, val, default, lo, hi, step, help, integer=False):
    v = as_num(val, default, integer)
    lo, hi = min(lo, v), max(hi, v)         # never reject a value that is already in the file
    if integer:
        return int(st.number_input(label, min_value=int(lo), max_value=int(hi), value=int(v),
                                   step=int(step), help=help))
    return round(float(st.number_input(label, min_value=float(lo), max_value=float(hi), value=float(v),
                                       step=float(step), format="%.2f", help=help)), 4)


def other_settings(cfg):
    """Plain settings (text, numbers, on/off) that have no widget of their own above."""
    out = []
    for sec, body in cfg.items():
        if isinstance(body, dict):
            for k, v in body.items():
                if (k not in KNOWN_SETTINGS.get(sec, ()) and not is_path_setting(k, v)
                        and isinstance(v, (bool, int, float, str))):
                    out.append((sec, k, v))
        elif (sec not in KNOWN_SETTINGS and not is_path_setting(sec, body)
              and isinstance(body, (bool, int, float, str))):
            out.append((None, sec, body))
    return out


def other_field(sec, key, val):
    pretty = key.replace("_", " ")
    label = pretty.capitalize() if sec is None else f"{sec.replace('_', ' ')}: {pretty}"
    wkey = f"cfg_{sec}_{key}"
    if isinstance(val, bool):
        return st.checkbox(label, value=val, key=wkey)
    if isinstance(val, int):
        return int(st.number_input(label, value=val, step=1, key=wkey))
    if isinstance(val, float):
        return float(st.number_input(label, value=val, format="%g", key=wkey))
    return st.text_input(label, value=val, key=wkey)


def config_view():
    cfg, problem = load_config()
    file_name = os.path.basename(CONFIG_PATH)

    st.header("Agent configuration")
    st.markdown("Settings for the watcher that collects new RBI notifications and for the model "
                "that checks them and answers questions.")
    if problem:
        st.error(f"`{file_name}` can't be read, so nothing can be changed here. Fix the file and "
                 f"reload this page. Details: {problem}")
        return
    if not cfg:
        st.info(f"`{file_name}` doesn't exist yet. Saving creates it with the values below.")
    st.caption(f"Loaded from and saved to `{file_name}`. Model names are shown for reference only. "
               "Settings that point to files or folders, and the watcher's own progress markers, "
               "stay in the file and are not shown here.")

    sched = section(cfg, "scraper_settings")
    hour_now = as_num(sched.get("time_scraper"), 9, integer=True)
    if not 0 <= hour_now <= 23:
        st.warning("The saved run time is not an hour between 0 and 23. Pick a new one below.")
        hour_now = 9
    runs_now = max(1, as_num(sched.get("run_count"), 1, integer=True))
    st.markdown(f"**Current schedule:** run time {hour_label(hour_now)}, "
                f"{runs_now} run{'' if runs_now == 1 else 's'} per day.")

    models, hp = section(cfg, "Model_Details"), section(cfg, "Hyper_Param")
    with st.form("agent_config"):
        st.subheader("Schedule")
        a, b = st.columns(2)
        with a:
            hour = st.selectbox("Run time", HOURS, index=hour_now, format_func=hour_label,
                                help="Hour of the day when the scraper runs. For example, 9:00 AM.")
        with b:
            runs = st.number_input("Runs per day", min_value=1, max_value=max(24, runs_now),
                                   value=runs_now, step=1,
                                   help="How many times the scraper runs in one day.")

        st.subheader("Models")
        st.caption("Set in the config file and shown here for reference. They can't be changed from this page.")
        a, b = st.columns(2)
        with a:
            st.text_input("Main model", value=str(models.get("Main_Model") or "Not set"), disabled=True,
                          help="The language model the agent uses to check notifications and answer questions.")
        with b:
            st.text_input("Embedding model", value=str(models.get("Embedding_Model") or "Not set"),
                          disabled=True, help="The model the agent uses to search the master directions.")

        st.subheader("Generation settings")
        st.caption("How the main model writes its answers.")
        values = {}
        cols = st.columns(3)
        for i, (key, label, integer, lo, hi, step, default, help_) in enumerate(HYPER_PARAMS):
            with cols[i % 3]:
                values[key] = setting_number(label, hp.get(key), default, lo, hi, step, help_, integer)

        extra_values = []
        other = other_settings(cfg)
        if other:
            st.subheader("Other settings")
            extra_values = [(sec, k, other_field(sec, k, v)) for sec, k, v in other]

        saved = st.form_submit_button("Save agent configuration", type="primary")

    if saved:
        # Model_Details and data_check are left out on purpose: deep_merge keeps them as they are in the file.
        new = {
            "scraper_settings": {"time_scraper": hour, "run_count": int(runs)},
            "Hyper_Param": values,
        }
        for sec, k, v in extra_values:
            if sec is None:
                new[k] = v
            else:
                new.setdefault(sec, {})[k] = v
        try:
            save_config(deep_merge(cfg, new))     # keeps every setting this page does not show
        except OSError as e:
            st.error(f"Could not save to `{file_name}`: {e}")
            return
        st.success("Agent configuration saved. If the watcher is already running, restart it so it "
                   "picks up the new values.")


if view == CHAT_VIEW:
    chat_view()
elif view == NOTES_VIEW:
    notifications_panel()
elif view == COMPANY_VIEW:
    company_view()
else:
    config_view()