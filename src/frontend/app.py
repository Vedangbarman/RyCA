import inspect
import io
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
# The weekly error files live in <project root>/data/error_logs. The project root is the folder
# that holds config.json.
ERROR_LOG_DIR = os.getenv("ERROR_LOG_DIR", os.path.join(
    os.path.dirname(os.path.abspath(CONFIG_PATH)), "data", "error_logs"))
CHAT_TIMEOUT = 1800        # a chat message waits behind any running notification batch
DEFAULT_NAME = "gatorade"
STARTERS = [
    "What is the Upper Layer in NBFCs?",
    "Which master directions apply to all NBFCs?",
    "What are the large exposure limits for IDF-NBFCs?",
]
CHAT_VIEW, NOTES_VIEW, COMPANY_VIEW, CONFIG_VIEW, ERRORS_VIEW = (
    "Chat", "Notifications", "Company details", "Agent configuration", "Error logs")

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
    view = st.radio("View", [CHAT_VIEW, NOTES_VIEW, COMPANY_VIEW, CONFIG_VIEW, ERRORS_VIEW],
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
LAYERS = ["BASE", "MIDDLE", "UPPER", "TOP"]       # lowest to highest
LEGACY = ["NBFC-D", "NBFC-ND", "NBFC-ND-SI"]
COR_STATUS = ["active", "suspended", "cancelled", "surrendered"]
SECURITISATION_ROLES = ["originator", "investor", "servicer"]
CUSTOMER_TYPES = ["MSME", "salaried", "self_employed", "retail", "corporate", "agriculture"]
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
EXPORT_IMPORT = ["none", "exports", "imports", "both"]
STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh",
    "Lakshadweep", "Puducherry",
]

# columns of the lending products table, in the order they appear in the profile
PRODUCT_COLS = ["product_type", "secured", "customer_segment", "share_of_aum_percent",
                "digitally_originated", "co_lent"]
EVENT_COLS = ["name", "relevant_if", "currently_relevant"]
TRI_TO_LABEL = {None: "Unknown", True: "Yes", False: "No"}
LABEL_TO_TRI = {"Yes": True, "No": False}


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
    """Read a nested value like g(p, 'gates.layer'). A missing or null value gives the default."""
    for k in path.split("."):
        if not isinstance(d, dict) or d.get(k) is None:
            return default
        d = d[k]
    return d


# --- input helpers. Each returns None for "unknown" so the agent can tell it apart from "no".

def txt(label, val, help):
    return st.text_input(label, value=val or "", help=help).strip() or None


def area(label, val, help, height=80):
    return st.text_area(label, value=val or "", help=help, height=height).strip() or None


def lines(label, val, help, height=100):
    raw = st.text_area(label, value="\n".join(val or []), help=help + " One per line.", height=height)
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


def num(label, val, help, errors, integer=False, lo=None, hi=None):
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
    if lo is not None and n < lo:
        errors.append(f"{label}: must be {lo:g} or more.")
        return val
    if hi is not None and n > hi:
        errors.append(f"{label}: must be {hi:g} or less.")
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


def profile_warnings(new):
    """Answers that contradict each other. These do not stop the save, they are shown after it."""
    gates, rel = new["gates"], new["relevance"]
    products = rel["lending_products"]
    funding = rel["funding_instruments"]
    digital, co = rel["digital_lending"], rel["co_lending"]
    ecb = rel["cross_border"]["has_foreign_borrowing_ecb"]
    out = []

    share = sum(r.get("share_of_aum_percent") or 0 for r in products)
    if products and abs(share - 100) > 0.5:
        out.append(f"Product shares add up to {share:g}%, not 100%.")

    layer, applicable = gates["layer"], gates["applicable_layers"]
    if layer in LAYERS:
        usual = LAYERS[:LAYERS.index(layer) + 1]
        if set(applicable) != set(usual):
            out.append(f"The layer is {layer} but the applicable layers are "
                       f"{', '.join(applicable) or 'empty'}. Lower-layer rules usually carry upward, "
                       f"so the usual list is {', '.join(usual)}.")

    total, group = gates["total_assets_crore"], gates["group_aggregate_nbfc_assets_crore"]
    if total is not None and group is not None and group < total:
        out.append("Group NBFC assets are lower than total assets, but the group figure includes "
                   "this company.")

    if "public_deposits" in funding and gates["is_deposit_taking"] is not True:
        out.append("Public deposits are listed as a funding source but \"Takes public deposits\" "
                   "is not Yes.")
    if "ecb" in funding and ecb is not True:
        out.append("ECB is listed as a funding source but \"Foreign borrowing (ECB)\" is not Yes.")
    if ecb is True and "ecb" not in funding:
        out.append("\"Foreign borrowing (ECB)\" is Yes but ECB is not listed as a funding source.")

    flagged = {r["product_type"] for r in products if r.get("co_lent") and r.get("product_type")}
    if co["active"] is False and (flagged or co["products"] or co["partner_types"]):
        out.append("Co-lending is marked not active, but co-lent products or partners are listed.")
    elif flagged != set(co["products"]):
        out.append("The products marked Co-lent on the Business tab differ from the co-lent products "
                   "on the Digital and co-lending tab.")

    if digital["offers_digital_loans"] is False and (
            digital["has_own_lending_app"] is True or any(r.get("digitally_originated") for r in products)):
        out.append("\"Offers digital loans\" is No, but there is a lending app or a digitally "
                   "originated product.")
    return out


def company_view():
    p = load_profile()
    errors = []

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
    rel = {}
    new["relevance"] = rel
    with st.form("company_form"):
        t_id, t_cls, t_dir, t_biz, t_dig, t_fund = st.tabs([
            "Identity", "Classification", "Applicable directions", "Business",
            "Digital and co-lending", "Funding and cross-border"])

        # ----- Identity -----
        with t_id:
            st.subheader("Profile")
            schema = g(p, "profile_meta.schema_version")
            if schema:
                st.caption(f"Profile format version {schema}.")
            a, b = st.columns(2)
            with a:
                profile_id = txt("Profile ID", g(p, "profile_meta.profile_id"),
                                 "A short label for this profile, for example NBFC-DEMO-0001.")
                as_of = date_txt("Figures as of", g(p, "profile_meta.as_of_date"),
                                 "Date the financial figures are taken from, usually the last balance "
                                 "sheet date.", errors)
                verified = date_txt("Last verified on", g(p, "profile_meta.last_verified_on"),
                                    "Date someone last checked these details against the source.", errors)
            with b:
                framework = txt("Regulatory framework", g(p, "profile_meta.framework"),
                                "The set of RBI directions this profile was prepared against, for "
                                "example RBI NBFC Directions 2025.")
                demo = st.checkbox("This is demo data", value=bool(g(p, "profile_meta.is_demo_data", False)),
                                   help="Tick this if the company is made up for testing.")
            new["profile_meta"] = {"profile_id": profile_id, "as_of_date": as_of, "last_verified_on": verified,
                                   "is_demo_data": demo, "framework": framework}

            st.subheader("Company")
            a, b = st.columns(2)
            with a:
                legal = txt("Legal name", g(p, "match_keys.legal_name"), "Registered name of the company.")
                cor_no = txt("RBI Certificate of Registration no.",
                             g(p, "match_keys.rbi_certificate_of_registration_no"),
                             "Number on your RBI Certificate of Registration, for example N-14.00000.")
            with b:
                cor_status = pick("Certificate status", COR_STATUS, g(p, "match_keys.cor_status"),
                                  "Current status of your Certificate of Registration.")
                state = pick("Registered office state", STATES, g(p, "match_keys.registered_office_state"),
                             "State where the registered office is located.")
            new["match_keys"] = {"legal_name": legal, "rbi_certificate_of_registration_no": cor_no,
                                 "cor_status": cor_status, "registered_office_state": state}

        # ----- Classification -----
        with t_cls:
            st.caption("This decides which RBI directions cover you, so it matters most.")
            st.subheader("Category and layer")
            a, b = st.columns(2)
            with a:
                entity = pick("Entity type", ENTITY_TYPES, g(p, "gates.entity_type"),
                              "The kind of regulated entity. NBFC for a non-banking financial company.")
                cat = pick("NBFC category", CATEGORIES, g(p, "gates.nbfc_category"),
                           "Your main category as stated on the Certificate of Registration.")
                sec = multi("Other categories", CATEGORIES, g(p, "gates.secondary_categories"),
                            "Any additional categories you are registered under.")
                legacy = multi("Older terms", LEGACY, g(p, "gates.legacy_terms"),
                               "Pre-2022 labels that older circulars still use, for example NBFC-ND-SI.")
            with b:
                layer = pick("Layer", LAYERS, g(p, "gates.layer"),
                             "Layer under RBI's scale-based regulation. For example, a non-deposit-taking "
                             "ICC with assets of Rs 1,000 crore or more is in the Middle Layer.")
                layer_label = txt("Layer label", g(p, "gates.layer_label"),
                                  "Short form of the layer, for example NBFC-ML for the Middle Layer.")
                layer_basis = txt("Why this layer", g(p, "gates.layer_basis"),
                                  "One line on how the layer was decided, for example non-deposit-taking "
                                  "with total assets of Rs 1,000 crore or more.")
                applicable_layers = multi("Layers whose notifications apply", LAYERS,
                                          g(p, "gates.applicable_layers"),
                                          "Notifications written for these layers apply to you. Lower-layer "
                                          "rules usually carry upward, so a Middle Layer company lists "
                                          "Base and Middle.")
            layer_note = area("Layer note", g(p, "gates.layer_note"),
                              "Free text explaining how layers apply to this company.")

            st.subheader("Status")
            a, b = st.columns(2)
            with a:
                dep = tri("Takes public deposits", g(p, "gates.is_deposit_taking"),
                          "Whether you accept public deposits.")
                pubfunds = tri("Accesses public funds", g(p, "gates.accesses_public_funds"),
                               "Whether you raise money from the public, for example through debentures.")
                cust = tri("Has customer interface", g(p, "gates.has_customer_interface"),
                           "Whether you deal directly with customers.")
            with b:
                govt = tri("Government owned", g(p, "gates.is_government_owned"),
                           "Whether the government owns you. Some circulars exclude these.")
                bank_sub = tri("Subsidiary of a bank", g(p, "gates.is_subsidiary_of_bank"),
                               "Whether a bank owns you.")
                exemptions = lines("Exemptions claimed", g(p, "gates.exemptions_claimed"),
                                   "Any RBI exemptions you rely on.")

            st.subheader("Size")
            st.caption("Amounts in Rs crore. Many circulars apply only above a size threshold.")
            a, b = st.columns(2)
            with a:
                ta = num("Total assets (Rs crore)", g(p, "gates.total_assets_crore"),
                         "Total assets on the balance sheet.", errors, lo=0)
            with b:
                grp = num("Group NBFC assets (Rs crore)", g(p, "gates.group_aggregate_nbfc_assets_crore"),
                          "Combined assets of all NBFCs in your group, including this one.", errors, lo=0)

            st.subheader("Does not apply to you")
            st.caption("Notifications aimed only at these do not apply to you.")
            a, b = st.columns(2)
            with a:
                na_cats = lines("Categories that do not apply", g(p, "gates.not_applicable_categories"),
                                "NBFC types you are not, for example NBFC-MFI or CIC.", height=220)
            with b:
                na_exempt = lines("Exempt entity types that do not apply",
                                  g(p, "gates.not_applicable_exempt_entities"),
                                  "Kinds of exempt entities you are not, for example insurance companies "
                                  "or stock brokers.", height=220)
            new["gates"] = {
                "entity_type": entity, "nbfc_category": cat, "secondary_categories": sec,
                "layer": layer, "layer_label": layer_label, "layer_basis": layer_basis,
                "applicable_layers": applicable_layers, "layer_note": layer_note,
                "legacy_terms": legacy, "is_deposit_taking": dep, "accesses_public_funds": pubfunds,
                "has_customer_interface": cust, "is_government_owned": govt,
                "is_subsidiary_of_bank": bank_sub, "exemptions_claimed": exemptions,
                "total_assets_crore": ta, "group_aggregate_nbfc_assets_crore": grp,
                "not_applicable_categories": na_cats, "not_applicable_exempt_entities": na_exempt}

        # ----- Applicable directions -----
        with t_dir:
            st.caption("Which RBI directions cover this company: the ones for all NBFCs, the ones "
                       "written for its category, and the ones that apply only when something happens.")
            a, b = st.columns(2)
            with a:
                title_pattern = txt("Direction title pattern", g(p, "applicable_directions.title_pattern"),
                                    "How the directions are titled. {name} stands for the name of one direction.")
            with b:
                source = txt("Source circular", g(p, "applicable_directions.source"),
                             "The RBI circular and paragraphs these lists come from.")
            all_nbfc = lines("Directions that apply to all NBFCs", g(p, "applicable_directions.apply_to_all_nbfcs"),
                             "The name of each direction.", height=320)
            a, b = st.columns(2)
            with a:
                type_specific = lines("Directions for your category only",
                                      g(p, "applicable_directions.type_specific"),
                                      "Directions written only for your NBFC category. Leave empty if there are none.")
            with b:
                type_note = area("Note on category-specific directions",
                                 g(p, "applicable_directions.type_specific_note"),
                                 "For example, why the list above is empty.", height=100)

            st.markdown("**Event or activity driven directions**")
            st.caption("Directions that apply only when something happens or you do something. "
                       "\"Currently relevant\" says whether that is true right now.")
            ev_rows = []
            for r in g(p, "applicable_directions.event_or_activity_driven", []):
                r = dict(r)
                r["currently_relevant"] = TRI_TO_LABEL.get(r.get("currently_relevant"), "Unknown")
                ev_rows.append(r)
            events = table(ev_rows, EVENT_COLS, {
                "name": st.column_config.TextColumn("Direction"),
                "relevant_if": st.column_config.TextColumn("Relevant if"),
                "currently_relevant": st.column_config.SelectboxColumn(
                    "Currently relevant", options=["Unknown", "Yes", "No"]),
            })
            for r in events:
                r["currently_relevant"] = LABEL_TO_TRI.get(r.get("currently_relevant"))
            new["applicable_directions"] = {
                "title_pattern": title_pattern, "source": source, "type_specific": type_specific,
                "type_specific_note": type_note, "apply_to_all_nbfcs": all_nbfc,
                "event_or_activity_driven": events}

        # ----- Business -----
        with t_biz:
            st.markdown("**Lending products**")
            st.caption("One row per product. Add rows with the + at the bottom of the table. "
                       "Shares of assets under management should add up to 100.")
            prod_rows = g(p, "relevance.lending_products", [])
            products = table(prod_rows, PRODUCT_COLS, {
                "product_type": st.column_config.SelectboxColumn(
                    "Product", options=with_existing(PRODUCT_TYPES, prod_rows, "product_type")),
                "secured": st.column_config.CheckboxColumn("Secured"),
                "customer_segment": st.column_config.SelectboxColumn(
                    "Segment", options=with_existing(CUSTOMER_TYPES, prod_rows, "customer_segment")),
                "share_of_aum_percent": st.column_config.NumberColumn("% of AUM", min_value=0, max_value=100),
                "digitally_originated": st.column_config.CheckboxColumn(
                    "Digital", help="Loans are originated through an app or online."),
                "co_lent": st.column_config.CheckboxColumn(
                    "Co-lent", help="Loans are made jointly with another lender."),
            })
            for r in products:
                for k in ("secured", "digitally_originated", "co_lent"):
                    r[k] = bool(r.get(k))
            rel["lending_products"] = products

            st.markdown("**Other activities**")
            a, b = st.columns(2)
            with a:
                roles = multi("Securitisation role", SECURITISATION_ROLES,
                              g(p, "relevance.other_activities.securitisation_roles"),
                              "Which side of securitisation or assignment deals you are on. "
                              "Leave empty if you do none.")
                insurance = tri("Insurance distribution", g(p, "relevance.other_activities.insurance_distribution"),
                                "Whether you sell insurance products as a corporate agent or broker.")
            with b:
                recovery = tri("Uses recovery agents", g(p, "relevance.other_activities.uses_recovery_agents"),
                               "Whether you use agents to recover overdue loans.")
                outsourced = multi("Outsourced functions", OUTSOURCED,
                                   g(p, "relevance.other_activities.outsourced_functions"),
                                   "Work you hand to outside firms.")
            rel["other_activities"] = {
                "securitisation_roles": roles, "insurance_distribution": insurance,
                "uses_recovery_agents": recovery, "outsourced_functions": outsourced}

        # ----- Digital and co-lending -----
        with t_dig:
            st.caption("Covers the RBI Digital Lending Directions and co-lending arrangements.")
            st.subheader("Digital lending")
            a, b = st.columns(2)
            with a:
                offers = tri("Offers digital loans", g(p, "relevance.digital_lending.offers_digital_loans"),
                             "Whether you give loans through digital channels.")
                has_app = tri("Has its own lending app", g(p, "relevance.digital_lending.has_own_lending_app"),
                              "Whether you run a lending app under your own name.")
            with b:
                uses_lsp = tri("Uses lending service providers",
                               g(p, "relevance.digital_lending.uses_lending_service_providers"),
                               "Whether third parties help you source, underwrite or collect on loans.")
                dlg = tri("Uses default loss guarantee",
                          g(p, "relevance.digital_lending.uses_default_loss_guarantee"),
                          "Whether a partner guarantees you against loan losses.")
            rel["digital_lending"] = {
                "offers_digital_loans": offers, "has_own_lending_app": has_app,
                "uses_lending_service_providers": uses_lsp, "uses_default_loss_guarantee": dlg}

            st.subheader("Co-lending")
            a, b = st.columns(2)
            with a:
                co_active = tri("Co-lending active", g(p, "relevance.co_lending.active"),
                                "Whether you currently co-lend with another lender.")
                co_partners = multi("Co-lending partner types", PARTNER_TYPES,
                                    g(p, "relevance.co_lending.partner_types"),
                                    "Kinds of lenders you co-lend with.")
            with b:
                co_products = multi("Co-lent products", PRODUCT_TYPES, g(p, "relevance.co_lending.products"),
                                    "Products you lend jointly with a partner.")
            rel["co_lending"] = {"active": co_active, "partner_types": co_partners, "products": co_products}

        # ----- Funding and cross-border -----
        with t_fund:
            st.subheader("Funding")
            funding = multi("Funding sources", FUNDING_TYPES, g(p, "relevance.funding_instruments"),
                            "Every source you borrow from.")
            rel["funding_instruments"] = funding

            st.subheader("Payments and cards")
            a, b = st.columns(2)
            with a:
                pay_auth = tri("Holds a payment system authorisation",
                               g(p, "relevance.payments_and_cards.holds_payment_authorisation"),
                               "Whether RBI has authorised you to run a payment system, for example "
                               "prepaid payment instruments.")
            with b:
                cards = tri("Issues credit cards", g(p, "relevance.payments_and_cards.issues_credit_cards"),
                            "Whether you issue credit cards.")
            rel["payments_and_cards"] = {"holds_payment_authorisation": pay_auth, "issues_credit_cards": cards}

            st.subheader("Cross-border")
            a, b = st.columns(2)
            with a:
                trade = pick("Exports or imports", EXPORT_IMPORT, g(p, "relevance.cross_border.exports_or_imports"),
                             "Whether the company has export or import business. Pick none if it has neither.")
                ad = tri("Authorised dealer", g(p, "relevance.cross_border.is_authorised_dealer"),
                         "Whether you hold an RBI authorised dealer licence for foreign exchange.")
                overseas = tri("Overseas investment made", g(p, "relevance.cross_border.overseas_investment_made"),
                               "Whether you have invested in entities outside India.")
            with b:
                ecb = tri("Foreign borrowing (ECB)", g(p, "relevance.cross_border.has_foreign_borrowing_ecb"),
                          "Whether you have external commercial borrowings.")
                foreign = num("Foreign holding (%)", g(p, "relevance.cross_border.foreign_holding_percent"),
                              "Share of the company held by foreign investors.", errors, lo=0, hi=100)
            rel["cross_border"] = {
                "exports_or_imports": trade, "is_authorised_dealer": ad, "has_foreign_borrowing_ecb": ecb,
                "foreign_holding_percent": foreign, "overseas_investment_made": overseas}

        submitted = st.form_submit_button("Save company details", type="primary")

    if submitted:
        if errors:
            st.error("Not saved. Please fix:\n\n" + "\n".join(f"- {e}" for e in errors))
            return
        try:
            save_profile(deep_merge(p, new))     # keeps every key this page does not edit
        except OSError as e:
            st.error(f"Could not save to `{PROFILE_PATH}`: {e}")
            return
        st.success("Company details saved.")
        for w in profile_warnings(new):
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

# What the retrieval filter and profile settings mean (they are read by build_retriever in the backend).
# "company" is accepted by the backend as another name for "all", so it has a label but no entry
# in the list of choices.
RETRIEVE_SOURCES = {
    "master_directories": "Master directions only",
    "notifications": "Notifications only",
    "both": "Notifications and master directions",
    "all": "Everything: notifications, master directions and company uploaded data",
}
RETRIEVE_SOURCE_LABELS = {**RETRIEVE_SOURCES, "company": "Everything (same as all)"}
RETRIEVE_PROFILES = {
    "default": "Default: read the company profile from the JSON file",
    "custom": "Custom: also search the profile passages stored with the documents",
}
DEVICES = {"cuda": "cuda (graphics card)", "cpu": "cpu (processor)"}
RETRIEVE_KEYS = {"filter", "answer_by_profile", "profile"}

# Settings this page handles itself. Model_Details is shown read-only and data_check (the watcher's
# reference dates and article link) is not shown at all. Both are listed here so they never turn up
# under "Other settings", and neither is ever written by this page, so the file keeps its values.
KNOWN_SETTINGS = {
    "scraper_settings": {"time_scraper", "run_count"},
    "data_check": {"ref_date", "delta_ref_date", "ref_article"},
    "Model_Details": {"Embedding_Model", "Main_Model"},
    "Hyper_Param": {h[0] for h in HYPER_PARAMS},
    "misc": {"tool_calling", "tool_calling_retries", "device", "jev"},
    "retrieve_chat": RETRIEVE_KEYS,
    "retrieve_watcher": RETRIEVE_KEYS,
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


def choice(label, options, val, default, labels, help, key=None):
    """Dropdown over fixed options. A value already in the file that is not one of them is kept."""
    opts = list(options)
    if val is not None and val not in opts:
        opts.append(val)
    current = val if val in opts else default
    return st.selectbox(label, opts, index=opts.index(current),
                        format_func=lambda o: labels.get(o, str(o)), help=help, key=key)


def switch(label, val, words, help, key, default=False):
    """On/off tick box for a setting stored as words ("on"/"off", "yes"/"no") or as a true/false."""
    on = default if val is None else (
        val is True or str(val).strip().lower() in ("on", "yes", "true", "1"))
    ticked = st.checkbox(label, value=on, help=help, key=key)
    return ticked if isinstance(val, bool) else (words[0] if ticked else words[1])


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

    models, hp, misc = section(cfg, "Model_Details"), section(cfg, "Hyper_Param"), section(cfg, "misc")
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

        st.subheader("Runtime")
        a, b = st.columns(2)
        with a:
            device = choice("Device", DEVICES, misc.get("device"), "cuda", DEVICES,
                            "Where the models run.", key="misc_device")
            tool_calling = switch("Tool calling", misc.get("tool_calling"), ("on", "off"),
                                  "When on, the model may call tools while it works on an answer.",
                                  key="misc_tool_calling")
        with b:
            retries = setting_number("Tool calling retries", misc.get("tool_calling_retries"), 1, 0, 10, 1,
                                     "How many times to retry a tool call that fails.", integer=True)
            jev = switch("JEV", misc.get("jev"), ("on", "off"),
                         "An on or off switch the backend reads. Leave it as it is unless you know what "
                         "it controls.", key="misc_jev")

        st.subheader("Retrieval")
        st.caption("What the model may look up before it answers. Chat is the question chat. "
                   "Watcher is the check of new notifications.")
        retrieve = {}
        cols = st.columns(2)
        for col, sec, title in [(cols[0], "retrieve_chat", "Chat"), (cols[1], "retrieve_watcher", "Watcher")]:
            body = section(cfg, sec)
            with col:
                st.markdown(f"**{title}**")
                retrieve[sec] = {
                    "filter": choice("Search in", RETRIEVE_SOURCES, body.get("filter"), "both",
                                     RETRIEVE_SOURCE_LABELS,
                                     "Which documents are searched for passages to answer from: the master "
                                     "directions, the notifications, or everything including the data your "
                                     "company uploaded.", key=f"{sec}_filter"),
                    "answer_by_profile": switch("Answer using the company profile",
                                                body.get("answer_by_profile"), ("yes", "no"),
                                                "Lets the model use your company details when it answers.",
                                                key=f"{sec}_answer_by_profile", default=True),
                    "profile": choice("Profile source", RETRIEVE_PROFILES, body.get("profile"), "default",
                                      RETRIEVE_PROFILES,
                                      "Default reads the company profile from the JSON file. Custom also "
                                      "searches profile passages stored with the other documents.",
                                      key=f"{sec}_profile"),
                }

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
            "misc": {"tool_calling": tool_calling, "tool_calling_retries": retries,
                     "device": device, "jev": jev},
            **retrieve,
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
        st.success("Agent configuration saved. If the backend or the watcher is already running, "
                   "restart it so it picks up the new values.")


# ---------- error logs view ----------

# The backend writes each error with these fields. Other names are mapped onto them when a file
# uses a different spelling, and any further columns are kept as they are.
LOG_COLUMNS = ["time", "error_file", "error_message", "error_count", "trace_back"]
LOG_ALIASES = {
    "time": "time", "timestamp": "time", "datetime": "time", "date": "time",
    "error_file": "error_file", "file": "error_file", "source": "error_file", "module": "error_file",
    "error_message": "error_message", "message": "error_message", "error": "error_message",
    "error_count": "error_count", "count": "error_count",
    "trace_back": "trace_back", "traceback": "trace_back", "trace": "trace_back",
}
LOG_EXTENSIONS = (".csv", ".json", ".jsonl", ".ndjson")
SHOWN_LOG_COLUMNS = {"time": "Time (UTC)", "error_file": "Where", "error_message": "Error",
                     "error_count": "Count", "log_file": "Weekly file"}
MAX_PICKABLE_ERRORS = 200


def json_records(obj):
    """Turn whatever a JSON file holds into a list of records."""
    if isinstance(obj, list):
        return [x if isinstance(x, dict) else {"error_message": str(x)} for x in obj]
    if isinstance(obj, dict):
        nested = [v for v in obj.values()
                  if isinstance(v, list) and v and all(isinstance(i, dict) for i in v)]
        return [i for v in nested for i in v] if nested else [obj]
    return [{"error_message": str(obj)}]


def read_log_file(path):
    """The rows of one weekly file. Raises if the file can't be understood."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    if not text.strip():
        return []
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False).to_dict("records")
    if ext == ".json":
        return json_records(json.loads(text))
    rows = []                                   # .jsonl / .ndjson: one JSON object per line
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                rows.extend(json_records(json.loads(line)))
            except ValueError:
                raise ValueError(f"line {n} is not valid JSON")
    return rows


def normalise_columns(df):
    seen, rename = set(), {}
    for c in df.columns:
        target = LOG_ALIASES.get(str(c).strip().lower().replace(" ", "_"))
        if target and target not in seen:
            seen.add(target)
            rename[c] = target
    df = df.rename(columns=rename)
    return df.loc[:, ~df.columns.duplicated()]


def list_log_files():
    """(supported files, other files), or None if the folder does not exist."""
    if not os.path.isdir(ERROR_LOG_DIR):
        return None
    supported, other = [], []
    for root, _, names in os.walk(ERROR_LOG_DIR):
        for n in names:
            (supported if n.lower().endswith(LOG_EXTENSIONS) else other).append(os.path.join(root, n))
    return sorted(supported), sorted(other)


@st.cache_data(show_spinner=False)
def load_error_logs(signature):
    """All weekly files as one table, newest first, plus a line per file saying what happened to it.
    The signature (path, modified time, size of every file) only serves as the cache key, so the
    table is rebuilt whenever a weekly file changes or a new one appears."""
    frames, files = [], []
    for path, _, _ in signature:
        name = os.path.relpath(path, ERROR_LOG_DIR)
        try:
            rows = read_log_file(path)
        except Exception as e:                  # one bad file must not hide the others
            files.append({"File": name, "Errors": 0, "Status": f"Could not read: {e}"})
            continue
        files.append({"File": name, "Errors": len(rows), "Status": "OK"})
        if rows:
            frame = normalise_columns(pd.DataFrame(rows))
            frame["log_file"] = name
            frames.append(frame)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for c in LOG_COLUMNS + ["log_file"]:
        if c not in df:
            df[c] = ""
    df = df.fillna("").astype(str)

    try:
        parsed = pd.to_datetime(df["time"], errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):             # older pandas has no format="mixed"
        parsed = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df["_ts"] = parsed
    df["time"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S").where(parsed.notna(), df["time"])
    df = df.sort_values("_ts", ascending=False, na_position="last", kind="stable").reset_index(drop=True)

    extra = [c for c in df.columns if c not in LOG_COLUMNS + ["log_file", "_ts"]]
    return df[LOG_COLUMNS + extra + ["log_file", "_ts"]], files


def one_line(text, limit=80):
    return " ".join(str(text).split())[:limit]


def errors_view():
    st.header("Error logs")
    st.caption(f"Every weekly file in Error Logs Folder, combined into one list with the newest first. "
               "The list reloads by itself when a weekly file changes.")
    st.button("Refresh")

    found = list_log_files()
    if found is None or not (found[0] or found[1]):
        st.info("No error files yet. They appear here once the backend has logged an error.")
        return
    supported, other = found
    if not supported:
        st.info(f"The folder has no files this page can read ({', '.join(LOG_EXTENSIONS)}).")
        return

    signature = []
    for path in supported:
        try:
            stat = os.stat(path)
        except OSError:
            continue                            # removed while we were looking
        signature.append((path, stat.st_mtime, stat.st_size))
    df, files = load_error_logs(tuple(signature))

    unreadable = [f for f in files if f["Status"] != "OK"]
    if unreadable:
        st.warning(f"{len(unreadable)} of {len(files)} files could not be read and are left out. "
                   "Details are under Files loaded.")
    with st.expander(f"Files loaded ({len(files)})"):
        st.dataframe(pd.DataFrame(files), hide_index=True)
        if other:
            st.caption("Skipped because of the file type: "
                       + ", ".join(f"`{os.path.relpath(f, ERROR_LOG_DIR)}`" for f in other))

    if df.empty:
        st.info("The weekly files are there but hold no errors.")
        return

    # ----- filters -----
    c1, c2, c3 = st.columns([2, 2, 3])
    wheres = c1.multiselect("Where it happened", sorted(x for x in df["error_file"].unique() if x))
    weeks = c2.multiselect("Weekly file", list(dict.fromkeys(df["log_file"])))
    query = c3.text_input("Search the message or traceback")
    view = df
    if wheres:
        view = view[view["error_file"].isin(wheres)]
    if weeks:
        view = view[view["log_file"].isin(weeks)]
    if query.strip():
        text = (view["error_message"] + " " + view["trace_back"]).str.lower()
        view = view[text.str.contains(query.strip().lower(), regex=False)]

    stamps = view["_ts"].dropna()
    m1, m2, m3 = st.columns(3)
    m1.metric("Errors", len(view))
    m2.metric("First", stamps.min().strftime("%d %b %Y %H:%M") if len(stamps) else "-")
    m3.metric("Latest", stamps.max().strftime("%d %b %Y %H:%M") if len(stamps) else "-")

    if view.empty:
        st.info("No errors match these filters.")
        return

    if len(stamps):
        st.caption("Errors per day (UTC)")
        st.bar_chart(stamps.dt.strftime("%Y-%m-%d").value_counts().sort_index())

    # ----- combined list -----
    shown = [c for c in view.columns if c not in ("_ts", "trace_back")]
    st.dataframe(
        view[shown], hide_index=True,
        column_config={c: st.column_config.TextColumn(label, width="large" if c == "error_message" else None)
                       for c, label in SHOWN_LOG_COLUMNS.items()})

    export = view.drop(columns=["_ts"])
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("Download as one CSV", data=export.to_csv(index=False).encode("utf-8-sig"),
                       file_name="error_logs_combined.csv", mime="text/csv")
    d2.download_button("Download as one JSON",
                       data=export.to_json(orient="records", force_ascii=False, indent=2),
                       file_name="error_logs_combined.json", mime="application/json")
    st.caption("The downloads hold the errors listed above: all of them when no filter is set.")

    # ----- one error in full -----
    st.subheader("Details")
    recent = view.head(MAX_PICKABLE_ERRORS).reset_index(drop=True)
    if len(view) > len(recent):
        st.caption(f"Showing the newest {len(recent)} of the {len(view)} matching errors. "
                   "Use the filters to narrow the list down.")
    labels = [f"{r.time}  |  {r.error_file or '-'}  |  {one_line(r.error_message)}"
              for r in recent.itertuples()]
    i = st.selectbox("Open an error", range(len(recent)), format_func=lambda n: labels[n])
    row = recent.iloc[i]
    st.markdown(f"**{row['error_file'] or 'Unknown source'}** at {row['time'] or 'unknown time'} "
                f"(weekly file `{row['log_file']}`)")
    if row["error_count"]:
        st.caption(f"Error count: {row['error_count']}")
    st.code(row["error_message"] or "-", language="text")
    if row["trace_back"] and row["trace_back"] != "Not Applicable":
        st.markdown("**Traceback**")
        st.code(row["trace_back"], language="text")


if view == CHAT_VIEW:
    chat_view()
elif view == NOTES_VIEW:
    notifications_panel()
elif view == COMPANY_VIEW:
    company_view()
elif view == CONFIG_VIEW:
    config_view()
else:
    errors_view()