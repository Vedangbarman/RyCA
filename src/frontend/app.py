import inspect
import os
import re

import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000")
CHAT_TIMEOUT = 1800        # a chat message waits behind any running notification batch
DEFAULT_NAME = "gatorade"
STARTERS = [
    "What is the Upper Layer in NBFCs?",
    "Which master directions apply to all NBFCs?",
    "What are the large exposure limits for IDF-NBFCs?",
]
CHAT_VIEW, NOTES_VIEW = "Chat", "Notifications"

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
    view = st.radio("View", [CHAT_VIEW, NOTES_VIEW], label_visibility="collapsed")

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


if view == CHAT_VIEW:
    chat_view()
else:
    notifications_panel()