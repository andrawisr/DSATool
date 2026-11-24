"""Password screen shown before anything else in the app.

In plain English: when the app is online, nobody sees anything until they type
the password. The password itself is never in the code. Online, you type it
into the app's Settings -> Secrets box on Streamlit Community Cloud as

    password = "your password here"

On your own Mac, with no password set, the app opens as before. You can also
set one locally by putting the same line in .streamlit/secrets.toml (kept out
of Git). If the app is online and no password has been set, it refuses to
open rather than showing everything.
"""

from __future__ import annotations

import hmac
import time

import streamlit as st

LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}


def _expected_password() -> str:
    try:
        return str(st.secrets.get("password", ""))
    except FileNotFoundError:  # no secrets file at all
        return ""


def _on_this_computer() -> bool:
    host = st.context.headers.get("Host", "")
    return host.rsplit(":", 1)[0] in LOCAL_HOSTS if host else False


def check(hero) -> None:
    """Stop the page here unless the visitor has signed in (or it's your own Mac)."""
    expected = _expected_password()
    if not expected:
        if _on_this_computer():
            return
        hero("DSA Tool", "This copy has no password set, so it won't open.")
        st.error('Add a line such as password = "..." under the app\'s Settings → Secrets, then reload.')
        st.stop()
    if st.session_state.get("signed_in"):
        return

    hero("DSA Tool", "Private. Enter the password to continue.")
    with st.form("sign_in"):
        attempt = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            if hmac.compare_digest(attempt.encode(), expected.encode()):
                st.session_state.signed_in = True
                st.rerun()
            time.sleep(2)  # slows down anyone guessing
            st.error("That password isn't right.")
    st.stop()
