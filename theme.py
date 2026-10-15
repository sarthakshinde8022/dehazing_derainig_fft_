import streamlit as st


def inject_custom_css():
    st.markdown(
        """
        <style>
        .clarity-divider {
            border: none;
            border-top: 1px solid rgba(128, 145, 160, 0.3);
            margin: 1.25rem 0;
        }

        .eyebrow {
            color: #93A5B1;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )