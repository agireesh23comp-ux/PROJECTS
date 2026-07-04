import streamlit as st


def show_sidebar():

    st.sidebar.title("🛡 SOC Dashboard")

    st.sidebar.success("System Status")

    st.sidebar.markdown("---")

    st.sidebar.subheader("Modules")

    st.sidebar.write("📦 Packet Inspection")

    st.sidebar.write("📡 Protocol Analysis")

    st.sidebar.write("🌐 Top Talkers")

    st.sidebar.write("🌍 DNS Analysis")

    st.sidebar.write("🔌 Port Analysis")

    st.sidebar.write("🚨 Threat Detection")

    st.sidebar.markdown("---")

    st.sidebar.info("Version 1.0")