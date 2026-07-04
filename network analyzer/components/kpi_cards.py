import streamlit as st


def show_kpis(results):

    threat_results = results["threats"]

    threats = 0

    if threat_results["FTP Detected"]:
        threats += 1

    if threat_results["Telnet Detected"]:
        threats += 1

    if threat_results["Port Scan Detected"]:
        threats += 1

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Packets",
            threat_results["Total Packets"]
        )

    with col2:

        st.metric(
            "Protocols",
            len(results["protocols"])
        )

    with col3:

        st.metric(
            "Threats",
            threats
        )

    with col4:

        st.metric(
            "Risk",
            threat_results["Risk Level"]
        )