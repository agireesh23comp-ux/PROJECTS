import streamlit as st


def show_threat_panel(threat_results):

    st.subheader("🚨 Threat Detection & Risk Assessment")

    # ==========================================
    # Risk Score
    # ==========================================

    risk_score = threat_results["Risk Score"]
    risk_level = threat_results["Risk Level"]

    st.metric(
        "Overall Risk Score",
        f"{risk_score}/100",
        risk_level
    )

    st.progress(risk_score / 100)

    st.divider()

    # ==========================================
    # Threat Status
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        if threat_results["FTP Detected"]:
            st.error("⚠ FTP Traffic Detected")
        else:
            st.success("✓ FTP Traffic Not Detected")

        if threat_results["Telnet Detected"]:
            st.error("⚠ Telnet Traffic Detected")
        else:
            st.success("✓ Telnet Traffic Not Detected")

        if threat_results["Port Scan Detected"]:
            st.error("⚠ Port Scan Detected")
        else:
            st.success("✓ No Port Scan Detected")

    with col2:

        if threat_results["ICMP Packets"] > 100:
            st.warning(
                f"⚠ ICMP Flood ({threat_results['ICMP Packets']} Packets)"
            )
        else:
            st.success(
                f"✓ ICMP Normal ({threat_results['ICMP Packets']})"
            )

        if threat_results["DNS Packets"] > 100:
            st.warning(
                f"⚠ DNS Flood ({threat_results['DNS Packets']} Packets)"
            )
        else:
            st.success(
                f"✓ DNS Normal ({threat_results['DNS Packets']})"
            )

    st.divider()

    # ==========================================
    # Threat Summary Table
    # ==========================================

    summary = {

        "Threat": [
            "FTP",
            "Telnet",
            "ICMP",
            "DNS",
            "Port Scan"
        ],

        "Status": [

            "Detected" if threat_results["FTP Detected"] else "Safe",

            "Detected" if threat_results["Telnet Detected"] else "Safe",

            "Flood" if threat_results["ICMP Packets"] > 100 else "Normal",

            "Flood" if threat_results["DNS Packets"] > 100 else "Normal",

            "Detected" if threat_results["Port Scan Detected"] else "Safe"
        ]
    }

    st.table(summary)

    st.divider()

    # ==========================================
    # Recommendations
    # ==========================================

    st.subheader("💡 Security Recommendations")

    recommendations = []

    if threat_results["FTP Detected"]:
        recommendations.append(
            "• Disable FTP and use SFTP."
        )

    if threat_results["Telnet Detected"]:
        recommendations.append(
            "• Replace Telnet with SSH."
        )

    if threat_results["ICMP Packets"] > 100:
        recommendations.append(
            "• Investigate possible ICMP Flood."
        )

    if threat_results["DNS Packets"] > 100:
        recommendations.append(
            "• Check excessive DNS requests."
        )

    if threat_results["Port Scan Detected"]:
        recommendations.append(
            "• Investigate Port Scanning activity."
        )

    if not recommendations:

        st.success(
            "No security recommendations. Network appears healthy."
        )

    else:

        for recommendation in recommendations:
            st.info(recommendation)