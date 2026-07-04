import os
import streamlit as st

# ===============================================
# Import Backend
# ===============================================

from main import main

# ===============================================
# Import Components
# ===============================================

from components.sidebar import show_sidebar
from components.header import show_header
from components.footer import show_footer

from components.kpi_cards import show_kpis
from components.protocol_chart import show_protocol_chart
from components.packet_table import show_packet_table
from components.top_talkers_panel import show_top_talkers
from components.dns_panel import show_dns_panel
from components.port_panel import show_port_panel
from components.threat_panel import show_threat_panel

# ===============================================
# PAGE CONFIGURATION
# ===============================================

st.set_page_config(
    page_title="Smart Network Traffic Analyzer",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================
# CUSTOM CSS
# ===============================================

st.markdown("""
<style>

/* Background */

.stApp{
    background-color:#0E1117;
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background-color:#161B22;
}

/* Metric Cards */

div[data-testid="metric-container"]{
    background:#161B22;
    border:1px solid #00FFAA;
    border-radius:15px;
    padding:15px;
    box-shadow:0px 0px 12px rgba(0,255,170,.25);
}

/* Buttons */

.stButton>button{
    background:#00C853;
    color:white;
    width:100%;
    height:50px;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
}

/* Tables */

[data-testid="stDataFrame"]{
    border:1px solid #00FFAA;
}

</style>
""", unsafe_allow_html=True)

# ===============================================
# Sidebar & Header
# ===============================================

show_sidebar()
show_header()

# ===============================================
# Upload Folder
# ===============================================

os.makedirs("uploads", exist_ok=True)

uploaded_path = "uploads/uploaded.pcap"

# ===============================================
# Upload PCAP
# ===============================================

uploaded_file = st.file_uploader(
    "📂 Upload a PCAP File",
    type=["pcap"]
)

# ===============================================
# Analyze
# ===============================================

if uploaded_file:

    st.success(f"Uploaded : {uploaded_file.name}")

    with open(uploaded_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    if st.button("🚀 Analyze Network"):

        try:

            with st.spinner("Analyzing Network Traffic..."):

                results = main(uploaded_path)

            st.success("✅ Analysis Completed Successfully")

            # ==========================================
            # KPI Cards
            # ==========================================

            show_kpis(results)

            st.divider()

            # ==========================================
            # Protocol Analysis
            # ==========================================

            show_protocol_chart(
                results["protocols"]
            )

            st.divider()

            # ==========================================
            # Packet Inspection
            # ==========================================

            show_packet_table(
                results["inspection"]
            )

            st.divider()

            # ==========================================
            # Top Talkers
            # ==========================================

            show_top_talkers(
                results["top_talkers"]
            )

            st.divider()

            # ==========================================
            # DNS Analysis
            # ==========================================

            show_dns_panel(
                results["dns"]
            )

            st.divider()

            # ==========================================
            # Port Analysis
            # ==========================================

            show_port_panel(
                results["ports"]
            )

            st.divider()

            # ==========================================
            # Threat Detection
            # ==========================================

            show_threat_panel(
                results["threats"]
            )

            st.divider()

            # ==========================================
            # Footer
            # ==========================================

            show_footer()

        except Exception as e:

            st.error("❌ Analysis Failed")

            st.exception(e)

else:

    st.info("👆 Upload a PCAP file to begin analysis.")