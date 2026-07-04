import streamlit as st
import pandas as pd


def show_packet_table(packet_results):

    st.subheader("📦 Packet Inspection")

    if not packet_results:

        st.warning("No packet information available.")
        return

    df = pd.DataFrame(packet_results)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.caption("Showing the first 5 packets from the uploaded PCAP.")