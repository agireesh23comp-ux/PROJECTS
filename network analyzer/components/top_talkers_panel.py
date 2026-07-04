import streamlit as st
import pandas as pd


def show_top_talkers(top_talkers):

    st.subheader("🌐 Top Talkers")

    # Get returned data
    source_ips = top_talkers["Top Source IPs"]
    destination_ips = top_talkers["Top Destination IPs"]

    # Convert to DataFrames
    source_df = pd.DataFrame(
        source_ips,
        columns=[
            "Source IP",
            "Packets"
        ]
    )

    destination_df = pd.DataFrame(
        destination_ips,
        columns=[
            "Destination IP",
            "Packets"
        ]
    )

    # Show only Top 5
    source_df = source_df.head(5)
    destination_df = destination_df.head(5)

    col1, col2 = st.columns(2)

    # ===========================
    # Source IP Table
    # ===========================

    with col1:

        st.markdown("### 🔵 Top Source IPs")

        st.dataframe(
            source_df,
            use_container_width=True,
            hide_index=True
        )

    # ===========================
    # Destination IP Table
    # ===========================

    with col2:

        st.markdown("### 🔴 Top Destination IPs")

        st.dataframe(
            destination_df,
            use_container_width=True,
            hide_index=True
        )