import streamlit as st
import pandas as pd


def show_dns_panel(dns_results):

    st.subheader("🌍 DNS Analysis")

    # Check if DNS data exists
    if not dns_results:

        st.info("No DNS traffic detected.")

        return

    # Convert list of tuples to DataFrame
    dns_df = pd.DataFrame(
        dns_results,
        columns=[
            "Domain",
            "Requests"
        ]
    )

    # Show only Top 10 domains
    dns_df = dns_df.head(10)

    # Display Table
    st.dataframe(
        dns_df,
        use_container_width=True,
        hide_index=True
    )

    # Show Total Domains
    st.metric(
        "Unique Domains",
        len(dns_results)
    )