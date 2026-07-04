import streamlit as st
import pandas as pd
import plotly.express as px


def show_protocol_chart(protocol_results):

    st.subheader("📡 Protocol Analysis")

    # Convert dictionary into DataFrame
    protocol_df = pd.DataFrame(
        list(protocol_results.items()),
        columns=["Protocol", "Packets"]
    )

    col1, col2 = st.columns(2)

    # ===============================
    # PIE CHART
    # ===============================
    with col1:

        fig = px.pie(
            protocol_df,
            names="Protocol",
            values="Packets",
            hole=0.45,
            title="Protocol Distribution"
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="white"),
            legend_title="Protocols"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ===============================
    # BAR CHART
    # ===============================
    with col2:

        fig = px.bar(
            protocol_df,
            x="Protocol",
            y="Packets",
            color="Protocol",
            text="Packets",
            title="Protocol Packet Count"
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="white"),
            showlegend=False,
            xaxis_title="Protocol",
            yaxis_title="Packets"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ===============================
    # TABLE
    # ===============================

    st.markdown("### 📋 Protocol Summary")

    st.dataframe(
        protocol_df,
        use_container_width=True,
        hide_index=True
    )