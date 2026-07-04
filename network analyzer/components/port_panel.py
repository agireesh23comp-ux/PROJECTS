import streamlit as st
import pandas as pd
import plotly.express as px


def show_port_panel(port_results):

    st.subheader("🔌 Port Analysis")

    if not port_results:

        st.warning("No Port Data Found")

        return

    # Convert to DataFrame
    port_df = pd.DataFrame(
        port_results,
        columns=[
            "Protocol",
            "Port",
            "Packets"
        ]
    )

    # Keep only Top 10
    port_df = port_df.head(10)

    # ==========================
    # Add Service Names
    # ==========================

    services = {

        80:"HTTP",
        443:"HTTPS",
        53:"DNS",
        22:"SSH",
        21:"FTP",
        23:"TELNET",
        25:"SMTP",
        110:"POP3",
        143:"IMAP",
        445:"SMB",
        3389:"RDP",
        8080:"HTTP-ALT"

    }

    port_df["Service"] = port_df["Port"].map(
        services
    ).fillna("OTHER")

    # Reorder Columns

    port_df = port_df[
        [
            "Port",
            "Protocol",
            "Service",
            "Packets"
        ]
    ]

    # ==========================
    # Display Table
    # ==========================

    st.dataframe(
        port_df,
        use_container_width=True,
        hide_index=True
    )

    # ==========================
    # Bar Chart
    # ==========================

    fig = px.bar(

        port_df,

        x="Packets",

        y="Port",

        color="Protocol",

        orientation="h",

        text="Packets",

        title="Top Active Ports"

    )

    fig.update_layout(

        template="plotly_dark",

        paper_bgcolor="#0E1117",

        plot_bgcolor="#0E1117",

        font=dict(color="white"),

        yaxis_title="Port",

        xaxis_title="Packets"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ==========================
    # Metrics
    # ==========================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(

            "Unique Ports",

            len(port_df)

        )

    with col2:

        st.metric(

            "Most Active Port",

            port_df.iloc[0]["Port"]

        )