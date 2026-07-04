from scapy.all import TCP, UDP


def analyze_ports(packets):

    # Dictionary to count ports
    port_count = {}

    # Read every packet
    for packet in packets:

        # ==========================
        # TCP
        # ==========================

        if packet.haslayer(TCP):

            source_port = packet[TCP].sport
            destination_port = packet[TCP].dport

            tcp_source = ("TCP", source_port)
            port_count[tcp_source] = port_count.get(tcp_source, 0) + 1

            tcp_destination = ("TCP", destination_port)
            port_count[tcp_destination] = port_count.get(tcp_destination, 0) + 1

        # ==========================
        # UDP
        # ==========================

        elif packet.haslayer(UDP):

            source_port = packet[UDP].sport
            destination_port = packet[UDP].dport

            udp_source = ("UDP", source_port)
            port_count[udp_source] = port_count.get(udp_source, 0) + 1

            udp_destination = ("UDP", destination_port)
            port_count[udp_destination] = port_count.get(udp_destination, 0) + 1

    # Sort by packet count
    sorted_ports = sorted(
        port_count.items(),
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    # Keep only Top 10
    for (protocol, port), count in sorted_ports[:10]:

        results.append(
            (protocol, port, count)
        )

    return results