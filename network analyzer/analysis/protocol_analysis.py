from scapy.all import IP


def analyze_protocols(packets):

    # Dictionary to store protocol counts
    protocol_count = {}

    # Loop through every packet
    for packet in packets:

        # Check if packet has an IP layer
        if packet.haslayer(IP):

            protocol_number = packet[IP].proto

            # Convert protocol number to protocol name
            if protocol_number == 6:
                protocol = "TCP"

            elif protocol_number == 17:
                protocol = "UDP"

            elif protocol_number == 1:
                protocol = "ICMP"

            else:
                protocol = "OTHER"

            # Count protocol occurrences
            protocol_count[protocol] = protocol_count.get(protocol, 0) + 1

    # Return dictionary instead of printing
    return protocol_count