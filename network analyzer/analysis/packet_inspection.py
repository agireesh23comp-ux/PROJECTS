from scapy.all import IP


def inspect_packets(packets):

    packet_details = []

    # Analyze first 5 packets
    for i, packet in enumerate(packets[:5], start=1):

        if packet.haslayer(IP):

            source_ip = packet[IP].src
            destination_ip = packet[IP].dst
            packet_length = len(packet)

            protocol_number = packet[IP].proto

            if protocol_number == 6:
                protocol = "TCP"

            elif protocol_number == 17:
                protocol = "UDP"

            elif protocol_number == 1:
                protocol = "ICMP"

            else:
                protocol = "OTHER"

            packet_details.append({

                "Packet": i,

                "Source IP": source_ip,

                "Destination IP": destination_ip,

                "Protocol": protocol,

                "Length": packet_length

            })

    return packet_details