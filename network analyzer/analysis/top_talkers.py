from scapy.all import IP


def analyze_top_talkers(packets):

    # Dictionary for Source IP count
    source_ip_count = {}

    # Dictionary for Destination IP count
    destination_ip_count = {}

    # Read every packet
    for packet in packets:

        if packet.haslayer(IP):

            source_ip = packet[IP].src
            destination_ip = packet[IP].dst

            # Count Source IPs
            source_ip_count[source_ip] = source_ip_count.get(source_ip, 0) + 1

            # Count Destination IPs
            destination_ip_count[destination_ip] = destination_ip_count.get(destination_ip, 0) + 1

    # Sort Source IPs
    sorted_source = sorted(
        source_ip_count.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Sort Destination IPs
    sorted_destination = sorted(
        destination_ip_count.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Return only Top 5
    return {

        "Top Source IPs": sorted_source[:5],

        "Top Destination IPs": sorted_destination[:5]

    }