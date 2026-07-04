from scapy.all import DNS, DNSQR


def analyze_dns(packets):

    # Dictionary to count DNS queries
    domain_count = {}

    # Read every packet
    for packet in packets:

        # Check if packet contains DNS
        if packet.haslayer(DNS):

            # Check if it is a DNS Query
            if packet.haslayer(DNSQR):

                try:

                    # Extract domain name
                    domain = packet[DNSQR].qname.decode()

                    # Remove trailing dot
                    domain = domain.rstrip(".")

                    # Count occurrences
                    domain_count[domain] = domain_count.get(domain, 0) + 1

                except:
                    continue

    # Sort by request count
    sorted_domains = sorted(

        domain_count.items(),

        key=lambda item: item[1],

        reverse=True

    )

    # Return Top 10 domains
    return sorted_domains[:10]