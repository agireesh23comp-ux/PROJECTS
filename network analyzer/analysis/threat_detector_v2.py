from scapy.all import IP, TCP, UDP, ICMP, DNS


def detect_threats(packets):

    # ==========================================
    # Variables
    # ==========================================

    ftp_detected = False
    telnet_detected = False

    icmp_count = 0
    dns_count = 0

    # Smart Port Scan Detection
    scan_targets = {}

    # Common ports to monitor
    interesting_ports = {
        20, 21, 22, 23, 25,
        53,
        67, 68,
        80,
        110,
        135,
        137, 138, 139,
        143,
        161,
        389,
        443,
        445,
        993,
        995,
        1433,
        1521,
        3306,
        3389,
        5432,
        5900,
        8080
    }

    # ==========================================
    # Read Every Packet
    # ==========================================

    for packet in packets:

        if not packet.haslayer(IP):
            continue

        source_ip = packet[IP].src
        destination_ip = packet[IP].dst

        # -------------------------
        # TCP
        # -------------------------

        if packet.haslayer(TCP):

            sport = packet[TCP].sport
            dport = packet[TCP].dport

            if sport == 21 or dport == 21:
                ftp_detected = True

            if sport == 23 or dport == 23:
                telnet_detected = True

            if dport in interesting_ports:

                key = (source_ip, destination_ip)

                if key not in scan_targets:
                    scan_targets[key] = set()

                scan_targets[key].add(dport)

        # -------------------------
        # UDP
        # -------------------------

        elif packet.haslayer(UDP):

            dport = packet[UDP].dport

            if dport in interesting_ports:

                key = (source_ip, destination_ip)

                if key not in scan_targets:
                    scan_targets[key] = set()

                scan_targets[key].add(dport)

        # -------------------------
        # ICMP
        # -------------------------

        if packet.haslayer(ICMP):
            icmp_count += 1

        # -------------------------
        # DNS
        # -------------------------

        if packet.haslayer(DNS):
            dns_count += 1

    # ==========================================
    # Thresholds
    # ==========================================

    ICMP_THRESHOLD = 100
    DNS_THRESHOLD = 100
    PORT_SCAN_THRESHOLD = 8

    # ==========================================
    # Port Scan Detection
    # ==========================================

    port_scan_detected = False

    for ports in scan_targets.values():

        if len(ports) >= PORT_SCAN_THRESHOLD:

            port_scan_detected = True
            break

    # ==========================================
    # Risk Score
    # ==========================================

    risk_score = 0

    if ftp_detected:
        risk_score += 10

    if telnet_detected:
        risk_score += 20

    if icmp_count > ICMP_THRESHOLD:
        risk_score += 20

    if dns_count > DNS_THRESHOLD:
        risk_score += 20

    if port_scan_detected:
        risk_score += 30

    # ==========================================
    # Risk Level
    # ==========================================

    if risk_score <= 20:
        risk_level = "LOW"

    elif risk_score <= 50:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"

    # ==========================================
    # Recommendations
    # ==========================================

    recommendations = []

    if ftp_detected:
        recommendations.append("Disable FTP and use SFTP.")

    if telnet_detected:
        recommendations.append("Replace Telnet with SSH.")

    if icmp_count > ICMP_THRESHOLD:
        recommendations.append("Investigate possible ICMP Flood.")

    if dns_count > DNS_THRESHOLD:
        recommendations.append("Investigate excessive DNS requests.")

    if port_scan_detected:
        recommendations.append("Investigate possible Port Scan.")

    if not recommendations:
        recommendations.append("No suspicious activity detected.")

    # ==========================================
    # Return Results
    # ==========================================

    return {

        "Total Packets": len(packets),

        "FTP Detected": ftp_detected,

        "Telnet Detected": telnet_detected,

        "ICMP Packets": icmp_count,

        "DNS Packets": dns_count,

        "Port Scan Detected": port_scan_detected,

        "Risk Score": risk_score,

        "Risk Level": risk_level,

        "Recommendations": recommendations
    }
