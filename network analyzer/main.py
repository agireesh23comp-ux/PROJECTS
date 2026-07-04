from scapy.all import rdpcap

# ===============================
# Import Analysis Modules
# ===============================

from analysis.packet_inspection import inspect_packets
from analysis.protocol_analysis import analyze_protocols
from analysis.top_talkers import analyze_top_talkers
from analysis.dns_analysis import analyze_dns
from analysis.port_analysis import analyze_ports
from analysis.threat_detector_v2 import detect_threats


# ===============================
# Main Function
# ===============================

def main(pcap_file):

    # Read PCAP File
    packets = rdpcap(pcap_file)

    # Run all analyses
    inspection_results = inspect_packets(packets)

    protocol_results = analyze_protocols(packets)

    top_talker_results = analyze_top_talkers(packets)

    dns_results = analyze_dns(packets)

    port_results = analyze_ports(packets)

    threat_results = detect_threats(packets)

    # Return everything
    return {

        "inspection": inspection_results,

        "protocols": protocol_results,

        "top_talkers": top_talker_results,

        "dns": dns_results,

        "ports": port_results,

        "threats": threat_results

    }


# ===============================
# Testing
# ===============================

if __name__ == "__main__":

    results = main("sample_pcaps/sample2.pcap")

    print(results)