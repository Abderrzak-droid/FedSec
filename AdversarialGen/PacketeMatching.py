import csv
import os
from datetime import datetime
from scapy.all import *
from scapy.utils import PcapWriter

def parse_timestamp(timestamp_str):
    """Parse timestamp string into datetime object."""
    return datetime.strptime(timestamp_str, "%d/%m/%Y %I:%M:%S %p")

def process_csv(csv_file, output_dir):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row
        for row in reader:
            if not row:  # Skip empty rows
                continue
            try:
                # Extract relevant fields from the CSV row
                flow_id = row[0]
                src_ip = row[1]
                src_port = int(row[2])
                dst_ip = row[3]
                dst_port = int(row[4])
                protocol = int(row[5])
                timestamp_str = row[6]
                flow_duration = float(row[7])
                total_fwd = int(row[8])
                total_bwd = int(row[9])
                pcap_path = row[-1]  # Last column is the pcap file path

                # Parse start time and convert to timestamp
                start_time = parse_timestamp(timestamp_str)
                start_time_ts = start_time.timestamp()

                # Calculate end time (assuming flow_duration is in milliseconds)
                flow_duration_seconds = flow_duration / 1000.0
                end_time_ts = start_time_ts + flow_duration_seconds

                # Read the pcap file
                if not os.path.exists(pcap_path):
                    print(f"PCAP file not found: {pcap_path}")
                    continue
                packets = rdpcap(pcap_path)

                matched_packets = []
                forward_count = 0
                backward_count = 0

                for pkt in packets:
                    # Check for IP layer (IPv4 or IPv6)
                    if pkt.haslayer(IP):
                        ip_layer = pkt[IP]
                    elif pkt.haslayer(IPv6):
                        ip_layer = pkt[IPv6]
                    else:
                        continue

                    # Check protocol
                    pkt_proto = ip_layer.proto
                    if pkt_proto != protocol:
                        continue

                    # Check if the packet time is within the flow's time window
                    pkt_time = pkt.time
                    if not (start_time_ts <= pkt_time < end_time_ts):
                        continue

                    # Check transport layer and ports
                    if protocol == 6:  # TCP
                        if not pkt.haslayer(TCP):
                            continue
                        transport = pkt[TCP]
                    elif protocol == 17:  # UDP
                        if not pkt.haslayer(UDP):
                            continue
                        transport = pkt[UDP]
                    else:
                        continue  # Other protocols not handled

                    pkt_sport = transport.sport
                    pkt_dport = transport.dport

                    # Check direction
                    if (ip_layer.src == src_ip and pkt_sport == src_port and
                        ip_layer.dst == dst_ip and pkt_dport == dst_port):
                        forward_count += 1
                        matched_packets.append(pkt)
                    elif (ip_layer.src == dst_ip and pkt_sport == dst_port and
                          ip_layer.dst == src_ip and pkt_dport == src_port):
                        backward_count += 1
                        matched_packets.append(pkt)

                # Optional check for packet counts
                if forward_count != total_fwd or backward_count != total_bwd:
                    print(f"Warning: Flow {flow_id} has mismatched packet counts. "
                          f"Expected Fwd: {total_fwd}, Bwd: {total_bwd}; "
                          f"Found Fwd: {forward_count}, Bwd: {backward_count}")

                # Save the matched packets to a new pcap file
                if matched_packets:
                    output_filename = f"{flow_id.replace('/', '_')}.pcap"
                    output_path = os.path.join(output_dir, output_filename)
                    wrpcap(output_path, matched_packets)
                    print(f"Saved {len(matched_packets)} packets for flow {flow_id} to {output_path}")
                else:
                    print(f"No packets matched for flow {flow_id}")

            except Exception as e:
                print(f"Error processing row {row}: {e}")
                continue

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <csv_file> <output_directory>")
        sys.exit(1)
    csv_file = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    process_csv(csv_file, output_dir)