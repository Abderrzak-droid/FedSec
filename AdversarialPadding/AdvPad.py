from scapy.all import rdpcap, wrpcap, IP, Raw, TCP, UDP
import random
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pcap_processing.log'),
        logging.StreamHandler()
    ]
)

def update_checksums(packet):
    """Force Scapy to recalculate IP and transport layer checksums."""
    if IP in packet:
        del packet[IP].chksum
    if TCP in packet:
        del packet[TCP].chksum
    elif UDP in packet:
        del packet[UDP].chksum
    return packet

def apply_adversarial_padding(
    packet, 
    loc_adv_pad="Start", 
    func="PC-HP", 
    overhead_percentage=30, 
    apply_padding=True
):
    """Add adversarial padding while maintaining valid headers."""
    if not (IP in packet and Raw in packet):
        return packet

    original_payload = packet[Raw].load
    payload_length = len(original_payload)

    if not apply_padding:
        return update_checksums(packet)

    ip_header_length = len(packet[IP])
    max_allowed_padding = 65535 - ip_header_length - payload_length
    pad_size = min(int(payload_length * (overhead_percentage / 100.0)), max_allowed_padding)

    if pad_size <= 0:
        return update_checksums(packet)

    random_pad = bytes(random.getrandbits(8) for _ in range(pad_size))

    if loc_adv_pad.lower() == "start":
        new_payload = random_pad + original_payload
    else:
        new_payload = original_payload + random_pad

    packet[Raw].load = new_payload
    return update_checksums(packet)

def process_single_pcap(args):
    """Process a single PCAP file with adversarial padding."""
    input_file, output_dir, params = args
    
    try:
        # Create output filename
        output_file = os.path.join(
            output_dir, 
            f"adversarial_{os.path.basename(input_file)}"
        )
        
        # Skip if output file already exists
        if os.path.exists(output_file):
            logging.info(f"Skipping {input_file} - output already exists")
            return {
                'file': input_file,
                'status': 'skipped',
                'packets_processed': 0,
                'packets_dropped': 0
            }

        packets = rdpcap(input_file)
        modified_packets = []
        dropped_count = 0

        for pkt in packets:
            try:
                modified_pkt = apply_adversarial_padding(
                    pkt,
                    loc_adv_pad=params['loc_adv_pad'],
                    func=params['func'],
                    overhead_percentage=params['overhead_percentage'],
                    apply_padding=params['apply_padding']
                )
                modified_packets.append(modified_pkt)
            except Exception as e:
                logging.warning(f"Error modifying packet in {input_file}: {e}")
                dropped_count += 1
                modified_packets.append(pkt)

        wrpcap(output_file, modified_packets)
        
        return {
            'file': input_file,
            'status': 'success',
            'packets_processed': len(modified_packets),
            'packets_dropped': dropped_count
        }

    except Exception as e:
        logging.error(f"Failed to process {input_file}: {e}")
        return {
            'file': input_file,
            'status': 'failed',
            'error': str(e),
            'packets_processed': 0,
            'packets_dropped': 0
        }

def process_pcap_directory(
    input_dir,
    output_dir,
    loc_adv_pad="Start",
    func="PC-HP",
    overhead_percentage=30,
    apply_padding=True,
    max_workers=None
):
    """Process all PCAP files in a directory with adversarial padding."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of PCAP files
    pcap_files = [
        f for f in Path(input_dir).glob('**/*') 
        if f.is_file() and f.suffix.lower() in ('.pcap', '.pcapng')
    ]
    
    if not pcap_files:
        logging.warning(f"No PCAP files found in {input_dir}")
        return

    # Prepare parameters for processing
    params = {
        'loc_adv_pad': loc_adv_pad,
        'func': func,
        'overhead_percentage': overhead_percentage,
        'apply_padding': apply_padding
    }
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        args = [(str(f), output_dir, params) for f in pcap_files]
        results = list(tqdm(
            executor.map(process_single_pcap, args),
            total=len(args),
            desc="Processing PCAP files"
        ))
    
    # Compile statistics
    stats = {
        'total_files': len(pcap_files),
        'successful': len([r for r in results if r['status'] == 'success']),
        'failed': len([r for r in results if r['status'] == 'failed']),
        'skipped': len([r for r in results if r['status'] == 'skipped']),
        'total_packets_processed': sum(r['packets_processed'] for r in results),
        'total_packets_dropped': sum(r['packets_dropped'] for r in results)
    }
    
    # Log results
    logging.info("\nProcessing Summary:")
    logging.info(f"Total files processed: {stats['total_files']}")
    logging.info(f"Successfully processed: {stats['successful']}")
    logging.info(f"Failed: {stats['failed']}")
    logging.info(f"Skipped: {stats['skipped']}")
    logging.info(f"Total packets processed: {stats['total_packets_processed']}")
    logging.info(f"Total packets dropped: {stats['total_packets_dropped']}")
    
    return stats

def main():
    # Example usage
    input_dir = "E:\\PFE2025\\Dataset\\Dataset_1\\extracted"
    output_dir = "E:\\PFE2025\\Dataset\\Dataset_1\\adversarial_pcaps"
    
    stats = process_pcap_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        loc_adv_pad="end",
        func="PC-HP",
        overhead_percentage=30,
        apply_padding=True,
        max_workers=os.cpu_count()  # Use all available CPU cores
    )

if __name__ == "__main__":
    main()