#define _BSD_SOURCE
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <pcap.h>
#include <time.h>  
#include <netinet/ip.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <net/ethernet.h>
#include <netinet/if_ether.h>
#include <signal.h>
#include <unistd.h>

#define MAX_FLOWS 10000
#define LINE_BUFFER 1024
#define DEBUG 0  // Set to 1 to enable debug prints
#define BATCH_SIZE 1000  // Process this many packets before writing to disk
#define PROGRESS_INTERVAL 10000  // Show progress every this many packets

/* Use uint8_t instead of u_char if not defined */
#ifndef u_char
typedef unsigned char u_char;
#endif

// Structure to hold flow information
typedef struct {
    char src_ip[16];
    uint16_t src_port;
    char dst_ip[16];
    uint16_t dst_port;
    uint8_t protocol;
    int flow_id;
} Flow;

// Structure to hold flow statistics
typedef struct {
    int flow_id;
    unsigned long packet_count;
} FlowStat;

// Structure to hold all user data
typedef struct {
    pcap_dumper_t* output_dumper;
    Flow* flows;
    int flow_count;
    FlowStat* flow_stats;
    int flow_stats_count;
    unsigned long matched_packets;
    unsigned long total_packets;
    time_t last_progress;
} UserData;

// Global flag for graceful termination
volatile sig_atomic_t keep_running = 1;

// Signal handler for graceful termination
void signal_handler(int signum) {
    keep_running = 0;
}

// Debug print function
void debug_print(const char* format, ...) {
    if (DEBUG) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
}

// Function to show progress
void show_progress(UserData* user_data) {
    time_t current_time = time(NULL);
    if (current_time - user_data->last_progress >= 1) {  // Update every second
        printf("\rProcessed: %lu packets, Matched: %lu packets", 
               user_data->total_packets, user_data->matched_packets);
        fflush(stdout);
        user_data->last_progress = current_time;
    }
}

// Function to read CSV file and parse flow information
int read_csv_flows(const char* csv_file, Flow* flows) {
    FILE* file = fopen(csv_file, "r");
    if (!file) {
        fprintf(stderr, "Error opening CSV file: %s\n", csv_file);
        return -1;
    }

    char line[LINE_BUFFER];
    int flow_count = 0;
    
    // Skip header line
    fgets(line, LINE_BUFFER, file);
    
    // Read each line of the CSV
    while (fgets(line, LINE_BUFFER, file) && flow_count < MAX_FLOWS) {
        char* token;
        
        // Flow ID
        if ((token = strtok(line, ",")) == NULL) continue;
        flows[flow_count].flow_id = atoi(token);
        
        // Source IP
        if ((token = strtok(NULL, ",")) == NULL) continue;
        strncpy(flows[flow_count].src_ip, token, 15);
        flows[flow_count].src_ip[15] = '\0';
        
        // Source Port
        if ((token = strtok(NULL, ",")) == NULL) continue;
        flows[flow_count].src_port = (uint16_t)atoi(token);
        
        // Destination IP
        if ((token = strtok(NULL, ",")) == NULL) continue;
        strncpy(flows[flow_count].dst_ip, token, 15);
        flows[flow_count].dst_ip[15] = '\0';
        
        // Destination Port
        if ((token = strtok(NULL, ",")) == NULL) continue;
        flows[flow_count].dst_port = (uint16_t)atoi(token);
        
        // Protocol
        if ((token = strtok(NULL, ",")) == NULL) continue;
        flows[flow_count].protocol = (uint8_t)atoi(token);
        
        flow_count++;
    }
    
    fclose(file);
    printf("Read %d flows from CSV file\n", flow_count);
    return flow_count;
}

// Function to find flow statistics index by flow_id
int find_flow_stat(FlowStat* stats, int count, int flow_id) {
    for (int i = 0; i < count; i++) {
        if (stats[i].flow_id == flow_id) {
            return i;
        }
    }
    return -1;
}

// Packet processing callback function
void packet_handler(u_char* user, const struct pcap_pkthdr* pkthdr, const u_char* packet) {
    UserData* user_data = (UserData*)user;
    
    // Check if we should continue processing
    if (!keep_running) {
        pcap_breakloop((pcap_t*)user_data->output_dumper);
        return;
    }

    user_data->total_packets++;
    show_progress(user_data);
    
    // Check if packet is truncated or too small
    if (pkthdr->caplen < sizeof(struct ether_header)) {
        return;
    }
    
    // Extract the ethernet header
    struct ether_header* eth_header = (struct ether_header*)packet;
    
    // Check if it's an IP packet
    if (ntohs(eth_header->ether_type) != ETHERTYPE_IP) {
        return;
    }
    
    // Check if packet is large enough to contain IP header
    if (pkthdr->caplen < sizeof(struct ether_header) + sizeof(struct iphdr)) {
        return;
    }
    
    // Get IP header
    const u_char* ip_packet = packet + sizeof(struct ether_header);
    struct iphdr* ip_header = (struct iphdr*)ip_packet;
    
    // Calculate minimum required packet size based on IP header length
    size_t ip_header_len = ip_header->ihl * 4;
    size_t min_packet_size = sizeof(struct ether_header) + ip_header_len;
    
    // Check if it's TCP or UDP
    int is_tcp = (ip_header->protocol == IPPROTO_TCP);
    int is_udp = (ip_header->protocol == IPPROTO_UDP);
    
    if (!is_tcp && !is_udp) {
        return;
    }
    
    // Add size check for transport layer header
    if (is_tcp) {
        min_packet_size += sizeof(struct tcphdr);
    } else {
        min_packet_size += sizeof(struct udphdr);
    }
    
    if (pkthdr->caplen < min_packet_size) {
        return;
    }
    
    // Get source and destination IP addresses
    char src_ip[16], dst_ip[16];
    struct in_addr source_addr, dest_addr;
    source_addr.s_addr = ip_header->saddr;
    dest_addr.s_addr = ip_header->daddr;
    inet_ntop(AF_INET, &source_addr, src_ip, INET_ADDRSTRLEN);
    inet_ntop(AF_INET, &dest_addr, dst_ip, INET_ADDRSTRLEN);
    
    // Get transport layer header
    const u_char* transport_packet = ip_packet + ip_header_len;
    uint16_t src_port, dst_port;
    
    if (is_tcp) {
        struct tcphdr* tcp_header = (struct tcphdr*)transport_packet;
        src_port = ntohs(tcp_header->source);
        dst_port = ntohs(tcp_header->dest);
    } else {
        struct udphdr* udp_header = (struct udphdr*)transport_packet;
        src_port = ntohs(udp_header->source);
        dst_port = ntohs(udp_header->dest);
    }
    
    // Check if packet belongs to any flow in our list
    int matched = 0;
    for (int i = 0; i < user_data->flow_count; i++) {
        // Check forward direction
        if (strcmp(user_data->flows[i].src_ip, src_ip) == 0 &&
            user_data->flows[i].src_port == src_port &&
            strcmp(user_data->flows[i].dst_ip, dst_ip) == 0 &&
            user_data->flows[i].dst_port == dst_port &&
            user_data->flows[i].protocol == ip_header->protocol) {
            
            matched = 1;
            int stat_idx = find_flow_stat(user_data->flow_stats, user_data->flow_stats_count, 
                                        user_data->flows[i].flow_id);
            if (stat_idx == -1 && user_data->flow_stats_count < MAX_FLOWS) {
                user_data->flow_stats[user_data->flow_stats_count].flow_id = user_data->flows[i].flow_id;
                user_data->flow_stats[user_data->flow_stats_count].packet_count = 1;
                user_data->flow_stats_count++;
            } else if (stat_idx >= 0) {
                user_data->flow_stats[stat_idx].packet_count++;
            }
            break;
        }
        
        // Check reverse direction
        if (strcmp(user_data->flows[i].src_ip, dst_ip) == 0 &&
            user_data->flows[i].src_port == dst_port &&
            strcmp(user_data->flows[i].dst_ip, src_ip) == 0 &&
            user_data->flows[i].dst_port == src_port &&
            user_data->flows[i].protocol == ip_header->protocol) {
            
            matched = 1;
            int stat_idx = find_flow_stat(user_data->flow_stats, user_data->flow_stats_count, 
                                        user_data->flows[i].flow_id);
            if (stat_idx == -1 && user_data->flow_stats_count < MAX_FLOWS) {
                user_data->flow_stats[user_data->flow_stats_count].flow_id = user_data->flows[i].flow_id;
                user_data->flow_stats[user_data->flow_stats_count].packet_count = 1;
                user_data->flow_stats_count++;
            } else if (stat_idx >= 0) {
                user_data->flow_stats[stat_idx].packet_count++;
            }
            break;
        }
    }
    
    if (matched) {
        pcap_dump((u_char*)user_data->output_dumper, pkthdr, packet);
        user_data->matched_packets++;
        
        // Flush to disk periodically
        if (user_data->matched_packets % BATCH_SIZE == 0) {
            pcap_dump_flush(user_data->output_dumper);
        }
    }
}

int extract_flow_packets(const char* pcap_file, const char* csv_file, const char* output_file) {
    printf("Reading CSV file...\n");
    Flow* flows = (Flow*)malloc(MAX_FLOWS * sizeof(Flow));
    if (!flows) {
        fprintf(stderr, "Memory allocation failed for flows\n");
        return -1;
    }
    
    int flow_count = read_csv_flows(csv_file, flows);
    if (flow_count <= 0) {
        free(flows);
        return -1;
    }
    
    printf("Opening PCAP file...\n");
    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t* handle = pcap_open_offline(pcap_file, errbuf);
    if (handle == NULL) {
        fprintf(stderr, "Could not open PCAP file %s: %s\n", pcap_file, errbuf);
        free(flows);
        return -1;
    }
    
    // Create output pcap file
    pcap_dumper_t* output_dumper = pcap_dump_open(handle, output_file);
    if (output_dumper == NULL) {
        fprintf(stderr, "Could not open output file %s\n", output_file);
        pcap_close(handle);
        free(flows);
        return -1;
    }
    
    // Allocate memory for flow statistics
    FlowStat* flow_stats = (FlowStat*)calloc(MAX_FLOWS, sizeof(FlowStat));
    if (!flow_stats) {
        fprintf(stderr, "Memory allocation failed for flow stats\n");
        pcap_dump_close(output_dumper);
        pcap_close(handle);
        free(flows);
        return -1;
    }
    
    // Initialize UserData structure
    UserData* user_data = (UserData*)calloc(1, sizeof(UserData));
    if (!user_data) {
        fprintf(stderr, "Memory allocation failed for user data\n");
        free(flow_stats);
        pcap_dump_close(output_dumper);
        pcap_close(handle);
        free(flows);
        return -1;
    }
    
    // Setup signal handler
    signal(SIGINT, signal_handler);
    
    user_data->output_dumper = output_dumper;
    user_data->flows = flows;
    user_data->flow_count = flow_count;
    user_data->flow_stats = flow_stats;
    user_data->flow_stats_count = 0;
    user_data->matched_packets = 0;
    user_data->total_packets = 0;
    user_data->last_progress = time(NULL);
    
    printf("\nProcessing packets...\n");
    // Process all packets in the file
    int result = pcap_loop(handle, -1, packet_handler, (u_char*)user_data);
    
    printf("\n\nExtraction Summary:\n");
    printf("--------------------------------------------------\n");
    for (int i = 0; i < user_data->flow_stats_count; i++) {
        printf("Flow %d: %lu packets extracted\n", 
               user_data->flow_stats[i].flow_id, 
               user_data->flow_stats[i].packet_count);
    }
    
    printf("--------------------------------------------------\n");
    printf("Total packets processed: %lu\n", user_data->total_packets);
    printf("Total flows processed: %d\n", user_data->flow_stats_count);
    printf("Total packets extracted: %lu\n", user_data->matched_packets);
    printf("Output saved to: %s\n", output_file);
    
    // Error handling for packet processing
    if (result == -1 && keep_running) {
        fprintf(stderr, "Error processing packets: %s\n", pcap_geterr(handle));
    } else if (!keep_running) {
        printf("\nProcessing interrupted by user. Saving progress...\n");
    }
    
    // Final flush of output file
    pcap_dump_flush(output_dumper);
    
    // Save matched packets count before cleanup
    unsigned long matched_packets = user_data->matched_packets;
    
    // Cleanup
    free(user_data);
    free(flow_stats);
    pcap_dump_close(output_dumper);
    pcap_close(handle);
    free(flows);
    
    return (keep_running && result >= 0) ? matched_packets : -1;
}

int main(int argc, char *argv[]) {
    
    const char* csv_file = "/home/brahim/Music/enp0s3-tcpdump-pvt-friday.pcap_Flow_non_benign.csv";
    const char* pcap_file = "/home/brahim/Music/enp0s3-tcpdump-pvt-friday.pcap";
    const char* output_file = "/home/brahim/Music/extracted_MALICIOUS_packets.pcap";
    
    
    printf("\nStarting packet extraction process...\n");
    printf("PCAP file: %s\n", pcap_file);
    printf("CSV file: %s\n", csv_file);
    printf("Output file: %s\n", output_file);
    
    int result = extract_flow_packets(pcap_file, csv_file, output_file);
    
    if (result < 0) {
        printf("Extraction process failed or was interrupted.\n");
        return 1;
    }
    
    return 0;
}