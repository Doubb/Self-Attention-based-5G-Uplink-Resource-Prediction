# 5G Uplink Resource Dataset

We collected a large amount of data and extensively measured 5G Dual Connectivity traffic over the commercial NSA 5G networks of three major Korean carriers (SKT, KT, and LG U+).
In these measurements, we used a rooted phone (Samsung Galaxy S20) and XCAL-Solo, which is a COTS monitoring tool, to acquire a cellular signal and uplink scheduling information directly from a 5G modem chipset.
We executed iPerf to generate massive UDP traffic to fully utilize the available uplink bandwidth allocated by BSs (eNodeB or gNodeB)

We collected measurement data in various locations, i.e., five highways, four downtown areas, and three stationary locations (residences, offices, and department stores).
Here, equal amounts of data were collected from each of the three carriers.

# Dataset Features and Configuration

Each .csv file consists of various wireless network features, i.e., TBS (Transport Block Size), MCS (Modulation and Coding Scheme), RSRP (Reference Signal Received Power) and CQI (Channel Quality Indicator).
The names of all the files were written in the following format.

{Collected_Date(YYYYMMDD)}_{Collection_Start_Time(HHMM)}_{Collection_End_Time(HHMM)}_{Location}_{Specific_Location}_{5G_Carrier}.csv

We make our test dataset available for future research.
