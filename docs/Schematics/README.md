## Components
1. [**Jetson AGX Orin**](#connection-breakdown)
2. [**ZED2i Stereo Camera**](#zed2i-stereo-camera)
3. [**Robosense Helios 16**](#robosense-helios-16-lidar)
4. [**TP Link Archer AX1500**](#tp-link-archer-ax1500-router)
5. [**CHCNAV CGI-410 INS**](#chcnav-cgi-410-ins-dual-gnss-antennas)
5. [**ANKER Prime 27,650 mAh**](#anker-prime-27560-mah-powerbank)

## Connection Breakdown
- **Jetson AGX Orin**
    - Connected Devices
        1. ZED2i Stereo Camera
        2. Robosense Helios 16
        3. CHCNAV CGI-410

    - Connected to
        1. TP Link Archer AX1500

## ZED2i Stereo Camera
- **Connection Mode:** USB 3.0
- **Connector:** USB Type-C → USB-A 3.0
- **Port connected on the Jetson:** USB-A Port

## Robosense Helios-16 LiDAR
- **Connection Mode:** Ethernet (UDP Broadcast)
- **Connector:** RJ45 (LiDAR Port) → RJ45 (Jetson/Router)
- **Port connected on the Jetson:** RJ45 Ethernet port or via Router as a switch

## CHCNAV CGI-410 INS (Dual GNSS Antennas)
- **Connection Mode:** Ethernet (Primary data), Serial (config/control), Dual GNSS
- **Connector:** RJ45 (CHCNAV) → Jetson (via Router)
- **Port connected on the Jetson:** RJ45 Ethernet port or serial via USB-serial adapter

## TP-Link Archer AX1500 Router
- **Connection Mode:** Ethernet LAN
- **Connector:** RJ45 (Router) → RJ45 (Jetson)
- **Port connected on the Jetson:** RJ45 Ethernet port on the Jetson

## Anker Prime 27,560 mAh Powerbank
- **Connection Mode:** Type C PD (160W Max)
- **Connector:** USB-C PD (Anker) → Jetson PD
- **Port connected on the Jetson:** Power delivery port on the Jetson
 
## Power Ratings