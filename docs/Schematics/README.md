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
| **Component**                         | **Typical Power Draw** | **Max Power Draw** | **Voltage Input** | **Source** |
|--------------------------------------|------------------------|--------------------|-------------------|------------|
| NVIDIA Jetson AGX Orin 64GB          | 15–60 W                | Up to 108 W        | 5V–20V (19V typical) | [NVIDIA Datasheet](https://openzeka.com/en/wp-content/uploads/2022/08/Jetson-AGX-Orin-Module-Series-Datasheet.pdf) |
| ZED 2i Stereo Camera (USB Mode)      | ~7 W                   | ~8 W               | 5V (USB 3.1)      | [ZED 2i Datasheet](https://static.generation-robots.com/media/stereolabs-zed-2i-datasheet.pdf) |
| Robosense Helios 16 LiDAR            | 11–12 W                | 15 W               | 9V–32V (12V typical) | [Helios-16P User Guide](https://static.generation-robots.com/media/RS-HELIOS-16P_USER_GUIDE_V1.0.1_EN.pdf) |
| TP-Link Archer AX1500 Router         | ~5 W                   | ~7.7 W             | 12V DC            | [TPCDB](https://www.tpcdb.com/list.php?query=TP-Link&type=11) |
| CHCNAV CGI-410 INS (Dual GNSS)       | ~5 W                   | ~6 W               | 9V–36V DC         | [CHCNAV CGI-610 Datasheet](https://navigation.chcnav.com/products/chcnav-CGI-610) |
| Anker Prime 27,650mAh Power Bank     | Variable (up to 140 W per port) | 250 W total | N/A (Internal Battery) | [Anker A1340 Specs](https://www.anker.com/products/a1340-250w-power-bank) |