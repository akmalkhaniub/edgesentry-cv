# 🆓 Free Tier Deployment Guide for EdgeSentry CV

Deploy **EdgeSentry CV Operations Dashboard** using **Koyeb Free Eco**, **Render Blueprints**, **HiveMQ Cloud Free Broker**, and **Cloudflare Tunnels**.

---

## 1. Free Operations Center: Koyeb / Render
Host the central spatial surveillance control room 100% free:
- **Koyeb**: Connect GitHub repo and deploy via `koyeb.yaml` on port 3005.
- **Render**: Connect GitHub repo and deploy via `render.yaml`.

---

## 2. Free IoT MQTT Broker: HiveMQ Cloud
1. Sign up for a free 100-connection cluster at [hivemq.com/mqtt-cloud-broker](https://www.hivemq.com/mqtt-cloud-broker/) (or use public broker `broker.hivemq.com`).
2. Edge devices publish spatial intrusion payloads without needing AWS IoT Core charges.

---

## 3. Remote Live Radar Demo: Cloudflare Tunnel
```powershell
# Windows
.\deploy\free\tunnel.ps1 -Port 3005

# Linux / macOS
./deploy/free/tunnel.sh 3005
```
Share the generated `https://*.trycloudflare.com` URL with judges to show real-time 3D spatial safety zones!
