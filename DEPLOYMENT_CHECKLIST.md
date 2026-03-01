# DRONE SURVEILLANCE - DEPLOYMENT CHECKLIST

## Pre-Deployment Validation

### Phase 1: Environment Setup
- [ ] Python 3.8+ installed
- [ ] Virtual environment created: `python -m venv venv`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] CUDA/GPU available (optional but recommended)
- [ ] FFmpeg installed for video codec support

### Phase 2: Model Verification
- [ ] YOLO detection model exists: `runs/detect/yolo11n/weights/best.pt`
- [ ] YOLO pose model exists: `runs/pose/human_pose_detector/weights/best.pt`
- [ ] Fighting LSTM model exists: `models/fighting_lstm_final.pt`
- [ ] All models load without errors
- [ ] GPU acceleration works (if using GPU)

### Phase 3: Code Verification
- [ ] drone_surveillance.py loads without syntax errors
- [ ] drone_surveillance_advanced.py loads without syntax errors
- [ ] Zone creator runs interactively: `python zone_creator.py`
- [ ] Test script completes: `python test_drone_surveillance.py`
- [ ] Output videos generated successfully

### Phase 4: Zone Configuration
- [ ] Zones defined for your surveillance area
- [ ] Zone file saved: `zones.json`
- [ ] Zone points verified (in pixel coordinates)
- [ ] Zone polygons visualized correctly
- [ ] Multiple zones tested (if applicable)

### Phase 5: Performance Baseline
- [ ] FPS measured on sample video
- [ ] GPU memory usage acceptable
- [ ] CPU temperature normal
- [ ] Output video quality acceptable
- [ ] Processing time within acceptable range

---

## Deployment Options

### Option A: Batch Video Processing
```bash
# Requirements
- Video files in 'videos/' directory
- Zones configured in 'zones.json'
- Output directory 'output/' exists

# Deploy
python scripts/batch_process.py

# Monitor
tail -f logs/batch_processing.log
```

### Option B: Real-Time Camera Surveillance
```bash
# Requirements
- Webcam or IP camera connected
- RTSP stream URL (if IP camera)
- Recording directory ready

# Deploy
python scripts/live_surveillance.py --camera 0  # or RTSP URL

# Monitor
- Real-time display window
- Logs saved to 'logs/live_surveillance.log'
```

### Option C: API Server
```bash
# Requirements
- Flask/FastAPI installed
- Video storage configured
- Database for results

# Deploy
python scripts/api_server.py --port 8000

# Usage
curl -X POST http://localhost:8000/process \
  -F "video=@video.mp4" \
  -F "zones=zones.json"
```

### Option D: Integration with Security System
```bash
# Requirements
- Security system API documentation
- Authentication credentials
- Webhook/alert endpoints

# Deploy
from drone_surveillance_advanced import AdvancedDroneSurveillance

class SecurityIntegration(AdvancedDroneSurveillance):
    def process_frame(self, frame):
        frame, stats = super().process_frame(frame)
        
        if stats['fighting_detected']:
            self.send_to_security_api(stats)
        
        return frame, stats
    
    def send_to_security_api(self, stats):
        # Call security system API
        pass
```

---

## Performance Checklist

### Detection Stage
- [ ] YOLO inference < 50ms per frame
- [ ] Person detection accurate (> 90% recall)
- [ ] No false positives in empty frames
- [ ] Handles multiple persons correctly

### Tracking Stage
- [ ] Drone acquires target within 5 frames
- [ ] Tracking smooth (no jittering)
- [ ] Recovers from temporary occlusion
- [ ] Handles persons leaving/entering zones

### Fighting Detection Stage
- [ ] Pose extraction accurate
- [ ] Feature engineering consistent
- [ ] LSTM inference < 100ms
- [ ] Accuracy meets requirements (> 95%)

### Output Stage
- [ ] HUD displays all required info
- [ ] Video codec compatible with players
- [ ] Frame rate matches input
- [ ] Output file size acceptable

---

## Monitoring & Logging Setup

### Log Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('surveillance.log'),
        logging.StreamHandler()
    ]
)
```

### Key Metrics to Monitor
- [ ] FPS (should be > 10)
- [ ] GPU memory usage (< 8GB recommended)
- [ ] CPU usage (< 80%)
- [ ] Intrusions detected per hour
- [ ] Fighting alerts per hour
- [ ] False positive rate
- [ ] Processing latency

### Alert Conditions
- [ ] FPS drops below 8
- [ ] GPU memory exceeds 8GB
- [ ] Processing latency > 1s per frame
- [ ] Fighting detected (send notification)
- [ ] Intrusion detected (send notification)

---

## Database Integration

### Option 1: SQLite (Local)
```python
import sqlite3

db = sqlite3.connect('surveillance.db')
c = db.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS alerts
    (id INTEGER PRIMARY KEY, timestamp TEXT, type TEXT, 
     confidence REAL, video_file TEXT, frame_number INTEGER)
''')
db.commit()
```

### Option 2: MySQL (Network)
```python
import mysql.connector

db = mysql.connector.connect(
    host='192.168.1.100',
    user='surveillance',
    password='password',
    database='surveillance_db'
)
cursor = db.cursor()
```

### Option 3: PostgreSQL (Scalable)
```python
import psycopg2

db = psycopg2.connect(
    host='192.168.1.100',
    user='surveillance',
    password='password',
    database='surveillance_db'
)
cursor = db.cursor()
```

---

## Cloud Integration

### AWS S3 Upload
```python
import boto3

s3 = boto3.client('s3')
s3.upload_file('output.mp4', 'surveillance-bucket', 'output.mp4')
```

### Firebase Alert
```python
import firebase_admin
from firebase_admin import db

firebase_admin.initialize_app()
ref = db.reference('alerts')
ref.push({
    'timestamp': datetime.now().isoformat(),
    'type': 'FIGHTING_DETECTED',
    'confidence': 0.95
})
```

### Slack Notification
```python
import slack_sdk

client = slack_sdk.WebClient(token='xoxb-xxx')
client.chat_postMessage(
    channel='#security-alerts',
    text='ALERT: Fighting detected in Zone 1'
)
```

---

## Hardware Requirements

### Minimum (CPU-only)
- [ ] Processor: 4-core Intel i5 or equivalent
- [ ] RAM: 8GB
- [ ] Storage: SSD 256GB
- [ ] Network: 1Gbps Ethernet

### Recommended (GPU-accelerated)
- [ ] Processor: 8-core Intel i7 or equivalent
- [ ] GPU: NVIDIA RTX 3060 or better
- [ ] RAM: 16GB
- [ ] Storage: SSD 512GB
- [ ] Network: 10Gbps Ethernet

### High-Scale (Multi-stream)
- [ ] Processor: 16+ cores XEON
- [ ] GPU: 2x NVIDIA A100
- [ ] RAM: 64GB
- [ ] Storage: NVMe RAID 1TB
- [ ] Network: 40Gbps

---

## Network Configuration

### IP Camera Setup
- [ ] Camera IP address configured
- [ ] RTSP stream accessible: `rtsp://camera_ip:554/stream`
- [ ] Firewall rules allow RTSP traffic
- [ ] Network latency < 50ms
- [ ] Bandwidth sufficient (4-10 Mbps per stream)

### Server Deployment
- [ ] Server IP assigned
- [ ] Firewall rules configured
- [ ] Port forwarding if needed
- [ ] HTTPS certificate installed (if web interface)
- [ ] VPN access configured (if remote)

---

## Security Checklist

- [ ] API endpoints protected with authentication
- [ ] Video files encrypted at rest
- [ ] Network traffic encrypted (HTTPS/TLS)
- [ ] Database credentials not in code
- [ ] Logs don't contain sensitive information
- [ ] Access control list configured
- [ ] Audit logging enabled
- [ ] Regular security backups

---

## Rollback Plan

### If Issues Detected
1. [ ] Stop surveillance: `Ctrl+C`
2. [ ] Collect logs for debugging
3. [ ] Revert to previous version: `git checkout v1.0`
4. [ ] Restart with known-good configuration
5. [ ] Diagnose issue in test environment
6. [ ] Fix and re-deploy

### Backup Strategy
- [ ] Daily backup of zones.json
- [ ] Weekly backup of surveillance logs
- [ ] Monthly backup of database
- [ ] Version control for all code changes
- [ ] Test restore procedures monthly

---

## Maintenance Schedule

### Daily
- [ ] Check system running
- [ ] Review alerts
- [ ] Monitor disk space

### Weekly
- [ ] Review performance metrics
- [ ] Check for model updates
- [ ] Verify zone accuracy

### Monthly
- [ ] Database cleanup (archive old data)
- [ ] Model performance validation
- [ ] Security audit
- [ ] Capacity planning

### Quarterly
- [ ] Accuracy validation on new data
- [ ] Performance benchmarking
- [ ] Disaster recovery test
- [ ] System upgrade evaluation

---

## Success Criteria

### Functional Requirements
- [ ] ≥95% detection accuracy on test videos
- [ ] ≥90% fighting detection accuracy
- [ ] Zero system crashes over 24 hours
- [ ] Real-time processing (FPS > 10)
- [ ] All alerts logged correctly

### Performance Requirements
- [ ] Latency < 1 second per frame
- [ ] GPU memory < 8GB
- [ ] CPU usage < 80%
- [ ] Network bandwidth < 10Mbps
- [ ] Disk I/O sustainable

### Reliability Requirements
- [ ] 99.9% uptime over month
- [ ] No data loss events
- [ ] Automated recovery from failures
- [ ] Health checks passing
- [ ] Logs clean (no errors)

---

## Sign-Off Checklist

**System Ready for Production When:**
- [ ] All validation tests passed
- [ ] Performance meets targets
- [ ] Documentation complete
- [ ] Team trained on operation
- [ ] Support procedures documented
- [ ] Disaster recovery tested
- [ ] Security audit passed
- [ ] Stakeholder approval obtained

---

## Production Deployment Command

```bash
#!/bin/bash

# Set environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Navigate to project
cd /path/to/fyp_detection

# Start surveillance
python -u scripts/production_surveillance.py \
    --zones zones.json \
    --cameras cameras.json \
    --output-dir /data/surveillance_output \
    --log-file surveillance.log \
    --daemon

# Verify running
ps aux | grep surveillance
tail -f surveillance.log
```

---

## Emergency Contacts

- **System Administrator**: +1-XXX-XXX-XXXX
- **Security Team**: security@organization.com
- **Technical Support**: support@organization.com
- **Emergency**: 911

---

**Last Updated**: 2024-12-18  
**Version**: 1.0  
**Status**: Ready for Deployment
