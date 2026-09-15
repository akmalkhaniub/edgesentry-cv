import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { SpatialHazardDetector } from './spatial_hazard_detector.js';
import { TemporalEventFilter } from './temporal_event_filter.js';
import { AWSServerlessDispatcher } from './aws_serverless_dispatcher.js';
import { DepthAIPipeline } from './depthai_pipeline.js';
import { GreengrassV2Client } from './greengrass_v2_client.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PUBLIC_DIR = path.join(__dirname, 'public');
const PORT = process.env.PORT || 3005;

const detector = new SpatialHazardDetector();
const filter = new TemporalEventFilter(1500);
const dispatcher = new AWSServerlessDispatcher();
const pipeline = new DepthAIPipeline({ modelName: 'yolo26-spatial-safety-fp16.blob' });
const greengrass = new GreengrassV2Client();

// Register default danger zone
detector.registerZone('zone_robot_weld_01', [
  [200, 200],
  [800, 200],
  [800, 700],
  [200, 700]
], 'Automated Robotic Weld Cell #1');

const server = http.createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // REST API Routes
  if (req.url === '/api/stream/frame' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', async () => {
      try {
        const { mockEntities, timestamp } = JSON.parse(body);
        const frame = pipeline.processFrame(mockEntities || []);
        const rawViolations = detector.evaluateDetections(mockEntities || []);
        const verifiedAlerts = filter.filter(rawViolations, timestamp || Date.now());

        if (verifiedAlerts.length > 0) {
          for (const alert of verifiedAlerts) {
            await greengrass.publishHazardAlert(alert);
            await dispatcher.publishSafetyAlert(alert);
          }
        }

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          frame,
          rawViolations,
          verifiedAlerts,
          greengrassEvents: greengrass.dispatchedEvents.slice(-5)
        }));
      } catch (err) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  if (req.url === '/api/status' && req.method === 'GET') {
    const shadow = await greengrass.updateDeviceShadow();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      camera: 'Luxonis OAK-D Pro PoE',
      shadow,
      zones: Array.from(detector.zones.values()),
      recentAlerts: greengrass.dispatchedEvents.slice(-10)
    }));
    return;
  }

  // Static files
  let filePath = path.join(PUBLIC_DIR, req.url === '/' ? 'index.html' : req.url);
  const ext = path.extname(filePath).toLowerCase();
  const mimeTypes = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8'
  };

  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === 'ENOENT') {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('404 Not Found');
      } else {
        res.writeHead(500, { 'Content-Type': 'text/plain' });
        res.end('Server Error: ' + err.code);
      }
    } else {
      res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream' });
      res.end(content);
    }
  });
});

server.listen(PORT, () => {
  console.log(`🎥 EdgeSentry CV Server running at http://localhost:${PORT}`);
});
