import assert from 'assert';
import { SpatialHazardDetector } from '../src/spatial_hazard_detector.js';
import { TemporalEventFilter } from '../src/temporal_event_filter.js';
import { AWSServerlessDispatcher } from '../src/aws_serverless_dispatcher.js';
import { DepthAIPipeline } from '../src/depthai_pipeline.js';
import { GreengrassV2Client } from '../src/greengrass_v2_client.js';

console.log('🧪 Starting EdgeSentry Automated Verification Suite (OpenCV AI Competition 2026)...\n');

const detector = new SpatialHazardDetector();
const filter = new TemporalEventFilter(1500); // 1.5s persistence threshold
const dispatcher = new AWSServerlessDispatcher();
const pipeline = new DepthAIPipeline({ modelName: 'yolo26-spatial-safety-fp16.blob' });
const greengrass = new GreengrassV2Client('EdgeSentry-OAK-D-Camera-01');

// Define polygonal danger zone around automated robotic weld cell
const robotCellPolygon = [
  [200, 200],
  [800, 200],
  [800, 700],
  [200, 700]
];
detector.registerZone('zone_robot_weld_01', robotCellPolygon, 'Automated Robotic Weld Cell #1');
console.log('1️⃣ Registered 4-Point Polygonal Restricted Zone: [200,200] to [800,700]');

// Test 1: Spatial Evaluation & Point-in-Polygon
console.log('2️⃣ Testing Spatial Polygon Evaluation...');
const detections = [
  {
    id: 'worker_alpha',
    bbox: [250, 350, 450, 450], // Foot coordinate: [400, 450] -> Inside polygon
    hasHardhat: false, // PPE violation
    hasHighVisVest: true
  },
  {
    id: 'worker_bravo',
    bbox: [250, 850, 450, 950], // Foot coordinate: [900, 450] -> Outside polygon
    hasHardhat: true,
    hasHighVisVest: true
  }
];

const rawViolations = detector.evaluateDetections(detections);
assert(rawViolations.length === 1, `Expected 1 violation inside polygon, detected ${rawViolations.length}`);
assert(rawViolations[0].entityId === 'worker_alpha', 'Must identify worker_alpha inside danger zone');
assert(rawViolations[0].violationType === 'ZONE_INTRUSION_MISSING_PPE', 'Must flag missing hardhat');
console.log(`   ✅ Successfully detected unauthorized intruder "${rawViolations[0].entityId}" at foot coordinates (${rawViolations[0].footCoordinate}) with violation "${rawViolations[0].violationType}".`);

// Test 2: Temporal False-Alarm Filtering
console.log('3️⃣ Testing Temporal Event Filtering (Transient Suppression)...');
const t0 = 1000000;
const instantAlerts = filter.filter(rawViolations, t0);
assert(instantAlerts.length === 0, 'Transient event (< 1.5s) must be suppressed');
console.log('   ✅ Transient detection suppressed correctly at t=0s.');

const t1 = t0 + 1600;
const persistentAlerts = filter.filter(rawViolations, t1);
assert(persistentAlerts.length === 1, 'Persistent violation (>= 1.5s) must trigger verified alarm');
assert(persistentAlerts[0].durationInsideMs === 1600, 'Duration must be 1600ms');
console.log(`   🚨 Verified Alarm Triggered after ${persistentAlerts[0].durationInsideMs}ms persistent presence in danger zone.`);

// Test 3: AWS Serverless Event Dispatch
console.log('4️⃣ Testing AWS EventBridge Serverless Cloud Dispatch...');
const eventResult = await dispatcher.publishSafetyAlert(persistentAlerts[0]);
assert(eventResult.status === 'PUBLISHED', 'EventBridge dispatch must succeed');
assert(eventResult.event.source === 'edgesentry.vision.edge', 'Event source must be edgesentry');
console.log('   ✅ Dispatched to AWS EventBridge Bus:');
console.log('      Event ID:', eventResult.event.id);
console.log('      Detail:', eventResult.event.detail.violationType);

// Test 4: Luxonis OAK-D & DepthAI 3D Spatial Pipeline
console.log('5️⃣ Testing Luxonis OAK-D 3D Spatial Calculation (YOLO26)...');
const frameResult = pipeline.processFrame(detections);
assert(frameResult.detectionsCount === 2, 'Must process 2 detections');
assert(typeof frameResult.detections[0].spatial3D.z === 'number', 'Must calculate depth Z in meters');
assert(frameResult.detections[0].spatial3D.z > 0, 'Depth Z must be positive');
console.log(`   🎯 Detected spatial coordinates: X=${frameResult.detections[0].spatial3D.x}m, Y=${frameResult.detections[0].spatial3D.y}m, Z=${frameResult.detections[0].spatial3D.z}m`);

// Test 5: AWS IoT Greengrass v2 IPC & MQTT Telemetry
console.log('6️⃣ Testing AWS IoT Greengrass v2 Edge IPC & Shadow Sync...');
const ggAlert = await greengrass.publishHazardAlert({
  violationType: 'CRITICAL_ZONE_INTRUSION',
  zoneId: 'zone_robot_weld_01',
  entityId: 'worker_alpha',
  spatial3D: frameResult.detections[0].spatial3D
});
assert(ggAlert.status === 'PUBLISHED', 'Greengrass alert must publish');
assert(ggAlert.payload.kinesisClipUrl.includes('kinesisvideo'), 'Must generate Kinesis clip URL');

const shadow = await greengrass.updateDeviceShadow({ fps: 30.0, temp: 42.1 });
assert(shadow.state.reported.status === 'HEALTHY', 'Device shadow must report HEALTHY');
console.log(`   📡 Greengrass v2 IPC published alert to: ${ggAlert.topic}`);
console.log(`   📹 Kinesis Video Clip linked: ${ggAlert.payload.kinesisClipUrl}`);

console.log('\n🎉 ALL 6 EDGESENTRY & OPENCV AI AWS TESTS PASSED WITH 100% SUCCESS!\n');
