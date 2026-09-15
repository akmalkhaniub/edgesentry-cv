import assert from 'assert';
import { SpatialHazardDetector } from '../src/spatial_hazard_detector.js';
import { TemporalEventFilter } from '../src/temporal_event_filter.js';
import { AWSServerlessDispatcher } from '../src/aws_serverless_dispatcher.js';

console.log('🧪 Starting EdgeSentry Automated Verification Suite (OpenCV AI AWS Hackathon)...\n');

const detector = new SpatialHazardDetector();
const filter = new TemporalEventFilter(1500); // 1.5s persistence threshold
const dispatcher = new AWSServerlessDispatcher();

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
// Frame at t=0s -> should be tracked but suppressed from firing immediately
const instantAlerts = filter.filter(rawViolations, t0);
assert(instantAlerts.length === 0, 'Transient event (< 1.5s) must be suppressed');
console.log('   ✅ Transient detection suppressed correctly at t=0s.');

// Frame at t=1.6s -> should now trigger verified persistent alarm
const t1 = t0 + 1600;
const persistentAlerts = filter.filter(rawViolations, t1);
assert(persistentAlerts.length === 1, 'Persistent violation (>= 1.5s) must trigger verified alarm');
assert(persistentAlerts[0].durationInsideMs === 1600, 'Duration must be 1600ms');
console.log(`   🚨 Verified Alarm Triggered after ${persistentAlerts[0].durationInsideMs}ms persistent presence in danger zone.`);

// Test 3: AWS Serverless Event Dispatch
console.log('4️⃣ Testing AWS Serverless Cloud Dispatch (Lambda / DynamoDB / S3)...');
const dispatchResult = await dispatcher.dispatch(persistentAlerts[0]);
assert(dispatchResult.status === 'DISPATCHED_TO_AWS', 'Event status must be DISPATCHED_TO_AWS');
assert(dispatchResult.s3SnapshotUri.includes('s3://edgesentry-vault-2026/snapshots/'), 'Must construct valid S3 URI');
console.log('   ☁️ Dispatched Event to AWS:');
console.log(`      Event ID: ${dispatchResult.eventId}`);
console.log(`      Target DynamoDB Table: ${dispatchResult.awsDestination.table}`);
console.log(`      Encrypted S3 Snapshot: ${dispatchResult.s3SnapshotUri}`);

console.log('\n🎉 ALL EDGESENTRY & OPENCV AI AWS TESTS PASSED WITH 100% SUCCESS!\n');
