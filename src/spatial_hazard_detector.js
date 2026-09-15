/**
 * SpatialHazardDetector - OpenCV-Style Polygonal Restricted Zone Evaluation
 * Implements ray-casting point-in-polygon algorithm matching cv2.pointPolygonTest.
 */

export class SpatialHazardDetector {
  constructor() {
    this.restrictedZones = new Map();
  }

  /**
   * Define a polygonal restricted hazard zone
   * @param {string} zoneId 
   * @param {Array<[number, number]>} polygonVertices [[x1, y1], [x2, y2], ...]
   * @param {string} zoneName 
   */
  registerZone(zoneId, polygonVertices, zoneName = 'Restricted Hazard Perimeter') {
    if (polygonVertices.length < 3) {
      throw new Error('A valid polygon requires at least 3 vertices.');
    }
    this.restrictedZones.set(zoneId, {
      zoneId,
      zoneName,
      vertices: polygonVertices
    });
  }

  /**
   * Check if a 2D coordinate is inside a polygon using ray-casting
   * Equivalent to OpenCV cv2.pointPolygonTest(polygon, (x, y), false) >= 0
   */
  isPointInPolygon(point, polygon) {
    const [x, y] = point;
    let inside = false;

    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i][0], yi = polygon[i][1];
      const xj = polygon[j][0], yj = polygon[j][1];

      const intersect = ((yi > y) !== (yj > y)) &&
        (x < (xj - xi) * (y - yi) / (yj - yi) + xi);

      if (intersect) inside = !inside;
    }

    return inside;
  }

  /**
   * Evaluate detected entities (e.g. workers, vehicles) against all registered zones
   */
  evaluateDetections(detections) {
    const violations = [];

    for (const det of detections) {
      // Use center base coordinate of bounding box (feet position on ground plane)
      const [ymin, xmin, ymax, xmax] = det.bbox;
      const footPoint = [Math.round((xmin + xmax) / 2), ymax];

      for (const zone of this.restrictedZones.values()) {
        const isInside = this.isPointInPolygon(footPoint, zone.vertices);
        if (isInside) {
          const isPPECompliant = det.hasHardhat && det.hasHighVisVest;
          violations.push({
            entityId: det.id,
            zoneId: zone.zoneId,
            zoneName: zone.zoneName,
            footCoordinate: footPoint,
            isPPECompliant,
            violationType: isPPECompliant ? 'UNAUTHORIZED_ZONE_INTRUSION' : 'ZONE_INTRUSION_MISSING_PPE',
            timestamp: Date.now()
          });
        }
      }
    }

    return violations;
  }
}
