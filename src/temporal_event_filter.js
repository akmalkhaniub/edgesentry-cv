/**
 * TemporalEventFilter - Suppresses transient false alarms in edge computer vision feeds.
 */

export class TemporalEventFilter {
  constructor(persistenceThresholdMs = 1500) {
    this.persistenceThresholdMs = persistenceThresholdMs;
    this.trackedViolations = new Map(); // key: entityId_zoneId -> { firstSeen, lastSeen, alertFired }
  }

  /**
   * Filter raw spatial detections; only return verified persistent hazards
   */
  filter(violations, currentTimestamp = Date.now()) {
    const verifiedAlerts = [];
    const activeKeys = new Set();

    for (const v of violations) {
      const key = `${v.entityId}_${v.zoneId}`;
      activeKeys.add(key);

      if (!this.trackedViolations.has(key)) {
        this.trackedViolations.set(key, {
          firstSeen: currentTimestamp,
          lastSeen: currentTimestamp,
          alertFired: false,
          violationData: v
        });
      } else {
        const record = this.trackedViolations.get(key);
        record.lastSeen = currentTimestamp;
        const durationInsideMs = currentTimestamp - record.firstSeen;

        // Trigger alarm once duration exceeds persistence threshold
        if (durationInsideMs >= this.persistenceThresholdMs && !record.alertFired) {
          record.alertFired = true;
          verifiedAlerts.push({
            ...v,
            durationInsideMs,
            status: 'VERIFIED_PERSISTENT_HAZARD'
          });
        }
      }
    }

    // Clean up tracks that have left the zone
    for (const [key, record] of this.trackedViolations.entries()) {
      if (!activeKeys.has(key)) {
        if (currentTimestamp - record.lastSeen > 2000) {
          this.trackedViolations.delete(key);
        }
      }
    }

    return verifiedAlerts;
  }
}
