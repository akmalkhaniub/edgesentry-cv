// ==============================================================================
// HiveMQ Cloud Free MQTT Broker Adapter for EdgeSentry CV
// Free 100 concurrent IoT MQTT connections for edge device telemetry.
// Docs: https://www.hivemq.com/mqtt-cloud-broker/
// ==============================================================================

export class FreeMqttAdapter {
  constructor(options = {}) {
    // Default to HiveMQ Public test broker or user's free cluster
    this.brokerUrl = options.brokerUrl || process.env.HIVEMQ_BROKER_URL || 'broker.hivemq.com';
    this.port = options.port || 1883;
    this.topicPrefix = options.topicPrefix || 'edgesentry/cv/alerts';
  }

  formatMqttPacket(alert) {
    return {
      topic: `${this.topicPrefix}/${alert.zoneId || 'general'}`,
      qos: 1,
      retain: false,
      payload: JSON.stringify({
        alertId: alert.alertId,
        violationType: alert.violationType,
        intruderId: alert.intruderId,
        footCoordinates: alert.footCoordinates,
        timestamp: new Date().toISOString(),
        broker: this.brokerUrl
      })
    };
  }
}
