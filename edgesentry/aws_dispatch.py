"""AWS serverless dispatch for verified violations.

Uploads a cropped violation snapshot to S3 and publishes an SNS alert. Uses boto3
when credentials + target ARNs/buckets are configured; otherwise records a
deterministic simulated dispatch so the pipeline is fully demoable offline.
"""
from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .temporal_filter import VerifiedAlert


@dataclass
class DispatchRecord:
    alert_id: str
    provider: str  # "aws" | "simulator"
    s3_key: str | None
    sns_message_id: str | None
    zone_id: str
    violation_type: str
    sent_at: float = field(default_factory=time.time)


class AWSServerlessDispatcher:
    def __init__(self, sns_topic_arn: str | None = None, s3_bucket: str | None = None, region: str | None = None, force_mock: bool = False) -> None:
        self.sns_topic_arn = sns_topic_arn or os.environ.get("EDGESENTRY_SNS_TOPIC_ARN")
        self.s3_bucket = s3_bucket or os.environ.get("EDGESENTRY_S3_BUCKET")
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.mock = force_mock or not (self.sns_topic_arn and self.s3_bucket and os.environ.get("AWS_ACCESS_KEY_ID"))
        self.records: list[DispatchRecord] = []
        self._sns: Any = None
        self._s3: Any = None

    def _clients(self) -> None:  # pragma: no cover - requires boto3 + AWS credentials
        if self._sns is None:
            import boto3  # imported lazily so the package works without boto3 installed

            self._sns = boto3.client("sns", region_name=self.region)
            self._s3 = boto3.client("s3", region_name=self.region)

    def dispatch(self, alert: VerifiedAlert, frame: np.ndarray | None = None) -> DispatchRecord:
        v = alert.violation
        alert_id = f"alert_{int(alert.violation.timestamp * 1000)}_{v.entity_id}"
        s3_key = f"violations/{v.zone_id}/{alert_id}.jpg"

        if self.mock:
            record = DispatchRecord(alert_id=alert_id, provider="simulator", s3_key=s3_key, sns_message_id=f"sim_{alert_id}", zone_id=v.zone_id, violation_type=v.violation_type)
            self.records.append(record)
            return record

        try:
            self._clients()
            sns_message_id = None
            if frame is not None:  # pragma: no cover - live AWS S3/SNS path
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    self._s3.upload_fileobj(io.BytesIO(buf.tobytes()), self.s3_bucket, s3_key)
            resp = self._sns.publish(  # pragma: no cover - live AWS SNS path
                TopicArn=self.sns_topic_arn,
                Subject=f"EdgeSentry {v.violation_type} in {v.zone_name}",
                Message=(
                    f"Verified persistent hazard: entity {v.entity_id} in zone {v.zone_id} "
                    f"({v.violation_type}) for {alert.duration_inside_s:.1f}s. Snapshot: s3://{self.s3_bucket}/{s3_key}"
                ),
            )
            sns_message_id = resp.get("MessageId")  # pragma: no cover
            record = DispatchRecord(alert_id=alert_id, provider="aws", s3_key=s3_key, sns_message_id=sns_message_id, zone_id=v.zone_id, violation_type=v.violation_type)  # pragma: no cover
        except Exception as exc:  # pragma: no cover - network path
            print(f"[EdgeSentry] AWS dispatch failed, using simulator: {exc}")
            record = DispatchRecord(alert_id=alert_id, provider="simulator", s3_key=s3_key, sns_message_id=f"sim_{alert_id}", zone_id=v.zone_id, violation_type=v.violation_type)

        self.records.append(record)
        return record
