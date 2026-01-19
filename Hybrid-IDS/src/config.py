"""Shared config mappings and helpers for Hybrid-IDS."""

from __future__ import annotations

from typing import Optional


ZERO_DAY_LABEL = "ZERO_DAY"
LEGACY_ZERO_DAY_LABEL = "UNKNOWN_ATTACK"

CLASS_CATEGORY_MAP = {
	"PORTSCAN": "Reconnaissance",
	"BRUTE_FORCE": "Credential Access",
	"DOS": "Impact",
	"DDOS": "Impact",
	"BOTNET": "Malware Command & Control",
	"INFILTRATION": "Exploitation",
	"WEB_ATTACK": "Exploitation",
	ZERO_DAY_LABEL: "Unknown / Emerging",
	"BENIGN": "Benign",
}

MITRE_MAP = {
	"PORTSCAN": "TA0043 Reconnaissance",
	"BRUTE_FORCE": "TA0006 Credential Access",
	"DOS": "TA0040 Impact",
	"DDOS": "TA0040 Impact",
	"BOTNET": "TA0011 Command and Control",
	"INFILTRATION": "TA0001 Initial Access",
	"WEB_ATTACK": "TA0001 Initial Access",
	ZERO_DAY_LABEL: "Unknown (Research Required)",
}

SEVERITY_LEVELS = ("CRITICAL", "HIGH", "MEDIUM", "LOW")


def normalize_label(label: str) -> str:
	if label == LEGACY_ZERO_DAY_LABEL:
		return ZERO_DAY_LABEL
	return label


def category_for_label(label: str) -> str:
	label = normalize_label(label)
	return CLASS_CATEGORY_MAP.get(label, "Unknown / Emerging")


def mitre_for_label(label: str) -> str:
	label = normalize_label(label)
	return MITRE_MAP.get(label, "Unknown (Research Required)")


def severity_for_event(
	*,
	label: str,
	category: Optional[str],
	risk: float,
	anomaly: float,
	anomaly_threshold_p95: Optional[float],
) -> str:
	label = normalize_label(label)
	category = category or category_for_label(label)

	if label == ZERO_DAY_LABEL:
		return "CRITICAL"
	if risk > 0.90 or (anomaly_threshold_p95 is not None and anomaly > anomaly_threshold_p95):
		return "HIGH"
	if category in {"Impact", "Exploitation"}:
		return "MEDIUM"
	return "LOW"


def enrich_event(
	event: dict,
	*,
	anomaly_threshold_p95: Optional[float] = None,
) -> dict:
	label = normalize_label(str(event.get("label", "")))
	category = event.get("category") or category_for_label(label)
	mitre = event.get("mitre") or mitre_for_label(label)
	anomaly_value = float(event.get("reconstruction_error", event.get("anomaly", 0.0)))
	risk = float(event.get("risk", 0.0))
	threshold = anomaly_threshold_p95
	if threshold is None:
		threshold = event.get("anomaly_threshold_p95")
	severity = event.get("severity") or severity_for_event(
		label=label,
		category=category,
		risk=risk,
		anomaly=anomaly_value,
		anomaly_threshold_p95=threshold if threshold is None else float(threshold),
	)
	if label == ZERO_DAY_LABEL and "legacy_label" not in event:
		event["legacy_label"] = LEGACY_ZERO_DAY_LABEL

	event.update(
		{
			"label": label,
			"category": category,
			"mitre": mitre,
			"severity": severity,
		}
	)
	return event
