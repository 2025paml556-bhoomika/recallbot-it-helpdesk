doc_id: KB-INCIDENT
version: 2026.1
valid_from: 2026-01-06
status: active

# Incident Severity

## The four levels

Severity describes impact, not how annoyed a caller is. The service desk sets it when the ticket is opened and can change it later.

SEV-1 is a complete loss of a core service for a large part of the company, or a confirmed exposure of sensitive data. Examples are the mail platform down for every site, or the VPN gateway rejecting all connections. SEV-1 pages the on-call lead immediately, at any hour.

SEV-2 is a core service degraded or unavailable for one site or one business unit, with a workaround that is painful. An example is one office unable to reach shared drives while other offices work normally.

SEV-3 is a fault with a workaround that lets the user carry on, such as one laptop that will not take the VPN while a spare machine works. Most service-desk tickets are SEV-3.

SEV-4 is a request or a cosmetic fault: a software install, a quota increase, a label printed wrong. SEV-4 follows the request queue rather than the incident queue.

## What callers should report

State what is broken, how many people are affected, and whether a workaround exists. Quote an error code if one is on screen. The desk maps that information to a severity; callers do not choose their own.

## Data exposure

Any suspected exposure of customer or employee data starts at SEV-1 regardless of how many records are involved, and is downgraded only by the security team.
