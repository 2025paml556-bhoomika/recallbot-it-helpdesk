doc_id: KB-SLA
version: 2026.1
valid_from: 2026-01-06
status: active

# Service Levels

## Response and restore targets

The clock starts when the ticket is logged, not when the user first noticed the fault. Times below are working time for SEV-3 and SEV-4, and elapsed time for SEV-1 and SEV-2.

SLA-P1 covers SEV-1. First response is within 15 minutes and the restore target is 4 hours. A named incident lead publishes an update every 30 minutes until the service is back.

SLA-P2 covers SEV-2. First response is within 1 hour and the restore target is 8 hours. Updates go out every 2 hours.

SLA-P3 covers SEV-3. First response is within 4 working hours and the target resolution is 3 working days.

SLA-P4 covers SEV-4. First response is within 1 working day and the target fulfilment is 5 working days. A software request under review uses this target, and the ten-day security review sits inside it.

## When the clock pauses

The clock pauses when the desk is waiting on the user for a decision or for access to the machine, and it resumes when the user replies. It does not pause while the desk waits on an internal team.

## Missed targets

A missed SLA-P1 or SLA-P2 target is reviewed the next working day by the service owner. A missed SLA-P3 or SLA-P4 target is reported in the weekly queue review. Missing a target does not by itself change the severity.
