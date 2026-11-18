doc_id: KB-OUTAGES
version: 2026.1
valid_from: 2026-01-06
status: active

# Known Outages and Planned Maintenance

## How outages are announced

Planned work is posted to the status page and to the all-hands chat at least three working days ahead. Emergency work is posted as it starts, with the severity taken from the incident note. The status page is reachable without the VPN so that a VPN fault can still be read about.

## Standing windows

The directory and the VPN gateway are patched on the second Tuesday of each month, between 22:00 and 23:30 India time. During that window a reconnect may be needed and a certificate warning is expected; it is not ERR-VPN-809. Mail is not patched in the same window.

Shared-drive firmware is updated on the first Sunday of each quarter, from 06:00 to 08:00 India time. Open files may be disconnected. Save and close large files before leaving on Friday.

## Current advisory

There is no standing advisory beyond the windows above. A fault that matches nothing posted on the status page is a new incident and should be ticketed with its error code rather than waited out.

## After a window

If a service is still failing thirty minutes after the published end time, treat it as a new incident. Quote the change number from the status page in the ticket so the desk can link the two.
