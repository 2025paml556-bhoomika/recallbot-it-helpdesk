doc_id: KB-BACKUP
version: 2026.1
valid_from: 2026-01-06
status: active

# Backup and Restore

## What is backed up

The backup service covers the home drive, the desktop and the documents folder of every company laptop, plus the shared drives. It runs every two hours while the device is online and on the VPN or on CORP-WIFI. A laptop that stays offline keeps a local cache of changes for seven days and uploads them at the next connection.

Mail is not part of this service. Mail retention is handled by the mailbox platform and is described in the email note.

## Retention

Daily restore points are kept for 30 days. Weekly restore points are kept for one year. A user can restore their own files from the last 30 days through the client. Anything older is a service-desk request and needs the manager's approval, because those restores come from the weekly set.

## What to do after data loss

Restore from the client first. If the file is absent there, raise a ticket and name the folder, the file name and the rough date it last existed. The desk targets a recovery within one working day for data under 5 GB.

## What is not covered

Removable drives, personal cloud storage and anything stored outside the profile folders are not backed up. The acceptable-use note explains why company data should not be in those places.
