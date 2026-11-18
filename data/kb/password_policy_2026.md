doc_id: KB-PASSWORD
version: 2026.1
valid_from: 2026-01-06
status: active

# Password and MFA Policy (2026)

This policy replaces the 2025 version. Where the two differ, this document wins for any request dated 6 January 2026 or later.

## Password rules

A network password must be at least 14 characters long. It must contain lowercase letters, uppercase letters and a digit. Symbols are allowed but no longer required. There is no periodic expiry: a password is changed only when there is a reason, such as a suspected leak, a shared use, or a leaver's account being reclaimed.

The previous ten passwords cannot be reused. The directory checks new passwords against a list of known leaked passwords and rejects a match.

Five wrong attempts lock the account for fifteen minutes. The service desk can unlock it sooner after verifying the caller against the employee number and the manager's name. The desk never asks for the password itself and never sets a password that it then reads back.

## Multi-factor authentication

MFA protects VPN, webmail, the self-service portal and the software portal. The approved factor is a push notification in the authenticator app, approved with biometrics or the app PIN. Typed six-digit codes are no longer the primary method; they remain only as a fallback when the phone has no data connection.

SMS codes are still not accepted. A lost or replaced phone is re-enrolled by the service desk during a video call. Until that call happens, the fallback codes issued at enrolment work. Each user holds ten fallback codes, and each code works once.

## Storage and sharing

Store the password in the approved company password manager, listed as SW-PWM-01. Browser password stores on unmanaged devices are not approved. A password written on paper, kept in a personal note, or sent by email or chat breaches this policy.

A manager cannot sign in as a member of their team. Shared mailboxes have their own credentials, held by the mailbox owner and not by individuals.
