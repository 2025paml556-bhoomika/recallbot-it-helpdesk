doc_id: KB-VPN
version: 2026.1
valid_from: 2026-01-06
status: active

# Remote Access VPN

## Who may connect

Every employee and every approved contractor uses the corporate VPN before opening an internal system from outside an office. The only approved client is Cisco AnyConnect, listed in the catalogue as SW-VPN-AC. Split tunnelling is off, so all traffic from the laptop travels through the gateway while the session is up.

A session lasts twelve hours and then drops. Reconnect with the same method. Do not leave a session running on a shared or borrowed machine.

## How to connect

1. Open Cisco AnyConnect from the Start menu or the Applications folder.
2. The gateway address is vpn.corp.example.com. Type it once; the client remembers it.
3. Sign in with your corporate email address and your network password, then approve the push notification from the authenticator app.
4. Wait until the padlock icon turns green before you open Outlook, SAP or a file share.

If the push notification never arrives, wait thirty seconds and request it again. Three failed approvals lock the VPN sign-in for fifteen minutes.

## Error codes

ERR-VPN-809 means the client reached the gateway but the certificate check failed. The usual cause is a laptop clock that is more than five minutes wrong. Set the clock automatically and retry. If the code remains, the machine certificate has expired and the service desk must re-enrol it. Quote ERR-VPN-809 when you raise the ticket.

ERR-VPN-412 means the account is not in the remote-access group. New starters are added on their first day by onboarding, and the group change can take one hour to reach the gateway. Quote ERR-VPN-412 if it still fails after that hour.

ERR-VPN-102 means the gateway rejected the password. Reset the password through the self-service page rather than guessing again, because five wrong passwords lock the account.

## Split tunnelling and home use

Split tunnelling stays off because the data-loss controls only see traffic that passes the gateway. A user who needs a local printer at home should print before connecting, or use the follow-me print queue described in the printer note, which works over the tunnel.

Personal streaming and personal cloud backups are blocked by the gateway while the VPN is up. Disconnect when the work is finished.

## Contractors and travel

Contractors use the same client and the same gateway, but their access group expires on the contract end date. An expired group returns ERR-VPN-412 even when the password is correct. The sponsoring manager extends the date in the identity console.

Roaming abroad needs no extra setting. The gateway accepts connections from every country the company operates in. A connection from anywhere else is refused and the security team is notified.
