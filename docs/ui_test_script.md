# UI test script

Type these in the chat at http://127.0.0.1:8002, one line at a time, in order.
Wait for the reply before sending the next line. The user box starts as U1.
Do not click End session between cases unless a case says to.

Pass means the reply is the sentence in **Expect**, or the same fact in
different words. A tool name in brackets is what the turn panel should show.

These cases use only facts the bot can store: office (`location`) and laptop
(`asset_id`). Name, manager, phone and operating system are not stored, so
they are not in this script.

## 0. Start clean

1. Set the user box to `U1`.
2. Click **Clear this user**. Confirm the facts panel says no facts.
3. Click **New session** so this tab is not carrying slots from an old chat.

## 1. Greeting is refused

Send:

```
Hi
```

**Expect:** `I can only help with IT issues.`
No fact is added. This is the current behaviour, not a crash.

## 2. A stated office is stored

Send:

```
I sit at the Pune office.
```

**Expect:** `Thanks, I'll keep that in mind.`
The facts panel gains one row: `location = Pune office`, status active.

## 3. A stated laptop is stored

Send:

```
My laptop is LAP-T14G3.
```

**Expect:** `Thanks, I'll keep that in mind.`
Facts panel: `asset_id = LAP-T14G3`, and the Pune row is still there.

## 4. Recall the office

Send:

```
Where do I sit?
```

**Expect:** `You sit at Pune office.`
Tool: `recall_fact`. No new fact row. The question itself is not stored.

## 5. Recall the laptop

Send:

```
Which laptop do I use?
```

**Expect:** `You use LAP-T14G3.`
Tool: `recall_fact`.

## 6. Changing office supersedes the old one

Send:

```
I have moved from Pune to Hyderabad.
```

**Expect:** `Thanks, I'll keep that in mind.`
Facts panel: `location = Hyderabad` active. The Pune row is still in the
file but its status is `superseded`, so it is not in the active list.

Then send:

```
Which office am I based at now?
```

**Expect:** `You sit at Hyderabad.` The reply must not say Pune.

## 7. Knowledge-base question

Send:

```
What does ERR-VPN-809 mean?
```

**Expect:** a sentence taken from the VPN page, naming the code. It must not
say the code is unknown.
Tool: `search_kb`. No new fact.

## 8. Out of scope, and it must not hit the knowledge base

Send:

```
Book me a flight to Delhi.
```

**Expect:** `I can only help with IT issues.`
No tool. The facts panel is unchanged.

## 9. Book a visit, one slot per turn

Send these five lines as five separate turns:

```
Book a slot for a VPN issue.
```

**Expect:** `Which office, floor, and desk should the engineer come to?`

```
The Hyderabad office, third floor, desk 14.
```

**Expect:** `Which day works for the visit?`

```
Thursday.
```

**Expect:** `What time of day works?`

```
4 pm.
```

**Expect:** `Shall I book that visit?`
Nothing is written to the database yet. The facts panel does not gain a
ticket, and there is no booking episode.

```
Yes.
```

**Expect:** a sentence that names Hyderabad, Thursday and 4 pm.
Tool: `book_it_slot`. An episode appears: `Booked a visit at ...`.

## 10. No does not book

Start this in the same session, after case 9.

```
Book another visit.
```

Answer the three questions with `Pune office`, then `Friday`, then `10 am`.

**Expect** after the third answer: `Shall I book that visit?`

```
No.
```

**Expect:** `Okay, I won't book that.`
No second booking episode. A later `Yes` on its own must reply
`There is nothing waiting to book.`

## 11. A new session still knows the facts

Click **New session**. Do not clear the user. Send:

```
Where do I sit?
```

**Expect:** `You sit at Hyderabad.`
Slots from the booking (the day, the time) are gone with the old session.
The office survived because it is a semantic fact, not a session slot.

## 12. A second user cannot see the first

1. Set the user box to `U2`. The page starts a new session on its own.
2. Send:

```
Where does the other user sit?
```

**Expect:** `I don't have that stored for you.`
The reply must not contain Hyderabad, Pune, or LAP-T14G3.

3. Send:

```
I work from the Bengaluru office and I use LAP-MBP14.
```

**Expect:** `Thanks, I'll keep that in mind.`
U2's facts panel shows Bengaluru and LAP-MBP14 only.

4. Switch the user box back to `U1` and send `Where do I sit?`

**Expect:** `You sit at Hyderabad.` U1 did not pick up Bengaluru.

## 13. Forget

As U1, type `Hyderabad` in the forget box and click **Forget**.

**Expect:** the note says one fact was removed. Send:

```
Where do I sit?
```

**Expect:** `I don't have that stored for you.`
LAP-T14G3 is still stored. Switch to U2 and ask `Which office am I in?`
**Expect:** Bengaluru. Forgetting U1's office did not touch U2.

## 14. Known gaps, so a failure here is the design

Do not treat these as bugs in the run. They show the limits.

| Send | What happens | Why |
|---|---|---|
| `My manager is Rohan Deshpande.` | Thanks, but no fact row | Only `location` and `asset_id` are stored |
| `Who is my manager?` | `I don't have that stored for you.` | Nothing was written for manager |
| `Please forget my phone number 98220 11435.` | The sentence is not a forget. The forget box is the only way. | No intent deletes a fact |
| `It's been failing all week. Raise a ticket and book Thursday at 4 pm at my desk.` | One intent is chosen. The other request is dropped. | One intent per turn |
| `Hi` | `I can only help with IT issues.` | Greetings are labelled out of scope |

## After the run

- Facts on disk: U1 has `asset_id = LAP-T14G3` only, if case 13 was done. U2 has Bengaluru and LAP-MBP14.
- Episodes: one booking for U1 (Hyderabad, Thursday, 4 pm). No Friday booking.
- `memory/sessions/` has one JSON file per session id you used.
