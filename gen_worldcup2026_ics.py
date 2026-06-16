#!/usr/bin/env python3
"""Generate an ICS calendar for all 104 FIFA World Cup 2026 matches.

Data compiled from public sources (FIFA / ESPN / Sky Sports / Al Jazeera) in
June 2026. Group draw and matchups are confirmed; kick-off times are best-effort
and should be verified against fifa.com. Times are stored in UTC (the 'Z'
suffix), so calendar apps will display them in the viewer's local time zone.
"""

from datetime import datetime, timedelta

# Each match: (stage, date, utc_time, home, away, venue, city)
#   date     : YYYY-MM-DD  (UTC calendar date of kickoff)
#   utc_time : HH:MM in UTC
# US/Canada are on DST in Jun/Jul 2026 (Eastern = UTC-4). Mexico has no DST
# (Central = UTC-6). Group-stage matchups are confirmed; knockout ties carry
# bracket-slot placeholders because qualifiers depend on results.

M = []

def add(stage, date, utc_time, home, away, venue, city):
    M.append((stage, date, utc_time, home, away, venue, city))

# ============================ GROUP STAGE (72) ============================
# ---- Matchday 1 ----
add("Group A", "2026-06-11", "19:00", "Mexico", "South Africa", "Estadio Azteca", "Mexico City")
add("Group A", "2026-06-12", "02:00", "Korea Republic", "Czechia", "Estadio Akron", "Guadalajara")
add("Group B", "2026-06-12", "23:00", "Canada", "Bosnia and Herzegovina", "BMO Field", "Toronto")
add("Group D", "2026-06-13", "01:00", "United States", "Paraguay", "SoFi Stadium", "Los Angeles")
add("Group B", "2026-06-13", "20:00", "Qatar", "Switzerland", "Levi's Stadium", "San Francisco Bay Area")
add("Group C", "2026-06-13", "23:00", "Brazil", "Morocco", "MetLife Stadium", "New York / New Jersey")
add("Group C", "2026-06-14", "17:00", "Haiti", "Scotland", "Gillette Stadium", "Boston")
add("Group E", "2026-06-14", "19:00", "Germany", "Curacao", "NRG Stadium", "Houston")
add("Group E", "2026-06-14", "21:00", "Cote d'Ivoire", "Ecuador", "Lincoln Financial Field", "Philadelphia")
add("Group D", "2026-06-14", "23:00", "Australia", "Turkiye", "BC Place", "Vancouver")
add("Group F", "2026-06-15", "01:00", "Netherlands", "Japan", "AT&T Stadium", "Dallas")
add("Group F", "2026-06-15", "17:00", "Tunisia", "Sweden", "Mercedes-Benz Stadium", "Atlanta")
add("Group G", "2026-06-15", "20:00", "Belgium", "Egypt", "Lumen Field", "Seattle")
add("Group G", "2026-06-15", "23:00", "Iran", "New Zealand", "Levi's Stadium", "San Francisco Bay Area")
add("Group H", "2026-06-16", "16:00", "Spain", "Cabo Verde", "Hard Rock Stadium", "Miami")
add("Group H", "2026-06-16", "19:00", "Saudi Arabia", "Uruguay", "Arrowhead Stadium", "Kansas City")
add("Group I", "2026-06-16", "22:00", "France", "Senegal", "MetLife Stadium", "New York / New Jersey")
add("Group I", "2026-06-17", "01:00", "Norway", "Iraq", "SoFi Stadium", "Los Angeles")
add("Group J", "2026-06-17", "17:00", "Argentina", "Jordan", "Mercedes-Benz Stadium", "Atlanta")
add("Group J", "2026-06-17", "20:00", "Algeria", "Austria", "Gillette Stadium", "Boston")
add("Group K", "2026-06-17", "22:00", "Portugal", "Congo DR", "AT&T Stadium", "Dallas")
add("Group K", "2026-06-18", "01:00", "Uzbekistan", "Colombia", "BC Place", "Vancouver")
add("Group L", "2026-06-18", "16:00", "England", "Croatia", "Lincoln Financial Field", "Philadelphia")
add("Group L", "2026-06-18", "19:00", "Ghana", "Panama", "NRG Stadium", "Houston")

# ---- Matchday 2 ----
add("Group A", "2026-06-18", "22:00", "Mexico", "Korea Republic", "Estadio Azteca", "Mexico City")
add("Group A", "2026-06-19", "00:00", "Czechia", "South Africa", "Mercedes-Benz Stadium", "Atlanta")
add("Group B", "2026-06-19", "16:00", "Switzerland", "Bosnia and Herzegovina", "SoFi Stadium", "Los Angeles")
add("Group B", "2026-06-19", "19:00", "Canada", "Qatar", "BMO Field", "Toronto")
add("Group D", "2026-06-19", "19:00", "United States", "Australia", "SoFi Stadium", "Los Angeles")
add("Group C", "2026-06-19", "22:00", "Scotland", "Morocco", "Hard Rock Stadium", "Miami")
add("Group C", "2026-06-20", "01:00", "Brazil", "Haiti", "Levi's Stadium", "San Francisco Bay Area")
add("Group D", "2026-06-20", "04:00", "Turkiye", "Paraguay", "Lumen Field", "Seattle")
add("Group F", "2026-06-20", "17:00", "Netherlands", "Sweden", "AT&T Stadium", "Dallas")
add("Group E", "2026-06-20", "20:00", "Germany", "Cote d'Ivoire", "NRG Stadium", "Houston")
add("Group E", "2026-06-21", "00:00", "Ecuador", "Curacao", "Lincoln Financial Field", "Philadelphia")
add("Group F", "2026-06-21", "04:00", "Tunisia", "Japan", "Levi's Stadium", "San Francisco Bay Area")
add("Group H", "2026-06-21", "16:00", "Spain", "Saudi Arabia", "Hard Rock Stadium", "Miami")
add("Group G", "2026-06-21", "19:00", "Belgium", "Iran", "Arrowhead Stadium", "Kansas City")
add("Group H", "2026-06-21", "22:00", "Uruguay", "Cabo Verde", "Mercedes-Benz Stadium", "Atlanta")
add("Group G", "2026-06-22", "01:00", "New Zealand", "Egypt", "BC Place", "Vancouver")
add("Group J", "2026-06-22", "17:00", "Argentina", "Austria", "MetLife Stadium", "New York / New Jersey")
add("Group I", "2026-06-22", "21:00", "France", "Iraq", "Gillette Stadium", "Boston")
add("Group I", "2026-06-23", "00:00", "Norway", "Senegal", "Lumen Field", "Seattle")
add("Group J", "2026-06-23", "03:00", "Jordan", "Algeria", "SoFi Stadium", "Los Angeles")
add("Group K", "2026-06-23", "17:00", "Portugal", "Uzbekistan", "AT&T Stadium", "Dallas")
add("Group L", "2026-06-23", "20:00", "England", "Ghana", "Lincoln Financial Field", "Philadelphia")
add("Group L", "2026-06-23", "23:00", "Panama", "Croatia", "MetLife Stadium", "New York / New Jersey")
add("Group K", "2026-06-24", "02:00", "Colombia", "Congo DR", "BC Place", "Vancouver")

# ---- Matchday 3 (simultaneous kickoffs per group) ----
add("Group A", "2026-06-24", "22:00", "South Africa", "Korea Republic", "Estadio Akron", "Guadalajara")
add("Group A", "2026-06-24", "22:00", "Mexico", "Czechia", "Estadio Azteca", "Mexico City")
add("Group B", "2026-06-25", "00:00", "Qatar", "Bosnia and Herzegovina", "Levi's Stadium", "San Francisco Bay Area")
add("Group B", "2026-06-25", "00:00", "Canada", "Switzerland", "BC Place", "Vancouver")
add("Group C", "2026-06-24", "22:00", "Scotland", "Brazil", "Hard Rock Stadium", "Miami")
add("Group C", "2026-06-24", "22:00", "Morocco", "Haiti", "Mercedes-Benz Stadium", "Atlanta")
add("Group D", "2026-06-26", "02:00", "Turkiye", "United States", "SoFi Stadium", "Los Angeles")
add("Group D", "2026-06-26", "02:00", "Paraguay", "Australia", "Lumen Field", "Seattle")
add("Group E", "2026-06-25", "20:00", "Ecuador", "Germany", "MetLife Stadium", "New York / New Jersey")
add("Group E", "2026-06-25", "20:00", "Curacao", "Cote d'Ivoire", "Lincoln Financial Field", "Philadelphia")
add("Group F", "2026-06-25", "23:00", "Japan", "Sweden", "AT&T Stadium", "Dallas")
add("Group F", "2026-06-25", "23:00", "Tunisia", "Netherlands", "Arrowhead Stadium", "Kansas City")
add("Group G", "2026-06-27", "03:00", "Egypt", "Iran", "Lumen Field", "Seattle")
add("Group G", "2026-06-27", "03:00", "New Zealand", "Belgium", "BC Place", "Vancouver")
add("Group H", "2026-06-27", "00:00", "Cabo Verde", "Saudi Arabia", "NRG Stadium", "Houston")
add("Group H", "2026-06-27", "00:00", "Uruguay", "Spain", "Estadio Akron", "Guadalajara")
add("Group I", "2026-06-26", "19:00", "Norway", "France", "Gillette Stadium", "Boston")
add("Group I", "2026-06-26", "19:00", "Senegal", "Iraq", "BMO Field", "Toronto")
add("Group J", "2026-06-27", "22:00", "Argentina", "Algeria", "Hard Rock Stadium", "Miami")
add("Group J", "2026-06-27", "22:00", "Austria", "Jordan", "Mercedes-Benz Stadium", "Atlanta")
add("Group K", "2026-06-27", "23:30", "Congo DR", "Uzbekistan", "AT&T Stadium", "Dallas")
add("Group K", "2026-06-27", "23:30", "Colombia", "Portugal", "NRG Stadium", "Houston")
add("Group L", "2026-06-28", "02:00", "Panama", "England", "MetLife Stadium", "New York / New Jersey")
add("Group L", "2026-06-28", "02:00", "Croatia", "Ghana", "Levi's Stadium", "San Francisco Bay Area")

# ============================ KNOCKOUT STAGE (32) ============================
# Teams are placeholders (bracket slots) until group results are final.
# ---- Round of 32 (Jun 28 - Jul 3) ----
add("Round of 32", "2026-06-28", "20:00", "Runner-up Group A", "Runner-up Group B", "SoFi Stadium", "Los Angeles")
add("Round of 32", "2026-06-29", "17:00", "Winner Group C", "Runner-up Group F", "NRG Stadium", "Houston")
add("Round of 32", "2026-06-29", "20:30", "Winner Group E", "Third Place A/B/C/D/F", "Gillette Stadium", "Boston")
add("Round of 32", "2026-06-30", "01:00", "Winner Group F", "Runner-up Group C", "Estadio BBVA", "Monterrey")
add("Round of 32", "2026-06-30", "17:00", "Runner-up Group E", "Runner-up Group I", "AT&T Stadium", "Dallas")
add("Round of 32", "2026-06-30", "21:00", "Winner Group I", "Third Place C/D/F/G/H", "MetLife Stadium", "New York / New Jersey")
add("Round of 32", "2026-07-01", "01:00", "Winner Group A", "Third Place C/E/F/H/I", "Estadio Azteca", "Mexico City")
add("Round of 32", "2026-07-01", "16:00", "Winner Group L", "Third Place E/H/I/J/K", "Mercedes-Benz Stadium", "Atlanta")
add("Round of 32", "2026-07-01", "20:00", "Winner Group G", "Third Place A/E/H/I/J", "Lumen Field", "Seattle")
add("Round of 32", "2026-07-02", "00:00", "Winner Group D", "Third Place B/E/F/I/J", "Levi's Stadium", "San Francisco Bay Area")
add("Round of 32", "2026-07-02", "17:00", "Runner-up Group K", "Runner-up Group L", "Hard Rock Stadium", "Miami")
add("Round of 32", "2026-07-02", "20:00", "Winner Group H", "Runner-up Group J", "Estadio Akron", "Guadalajara")
add("Round of 32", "2026-07-02", "23:00", "Winner Group B", "Third Place E/F/G/I/J", "BC Place", "Vancouver")
add("Round of 32", "2026-07-03", "16:00", "Winner Group J", "Runner-up Group H", "Arrowhead Stadium", "Kansas City")
add("Round of 32", "2026-07-03", "19:00", "Winner Group K", "Third Place D/E/I/J/L", "Lincoln Financial Field", "Philadelphia")
add("Round of 32", "2026-07-03", "23:00", "Runner-up Group D", "Runner-up Group G", "BMO Field", "Toronto")

# ---- Round of 16 (Jul 4 - 7) ----
add("Round of 16", "2026-07-04", "17:00", "Winner Match 74", "Winner Match 77", "Lincoln Financial Field", "Philadelphia")
add("Round of 16", "2026-07-04", "21:00", "Winner Match 73", "Winner Match 75", "NRG Stadium", "Houston")
add("Round of 16", "2026-07-05", "17:00", "Winner Match 76", "Winner Match 78", "MetLife Stadium", "New York / New Jersey")
add("Round of 16", "2026-07-05", "21:00", "Winner Match 80", "Winner Match 82", "Estadio Azteca", "Mexico City")
add("Round of 16", "2026-07-06", "17:00", "Winner Match 79", "Winner Match 81", "AT&T Stadium", "Dallas")
add("Round of 16", "2026-07-06", "21:00", "Winner Match 83", "Winner Match 84", "Hard Rock Stadium", "Miami")
add("Round of 16", "2026-07-07", "17:00", "Winner Match 86", "Winner Match 88", "SoFi Stadium", "Los Angeles")
add("Round of 16", "2026-07-07", "21:00", "Winner Match 85", "Winner Match 87", "Lumen Field", "Seattle")

# ---- Quarter-finals (Jul 9 - 11) ----
add("Quarter-final", "2026-07-09", "21:00", "Winner Match 89", "Winner Match 90", "Gillette Stadium", "Boston")
add("Quarter-final", "2026-07-10", "21:00", "Winner Match 93", "Winner Match 94", "SoFi Stadium", "Los Angeles")
add("Quarter-final", "2026-07-11", "16:00", "Winner Match 91", "Winner Match 92", "Arrowhead Stadium", "Kansas City")
add("Quarter-final", "2026-07-11", "20:00", "Winner Match 95", "Winner Match 96", "Hard Rock Stadium", "Miami")

# ---- Semi-finals (Jul 14 - 15) ----
add("Semi-final", "2026-07-14", "23:00", "Winner Match 97", "Winner Match 98", "AT&T Stadium", "Dallas")
add("Semi-final", "2026-07-15", "23:00", "Winner Match 99", "Winner Match 100", "Mercedes-Benz Stadium", "Atlanta")

# ---- Third-place play-off (Jul 18) ----
add("Third-place play-off", "2026-07-18", "20:00", "Loser Match 101", "Loser Match 102", "Hard Rock Stadium", "Miami")

# ---- Final (Jul 19) ----
add("Final", "2026-07-19", "19:00", "Winner Match 101", "Winner Match 102", "MetLife Stadium", "New York / New Jersey")


def esc(text):
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")

lines = [
    "BEGIN:VCALENDAR",
    "VERSION:2.0",
    "PRODID:-//worldcup2026//ICS//EN",
    "CALSCALE:GREGORIAN",
    "METHOD:PUBLISH",
    "X-WR-CALNAME:FIFA World Cup 2026",
    "X-WR-TIMEZONE:UTC",
    "X-WR-CALDESC:All 104 matches of the 2026 FIFA World Cup (USA/Canada/Mexico). "
    "Compiled from public sources June 2026; verify kick-off times on fifa.com.",
]

stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

for i, (stage, date, t, home, away, venue, city) in enumerate(M, start=1):
    start = datetime.strptime(f"{date} {t}", "%Y-%m-%d %H:%M")
    end = start + timedelta(hours=2)
    summary = f"{home} vs {away}"
    desc = f"{stage} - Match {i}. Venue: {venue}, {city}. " \
           f"Times are best-effort (UTC); verify on fifa.com."
    lines += [
        "BEGIN:VEVENT",
        f"UID:wc2026-match-{i}@worldcup2026",
        f"DTSTAMP:{stamp}",
        f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}Z",
        f"DTEND:{end.strftime('%Y%m%dT%H%M%S')}Z",
        f"SUMMARY:{esc(summary)} ({esc(stage)})",
        f"LOCATION:{esc(venue + ', ' + city)}",
        f"DESCRIPTION:{esc(desc)}",
        "END:VEVENT",
    ]

lines.append("END:VCALENDAR")

with open("worldcup2026.ics", "w", newline="\r\n") as f:
    f.write("\r\n".join(lines) + "\r\n")

print(f"Wrote worldcup2026.ics with {len(M)} matches.")
