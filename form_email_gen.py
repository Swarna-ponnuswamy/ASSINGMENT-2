import random, os, csv

random.seed(42)

N = 200
FIRST = ["Alex","Jordan","Taylor","Casey","Morgan","Sam","Avery","Jamie","Riley","Quinn"]
ORG   = ["Finance","Marketing","Research","Operations","Support","Legal","Security","HR","Design","Analytics"]
ACT   = ["submit the report","reschedule the meeting","review the draft","update the budget","confirm attendance",
         "share the slides","approve the request","finalize the agenda","prepare the summary","send the invoice"]
DATE  = ["Monday","Tuesday","Wednesday","Thursday","Friday","next week","this afternoon","tomorrow morning"]
LOC   = ["Sydney","Melbourne","Brisbane","Perth","Adelaide","Canberra","Hobart"]

os.makedirs("data/form_email", exist_ok=True)
with open("data/form_email/original.txt", "w") as ftxt, open("data/form_email/original.csv", "w", newline="") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(["id","sentence"])
    for i in range(N):
        s = (f"Hi {random.choice(FIRST)}, please {random.choice(ACT)} "
             f"by {random.choice(DATE)} for the {random.choice(ORG)} team in {random.choice(LOC)}.")
        ftxt.write(s + "\n")
        w.writerow([i, s])

print("✅ Wrote data/form_email/original.{txt,csv}")
