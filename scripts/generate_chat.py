"""
ChatScope AI - Synthetic Hinglish Group Chat Generator
========================================================
Generates a deterministic (seeded) synthetic WhatsApp-style group chat dataset
for 8 personas over ~6 months, with 3 embedded "anchored decision threads"
(Manali trip, group project tech stack, dinner meetup) plus generic daily
chatter, noise, forwards, typos, emojis, and one-word replies.

Run:
    python3 scripts/generate_chat.py

Output:
    data/synthetic_chat.json
"""

import json
import random
import os
import datetime as dt

import numpy as np

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_chat.json")

START_DATE = dt.date(2026, 1, 1)
END_DATE = dt.date(2026, 6, 30)   # ~6 months
TOTAL_DAYS = (END_DATE - START_DATE).days + 1

# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------
# activity_weight: relative likelihood of posting on a given generic message slot
# emoji_rate / typo_rate / hinglish_rate: probability knobs per message
# short_reply_rate: probability persona sends a short reply instead of a full template
PERSONAS = {
    "Priya Sharma": dict(activity_weight=1.0, emoji_rate=0.15, typo_rate=0.05,
                          hinglish_rate=0.55, short_reply_rate=0.15, style="analytical"),
    "Rahul Verma": dict(activity_weight=1.3, emoji_rate=0.55, typo_rate=0.20,
                         hinglish_rate=0.85, short_reply_rate=0.35, style="casual"),
    "Aman Singh": dict(activity_weight=1.2, emoji_rate=0.25, typo_rate=0.08,
                        hinglish_rate=0.60, short_reply_rate=0.15, style="organizer"),
    "Neha Kapoor": dict(activity_weight=0.9, emoji_rate=0.10, typo_rate=0.03,
                         hinglish_rate=0.40, short_reply_rate=0.10, style="structured"),
    "Arjun Mehta": dict(activity_weight=0.8, emoji_rate=0.08, typo_rate=0.04,
                         hinglish_rate=0.30, short_reply_rate=0.20, style="technical"),
    "Sneha Patel": dict(activity_weight=1.3, emoji_rate=0.60, typo_rate=0.18,
                         hinglish_rate=0.80, short_reply_rate=0.30, style="social"),
    "Rohan Gupta": dict(activity_weight=1.1, emoji_rate=0.20, typo_rate=0.25,
                         hinglish_rate=0.70, short_reply_rate=0.55, style="terse"),
    "Kavya Iyer": dict(activity_weight=0.9, emoji_rate=0.20, typo_rate=0.05,
                        hinglish_rate=0.50, short_reply_rate=0.10, style="detailed"),
}
NAMES = list(PERSONAS.keys())

EMOJIS = ["😂", "🔥", "😅", "👍", "🙌", "😍", "🥲", "😭", "🤔", "✅", "🎉", "😴", "🙃", "💀", "🚀", "☕", "🍕", "🌧️"]
SHORT_REPLIES = ["haan", "ok", "kk", "bilkul", "nahi yaar", "sahi hai", "achha", "hmm",
                 "😂😂", "theek hai", "sure", "kyu nahi", "sahi baat", "chalega", "yaar sach",
                 "lol", "haha true", "exactly", "arre wah", "no way", "seriously?", "kb tak",
                 "kal batata hu", "abhi busy hu", "1 sec", "done", "great", "nice one"]

HINGLISH_CONNECTORS = ["bhai", "yaar", "arre", "abe", "toh", "matlab", "scene ye hai ki",
                        "waise", "vaise bhi", "seriously", "honestly", "buddy"]

TYPO_SUBS = {
    "the": "teh", "and": "nad", "you": "u", "are": "r", "for": "4", "to": "2",
    "please": "plz", "tomorrow": "tmrw", "because": "bcoz", "with": "wid",
    "what": "wat", "message": "msg", "before": "b4", "okay": "ok",
}

# ---------------------------------------------------------------------------
# Generic topic template banks (slot-filled)
# ---------------------------------------------------------------------------
FOODS = ["maggi", "biryani", "pizza", "chole bhature", "momos", "dosa", "paratha",
         "pav bhaji", "chai aur samosa", "vada pav", "pasta", "shawarma"]
SHOWS = ["that new Netflix show", "Panchayat season", "the new cricket highlights",
         "that YouTube video", "the new trailer", "that Instagram reel", "the meme page post"]
SUBJECTS = ["DBMS assignment", "stats submission", "project report", "internship application",
            "office deck", "client email", "exam prep", "lab file"]
WEEKDAYS_TALK = ["Monday", "weekend", "Friday evening", "Sunday", "next week"]
WEATHER = ["itni garmi ho gyi hai", "kal se baarish chalu hai", "AC kharab ho gya mera",
           "thand shuru ho gyi finally", "mausam mast hai aaj"]
GYM_HEALTH = ["gym chala aaj subah", "running kar raha tha", "kal se diet strict",
              "neend puri nahi hui", "headache ho raha hai thoda"]

GENERIC_TEMPLATES = [
    lambda: f"kal {random.choice(FOODS)} khaane chalte hain kya",
    lambda: f"{random.choice(SHOWS)} dekha kisine? bohot mast tha",
    lambda: f"yr {random.choice(SUBJECTS)} ka deadline kab hai exactly",
    lambda: f"{random.choice(WEEKDAYS_TALK)} ko free ho tum log?",
    lambda: f"{random.choice(WEATHER)} 😩",
    lambda: f"{random.choice(GYM_HEALTH)}, kal se sudharunga",
    lambda: f"koi match dekh raha hai aaj raat?",
    lambda: f"office/college mein itna kaam pada hai yaar, thak gaya",
    lambda: f"{random.choice(FOODS)} order karu ya bahar chale?",
    lambda: f"kisi ne notes bheje the woh bhejo dobara please",
    lambda: f"itni neend aa rahi hai abhi meeting mein",
    lambda: f"wifi phir se down ho gya mera, itna irritating hai",
    lambda: f"{random.choice(SHOWS)} ka new episode kab aa raha hai pata hai kisiko?",
    lambda: f"assignment submit kar diya finally, chain aa gyi",
    lambda: f"kal ka plan kya hai guys, kuch decide karte hain",
    lambda: f"itna traffic tha aaj, 1 ghanta late pahucha",
    lambda: f"phone ki battery hi nahi chal rahi aajkal",
    lambda: f"koi movie chalen weekend pe?",
    lambda: f"result kab aane wala hai, pata hai kisiko?",
    lambda: f"{random.choice(FOODS)} ka craving ho raha hai bohot",
]

FORWARD_TEMPLATES = [
    "🌸 Good morning everyone, have a blessed day ahead 🙏",
    "Forwarded: Scientists say drinking water first thing in the morning boosts metabolism by 30%!! Share with 10 people or bad luck for a week",
    "Happy Sunday to all! Stay blessed 🌞🙏",
    "Forwarded: Breaking - new WhatsApp update will make chats disappear after 24 hrs, save this msg to confirm",
    "🎉 Happy Diwali to you and your family! May this festival bring joy and prosperity 🪔",
    "Forwarded: 5 tips to save money every month that banks don't want you to know",
    "Good night everyone, sweet dreams 🌙✨",
    "Forwarded: This joke will make your day - why did the developer go broke? Because he used up all his cache",
]

# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def apply_typos(text, rate):
    words = text.split(" ")
    out = []
    for w in words:
        lw = w.lower().strip(",.!?")
        if lw in TYPO_SUBS and random.random() < rate * 2.5:
            repl = TYPO_SUBS[lw]
            out.append(repl)
        elif len(w) > 4 and random.random() < rate * 0.5:
            # drop one interior character to mimic a typo
            i = random.randint(1, len(w) - 2)
            out.append(w[:i] + w[i + 1:])
        else:
            out.append(w)
    return " ".join(out)


def style_message(base_text, persona):
    p = PERSONAS[persona]
    text = base_text
    if random.random() < p["hinglish_rate"] * 0.4:
        text = f"{random.choice(HINGLISH_CONNECTORS)} {text}"
    if random.random() < p["emoji_rate"]:
        text = f"{text} {random.choice(EMOJIS)}"
    if p["style"] == "terse" and random.random() < 0.4:
        # trim to first ~6 words for terse persona
        words = text.split(" ")
        text = " ".join(words[:max(3, min(6, len(words)))])
    text = apply_typos(text, p["typo_rate"])
    if p["style"] == "structured" and random.random() < 0.3:
        text = text.capitalize()
    return text.strip()


def random_time_on(date_obj):
    # message activity skewed towards evening (group chat behaviour)
    hour_weights = np.array([1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 5, 6, 6, 5, 5, 5, 6, 7, 9, 10, 9, 7, 4, 2], dtype=float)
    hour_weights = hour_weights / hour_weights.sum()
    hour = int(np.random.choice(np.arange(24), p=hour_weights))
    minute = int(np.random.randint(0, 60))
    second = int(np.random.randint(0, 60))
    return dt.datetime.combine(date_obj, dt.time(hour, minute, second))


def make_day_clock():
    """Returns a function that hands out strictly increasing timestamps for a
    given date, so multi-message scripted conversations on the same day stay
    in narrative order after the global chronological sort."""
    state = {}

    def get_time(date_obj):
        if date_obj not in state:
            start_hour = int(np.random.randint(9, 15))
            state[date_obj] = dt.datetime.combine(date_obj, dt.time(start_hour, int(np.random.randint(0, 60))))
        else:
            state[date_obj] += dt.timedelta(minutes=int(np.random.randint(4, 45)), seconds=int(np.random.randint(0, 60)))
        return state[date_obj]

    return get_time


def make_message(sender, text, timestamp, thread_id, topic, is_forwarded=False):
    return {
        "message_id": None,  # assigned after global chronological sort
        "timestamp": timestamp.isoformat(),
        "sender": sender,
        "message": text,
        "message_type": "text",
        "thread_id": thread_id,
        "metadata": {
            "date": timestamp.date().isoformat(),
            "month": timestamp.strftime("%Y-%m"),
            "is_forwarded": is_forwarded,
            "topic": topic,
        },
    }


def weighted_persona(exclude=None):
    pool = [n for n in NAMES if n != exclude] if exclude else NAMES
    weights = np.array([PERSONAS[n]["activity_weight"] for n in pool], dtype=float)
    weights = weights / weights.sum()
    return str(np.random.choice(pool, p=weights))


# ---------------------------------------------------------------------------
# Generic daily chatter
# ---------------------------------------------------------------------------

def generate_generic_day(date_obj, count):
    msgs = []
    last_sender = None
    for _ in range(count):
        sender = weighted_persona()
        p = PERSONAS[sender]
        if random.random() < p["short_reply_rate"] and last_sender is not None:
            text = random.choice(SHORT_REPLIES)
            topic = "chitchat"
        elif random.random() < 0.06:
            text = random.choice(FORWARD_TEMPLATES)
            topic = "forward"
            ts = random_time_on(date_obj)
            msgs.append(make_message(sender, text, ts, f"thread_general_{date_obj.isoformat()}", topic, is_forwarded=True))
            last_sender = sender
            continue
        else:
            text = style_message(random.choice(GENERIC_TEMPLATES)(), sender)
            topic = random.choice(["food", "college_work", "entertainment", "weather", "health", "chitchat", "logistics"])
        ts = random_time_on(date_obj)
        msgs.append(make_message(sender, text, ts, f"thread_general_{date_obj.isoformat()}", topic))
        last_sender = sender
    return msgs


# ---------------------------------------------------------------------------
# Anchored Thread 1: Manali Trip (Jan 10 - Jan 25)
# ---------------------------------------------------------------------------

def generate_trip_thread():
    thread_id = "thread_trip_001"
    msgs = []
    d0 = dt.date(2026, 1, 10)
    clock = make_day_clock()

    script = [
        (0, "Aman Singh", "guys long weekend aa raha hai, trip plan karein kya?"),
        (0, "Sneha Patel", "haan yaar bohot time ho gya kahi gaye hue"),
        (0, "Rahul Verma", "bilkul chalo kahi chalte hain 🔥"),
        (1, "Aman Singh", "options soch raha tha - Manali, Rishikesh ya Goa"),
        (1, "Priya Sharma", "budget kitna rakhna hai per person, Goa thoda costly pad sakta hai flights ke wajah se"),
        (1, "Neha Kapoor", "date bhi to fix karni hogi, long weekend mein exact kaunse din free hain sabke"),
        (2, "Arjun Mehta", "Rishikesh mein river rafting hai, Manali mein snow milega is time"),
        (2, "Kavya Iyer", "mujhe personally Manali zyada pasand aayega kyuki snowfall abhi tak chal raha hoga wahan, aur Rishikesh hum pehle already ja chuke hain do saal pehle"),
        (2, "Rohan Gupta", "manali +1"),
        (3, "Priya Sharma", "Manali ka budget dekha maine, bus se jaayein toh per person around 6-7k mein ho jayega including stay"),
        (3, "Sneha Patel", "omg itna kam? toh Manali hi final karte hain na"),
        (3, "Aman Singh", "wait thoda ruko, Goa bhi ek baar dekh lete hain flight prices"),
        (4, "Arjun Mehta", "Goa flights are almost double the Manali bus cost right now, not worth it for this budget"),
        (4, "Neha Kapoor", "matlab Manali is clearly cheaper aur weather bhi better hoga is season mein"),
        (5, "Rahul Verma", "bhai Manali chalte hain na, itna discuss kyu kar rahe ho 😂"),
        (5, "Rohan Gupta", "haan bhai manali finalize karo"),
        (6, "Aman Singh", "theek hai, poll kar leta hu quickly - Manali ya Goa ya Rishikesh"),
        (6, "Kavya Iyer", "mera vote Manali, especially snow ke liye"),
        (6, "Priya Sharma", "Manali se agree, budget bhi fit ho raha hai"),
        (7, "Neha Kapoor", "sabne vote kar diya, majority Manali pe hai"),
        (7, "Sneha Patel", "yesss Manali final karte hain toh 🎉"),
        (8, "Aman Singh", "dates ka kya, 24 se 27 January chalega sabko?"),
        (8, "Priya Sharma", "mere liye 24-27 fine hai, us weekend office bhi nahi hai"),
        (8, "Arjun Mehta", "24-27 works for me too"),
        (9, "Neha Kapoor", "confirm kar rahi hu list - Aman, Sneha, Rahul, Priya, Arjun, Kavya, Rohan sab available 24-27?"),
        (9, "Rohan Gupta", "haan available"),
        (9, "Kavya Iyer", "haan mai bhi free hu us weekend"),
        (10, "Rahul Verma", "budget final kar do bhai, hotel bhi dekhna hai"),
        (10, "Priya Sharma", "stay ke liye ek decent hostel dekha hai, 800 per night per person, dorm style"),
        (10, "Aman Singh", "chalo woh book kar lete hain, sabko bata dena payment ke liye"),
        (11, "Sneha Patel", "wait ek baar Old Manali wale area mein bhi dekh lein? zyada scenic hai"),
        (11, "Neha Kapoor", "utna difference nahi hoga price mein, jo mila hai wahi book kar lo time bacha ke"),
        (12, "Aman Singh", "bhai Manali fix hai, ab bas tickets dekhte hain 🔥"),
        (12, "Rahul Verma", "finally 🎉🎉 ab bus tickets bhi book kar do jaldi"),
        (12, "Kavya Iyer", "great, main share kar deti hu ek Google sheet expenses track karne ke liye"),
    ]

    for day_offset, sender, text in script:
        date_obj = d0 + dt.timedelta(days=day_offset)
        ts = clock(date_obj)
        msgs.append(make_message(sender, text, ts, thread_id, "trip"))

    # a few follow-up logistics messages after the decision
    followups = [
        (13, "Rahul Verma", "tickets book ho gaye maine apna, aap log bhi kar lo jaldi"),
        (14, "Sneha Patel", "packing list bana rahi hu, jacket zaroor le lena sab log thand bohot hogi"),
        (15, "Rohan Gupta", "kitna paisa lekar chalna hai total"),
    ]
    for day_offset, sender, text in followups:
        date_obj = d0 + dt.timedelta(days=day_offset)
        ts = clock(date_obj)
        msgs.append(make_message(sender, text, ts, thread_id, "trip"))

    return msgs


# ---------------------------------------------------------------------------
# Anchored Thread 2: Group Project Tech Stack (Mar 1 - Mar 28)
# ---------------------------------------------------------------------------

def generate_project_thread():
    thread_id = "thread_project_001"
    msgs = []
    d0 = dt.date(2026, 3, 1)
    clock = make_day_clock()

    script = [
        (0, "Neha Kapoor", "guys hume group project ka tech stack finalize karna hai is week"),
        (0, "Arjun Mehta", "backend ke liye mai suggest karunga either Django or Node with Express"),
        (1, "Priya Sharma", "kaunsa jaldi seekhne mein easy hoga, hum sab ke paas time kam hai"),
        (1, "Arjun Mehta", "Node thoda faster to prototype hai agar sab JS jaante ho, Django zyada batteries-included hai"),
        (2, "Aman Singh", "frontend ke liye React chalte hain, sabne college mein use kiya hai already"),
        (2, "Kavya Iyer", "React se agree, lekin state management ke liye kuch decide karna hoga - Redux ya Context API"),
        (3, "Rohan Gupta", "context api simple hai bas usi se karo"),
        (3, "Neha Kapoor", "roles bhi assign karne honge - kaun frontend, kaun backend, kaun database dekhega"),
        (4, "Arjun Mehta", "database ke liye PostgreSQL suggest karunga, relational data hai humara"),
        (4, "Priya Sharma", "MongoDB ke baare mein kya khayal hai, thoda flexible rahega schema changes ke liye"),
        (5, "Arjun Mehta", "our data has clear relations between users, tasks and projects, PostgreSQL fits better here honestly"),
        (5, "Kavya Iyer", "makes sense, relational structure ke liye SQL zyada sahi rahega long term"),
        (6, "Aman Singh", "toh final list ban rahi hai - React frontend, kya backend pe decide hua?"),
        (6, "Neha Kapoor", "abhi tak Node vs Django pe clarity nahi hai, vote kar lete hain"),
        (7, "Rahul Verma", "mujhe jo bhi chale mai theek hu, jaldi decide kar lo bas 😅"),
        (7, "Sneha Patel", "Node try karte hain na, sabko thoda JS aata hai already toh easier hoga"),
        (8, "Arjun Mehta", "fine with Node + Express, faster for our timeline given everyone knows JS already"),
        (8, "Priya Sharma", "ok Node se agree, PostgreSQL bhi with Node achhe se kaam karta hai"),
        (9, "Neha Kapoor", "roles: Arjun aur Priya backend + database, Aman aur Kavya frontend, Sneha aur Rahul UI/testing, Rohan aur main deployment/docs"),
        (9, "Rohan Gupta", "theek hai chalega"),
        (10, "Aman Singh", "deadline kya rakhein final submission ke liye"),
        (10, "Kavya Iyer", "professor ne 28th March bola tha last date, usse 2 din pehle apna internal deadline rakh lete hain"),
        (11, "Neha Kapoor", "toh internal deadline 26th March, final submission 28th"),
        (12, "Arjun Mehta", "sounds good, I'll set up the repo today with the initial folder structure"),
        (13, "Aman Singh", "great, sabko access de dena repo ka"),
        (14, "Priya Sharma", "database schema draft kal tak bhej dungi review ke liye"),
        (15, "Neha Kapoor", "final tech stack confirm kar rahi hu sabke liye: React + Node/Express + PostgreSQL, deadline 26th internal, 28th final submission"),
        (16, "Sneha Patel", "perfect, sab clear hai ab 👍"),
        (16, "Rahul Verma", "chalo shuru karte hain toh finally 🔥"),
    ]

    for day_offset, sender, text in script:
        date_obj = d0 + dt.timedelta(days=day_offset)
        ts = clock(date_obj)
        msgs.append(make_message(sender, text, ts, thread_id, "project"))

    followups = [
        (18, "Arjun Mehta", "repo bana diya hai, link daal raha hu yahan"),
        (20, "Priya Sharma", "schema draft bhej diya, ek baar sab review kar lena"),
        (24, "Neha Kapoor", "guys deadline paas aa raha hai, apna apna status update karo yahan"),
    ]
    for day_offset, sender, text in followups:
        date_obj = d0 + dt.timedelta(days=day_offset)
        ts = clock(date_obj)
        msgs.append(make_message(sender, text, ts, thread_id, "project"))

    return msgs


# ---------------------------------------------------------------------------
# Anchored Thread 3: Dinner / Meetup (May 5 - May 12)
# ---------------------------------------------------------------------------

def generate_dinner_thread():
    thread_id = "thread_dinner_001"
    msgs = []
    d0 = dt.date(2026, 5, 5)
    clock = make_day_clock()

    script = [
        (0, "Sneha Patel", "bohot din ho gaye sab milke hangout kiye hue, dinner plan karein?"),
        (0, "Rahul Verma", "haan yaar bohot time ho gya, chalo kahi chalte hain 🍕"),
        (1, "Aman Singh", "kaunsi jagah? koi suggest karo achhi si"),
        (1, "Kavya Iyer", "woh naya restaurant khula hai market mein, achhi reviews sun rahi hu uske"),
        (1, "Priya Sharma", "price range kya hai wahan ka, thoda budget friendly rahe toh better hoga"),
        (2, "Kavya Iyer", "around 500-600 per person for a full meal, not too expensive"),
        (2, "Neha Kapoor", "headcount confirm karna hoga pehle, kitne log aa rahe hain exactly"),
        (3, "Rohan Gupta", "mai aa raha hu"),
        (3, "Arjun Mehta", "count me in too"),
        (3, "Sneha Patel", "mai aur Rahul dono aayenge"),
        (4, "Aman Singh", "toh total 7 log ho gaye confirm, sabko headcount pata chal gya"),
        (4, "Priya Sharma", "us restaurant mein 7 logo ke liye table milega na easily?"),
        (5, "Kavya Iyer", "haan unhone bola reservation kara denge, weekday hai toh crowd kam hoga"),
        (5, "Neha Kapoor", "time kya rakhein, 8 baje thik rahega sabke liye?"),
        (6, "Rahul Verma", "8 baje perfect hai mere liye"),
        (6, "Rohan Gupta", "8 chalega"),
        (7, "Aman Singh", "sabko confirm - 8 baje us naye restaurant mein, 7 log, Kavya reservation kara degi"),
        (7, "Sneha Patel", "yesss finally dinner plan ho gaya 🎉"),
        (7, "Kavya Iyer", "dinner confirm, Friday 8pm, table for 7, reservation ho jayegi kal tak"),
    ]

    for day_offset, sender, text in script:
        date_obj = d0 + dt.timedelta(days=day_offset)
        ts = clock(date_obj)
        msgs.append(make_message(sender, text, ts, thread_id, "meetup"))

    return msgs


# ---------------------------------------------------------------------------
# Assemble full dataset
# ---------------------------------------------------------------------------

def main():
    all_msgs = []

    # 1. anchored threads
    all_msgs.extend(generate_trip_thread())
    all_msgs.extend(generate_project_thread())
    all_msgs.extend(generate_dinner_thread())

    # 2. generic daily chatter across the full ~6 month span
    for day_idx in range(TOTAL_DAYS):
        date_obj = START_DATE + dt.timedelta(days=day_idx)
        # vary daily volume: quiet days ~5-15, medium ~15-35, active ~35-55
        base = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
        if base == 1:
            count = int(np.random.randint(5, 16))
        elif base == 2:
            count = int(np.random.randint(16, 36))
        else:
            count = int(np.random.randint(36, 56))
        all_msgs.extend(generate_generic_day(date_obj, count))

    # 3. sort chronologically and assign sequential message_ids
    all_msgs.sort(key=lambda m: m["timestamp"])
    for i, m in enumerate(all_msgs, start=1):
        m["message_id"] = f"msg_{i:06d}"

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_msgs, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(all_msgs)} messages -> {OUT_PATH}")
    print(f"Date range: {all_msgs[0]['metadata']['date']} to {all_msgs[-1]['metadata']['date']}")
    thread_counts = {}
    for m in all_msgs:
        tid = m["thread_id"]
        thread_counts[tid] = thread_counts.get(tid, 0) + 1
    for tid in ["thread_trip_001", "thread_project_001", "thread_dinner_001"]:
        print(f"  {tid}: {thread_counts.get(tid, 0)} messages")


if __name__ == "__main__":
    main()
