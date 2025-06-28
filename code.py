# -*- coding: utf-8 -*-
"""
updated_timetable_generator.py

Python port of the Next.js LTPSC scheduling logic, updated to:
• Enforce a 30-min mid-morning break (11:00–11:30), 2-hour lunch (12:30–14:30),
  30-min snack break (16:30–17:00), and no classes after 19:00.
• Schedule B1	+B2 electives in parallel, and B3	+B4 electives in parallel.
• Export all formatted CSVs with spaces instead of underscores and remove “.0.”
"""

import os, random, traceback, re
import pandas as pd

WEEKDAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday"]

TIME_SLOTS = [
    "08:00-09:00","09:00-10:00","10:00-11:00",
    "11:00-11:30","11:30-12:30",
    "12:30-13:30","13:30-14:30",
    "14:30-15:30","15:30-16:30",
    "16:30-17:00","17:00-18:00","18:00-19:00"
]

BREAK_SLOTS = {
    "11:00-11:30": "Mid-Morning Break",
    "12:30-13:30": "Lunch Break",
    "13:30-14:30": "Lunch Break",
    "16:30-17:00": "Snack Break"
}

# Room capacities
CLASSROOM_CAPACITIES = {
    "C002": 120, "C003": 120, "C004": 240,
    "C101": 86, "C102": 86, "C104": 86,
    "C202": 86, "C203": 86, "C204": 86, "C206": 86,
    "C302": 86, "C303": 86, "C304": 86, "C305": 86
}

LAB_CAPACITIES = {
    "L105": 50, "L106": 50, "L107": 50,
    "L206": 50, "L207": 50, "L208": 50, "L209": 50
}

CLASSROOMS = sorted(list(CLASSROOM_CAPACITIES.keys()))
LABS = sorted(list(LAB_CAPACITIES.keys()))

def create_dirs():
    for d in [
        "Timetables",
        "Timetables/By Branch",
        "Timetables/By Semester",
        "Timetables/By Branch Semester",
        "Timetables/By Faculty",
        "Timetables/By Room"
    ]:
        os.makedirs(d, exist_ok=True)

def save_faculty_timetables(tt):
    faculty_map = {}

    # First pass: collect all unique faculty names and standardize them
    for d in WEEKDAYS:
        for sl in TIME_SLOTS:
            for r in CLASSROOMS + LABS:
                entry = tt[d][sl][r]
                if entry and entry[0] != "BREAK":
                    fac_str = entry[1]
                    # Split combined faculty names by various separators
                    for f_part in re.split(r'\s*/\s*|\s*&\s*|\s+and\s+|\s*\(Course\s+Coordinator:\s*|\)\s*' , str(fac_str)):
                        f_part = f_part.strip()
                        if not f_part:
                            continue
                        
                        # Cleaned name for key: lowercase, no titles/dots/spaces
                        cleaned = re.sub(r'^(dr|prof|mr|ms)\.?\s*', '', f_part.lower()).replace('.', '').replace(' ', '')
                        cleaned = re.sub(r'\(.*?\)', '', cleaned) # remove coordinator info

                        if cleaned and cleaned not in faculty_map:
                            faculty_map[cleaned] = f_part # Use the first-seen version for display

    # Create timetables for each unique faculty
    for cleaned_name, display_name in sorted(faculty_map.items()):
        pivot = pd.DataFrame(index=WEEKDAYS, columns=TIME_SLOTS).fillna("")
        for d in WEEKDAYS:
            for sl in TIME_SLOTS:
                entries_for_slot = []
                for r in CLASSROOMS + LABS:
                    entry = tt[d][sl][r]
                    if entry and entry[0] != "BREAK":
                        fac_str = entry[1]
                        is_involved = False
                        for f_part in re.split(r'\s*/\s*|\s*&\s*|\s+and\s+|\s*\(Course\s+Coordinator:\s*|\)\s*', str(fac_str)):
                            f_part_cleaned = re.sub(r'^(dr|prof|mr|ms)\.?\s*', '', f_part.strip().lower()).replace('.', '').replace(' ', '')
                            f_part_cleaned = re.sub(r'\(.*?\)', '', f_part_cleaned)
                            if f_part_cleaned == cleaned_name:
                                is_involved = True
                                break
                        
                        if is_involved:
                            entry_groups = entry[4]
                            branch_sem_str = ""
                            if isinstance(entry_groups, list):
                                branch_sem_str = ", ".join([f"{b}-{s.replace('.0','')}" for b, s in entry_groups])
                            else:
                                branch_sem_str = f"{entry_groups[0]}-{entry_groups[1].replace('.0','')}"
                            
                            entries_for_slot.append(f"{entry[2]}-{entry[3]}-{r}-{branch_sem_str}")
                if entries_for_slot:
                    pivot.loc[d, sl] = " | ".join(sorted(list(set(entries_for_slot))))
        
        safe_filename = re.sub(r'[^a-zA-Z0-9_ .-]', '', display_name).strip()
        os.makedirs("Timetables/By Faculty", exist_ok=True)
        pivot.to_csv(f"Timetables/By Faculty/{safe_filename}.csv")
        print(f"Faculty Timetable -> {safe_filename}.csv")

def save_room_timetables(df):
    real = df[df['Session Type'] != "Break"]
    all_rooms = set(CLASSROOMS + LABS)

    for room in sorted(list(all_rooms)):
        sub = real[real['Room'] == room]
        if sub.empty:
            continue
        
        pivot = pd.DataFrame(index=WEEKDAYS, columns=TIME_SLOTS).fillna("")
        for sl in TIME_SLOTS:
            for d in WEEKDAYS:
                entries = sub[(sub['Day'] == d) & (sub['Time'] == sl)]
                if not entries.empty:
                    pivot.loc[d, sl] = " | ".join(f"{r['Course Title']}-{r['Session Type']}-{r['Faculty']}-{r['Branch']}-{r['Semester']}" for _, r in entries.iterrows())
        
        pivot.to_csv(f"Timetables/By Room/{room}.csv")
        print(f"Room Timetable -> {room}.csv")

def extract_ltpsc(row):
    try:
        return [int(row.get(f,0) or 0) for f in
                ("Lectures","Tutorials","Practicals","Self-Study","Credits")]
    except:
        return None

def init_timetable():
    rooms = CLASSROOMS + LABS
    tt = {day:{slot:{r:None for r in rooms} for slot in TIME_SLOTS}
          for day in WEEKDAYS}
    for day in WEEKDAYS:
        for slot,label in BREAK_SLOTS.items():
            for r in rooms:
                tt[day][slot][r] = ("BREAK", label, "", "", "", "")
    teacher_schedule = {day: {slot: {} for slot in TIME_SLOTS} for day in WEEKDAYS}
    return tt, teacher_schedule

def find_available_rooms(tt, day, slot, strength, is_practical, semester=None):
    """Finds available rooms or labs based on strength, session type, and semester floor constraints."""
    
    possible_rooms = []
    if is_practical:
        rooms_to_check = LABS
        capacities = LAB_CAPACITIES
    else:
        rooms_to_check = CLASSROOMS
        capacities = CLASSROOM_CAPACITIES

    # Apply floor constraints for non-practical sessions
    floor_prefix = None
    if semester and not is_practical: # Floor constraints only for core subjects
        if '2.0' in semester:
            floor_prefix = 'C2' # 1st Year -> 2nd Floor (Rooms C2xx)
        elif '4.0' in semester:
            floor_prefix = 'C3' # 2nd Year -> 3rd Floor (Rooms C3xx)
        elif '6.0' in semester:
            floor_prefix = None # 3rd Year -> Any Floor

    preferred_floor_rooms = []
    other_floor_rooms = []

    for r in rooms_to_check:
        cap = capacities.get(r, 0)
        effective_cap = 90 if cap == 86 and not is_practical else cap # 86-seater rooms can hold up to 90 for non-practicals
        
        if tt[day][slot][r] is None and effective_cap >= strength:
            if floor_prefix and r.startswith(floor_prefix):
                preferred_floor_rooms.append(r)
            else:
                other_floor_rooms.append(r)

    # Prioritize preferred floor, then other floors, sorted by capacity (descending)
    sorted_preferred = sorted(preferred_floor_rooms, key=lambda r: capacities.get(r, 0), reverse=True)
    sorted_other = sorted(other_floor_rooms, key=lambda r: capacities.get(r, 0), reverse=True)
    
    available_rooms = sorted_preferred + sorted_other

    if not available_rooms:
        room_type = "lab" if is_practical else "classroom"
        print(f"WARNING: No suitable {room_type} available for strength {strength} on {day} at {slot} (Sem: {semester}).")
    
    return available_rooms

def has_conflict(tt, day, slot, fac, branch, sem, teacher_schedule):
    # Check for faculty conflict
    faculties = [f.strip() for f in re.split(r'\s*&\s*|\s*and\s*|\s*/\s*', fac)]
    for f in faculties:
        if f in teacher_schedule[day][slot]:
            return True

    for r in CLASSROOMS + LABS:
        e = tt[day][slot][r]
        if isinstance(e,tuple) and e[0]!="BREAK":
            d = e[4:]
            if len(d)==1 and (branch,sem) in d[0]:
                return True
            if len(d)==2 and (d[0],d[1])==(branch,sem):\
                return True
    return False

def count_subject_sessions_on_day(tt, day, code, branch, sem, session_type=None):
    cnt=0
    for slot in TIME_SLOTS:
        for r in CLASSROOMS+LABS:
            e=tt[day][slot][r]
            if not isinstance(e,tuple) or e[0]=="BREAK":
                continue
            c,_,_,t=e[:4]
            grp=e[4]
            if isinstance(grp,list):
                relevant=(branch,sem) in grp
            else:
                relevant=(e[4],e[5])==(branch,sem)
            if c==code and relevant and (session_type is None or t==session_type):
                cnt+=1
    return cnt

def get_semester_from_code(code):
    m=re.search(r'\d',str(code))
    if not m: return None
    return {'1':'2.0','2':'4.0','3':'6.0','4':'6.0'}.get(m.group(0))

def load_and_group_electives(filepath):
    if not os.path.exists(filepath):
        print(f"WARNING: '{filepath}' not found, skipping electives.")
        return {}
    df=pd.read_csv(filepath, header=1)
    electives={}
    # Each basket has 4 columns: Title, Code, Faculty, Strength
    for i in range(0, len(df.columns), 4):
        bcol, ccol, fcol, scol = df.columns[i], df.columns[i+1], df.columns[i+2], df.columns[i+3]
        if "Basket" not in bcol:
            continue
        basket = bcol.strip()
        electives[basket] = {}
        # Select the 4 columns for the current basket and drop rows with any NaN values
        temp = df[[bcol, ccol, fcol, scol]].dropna().copy()
        temp.columns = ['title', 'code', 'faculty', 'strength']
        for _, r in temp.iterrows():
            sem = get_semester_from_code(r['code'])
            if sem:
                electives[basket].setdefault(sem, []).append({
                    'title': r['title'].strip(),
                    'code': str(r['code']).strip(),
                    'faculty': r['faculty'].strip(),
                    'strength': int(r['strength'])
                })
    return electives

def schedule_electives(tt, electives_data, all_branches, sched, teacher_schedule):
    print("--- Scheduling Electives: pairing B1+B2 and B3+B4 ---")
    pairs = [("Basket-1", "Basket-2"), ("Basket-3", "Basket-4")]
    
    baskets_to_schedule = []
    grouped_for_return = []

    for b1, b2 in pairs:
        sems = set(electives_data.get(b1, {})) | set(electives_data.get(b2, {}))
        for sem in sems:
            subs1 = electives_data.get(b1, {}).get(sem, [])
            subs2 = electives_data.get(b2, {}).get(sem, [])
            subs = subs1 + subs2
            if not subs:
                continue

            # For 6th sem B1+B2, if subjects > 6, split them
            if sem == '6.0' and b1 == "Basket-1" and b2 == "Basket-2" and len(subs) > 6:
                print(f"INFO: Splitting basket {b1}+{b2} for Sem {sem} due to large size.")
                if subs1:
                    baskets_to_schedule.append({'label': b1, 'sem': sem, 'subs': subs1})
                if subs2:
                    baskets_to_schedule.append({'label': b2, 'sem': sem, 'subs': subs2})
                continue

            # For other cases, try to schedule combined, split if cannot.
            needed_sessions = 3  # 2 lectures, 1 tutorial
            possible_slots = []
            slots = [(d, sl) for d in WEEKDAYS for sl in TIME_SLOTS if sl not in BREAK_SLOTS]
            random.shuffle(slots)
            
            for day, slot in slots:
                if any(has_conflict(tt, day, slot, "any_faculty", b, sem, teacher_schedule) for b in all_branches):
                    continue
                rooms_free = [r for r in CLASSROOMS if tt[day][slot][r] is None]
                if len(rooms_free) >= len(subs):
                    possible_slots.append((day, slot, rooms_free))

            if len(possible_slots) >= needed_sessions:
                baskets_to_schedule.append({'label': f"{b1}+{b2}", 'sem': sem, 'subs': subs})
            else:
                print(f"INFO: Could not find enough slots for combined basket {b1}+{b2} for Sem {sem}. Splitting it.")
                if subs1:
                    for sub in subs1:
                        baskets_to_schedule.append({'label': b1, 'sem': sem, 'subs': [sub]})
                if subs2:
                    for sub in subs2:
                        baskets_to_schedule.append({'label': b2, 'sem': sem, 'subs': [sub]})

    for basket in baskets_to_schedule:
        label, sem, subs = basket['label'], basket['sem'], basket['subs']
        grouped_for_return.append((label, sem, subs))
        needed = {'Lecture': 2, 'Tutorial': 1}
        placed = {s['code']: {'Lecture': 0, 'Tutorial': 0} for s in subs}
        
        for stype, req in needed.items():
            for _ in range(req):
                found = False
                slots = [(d, sl) for d in WEEKDAYS for sl in TIME_SLOTS if sl not in BREAK_SLOTS]
                random.shuffle(slots)
                for day, slot in slots:
                    if any(has_conflict(tt, day, slot, "any_faculty", b, sem, teacher_schedule) for b in all_branches):
                        continue
                    
                    rooms_free = [r for r in CLASSROOMS if tt[day][slot][r] is None]
                    if len(rooms_free) < len(subs):
                        continue
                    
                    # Assign rooms based on strength
                    subs.sort(key=lambda s: s['strength'], reverse=True) # Prioritize larger electives
                    rooms_assigned = 0
                    for sub in subs:
                        # Find a suitable room for this elective
                        suitable_room_found = False
                        for room in rooms_free:
                            if CLASSROOM_CAPACITIES.get(room, 0) >= sub['strength']:
                                entry = (sub['code'], sub['faculty'], sub['title'], stype, [(b, sem) for b in all_branches])
                                tt[day][slot][room] = entry
                                faculties = [f.strip() for f in re.split(r'\s*&\s*|\s*and\s*|\s*/\s*', sub['faculty'])]
                                for f in faculties:
                                    teacher_schedule[day][slot][f] = sub['code']
                                sched[f"{label}-{stype}"] = sched.get(f"{label}-{stype}", 0) + 1
                                placed[sub['code']][stype] += 1
                                rooms_free.remove(room) # Room is now taken
                                rooms_assigned += 1
                                suitable_room_found = True
                                break
                        if not suitable_room_found:
                            print(f"WARNING: Could not find a suitable room for elective {sub['code']} with strength {sub['strength']}")
                            # If a suitable room is not found, do not schedule this elective
                            continue

                    if rooms_assigned == len(subs):
                        found = True
                        break
                
                if not found:
                    print(f"WARNING: Could not schedule all {stype} for {label} Sem {sem}")

    print("--- Finished Paired Electives ---")
    return tt, grouped_for_return, teacher_schedule


def load_data(branches):
    parts=[]
    for b in branches:
        fn=f"Even {b.upper()}.csv"
        if os.path.exists(fn):
            d=pd.read_csv(fn)
            d['branch']=b.upper()
            parts.append(d)
        else:
            print(f"Missing {fn}, skipping.")
    if not parts: return pd.DataFrame()
    df=pd.concat(parts,ignore_index=True).fillna('')
    df['LTPSC']=df.apply(extract_ltpsc,axis=1)
    df=df[df['LTPSC'].notnull()]
    df[['L','T','P','S','C']]=pd.DataFrame(df['LTPSC'].tolist(),index=df.index)
    df['Strength'] = pd.to_numeric(df.get('Strength', 0), errors='coerce').fillna(0).astype(int)
    df.drop_duplicates(subset=["Course Code","branch"],inplace=True)
    df['Semester']=df['Semester'].astype(str)
    return df

def generate(branches, electives_filepath):
    df=load_data(branches)
    if df.empty:
        print("🚫 No input data. Exiting."); return None,None,None,None,None
    tt, teacher_schedule = init_timetable()
    sched={}
    electives_data=load_and_group_electives(electives_filepath)
    all_branches=[b.upper() for b in branches]
    tt,grouped,teacher_schedule=schedule_electives(tt,electives_data,all_branches,sched, teacher_schedule)

    # Exclude electives from core scheduling
    elective_codes=set()
    for bag, sems in electives_data.items():
        for sem,subs in sems.items():
            for sub in subs:
                elective_codes.add(sub['code'])

    core_df = df[~df['Course Code'].isin(elective_codes)]

    # Merge elective rows back with grouped titles
    extra=[]
    for label,sem,subs in grouped:
        for b in all_branches:
            extra.append({
                'Course Code':label,
                'Course Title':f"{label} Elective",
                'Faculty': '', # Faculty for grouped electives is not directly available here
                'Semester':sem,
                'branch':b,
                'Strength': 0, # Electives don't have strength
            })
    if extra:
        e_df=pd.DataFrame(extra)
        df=pd.concat([core_df,e_df], ignore_index=True)
    else:
        df = core_df

    print("--- Scheduling Core Courses ---")
    sessions=[]
    for _, row in core_df.iterrows():
        code, title, fac_str, branch, sem, strength, L, T, P = (
            row['Course Code'], row['Course Title'], row['Faculty'], 
            row['branch'], row['Semester'], row['Strength'],
            int(row['L']), int(row['T']), int(row['P'])
        )

        faculties = [f.strip() for f in re.split(r'\s*&\s*|\s*and\s*|\s*/\s*', fac_str)]
        is_cse_core = branch.upper() == 'CSE'

        if is_cse_core:
            if len(faculties) == 1: # Single teacher, combined class
                combined_strength = strength * 2 # Assuming strength is per section
                sessions += [{'code':code,'title':title,'fac':faculties[0],'groups':[(branch, sem)],'type':'Lecture', 'strength': combined_strength}]*L
                sessions += [{'code':code,'title':title,'fac':faculties[0],'groups':[(branch, sem)],'type':'Tutorial', 'strength': combined_strength}]*T
                sessions += [{'code':code,'title':title,'fac':faculties[0],'groups':[(branch, sem)],'type':'Practical', 'strength': combined_strength}]*P
            else: # Multiple teachers, split classes
                strength_per_section = strength
                sessions += [{'code':code,'title':title,'fac':faculties[0],'groups':[('CSE A', sem)],'type':'Lecture', 'strength': strength_per_section}]*L
                sessions += [{'code':code,'title':title,'fac':faculties[1],'groups':[('CSE B', sem)],'type':'Lecture', 'strength': strength_per_section}]*L
                sessions += [{'code':code,'title':title,'fac':faculties[0],'groups':[('CSE A', sem)],'type':'Tutorial', 'strength': strength_per_section}]*T
                sessions += [{'code':code,'title':title,'fac':faculties[1],'groups':[('CSE B', sem)],'type':'Tutorial', 'strength': strength_per_section}]*T
                sessions += [{'code':code,'title':title,'fac':faculties[0],'groups':[('CSE A', sem)],'type':'Practical', 'strength': strength_per_section}]*P
                sessions += [{'code':code,'title':title,'fac':faculties[1],'groups':[('CSE B', sem)],'type':'Practical', 'strength': strength_per_section}]*P
        else: # Non-CSE subjects
            sessions += [{'code':code,'title':title,'fac':faculties[0],'groups':[(branch, sem)],'type':'Lecture', 'strength': strength}]*L
            sessions += [{'code':code,'title':title,'fac':faculties[0],'groups':[(branch, sem)],'type':'Tutorial', 'strength': strength}]*T
            sessions += [{'code':code,'title':title,'fac':faculties[0],'groups':[(branch, sem)],'type':'Practical', 'strength': strength}]*P

    max_attempts=5
    for attempt in range(max_attempts):
        if not sessions: break
        placed=0
        slots=[(d,sl) for d in WEEKDAYS for sl in TIME_SLOTS if sl not in BREAK_SLOTS]
        random.shuffle(slots)
        for day,slot in slots:
            used_grp=set()
            for e in tt[day][slot].values():
                if e and e[0]!="BREAK":
                    grp=e[4]
                    if isinstance(grp,list): used_grp.update(grp)
                    else: used_grp.add((e[4],e[5]))
            for sess in sessions[:]:
                is_prac = sess['type']=='Practical'
                
                if is_prac:
                    # Try to schedule a 2-hour block first
                    next_slot_index = TIME_SLOTS.index(slot) + 1
                    if next_slot_index < len(TIME_SLOTS) and TIME_SLOTS[next_slot_index] not in BREAK_SLOTS:
                        rooms = find_available_rooms(tt, day, slot, sess['strength'], is_prac, sess['groups'][0][1])
                        if rooms:
                            room = rooms[0]
                            if tt[day][TIME_SLOTS[next_slot_index]][room] is None:
                                entry=(sess['code'],sess['fac'],sess['title'],sess['type'],sess['groups'])
                                tt[day][slot][room]=entry
                                tt[day][TIME_SLOTS[next_slot_index]][room]=entry
                                faculties = [f.strip() for f in re.split(r'\s*&\s*|\s*and\s*|\s*/\s*', sess['fac'])]
                                for f in faculties:
                                    teacher_schedule[day][slot][f] = sess['code']
                                    teacher_schedule[day][TIME_SLOTS[next_slot_index]][f] = sess['code']
                                used_grp.update(sess['groups'])
                                sessions.remove(sess)
                                sched.setdefault(sess['code'], {'Lecture': 0, 'Tutorial': 0, 'Practical': 0})[sess['type']] += 1
                                placed+=1
                                continue

                    
                
                # Check faculty and group conflicts
                if has_conflict(tt, day, slot, sess['fac'], sess['groups'][0][0], sess['groups'][0][1], teacher_schedule):
                    continue
                if any(g in used_grp for g in sess['groups']):
                    continue

                # CSE Section Split Logic
                is_cse_core = any(b.upper() == 'CSE' for b, s in sess['groups']) and not is_prac
                if is_cse_core:
                    strength_per_section = sess['strength'] // 2
                    rooms = find_available_rooms(tt, day, slot, strength_per_section, is_prac, sess['groups'][0][1])
                    if len(rooms) < 2:
                        print(f"WARNING: Not enough rooms for CSE sections for {sess['code']} on {day} at {slot}.")
                        # Try to schedule as a combined class if split fails
                        rooms = find_available_rooms(tt, day, slot, sess['strength'], is_prac, sess['groups'][0][1])
                        if not rooms:
                            continue
                        room = rooms[0]
                        entry=(sess['code'],sess['fac'],sess['title'],sess['type'],sess['groups'])
                        tt[day][slot][room]=entry
                        faculties = [f.strip() for f in re.split(r'\s*&\s*|\s*and\s*|\s*/\s*', sess['fac'])]
                        for f in faculties:
                            teacher_schedule[day][slot][f] = sess['code']
                        used_grp.update(sess['groups'])
                        sessions.remove(sess)
                        sched.setdefault(sess['code'], {'Lecture': 0, 'Tutorial': 0, 'Practical': 0})[sess['type']] += 1
                        placed+=1
                        continue
                    
                    room_a, room_b = rooms[0], rooms[1]
                    
                    # Assign Section A
                    groups_a = [(b, s) if b.upper() != 'CSE' else ('CSE A', s) for b, s in sess['groups']]
                    entry_a = (sess['code'], sess['fac'], sess['title'], sess['type'], groups_a)
                    tt[day][slot][room_a] = entry_a

                    # Assign Section B
                    groups_b = [(b, s) if b.upper() != 'CSE' else ('CSE B', s) for b, s in sess['groups']]
                    entry_b = (sess['code'], sess['fac'], sess['title'], sess['type'], groups_b)
                    tt[day][slot][room_b] = entry_b
                    
                    faculties = [f.strip() for f in re.split(r'\s*&\s*|\s*and\s*|\s*/\s*', sess['fac'])]
                    for f in faculties:
                        teacher_schedule[day][slot][f] = sess['code']
                    used_grp.update(sess['groups'])
                    sessions.remove(sess)
                    sched.setdefault(sess['code'], {'Lecture': 0, 'Tutorial': 0, 'Practical': 0})[sess['type']] += 1
                    placed+=1
                    continue

                # Non-CSE core subjects and tutorials
                rooms = find_available_rooms(tt, day, slot, sess['strength'], is_prac, sess['groups'][0][1])
                if not rooms:
                    print(f"DEBUG: No room for {sess['type']} {sess['code']} on {day} at {slot}.")
                    continue
                
                room = rooms[0]
                entry=(sess['code'],sess['fac'],sess['title'],sess['type'],sess['groups'])
                tt[day][slot][room]=entry
                faculties = [f.strip() for f in re.split(r'\s*&\s*|\s*and\s*|\s*/\s*', sess['fac'])]
                for f in faculties:
                    teacher_schedule[day][slot][f] = sess['code']
                used_grp.update(sess['groups'])
                sessions.remove(sess)
                sched.setdefault(sess['code'], {'Lecture': 0, 'Tutorial': 0, 'Practical': 0})[sess['type']] += 1
                placed+=1
                continue
                
                # Check faculty and group conflicts
                if has_conflict(tt, day, slot, sess['fac'], sess['groups'][0][0], sess['groups'][0][1], teacher_schedule):
                    continue
                if any(g in used_grp for g in sess['groups']):
                    continue

                # CSE Section Split Logic
                is_cse_core = any(b.upper() == 'CSE' for b, s in sess['groups']) and not is_prac
                if is_cse_core:
                    strength_per_section = sess['strength'] // 2
                    rooms = find_available_rooms(tt, day, slot, strength_per_section, is_prac, sess['groups'][0][1])
                    if len(rooms) < 2:
                        print(f"WARNING: Not enough rooms for CSE sections for {sess['code']} on {day} at {slot}.")
                        continue
                    
                    room_a, room_b = rooms[0], rooms[1]
                    
                    # Assign Section A
                    groups_a = [(b, s) if b.upper() != 'CSE' else ('CSE A', s) for b, s in sess['groups']]
                    entry_a = (sess['code'], sess['fac'], sess['title'], sess['type'], groups_a)
                    tt[day][slot][room_a] = entry_a

                    # Assign Section B
                    groups_b = [(b, s) if b.upper() != 'CSE' else ('CSE B', s) for b, s in sess['groups']]
                    entry_b = (sess['code'], sess['fac'], sess['title'], sess['type'], groups_b)
                    tt[day][slot][room_b] = entry_b
                    
                    faculties = [f.strip() for f in re.split(r'\s*&\s*|\s*and\s*|\s*/\s*', sess['fac'])]
                    for f in faculties:
                        teacher_schedule[day][slot][f] = sess['code']
                    used_grp.update(sess['groups'])
                    sessions.remove(sess)
                    sched.setdefault(sess['code'], {'Lecture': 0, 'Tutorial': 0, 'Practical': 0})[sess['type']] += 1
                    placed+=1
                    continue

                # Historical conflict
                conflict=False
                for b,s in sess['groups']:
                    if count_subject_sessions_on_day(tt,day,sess['code'],b,s)>=2:
                        conflict=True; break
                    if sess['type']=='Lecture' and count_subject_sessions_on_day(tt,day,sess['code'],b,s,'Lecture')>0:
                        conflict=True; break
                    if sess['type']=='Practical' and count_subject_sessions_on_day(tt,day,sess['code'],b,s,'Lecture')>0:
                        conflict=True; break
                    if sess['type']=='Lecture' and count_subject_sessions_on_day(tt,day,sess['code'],b,s,'Practical')>0:
                        conflict=True; break
                if conflict: continue

                # CSE Section Split Logic
                is_cse_core = any(b.upper() == 'CSE' for b, s in sess['groups']) and not is_prac
                if is_cse_core:
                    strength_per_section = sess['strength'] // 2
                    rooms = find_available_rooms(tt, day, slot, strength_per_section, is_prac, sess['groups'][0][1])
                    if len(rooms) < 2:
                        # If not enough rooms for split sections, try to schedule as a combined class
                        rooms = find_available_rooms(tt, day, slot, sess['strength'], is_prac, sess['groups'][0][1])
                        if not rooms:
                            continue
                        room = rooms[0]
                        entry=(sess['code'],sess['fac'],sess['title'],sess['type'],sess['groups'])
                        tt[day][slot][room]=entry
                        faculties = [f.strip() for f in re.split(r'\s*&\s*|\s*and\s*|\s*/\s*', sess['fac'])]
                        for f in faculties:
                            teacher_schedule[day][slot][f] = sess['code']
                        used_grp.update(sess['groups'])
                        sessions.remove(sess)
                        sched.setdefault(sess['code'], {'Lecture': 0, 'Tutorial': 0, 'Practical': 0})[sess['type']] += 1
                        placed+=1
                    else:
                        room_a, room_b = rooms[0], rooms[1]
                        
                        # Assign Section A
                        groups_a = [(b, s) if b.upper() != 'CSE' else ('CSE A', s) for b, s in sess['groups']]
                        entry_a = (sess['code'], sess['fac'], sess['title'], sess['type'], groups_a)
                        tt[day][slot][room_a] = entry_a

                        # Assign Section B
                        groups_b = [(b, s) if b.upper() != 'CSE' else ('CSE B', s) for b, s in sess['groups']]
                        entry_b = (sess['code'], sess['fac'], sess['title'], sess['type'], groups_b)
                        tt[day][slot][room_b] = entry_b
                        
                        faculties = [f.strip() for f in re.split(r'\s*&\s*|\s*and\s*|\s*/\s*', sess['fac'])]
                        for f in faculties:
                            teacher_schedule[day][slot][f] = sess['code']
                        used_grp.update(sess['groups'])
                        sessions.remove(sess)
                        sched.setdefault(sess['code'], {'Lecture': 0, 'Tutorial': 0, 'Practical': 0})[sess['type']] += 1
                        placed+=1
                else:
                    rooms = find_available_rooms(tt, day, slot, sess['strength'], is_prac, sess['groups'][0][1])
                    if not rooms:
                        continue
                    
                    room = rooms[0]
                    entry=(sess['code'],sess['fac'],sess['title'],sess['type'],sess['groups'])
                    tt[day][slot][room]=entry
                    faculties = [f.strip() for f in re.split(r'\s*&\s*|\s*and\s*|\s*/\s*', sess['fac'])]
                    for f in faculties:
                        teacher_schedule[day][slot][f] = sess['code']
                    used_grp.update(sess['groups'])
                    sessions.remove(sess)
                    sched.setdefault(sess['code'], {'Lecture': 0, 'Tutorial': 0, 'Practical': 0})[sess['type']] += 1
                    placed+=1
        if placed==0 and attempt==max_attempts-1 and sessions:
            print(f"Warning: Couldn't place all sessions; {len(sessions)} remain. Details:")
            for sess in sessions:
                print(f"  - Code: {sess['code']}, Type: {sess['type']}, Strength: {sess['strength']}, Groups: {sess['groups']}")
    return tt,df,sched,electives_data,teacher_schedule

def verify_and_export(sched,df):
    recs=[]
    for c in ['L','T','P']:
        df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0)
    for _,r in df.iterrows():
        code,br,sem=r['Course Code'],r['branch'],str(r['Semester'])
        need={"Lecture":int(r['L']),"Tutorial":int(r['T']),"Practical":int(r['P'])}
        got={st:sched.get(code, {}).get(st, 0) for st in need}
        is_elec='F_' in str(r['Faculty'])
        ok=all(got[st]>=need[st] for st in need)
        recs.append({**{"Course Code":code,"Branch":br,"Semester":sem},
                     **{f"{st}Req":need[st] for st in need},
                     **{f"{st}Sch":got[st] for st in need},
                     "All Met":ok})
    pd.DataFrame(recs).to_csv("Timetables/LTPSC_Verification.csv",index=False)
    print("Saved -> Timetables/LTPSC Verification.csv")

def save_combined(tt):
    rows=[]
    for d in WEEKDAYS:
        for sl in TIME_SLOTS:
            for r in CLASSROOMS+LABS:
                e=tt[d][sl][r]
                if not e: continue
                if e[0]=="BREAK":
                    rows.append({"Day":d,"Time":sl,"Room":r,"Session Type":"Break","Info":e[1]})
                else:
                    c,f,t,st=e[:4]
                    grp=e[4]
                    if isinstance(grp,list):
                        for b,s in grp:
                            rows.append({"Day":d,"Time":sl,"Room":r,
                                         "Course Code":c,"Course Title":t,
                                         "Faculty":f,"Session Type":st,
                                         "Branch":b,"Semester":s})
                    else:
                        rows.append({"Day":d,"Time":sl,"Room":r,
                                     "Course Code":c,"Course Title":t,
                                     "Faculty":f,"Session Type":st,
                                     "Branch":grp[0],"Semester":grp[1]})
    df=pd.DataFrame(rows)
    df.to_csv("Timetables/Combined_Timetable.csv",index=False)
    print("Saved -> Timetables/Combined Timetable.csv")
    return df

def split_by_branch(df):
    real=df[df['Session Type']!="Break"]
    for b in real['Branch'].unique():
        sub=real[real['Branch']==b]
        sub.to_csv(f"Timetables/By Branch/{b}.csv",index=False)
        pivot=pd.DataFrame(index=WEEKDAYS,columns=TIME_SLOTS).fillna("")
        for sl in TIME_SLOTS:
            pivot[sl]=[
                " | ".join(f"{r['Course Title']}-{r['Session Type']}-{r['Room']}-{r['Semester']}"
                           for _,r in sub[(sub['Day']==d)&(sub['Time']==sl)].iterrows())
                for d in WEEKDAYS
            ]
        pivot.to_csv(f"Timetables/By Branch/{b} Formatted.csv")
        print(f"Branch Formatted -> {b} Formatted.csv")

def split_by_semester(df):
    real=df[df['Session Type']!="Break"]
    for sem in real['Semester'].unique():
        clean=sem.replace('.0','')
        sub=real[real['Semester']==sem]
        sub.to_csv(f"Timetables/By Semester/Semester {clean}.csv",index=False)
        pivot=pd.DataFrame(index=WEEKDAYS,columns=TIME_SLOTS).fillna("")
        for sl in TIME_SLOTS:
            pivot[sl]=[
                " | ".join(f"{r['Course Title']}-{r['Session Type']}-{r['Room']}"
                           for _,r in sub[(sub['Day']==d)&(sub['Time']==sl)].iterrows())
                for d in WEEKDAYS
            ]
        pivot.to_csv(f"Timetables/By Semester/Semester {clean} Formatted.csv")
        print(f"Semester Formatted -> Semester {clean} Formatted.csv")

def split_by_branch_sem(df, electives_data, tt):
    real = df[df['Session Type'] != "Break"]
    all_branches = real['Branch'].unique()
    
    # Add CSE A and CSE B if CSE is present
    if 'CSE' in all_branches:
        all_branches = list(all_branches)
        if 'CSE A' not in all_branches: all_branches.append('CSE A')
        if 'CSE B' not in all_branches: all_branches.append('CSE B')

    for b in all_branches:
        for sem in real['Semester'].unique():
            clean = sem.replace('.0', '')
            
            # Filter for the specific branch/section and semester
            if b in ['CSE A', 'CSE B']:
                # Include general CSE courses and electives in section timetables
                sub = real[((real['Branch'] == b) | (real['Branch'] == 'CSE')) & (real['Semester'] == sem)]
            else:
                sub = real[(real['Branch'] == b) & (real['Semester'] == sem)]

            if sub.empty:
                continue
            
            sub.to_csv(f"Timetables/By Branch Semester/{b} Semester {clean}.csv", index=False)
            
            pivot = pd.DataFrame(index=WEEKDAYS, columns=TIME_SLOTS).fillna("")
            for d in WEEKDAYS:
                for sl in TIME_SLOTS:
                    entries_for_slot = []
                    for r in CLASSROOMS + LABS:
                        entry = tt[d][sl][r]
                        if entry and entry[0] != "BREAK":
                            entry_groups = entry[4]
                            if isinstance(entry_groups, list):
                                if (b, sem) in entry_groups or (b.replace(' ', ''), sem) in entry_groups:
                                    entries_for_slot.append(f"{entry[2]}-{entry[3]}-{r}")

                    pivot.loc[d, sl] = " | ".join(sorted(list(set(entries_for_slot))))

            pivot.to_csv(f"Timetables/By Branch Semester/{b} Semester {clean} Formatted.csv")
            print(f"B+S Formatted -> {b} Semester {clean} Formatted.csv")

def main():
    try:
        create_dirs()
        branches=["dsai","ece","cse"]
        electives_filepath = os.path.join(os.getcwd(), "Electives.csv")
        tt,df,sched,electives_data,teacher_schedule=generate(branches,electives_filepath)
        if tt is None: return
        verify_and_export(sched,df)
        combined=save_combined(tt)
        split_by_branch(combined)
        split_by_semester(combined)
        split_by_branch_sem(combined, electives_data, tt)
        save_faculty_timetables(tt)
        save_room_timetables(combined)
    except Exception as e:
        print("Error:",e)
        traceback.print_exc()

if __name__=="__main__":
    main()
