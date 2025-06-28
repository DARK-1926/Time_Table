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

CLASSROOMS = ["C002","C003","C004","C101","C102","C104",
              "C202","C203","C204","C206","C302","C303","C304","C305"]
LABS       = ["L105","L106","L107","L206","L207","L208"]

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
    all_faculties = set()
    for d in WEEKDAYS:
        for sl in TIME_SLOTS:
            for r in CLASSROOMS + LABS:
                entry = tt[d][sl][r]
                if entry and entry[0] != "BREAK":
                    fac_str = entry[1]
                    for f in str(fac_str).split(' / '):
                        all_faculties.add(f.strip())

    for fac in sorted(list(all_faculties)):
        pivot = pd.DataFrame(index=WEEKDAYS, columns=TIME_SLOTS).fillna("")
        for d in WEEKDAYS:
            for sl in TIME_SLOTS:
                entries_for_slot = []
                for r in CLASSROOMS + LABS:
                    entry = tt[d][sl][r]
                    if entry and entry[0] != "BREAK":
                        fac_str = entry[1]
                        if fac in [f.strip() for f in fac_str.split(' / ')]:
                            # Determine branch and semester for display
                            entry_groups = entry[4]
                            branch_sem_str = ""
                            if isinstance(entry_groups, list):
                                branch_sem_str = ", ".join([f"{b}-{s}" for b, s in entry_groups])
                            else:
                                branch_sem_str = f"{entry_groups[0]}-{entry_groups[1]}"
                            
                            entries_for_slot.append(f"{entry[2]}-{entry[3]}-{r}-{branch_sem_str}")
                if entries_for_slot:
                    pivot.loc[d, sl] = " | ".join(entries_for_slot)
        
        # Create the directory if it doesn't exist
        os.makedirs("Timetables/By Faculty", exist_ok=True)
        pivot.to_csv(f"Timetables/By Faculty/{fac}.csv")
        print(f"Faculty Timetable -> {fac}.csv")

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
    return tt

def has_conflict(tt, day, slot, fac, branch, sem):
    for r in CLASSROOMS + LABS:
        e = tt[day][slot][r]
        if isinstance(e,tuple) and e[0]!="BREAK":
            # Check for faculty conflict
            existing_facs = [f.strip() for f in e[1].split(' / ')]
            new_facs = [f.strip() for f in fac.split(' / ')]
            if any(f in existing_facs for f in new_facs):
                return True
            
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
    for i in range(0,len(df.columns),2):
        bcol,ccol=df.columns[i],df.columns[i+1] if i+1<len(df.columns) else (None,None)
        if "Basket" not in bcol or ccol is None: continue
        basket=bcol.strip()
        electives[basket]={}
        temp=df[[bcol,ccol]].dropna().copy()
        temp.columns=['title','code']
        for _,r in temp.iterrows():
            sem=get_semester_from_code(r['code'])
            if sem:
                electives[basket].setdefault(sem,[]).append({
                    'title':r['title'].strip(),
                    'code':str(r['code']).strip()
                })
    return electives

def schedule_electives(tt, electives_data, all_branches, sched):
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
                if any(has_conflict(tt, day, slot, "any_faculty", b, sem) for b in all_branches):
                    continue
                rooms_free = [r for r in CLASSROOMS if tt[day][slot][r] is None]
                if len(rooms_free) >= len(subs):
                    possible_slots.append((day, slot, rooms_free))

            if len(possible_slots) >= needed_sessions:
                baskets_to_schedule.append({'label': f"{b1}+{b2}", 'sem': sem, 'subs': subs})
            else:
                print(f"INFO: Could not find enough slots for combined basket {b1}+{b2} for Sem {sem}. Splitting it.")
                if subs1:
                    baskets_to_schedule.append({'label': b1, 'sem': sem, 'subs': subs1})
                if subs2:
                    baskets_to_schedule.append({'label': b2, 'sem': sem, 'subs': subs2})

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
                    if any(has_conflict(tt, day, slot, "any_faculty", b, sem) for b in all_branches):
                        continue
                    
                    rooms_free = [r for r in CLASSROOMS if tt[day][slot][r] is None]
                    if len(rooms_free) < len(subs):
                        continue
                    
                    chosen = random.sample(rooms_free, len(subs))
                    for i, sub in enumerate(subs):
                        if placed[sub['code']][stype] >= req:
                            continue
                        
                        room=chosen[i]
                        # Retain original faculty name if available, otherwise use a placeholder
                        fac = sub.get('Faculty', "") 
                        entry=(sub['code'], fac, sub['title'],stype,
                               [(b,sem) for b in all_branches])
                        tt[day][slot][room] = entry
                        sched[f"{label}-{stype}"] = sched.get(f"{label}-{stype}", 0) + 1
                        placed[sub['code']][stype] += 1
                    found = True
                    break
                
                if not found:
                    print(f"WARNING: Could not schedule all {stype} for {label} Sem {sem}")

    print("--- Finished Paired Electives ---")
    return tt, grouped_for_return


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
    df.drop_duplicates(subset=["Course Code","branch"],inplace=True)
    df['Semester']=df['Semester'].astype(str)
    return df

def generate(branches, electives_filepath):
    df=load_data(branches)
    if df.empty:
        print("🚫 No input data. Exiting."); return None,None,None
    tt,sched=init_timetable(),{}
    electives_data=load_and_group_electives(electives_filepath)
    all_branches=[b.upper() for b in branches]
    tt,grouped=schedule_electives(tt,electives_data,all_branches,sched)

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
                'branch':b,'L':2,'T':1,'P':0,'S':0,'C':3
            })
    if extra:
        e_df=pd.DataFrame(extra)
        print("DEBUG: Elective DataFrame before concat:")
        print(e_df.head())
        df=pd.concat([core_df,e_df],
                     ignore_index=True)
    else:
        df = core_df

    print("--- Scheduling Core Courses ---")
    sessions=[]
    for (code,fac_str),grp in df.groupby(['Course Code','Faculty']):
        title=grp['Course Title'].iloc[0]
        groups=list(zip(grp['branch'],grp['Semester']))
        L,T,P=int(grp['L'].max()),int(grp['T'].max()),int(grp['P'].max())
        
        faculties = [f.strip() for f in fac_str.split(' / ')]
        
        for fac in faculties:
            sessions += [{'code':code,'title':title,'fac':fac,'groups':groups,'type':'Lecture', 'needed_L': L, 'needed_T': T, 'needed_P': P}]*L
            sessions += [{'code':code,'title':title,'fac':fac,'groups':groups,'type':'Tutorial', 'needed_L': L, 'needed_T': T, 'needed_P': P}]*T
            sessions += [{'code':code,'title':title,'fac':fac,'groups':groups,'type':'Practical', 'needed_L': L, 'needed_T': T, 'needed_P': P}]*P

    max_attempts=5
    for attempt in range(max_attempts):
        if not sessions: break
        placed=0
        slots=[(d,sl) for d in WEEKDAYS for sl in TIME_SLOTS if sl not in BREAK_SLOTS]
        random.shuffle(slots)
        for day,slot in slots:
            avail_rooms=[r for r in CLASSROOMS if tt[day][slot][r] is None]
            avail_labs =[r for r in LABS       if tt[day][slot][r] is None]
            used_fac={e[1] for e in tt[day][slot].values() if e and e[0]!="BREAK"}
            used_grp=set()
            for e in tt[day][slot].values():
                if e and e[0]!="BREAK":
                    grp=e[4]
                    if isinstance(grp,list): used_grp.update(grp)
                    else: used_grp.add((e[4],e[5]))
            for sess in sessions[:]:
                print(f"DEBUG: Considering session: {sess['code']}-{sess['type']} by {sess['fac']} for {sess['groups']}")
                print(f"DEBUG: Needed: L={sess['needed_L']}, T={sess['needed_T']}, P={sess['needed_P']}")
                is_prac = sess['type']=='Practical'
                
                # Check room availability
                if (is_prac and not avail_labs) or (not is_prac and not avail_rooms):
                    print(f"DEBUG: Skipping {sess['code']}-{sess['type']}: No available {'lab' if is_prac else 'room'}.")
                    continue
                
                # Check faculty and group conflicts
                if sess['fac'] in used_fac:
                    print(f"DEBUG: Skipping {sess['code']}-{sess['type']}: Faculty {sess['fac']} already busy.")
                    continue
                if any(g in used_grp for g in sess['groups']):
                    print(f"DEBUG: Skipping {sess['code']}-{sess['type']}: Group conflict.")
                    continue

                # NEW LTPSC CHECK
                current_scheduled_count = sched.get(sess['code'], {}).get(sess['type'], 0)
                needed_count = sess[f'needed_{sess["type"][0]}'] # e.g., 'needed_L' for 'Lecture'
                print(f"DEBUG: Current scheduled count for {sess['code']}-{sess['type']}: {current_scheduled_count}/{needed_count}")
                if current_scheduled_count >= needed_count:
                    print(f"DEBUG: Skipping {sess['code']}-{sess['type']}: LTPSC requirement already met.")
                    continue # Already scheduled enough sessions of this type for this subject

                # Historical conflict
                conflict=False
                for b,s in sess['groups']:
                    if count_subject_sessions_on_day(tt,day,sess['code'],b,s)>=2:
                        print(f"DEBUG: Skipping {sess['code']}-{sess['type']}: Historical conflict (more than 2 sessions on day for branch-sem). Branch: {b}, Sem: {s}")
                        conflict=True; break
                    if sess['type']=='Lecture' and count_subject_sessions_on_day(tt,day,sess['code'],b,s,'Lecture')>0:
                        print(f"DEBUG: Skipping {sess['code']}-{sess['type']}: Historical conflict (more than 1 lecture on day for branch-sem). Branch: {b}, Sem: {s}")
                        conflict=True; break
                    if sess['type']=='Practical' and count_subject_sessions_on_day(tt,day,sess['code'],b,s,'Lecture')>0:
                        print(f"DEBUG: Skipping {sess['code']}-{sess['type']}: Historical conflict (practical after lecture on day for branch-sem). Branch: {b}, Sem: {s}")
                        conflict=True; break
                    if sess['type']=='Lecture' and count_subject_sessions_on_day(tt,day,sess['code'],b,s,'Practical')>0:
                        print(f"DEBUG: Skipping {sess['code']}-{sess['type']}: Historical conflict (lecture after practical on day for branch-sem). Branch: {b}, Sem: {s}")
                        conflict=True; break
                if conflict: continue

                room = avail_labs.pop(0) if is_prac else avail_rooms.pop(0)
                entry=(sess['code'],sess['fac'],sess['title'],sess['type'],sess['groups'])
                tt[day][slot][room]=entry
                used_fac.add(sess['fac'])
                used_grp.update(sess['groups'])
                sessions.remove(sess)
                # Update sched correctly
                sched.setdefault(sess['code'], {'Lecture': 0, 'Tutorial': 0, 'Practical': 0})[sess['type']] += 1
                placed+=1
                print(f"DEBUG: Placed {sess['code']}-{sess['type']} at {day} {slot} in {room}.")
        if placed==0 and attempt==max_attempts-1 and sessions:
            print(f"Warning: Couldn't place all sessions; {len(sessions)} remain.")
    return tt,df,sched,electives_data

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
    for b in real['Branch'].unique():
        for sem in real['Semester'].unique():
            clean = sem.replace('.0', '')
            sub = real[(real['Branch'] == b) & (real['Semester'] == sem)]
            if sub.empty:
                continue
            sub.to_csv(f"Timetables/By Branch Semester/{b} Semester {clean}.csv", index=False)
            
            pivot = pd.DataFrame(index=WEEKDAYS, columns=TIME_SLOTS).fillna("")
            for d in WEEKDAYS: # Iterate through each day
                for sl in TIME_SLOTS: # Iterate through each time slot
                    entries_for_slot = []
                    for r in CLASSROOMS + LABS:
                        entry = tt[d][sl][r]
                        if entry and entry[0] != "BREAK":
                            entry_groups = entry[4]
                            if isinstance(entry_groups, list):
                                if (b, sem) in entry_groups:
                                    entries_for_slot.append((entry[0], entry[1], entry[2], entry[3], r))
                            else:
                                if (entry_groups[0], entry_groups[1]) == (b, sem):
                                    entries_for_slot.append((entry[0], entry[1], entry[2], entry[3], r))

                    # Group electives by basket for display
                    grouped_electives = {}
                    non_electives_strs = []

                    for entry_code, entry_fac, entry_title, entry_type, entry_room in entries_for_slot:
                        if "Elective" in entry_title:
                            basket_name = entry_code  # Use Course Code as basket name
                            if basket_name not in grouped_electives:
                                grouped_electives[basket_name] = {'type': entry_type, 'rooms': []}
                            grouped_electives[basket_name]['rooms'].append(entry_room)
                        else:
                            non_electives_strs.append(f"{entry_title}-{entry_type}-{entry_room}")

                    # Format grouped electives
                    elective_strs = []
                    for basket_name, data in grouped_electives.items():
                        # Get subject titles for the basket to sort rooms correctly
                        sem_electives = electives_data.get(basket_name, {}).get(sem, [])
                        if not sem_electives and '+' in basket_name: # Handle combined baskets like Basket-1+Basket-2
                            b1, b2 = basket_name.split('+')
                            sem_electives.extend(electives_data.get(b1, {}).get(sem, []))
                            sem_electives.extend(electives_data.get(b2, {}).get(sem, []))

                        # Create a mapping from subject code to room
                        # This needs to be based on the original elective subjects, not the combined entries
                        # We need to find the room for each individual subject within the basket
                        subject_room_map = {}
                        for sub_entry in entries_for_slot:
                            if sub_entry[0] in [s['code'] for s in sem_electives]:
                                subject_room_map[sub_entry[0]] = sub_entry[4]

                        sorted_rooms = [subject_room_map.get(sub['code']) for sub in sem_electives if sub['code'] in subject_room_map]
                        
                        elective_strs.append(f"{basket_name} Elective-{data['type']}-{' | '.join(sorted_rooms)}")

                    # Combine all entries for the cell
                    all_entries_strs = non_electives_strs + elective_strs
                    pivot.loc[d, sl] = " | ".join(all_entries_strs)

            pivot.to_csv(f"Timetables/By Branch Semester/{b} Semester {clean} Formatted.csv")
            print(f"B+S Formatted -> {b} Semester {clean} Formatted.csv")

def main():
    try:
        create_dirs()
        branches=["dsai","ece","cse"]
        electives_filepath = os.path.join(os.getcwd(), "Electives.csv")
        tt,df,sched,electives_data=generate(branches,electives_filepath)
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
