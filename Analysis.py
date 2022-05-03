#!/usr/bin/env python
# coding: utf-8

# # Step 0: Preprocessing

# ## Imports and definitions

# In[ ]:


import math
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pm4py
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net.variants import pydotplus_vis
from pm4py.visualization.heuristics_net import visualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns


# In[ ]:


subfolder = "data" + os.sep
no_students = pd.read_csv(subfolder + "no_students.csv", header=None)[0].tolist()
select_courses = [5]
select_sections_per_course = {5: [1,2,3,4]}


# ## Load scrolling data from pickle file

# In[ ]:


df_scroll = pd.read_csv(subfolder + "scroll.csv", sep=";", parse_dates=["timecreated", "value.utc"])
df_scroll


# In[ ]:


df_scroll.dtypes


# In[ ]:


df_scroll.columns


# ## Remove unknown pages and old versions of scrolling info
# ## Select courses and students

# In[ ]:


user_acceptances = pd.read_csv(subfolder + "user_acceptances.csv", sep=";")


# In[ ]:


df_scroll = df_scroll[(~df_scroll["value.pageid"].isna()) & (~df_scroll["value.utc"].isna())]
df_scroll = df_scroll[df_scroll["courseid"].isin(select_courses)]
df_scroll = df_scroll[(~df_scroll["userid"].isin(no_students)) & (df_scroll["userid"].isin(user_acceptances[user_acceptances["Einwilligungserklärung zur Forschungsbeteiligung"] == "Akzeptiert"]["userid"]))]
df_scroll


# ## Derive reading sessions from scrolling data

# In[ ]:


df_scroll = df_scroll.sort_values("value.utc")


# In[ ]:


lim = (1000*60*10) #10mins
lim


# In[ ]:


diffprev = df_scroll.groupby(["userid", "value.pageid"], sort=False)[["value.utc","value.relativeTime"]].apply(func=lambda x: x.diff())

diffprev


# In[ ]:


diffnext = df_scroll.groupby(["userid", "value.pageid"], sort=False)[["value.utc","value.relativeTime"]].apply(func=lambda x: x.diff(-1))
diffnext


# ### Define session borders

# In[ ]:


df_scroll[["diffprev_utc","diffprev_relativeTime"]] = diffprev
df_scroll[["diffnext_utc","diffnext_relativeTime"]] = diffnext
df_scroll["sessionstart"] = (df_scroll["diffprev_relativeTime"].isna()) | (df_scroll["diffprev_relativeTime"] < -1*lim) | (df_scroll["diffprev_utc"] > "10 minutes")
df_scroll["sessionend"] = (df_scroll["diffnext_relativeTime"].isna()) | (df_scroll["diffnext_relativeTime"] > lim) | (df_scroll["diffnext_utc"] < "-10 minutes")
df_scroll["sessionbreakstart"] = (df_scroll["diffnext_relativeTime"] < -1*0.5*lim) & (~df_scroll["sessionend"])
df_scroll["sessionbreakend"] = (df_scroll["diffprev_relativeTime"] > 0.5*lim) & (~df_scroll["sessionstart"]) 

#remove single scroll events
df_scroll = df_scroll[~(df_scroll["sessionstart"] & df_scroll["sessionend"])]

#separate events where start and breakstart are identical
dd = df_scroll[(df_scroll["sessionstart"] & df_scroll["sessionbreakstart"])]
df_scroll.loc[(df_scroll["sessionstart"]) & (df_scroll["sessionbreakstart"]), "sessionbreakstart"] = False
dd["sessionstart"] = False
dd["value.utc"] = dd["value.utc"] + pd.offsets.Second(1)
df_scroll = pd.concat([df_scroll, dd])

#separate events where end and breakend are identical
dd = df_scroll[(df_scroll["sessionend"] & df_scroll["sessionbreakend"])]
df_scroll.loc[(df_scroll["sessionend"]) & (df_scroll["sessionbreakend"]), "sessionbreakend"] = False
dd["sessionend"] = False
dd["value.utc"] = dd["value.utc"] - pd.offsets.Second(1)
df_scroll = pd.concat([df_scroll, dd])

#separate events where breakstart and breakend are identical
dd = df_scroll[(df_scroll["sessionbreakstart"] & df_scroll["sessionbreakend"])]
df_scroll.loc[(df_scroll["sessionbreakstart"]) & (df_scroll["sessionbreakend"]), "sessionbreakend"] = False
dd["sessionbreakstart"] = False
dd["value.utc"] = dd["value.utc"] - pd.offsets.Second(1)
df_scroll = pd.concat([df_scroll, dd])

df_scroll = df_scroll.sort_values("value.utc")
del diffnext, diffprev, dd, lim


# In[ ]:


mpl.rcParams['figure.figsize'] = [20, 2]

d = df_scroll[(df_scroll.userid == 1116) & (df_scroll.courseid == 5) & (df_scroll["value.pageid"] == 12) & (df_scroll["value.utc"] > "10.09.2020 15:00") & (df_scroll["value.utc"] < "10.10.2020 05:00")]
fig = d.plot.scatter("value.utc", "value.scrollYDistance", xlabel="Time created", ylabel="Vertical scroll position in px")
fig.invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")) 
plt.gca().vlines(d.loc[d["sessionstart"]==1, "value.utc"], 0, 10000, colors="green", label="reading_start") 
plt.gca().vlines(d.loc[d["sessionbreakstart"]==1, "value.utc"], 0, 10000, colors="red", linestyles="dashed", label="reading_pause")
plt.gca().vlines(d.loc[d["sessionbreakend"]==1, "value.utc"], 0, 10000, colors="green", linestyles="dashed", label="reading_continue")
plt.gca().vlines(d.loc[d["sessionend"]==1, "value.utc"], 0, 10000, colors="red", label="reading_end")
fig.legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig
plt.rc('font', size=15)
plt.rc("axes", titlesize=15)
plt.rc("legend", fontsize=12)
plt.savefig("reading_session.png", dpi=300, bbox_inches='tight')
del d


# ### Filter scroll logs between start and end

# In[ ]:


df_session = df_scroll[(df_scroll["sessionstart"] == True) | (df_scroll["sessionend"] == True) | (df_scroll["sessionbreakstart"] == True) | (df_scroll["sessionbreakend"] == True)]
df_session


# In[ ]:


df_session["state"] = np.select([df_session["sessionstart"], df_session["sessionend"], df_session["sessionbreakstart"], df_session["sessionbreakend"]], ["sessionstart", "sessionend", "sessionbreakstart", "sessionbreakend"])
df_session["state"]


# In[ ]:


df_session["sessionstart_or_breakend"] = df_session["sessionstart"] + df_session["sessionbreakend"] 
df_session["sessionid"] = df_session.groupby(["userid", "value.pageid"])["sessionstart_or_breakend"].cumsum()


# In[ ]:


df_session = df_session[["userid", "value.pageid", "sessionid", "state", "value.utc", "sessionstart", "sessionbreakstart", "sessionbreakend", "sessionend"]]


# In[ ]:


df_session = df_session.pivot(index=["userid", "value.pageid", "sessionid"], columns="state", values="value.utc")[["sessionstart", "sessionbreakstart", "sessionbreakend", "sessionend"]].reset_index()
df_session


# ### Filter pure breaks

# In[ ]:


df_session = df_session[~((~df_session["sessionbreakstart"].isna()) & (~df_session["sessionbreakend"].isna()))]
df_session


# ## Load and preprocess quiz attempts

# In[ ]:


df_attempts = pd.read_csv(subfolder + "m_quiz_attempts.csv", sep=";")
df_attempts = df_attempts[(~df_attempts["userid"].isin(no_students)) & (df_attempts["userid"].isin(user_acceptances[user_acceptances["Einwilligungserklärung zur Forschungsbeteiligung"] == "Akzeptiert"]["userid"]))] 
df_attempts


# In[ ]:


df_attempts["timestart"] = pd.to_datetime(df_attempts["timestart"], unit="s")
df_attempts["timefinish"] = pd.to_datetime(df_attempts["timefinish"], unit="s")
df_attempts["timemodified"] = pd.to_datetime(df_attempts["timemodified"], unit="s")


# In[ ]:


df_quiz = pd.read_csv(subfolder + "m_quiz.csv", sep=";")
df_quiz = df_quiz[(df_quiz.course.isin(select_courses))]
df_quiz


# ## Merge quiz and attempts and define quiz success or fail

# In[ ]:


df_attempts = df_attempts.merge(df_quiz, left_on="quiz", right_on="id", suffixes=["_attempt", "_quiz"])
df_attempts["result"] = "fail"
df_attempts.loc[(df_attempts["sumgrades_attempt"] / df_attempts["sumgrades_quiz"]) > 0.8, "result"] = "success"


# ## Assignments and Grades

# In[ ]:


df_assignments = pd.read_csv(subfolder + "m_assign.csv", sep=";")
df_assignments = df_assignments[df_assignments["course"].isin(select_courses)]


# In[ ]:


df_assignment_grades = pd.read_csv(subfolder + "m_assign_grades.csv", sep=";")
df_assignment_grades = df_assignment_grades[~df_assignment_grades["userid"].isin(no_students)]
df_assignment_grades = df_assignment_grades[df_assignment_grades["grade"] >= 0]
df_assignment_grades = df_assignment_grades.merge(df_assignments, left_on="assignment", right_on="id", suffixes=["_grades", "_assign"])
df_assignment_grades["result"] = df_assignment_grades["grade_grades"] / df_assignment_grades["grade_assign"]
df_assignment_grades["duedate"] = pd.to_datetime(df_assignment_grades["duedate"], unit="s")
df_assignment_grades["timecreated"] = pd.to_datetime(df_assignment_grades["timecreated"], unit="s")
df_assignment_grades["allowsubmissionsfromdate"] = pd.to_datetime(df_assignment_grades["allowsubmissionsfromdate"], unit="s")
df_assignment_grades


# In[ ]:


ax = df_assignment_grades.groupby(df_assignment_grades["timecreated"].dt.date)["timecreated"].count().plot(label="Assignments")

df_attempts[(df_attempts["state"] == "finished")].resample("D", on="timefinish")["timefinish"].count().plot(ax=ax, label="Quiz")

ax.vlines(df_assignment_grades["duedate"].dt.date, 0, 200, colors="black", label="Due Date")
ax.legend()


# ## Modules and sections

# In[ ]:


# Module IDS:
# 1 assign
# 15 page
# 16 quiz / assessment


# In[ ]:


df_course_modules = pd.read_csv(subfolder + "m_course_modules.csv", sep=";")
df_course_modules = pd.merge(df_course_modules, pd.read_csv(subfolder + "m_course_sections.csv", sep=";"), left_on="section", right_on="id", suffixes=["_modules", "_sections"])
df = pd.DataFrame()
for key, val in select_sections_per_course.items():
    df = df.append(df_course_modules[(df_course_modules["course_modules"] == key) & (df_course_modules["section_sections"].isin(val))])
df_course_modules = df.copy()
del df
df_course_modules


# ## Merge section to reading

# In[ ]:


df_session = pd.merge(df_session, df_course_modules.loc[df_course_modules["module"] == 15, ["instance", "name", "section_sections"]], left_on="value.pageid", right_on="instance", suffixes=["_session", "_sections"]).drop("instance", axis=1).rename({"section_sections":"section"}, axis=1)
df_session


# ## Merge section to assessment

# In[ ]:


df_attempts = pd.merge(df_attempts, df_course_modules.loc[df_course_modules["module"] == 16, ["instance", "name", "section_sections"]], left_on="quiz", right_on="instance", suffixes=["_attempts", "_sections"]).drop("instance", axis=1).rename({"section_sections":"section"}, axis=1)
df_attempts


# ## Merge section to assignment

# In[ ]:


df_assignment_grades = pd.merge(df_assignment_grades, df_course_modules.loc[df_course_modules["module"] == 1, ["instance", "name", "section_sections"]], left_on="assignment", right_on="instance", suffixes=["_grades", "_sections"]).drop("instance", axis=1).rename({"section_sections":"section"}, axis=1)
df_assignment_grades


# ## Action types coding

# ### Reading

# In[ ]:


# Codes: 
# Reading - reading_start, reading_short, reading_medium, reading_long, reading_pause, reading_continue, reading_end


# In[ ]:


_, bins = pd.qcut(df_session["sessionbreakstart"] - df_session["sessionstart"], 3, retbins=True)
bins


# In[ ]:


_, bins = pd.qcut(df_session["sessionend"] - df_session["sessionstart"], 3, retbins=True)
bins


# In[ ]:


_, bins = pd.qcut(df_session["sessionend"] - df_session["sessionbreakend"], 3, retbins=True)
bins


# In[ ]:


df_session.loc[(df_session["sessionbreakstart"] - df_session["sessionstart"]) > "5 minutes", "action"] = "reading_long"
df_session.loc[(df_session["sessionbreakstart"] - df_session["sessionstart"]) < "5 minutes", "action"] = "reading_medium"
df_session.loc[(df_session["sessionbreakstart"] - df_session["sessionstart"]) < "1 minute", "action"] = "reading_short"
df_session.loc[(~df_session["sessionbreakstart"].isna()) & (~df_session["sessionstart"].isna()), "timestamp"] = df_session.loc[(~df_session["sessionbreakstart"].isna()) & (~df_session["sessionstart"].isna()), "sessionstart"] 

df_session.loc[(df_session["sessionend"] - df_session["sessionstart"]) > "5 minutes", "action"] = "reading_long"
df_session.loc[(df_session["sessionend"] - df_session["sessionstart"]) < "5 minutes", "action"] = "reading_medium"
df_session.loc[(df_session["sessionend"] - df_session["sessionstart"]) < "1 minute", "action"] = "reading_short"
df_session.loc[(~df_session["sessionend"].isna()) & (~df_session["sessionstart"].isna()), "timestamp"] = df_session.loc[(~df_session["sessionend"].isna()) & (~df_session["sessionstart"].isna()), "sessionstart"] 


df_session.loc[(df_session["sessionend"] - df_session["sessionbreakend"]) > "5 minutes", "action"] = "reading_long"
df_session.loc[(df_session["sessionend"] - df_session["sessionbreakend"]) < "5 minutes", "action"] = "reading_medium"
df_session.loc[(df_session["sessionend"] - df_session["sessionbreakend"]) < "1 minute", "action"] = "reading_short"
df_session.loc[(~df_session["sessionend"].isna()) & (~df_session["sessionbreakend"].isna()), "timestamp"] = df_session.loc[(~df_session["sessionend"].isna()) & (~df_session["sessionbreakend"].isna()), "sessionbreakend"] 

df_session.loc[df_session["action"].isin(["reading_short", "reading_medium", "reading_long"]), "timestamp"] += pd.DateOffset(milliseconds=1)

d1 = df_session[~df_session["sessionstart"].isna()]
d1["action"] = "reading_start"
d1["timestamp"] = d1["sessionstart"]

d2 = df_session[~df_session["sessionbreakstart"].isna()]
d2["action"] = "reading_pause"
d2["timestamp"] = d2["sessionbreakstart"]

d3 = df_session[~df_session["sessionbreakend"].isna()]
d3["action"] = "reading_continue"
d3["timestamp"] = d3["sessionbreakend"]

d4 = df_session[~df_session["sessionend"].isna()]
d4["action"] = "reading_end"
d4["timestamp"] = d4["sessionend"]
df_session = pd.concat([df_session, d1, d2, d3, d4])

df_session = df_session.sort_values("timestamp")
del d1, d2, d3, d4


# ### Quiz

# In[ ]:


# Codes:
# Quiz - quiz_start, quiz_repeat_same, quiz_repeat_other, quiz_success, quiz_fail


# In[ ]:


df_attempts["timestamp"] = df_attempts["timestart"]

df_attempts.loc[df_attempts["attempt"] == 1, "action"] = "quiz_start"
df_attempts.loc[df_attempts["attempt"] > 1, "action"] = "quiz_repeat_other"

d = df_attempts[df_attempts["state"] == "finished"]
d.loc[d["result"] == "success", "action"] = "quiz_success"
d.loc[d["result"] == "fail", "action"] = "quiz_fail"
d["timestamp"] = d["timefinish"]

df_attempts = pd.concat([df_attempts, d])
df_attempts = df_attempts.sort_values("timestamp")
del d


# ## Merge quizzes and attempts

# In[ ]:


df_action = pd.concat([df_session[["userid", "value.pageid","timestamp", "action"]], df_attempts[["userid", "quiz","timestamp", "action", "sumgrades_attempt", "sumgrades_quiz"]]])
df_action["timestamp"] = df_action["timestamp"].astype("datetime64")
df_action = df_action.sort_values("timestamp").reset_index(drop=True)
df_action


# ## Define session (break) length

# In[ ]:


df_action["sessionid"] = (df_action.groupby(["userid"])["timestamp"].apply(lambda x: (x.diff() > "45 minutes").cumsum()))
df_action


# ## Define quiz repeat

# In[ ]:


df_action["action"].value_counts()


# In[ ]:


df_action.loc[(df_action["action"] == "quiz_repeat_other") & (df_action.groupby(["userid", "sessionid", "quiz"])["timestamp"].rank() > 1), "action"] = "quiz_repeat_same"


# ## Action as names and integers

# In[ ]:


codes, actions = df_action["action"].factorize(sort=True)
codes, actions


# In[ ]:


l = actions.tolist()
df_action["action_name"] = df_action["action"]
df_action["action"] = df_action["action"].apply(lambda x: l.index(x))
del l
df_action["action"]


# ## Descriptive statistics

# ### Number of user sessions

# In[ ]:


len(df_action.groupby(["userid", "sessionid"]))


# ### Number of log entries for reading and quiz

# In[ ]:


df_action.shape


# ### Max length of user sessions

# In[ ]:


df_action.groupby(["userid", "sessionid"]).apply(lambda x: len(x)).max()


# ### Users reading

# In[ ]:


df_action[df_action["action_name"].str.startswith("reading")]["userid"].nunique()


# # Step 1: Modeling

# ## Process mining

# In[ ]:


X = df_action.copy()
X["caseid"] = X["userid"].astype(str) + "_" + X["sessionid"].astype(str)
X = X[["timestamp", "action_name", "caseid", "userid", "sessionid"]]
X


# ## Feature generation

# In[ ]:


pd.qcut(X.groupby("caseid")["caseid"].count(), 3)


# In[ ]:


X["begin_reading"] = X.groupby("caseid").transform("first")["action_name"].str.startswith("reading").astype(int)
X["end_reading"] = X.groupby("caseid").transform("last")["action_name"].str.startswith("reading").astype(int)
X["seqlen_low"] = (X.groupby("caseid").transform("count")["action_name"] <= 3).astype(int)
X["seqlen_high"] = (X.groupby("caseid").transform("count")["action_name"] > 7).astype(int)
X["seqlen_mid"] = ((X.groupby("caseid").transform("count")["action_name"] > 3) & ~(X["seqlen_high"])).astype(int)
X["mid_reading"] = X.groupby("caseid")["action_name"].transform(lambda x: x.value_counts().index[0]).str.startswith("reading").astype(int)


# In[ ]:


plt.figure(figsize=(5,5))
kmeans = KMeans(random_state=1)
visualizer = KElbowVisualizer(kmeans, k=(2,30), metric="distortion")

from sklearn.datasets import make_blobs
Xx, y = make_blobs(n_samples=1, n_features=12, centers=8, shuffle=True, random_state=42)
print(Xx)

aa = X.groupby("caseid").first()[["begin_reading", "end_reading", "seqlen_low", "seqlen_mid", "seqlen_high", "mid_reading"]]
print(aa)
visualizer.fit([aa])
#visualizer.show()
n_clusters = visualizer.elbow_value_
kmeans = KMeans(n_clusters=n_clusters, random_state=1)
kmeans.fit(X.groupby("caseid").first()[["begin_reading", "end_reading", "seqlen_low", "seqlen_mid", "seqlen_high", "mid_reading"]])
kmeans.labels_


# In[ ]:


silhouette_avg = silhouette_score(X.groupby("caseid").first()[["begin_reading", "end_reading", "seqlen_low", "seqlen_mid", "seqlen_high", "mid_reading"]], kmeans.labels_)
sample_silhouette_values = silhouette_samples(X.groupby("caseid").first()[["begin_reading", "end_reading", "seqlen_low", "seqlen_mid", "seqlen_high", "mid_reading"]], kmeans.labels_)
y_lower = 10
fig = plt.figure(figsize=(10, 10))
for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[kmeans.labels_ == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = mpl.cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
del y_lower, y_upper
silhouette_avg


# ### Assign clusters to data

# In[ ]:


i = 0
def cluster_to_groups(g):
    global i
    i += 1
    return kmeans.labels_[i-1]+1

X["cluster"] = X.groupby("caseid")["caseid"].transform(cluster_to_groups)
del i


# ## Plot subprocesses

# In[ ]:


fig = plt.figure(figsize=(50,50))
plt.subplots_adjust(hspace=0.2)

for cluster in range(n_clusters):
    x = X[X["cluster"] == cluster+1]
    lg = pm4py.format_dataframe(x, case_id='caseid', activity_key='action_name', timestamp_key='timestamp')

    net = heuristics_miner.apply_heu(lg, parameters={
        heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.9,
        }, variant=pm4py.algo.discovery.heuristics.variants.classic)

    sumstart = sum(net.start_activities[0].values())
    sumend = sum(net.end_activities[0].values())
    sumnodes = sum(map(lambda x: x[1].node_occ, net.nodes.items()))
    
    for x in net.nodes.copy().items():
        if x[1].node_occ / sumnodes < 0.05:
            net.nodes.pop(x[0])

    graph = pydotplus_vis.get_graph(net)

    for edge in graph.get_edges().copy():
        src = edge.get_source()
        dst = edge.get_destination()

        graph.get_edge(src, dst)[0].set_fontname("Helvetica")

        if src == "start_0":
            val = net.start_activities[0][dst] / sumstart
        elif dst == "end_0" or src in net.dependency_matrix and dst in net.dependency_matrix[src]:
            val = int(edge.get_label()) / (sum([net.nodes[src].output_connections[y][0].repr_value for y in [x for x in net.nodes[src].output_connections]]) + (net.end_activities[0][src] if src in net.end_activities[0] else 0))#net.dependency_matrix[src][dst]

        if val < 0.01:
            graph.del_edge(src, dst)
        else:
            edge = graph.get_edge(src, dst)[0]
            edge.set_label("{:.2f}".format(val))
            edge.set_penwidth(1.0 + math.log(1 + val)*2)
    
    edges = graph.get_edges()
    for node in graph.get_nodes().copy():
        name = node.obj_dict["name"]
        nxt = next((x for x in edges if x.obj_dict["points"][0] != name and x.obj_dict["points"][1] == name), None)
        node = graph.get_node(node.obj_dict["name"])[0]
        node.set_fontname("Helvetica")
        if name not in ["start_0", "end_0"]:
            factor = math.log(net.activities_occurrences[name])*2
            node.set_fontsize(factor)
    
    file_name = "graph_"+str(cluster+1)+".png"
    graph.write(file_name, format="png")
    img = mpl.image.imread(file_name)
    
    plt.subplot(n_clusters, 1, cluster+1)
    plt.title("Study pattern " + str(cluster+1), loc="center")
    plt.grid(False)
    plt.axis("off")
    plt.imshow(img)

    graph.write_dot("graph_"+str(cluster+1)+".txt")
    
del sumstart, sumend, sumnodes


# ## Define study pattern from cluster

# In[ ]:


for cluster in range(n_clusters):
    X.loc[X["cluster"] == cluster+1, "study_pattern"] = "Sub-process " + str(cluster+1)


# ## Descriptive Statistics

# ### Users per subprocess

# In[ ]:


X.groupby(["study_pattern"])["userid"].nunique()


# ### Number of sessions per subprocess

# In[ ]:


X.groupby(["caseid"]).first()["study_pattern"].value_counts(sort=False).sort_index()


# In[ ]:


X.groupby(["caseid"]).first()["study_pattern"].value_counts(sort=False).sort_index().plot(kind="bar", figsize=(20,10))
plt.savefig("subprocess_hist.png")


# ### Mean sessions per user

# In[ ]:


X.groupby(["study_pattern", "userid"])["action_name"].count().groupby(["study_pattern"]).agg(["mean", "std"])


# In[ ]:


X.groupby(["study_pattern", "userid"])["action_name"].count().agg(["mean", "std"])


# In[ ]:


X["userid"].nunique()


# # Step 2: Cluster students by study pattern over study periods

# In[ ]:


df_patterns = X.copy()


# ## Define study period as times between assignment due dates (plus tail)

# In[ ]:


periods = df_assignment_grades["duedate"].sort_values().unique()
periods = np.append(periods, [np.datetime64('2021-02-28T22:59:00.000000000'), np.datetime64('2021-03-31T22:59:00.000000000')])
periods


# ## Define period names

# In[ ]:


period_names = ["P1", "P2", "P3", "P4", "P5", "P6"]


# ## Name study periods

# In[ ]:


for i, duedate in enumerate(reversed(periods)):
    df_patterns.loc[df_patterns["timestamp"] < duedate, "period"] = period_names[-(i+1)]


# ## Descriptive statistics

# In[ ]:


df_patterns.groupby(["caseid"]).first()[["period","study_pattern"]].value_counts().unstack().plot(kind="bar", stacked=True, figsize=(20,10))
plt.savefig("period_hist.png")


# ## Manually combine subprocesses

# In[ ]:


df_patterns = df_patterns.replace({
    "Sub-process 1": "Mainly quiz", 
    "Sub-process 2": "Mainly reading",
    "Sub-process 3": "Mainly quiz",
    "Sub-process 4": "Mainly quiz",
    "Sub-process 5": "Reading and quiz",
    "Sub-process 6": "Mainly reading"
    })
df_patterns


# ## Distribution of subprocesses per student and period

# In[ ]:


d = df_patterns.copy()
d["study_pattern"] = d["study_pattern"].astype("category")
labels = d["study_pattern"].cat.categories.tolist()
df_patterns = d.pivot_table(index="userid", values="study_pattern", columns="period", aggfunc=lambda x: labels[x.value_counts(normalize=True, sort=False).argmax()]).replace(np.nan, "(No reading/quiz)").astype(str).reset_index()
del d, labels
df_patterns


# ## Clustering

# ### One-Hot Encoding

# In[ ]:


df_patterns_dummies = pd.get_dummies(df_patterns.drop("userid", axis=1))
df_patterns_dummies


# ### KMeans

# In[ ]:


kmeans = KMeans(random_state=1)
plt.figure(figsize=(5,5))
visualizer = KElbowVisualizer(kmeans, k=(2,20), metric="silhouette")
visualizer.fit(df_patterns_dummies)
plt.figure(figsize=(5,5))
visualizer = KElbowVisualizer(kmeans, k=(2,20), metric="distortion")
visualizer.fit(df_patterns_dummies)
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=1)
labels = kmeans.fit_predict(df_patterns_dummies)
labels


# ### Get distortion score

# In[ ]:


visualizer.k_scores_[n_clusters-1]


# In[ ]:


silhouette_avg = silhouette_score(df_patterns_dummies, labels)
sample_silhouette_values = silhouette_samples(df_patterns_dummies, labels)
y_lower = 10
fig = plt.figure(figsize=(10, 10))
for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = mpl.cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
silhouette_avg


# ### Assign clusters to data

# In[ ]:


df_patterns["cluster"] = labels
df_patterns


# In[ ]:


values = list(np.sort(pd.unique(df_patterns.drop(["cluster", "userid"], axis=1).values.ravel())))

value_to_int = {j:i for i,j in enumerate(values)}
n = len(value_to_int)
cmap = sns.color_palette("Paired", n) 


cmap = [[0.99, 0.99, 0.99],'#333333','#cecece',   '#7c7c7c']

sns.set(rc={'figure.figsize':(8,10)})


fig, ax = plt.subplots(n_clusters,1, sharex=True, 
                    constrained_layout=True,
                 gridspec_kw={'height_ratios':df_patterns.groupby("cluster")["cluster"].count()})

for i,x in df_patterns.groupby("cluster"):
    sns.heatmap(ax=ax[i], data=x.drop(["cluster", "userid"], axis=1).replace(value_to_int),     cmap=cmap, cbar_kws = dict(use_gridspec=False,location="right", shrink=4.5, anchor=(0,4), drawedges=True), yticklabels=False)
    
    plt.subplots_adjust(hspace=0.2)
    # modify colorbar:
    colorbar = ax[i].collections[0].colorbar 
    if i == 0:
        r = colorbar.vmax - colorbar.vmin 
        colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
        colorbar.ax.set_yticklabels(list(value_to_int.keys()),rotation=0)
    else:
        colorbar.remove()

    ax[i].set_title("C" + str(i+1), rotation='vertical',x=-0.05,y=len(x)/100)
    ax[i].set_xlabel("")
    ax[i].set_ylabel("N = " + str(len(x)) + "")

fig.savefig("periods_clustering.png", bbox_inches="tight", dpi=300)


# ## Descriptive statistics

# ### No. of sessions per cluster

# In[ ]:


i = 1
d = pd.DataFrame()
for userids in df_patterns.groupby("cluster")["userid"].unique():
    cnt = pd.DataFrame(X[X["userid"].isin(userids)].replace({
    "Sub-process 1": "Mainly quiz", 
    "Sub-process 2": "Mainly reading",
    "Sub-process 3": "Mainly quiz",
    "Sub-process 4": "Mainly quiz",
    "Sub-process 5": "Reading and quiz",
    "Sub-process 6": "Mainly reading"
    }).groupby(["caseid"]).first()["study_pattern"].value_counts())
    cnt["cluster"] = i
    d = pd.concat([d, cnt])
    i += 1
d = d.rename_axis("index").sort_values(by=["cluster", "index"] )
print(d)
d.groupby(d.index).sum(), d.groupby("cluster").sum(), d.groupby(d.index).sum().sum()


# ### Sessions per learner per cluster

# In[ ]:


i = 1
for userids in df_patterns.groupby("cluster")["userid"].unique():
    print("Cluster " + str(i))
    i += 1
    print(X[X["userid"].isin(userids)].groupby(["userid", "caseid"])["userid"].count().agg(["mean", "std"]))

