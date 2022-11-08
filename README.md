# Interactions of reading and assessment activities in Moodle
Reading and assessment are elementary activities for knowledge acquisition in online learning. Assessments represented as quizzes can help learners to identify gaps in their knowledge and understanding, which they can then overcome by reading the corresponding text-based course material. Reversely, quizzes can be used to evaluate reading comprehension. In this paper, we ex- amine the interactions between reading and quiz activities using scroll and log data from an online undergraduate course (N=142). By analyzing processes and sequential patterns in user sessions, we identified six session clusters for characteristic reading and quiz patterns potentially relevant for adaptive learning support. Using these session clusters, we further clustered students by their reading and quiz behavior over six time periods within the semester. The results hypothesize a personalization for seven groups of learners characterized by their temporal activity and predominant quiz and reading behavior.


## Pre-requisits and install instructions
1. Make sure Python v3.9 is installed on your system.
2. conda env create -f environment.yml
3. conda activate analysis
4. To replicate the analysis open the file Ananlysis.ipynb and execute the code blocks one by one or all together.

`jupyter nbconvert --to python Analysis.ipynb`

`jupyter notebook Analysis.py`


## Files and folders

* (File) Analysis.ipynb: Python Notebook containing all applied code blocks applied for data analysis.
* (File) requirements.txt: List of python modules to be installed to fulfill the requirements of the Analysis.ipynb script.
* (Folder) data: The folder contains anonymized CSV files for each Moodle database table that was necessary for the data analysis. All data files are text files encoded in UTF-8. The columns are separated with a semicolon (";"), and rows are indicated by line breaks ("\n").
  * m_assign.csv: ...
  * m_course_modules.csv: ...
  * m_quiz.csv: ...
  * no_students.csv: ...
  * user_acceptances.csv: ...
  * m_assign_grades.csv: ...
  * m_course_sections.csv: ...
  * m_quiz_attempts.csv: ...
  * scroll.csv: ...


  # Publications and citation

**Publications**
* Seidel, N., & Menze, D. (2022). Interactions of reading and assessment activities. In S. Sosnovsky, P. Brusilovsky, & A. Lan (Eds.), 4th Workshop on Intelligent Textbooks, 2022 (pp. 64–76). CEUR-WS. http://ceur-ws.org/Vol-3192/
* Menze, D., Seidel, Ni., & Kasakowskij, R. (2022). Interaction of reading and assessment behavior. In P. A. Henning, M. Striewe, & M. Wölfel (Eds.), DELFI 2022 – Die 21. Fachtagung Bildungstechnologien der Gesellschaft für Informatik e.V. (pp. 27–38). Gesellschaft für Informatik. https://doi.org/10.18420/delfi2022-011

**Citation of the Dataset**

* Seidel. Niels, & Menze, Dennis. (2022). Data and Analysis of Reading and Assessment Activities in Moodle (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7300070

The source code and data are maintained at GitHub: https://github.com/nise/delfi22

**Acknowledgments** This research was supported by CATALPA - Center of Advanced Technology for Assisted Learning and Predictive Analytics of the FernUniversität in Hagen, Germany.
