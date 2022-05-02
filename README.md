# delfi2022-reading-assessment



## Pre-requisits and install instructions
1. Make sure Python v3.9 is installed on your system.
2. Install depending modules: 
  conda install -c conda-forge cvxopt 
  conda install -c conda-forge pm4py 
  conda install -c conda-forge seaborn
  conda install -c conda-forge sklearn
  conda install -c conda-forge pandas
  conda install -c conda-forge numpy
  conda install -c conda-forge yellowbrick
  conda install -c conda-forge matplotlib
4. To replicate the analysis open the file Ananlysis.ipynb and execute the code blocks one by one or all together.

## Files and folders
- (File) Analysis.ipynb: Pythone Notebook containing all applied code blocks applied for data analyis.
- (File) requirements.txt: List of python modules to be installed to fulfil the requirements of the Analysis.ipynb script.
- (Folder) data: Folder containes anonymized CSV files for each Moodle database tables that was necessary for the data analysis. All data files are text files encoded in UTF-8. The columns are separated with a semicolon (";"), rows are indicated by line breaks ("\n").
  * m_assign.csv: ...
  * m_course_modules.csv: ...
  * m_quiz.csv: ...
  * no_students.csv: ...
  * user_acceptances.csv: ...
  * m_assign_grades.csv: ...
  * m_course_sections.csv: ...
  * m_quiz_attempts.csv: ...
  * scroll.csv: ...



jupyter nbconvert --to python Analysis.ipynb

jupyter notebook Analysis.py
