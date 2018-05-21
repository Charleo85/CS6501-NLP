# CS 6501 Natural Language Processing

**Under Construction**

- Instructor: [Yangfeng Ji](http://yangfengji.net)
- Semester: Fall 2018
- Time: TBD

## Course Description

Natural language processing (NLP) seeks to provide computers with the ability to process and understand human language intelligently. Examples of NLP techniques include (i) automatically translating from one natural language to another, (ii) analyzing documents to answer related questions or make related predictions, and (iii) generating texts to help story writing or build conversational agents. This course, consisting of one fundamental part and one advanced part, will give an overview of modern NLP techniques. 

Topics of this course include 

1. POS tagging, syntactic parsing; 
2. Discourse processing and coreference resolution; 
3. Distributed semantics and representation learning; 
4. Neural language models, seq2seq models and neural machine translation;
5. NLP applications: QA, text generation, etc.

## Syllabus

- [Tentative topics per week](https://docs.google.com/spreadsheets/d/1vSIUQCYgkmJqzUKRi5VX7ScRci2puRBZUOIU5r4acP8/edit?usp=sharing)

## Textbooks

- Jurafsky and Martin, [Speech and Language Processing](https://web.stanford.edu/%7Ejurafsky/slp3/), 3rd Edition, 2017

### Supplemental textbooks

- Smith, [Linguistic Structure Prediction](https://www.morganclaypool.com/doi/abs/10.2200/S00361ED1V01Y201105HLT013), 2009
- Goodfellow, Bengio and Courville, [Deep Learning](http://www.deeplearningbook.org), 2016
- Bender, [Linguistic Fundamentals for Natural Language Processing](https://www.morganclaypool.com/doi/abs/10.2200/S00493ED1V01Y201303HLT020), 2013
- Eisenstein, [Natural Language Processing](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf), 2018

### Additional Readings

- [Reading List](readings.md)

## Grading

The graded material for the course will consist of:

- Seven short homework assignments, of which you must do five. Most of these involve performing linguistic annotation on some text of your choice. The purpose is to get a basic understanding of key linguistic concepts. Each assignment should take less than an hour. Each homework is worth 2 points (10 total). 
- Five assigned problem sets. These involve building and using NLP techniques which are at or near the state-of-the-art. The purpose is to learn how to implement natural language processing software, and to have fun. These assignments must be done individually. Each problem set is worth ten points (50 total). Students enrolled in CS 7650 will have an additional, research-oriented component to the problem sets.
- An in-class midterm exam, worth 20 points, and a final exam, worth 20 points. The purpose of these exams are to assess understanding of the core theoretical concepts, and to encourage you to review and synthesize your understanding of these concepts. Barring a personal emergency or an institute-approved absence, you must take the exam on the days indicated in the schedule. See [here](http://www.deanofstudents.gatech.edu/content/25/absences) and [here](http://www.registrar.gatech.edu/students/formlanding/iaabsences.php) for more information on GT policy about absences.

### Late policy

Problem sets will be accepted up to 72 hours late, at a penalty of 20% per 24 hours. (Maximum score after missing the deadline: 8/10; maximum score 24 hours after the deadline: 6/10, etc.)  It is usually best just to turn in what you have at the due date. Late homeworks will not be accepted. This late policy is intended to ensure fair and timely evaluation.


## Office hours

- TODO

### Online help

Please use Piazza rather than personal email to ask questions. This helps other students, who may have the same question. Personal emails may not be answered. If you cannot make it to office hours, please use Piazza to make an appointment. It is unlikely that I will be able to chat if you make an unscheduled visit to my office. The same is true for the TAs.

## Prerequisites

The official prerequisite for CS 4650 is CS 3510/3511, "Design and Analysis of Algorithms." This prerequisite is essential because understanding natural language processing algorithms requires familiarity with dynamic programming, as well as automata and formal language theory: finite-state and context-free languages, NP-completeness, etc. While course prerequisites are not enforced for graduate students, prior exposure to analysis of algorithms is very strongly recommended.

Furthermore, this course assumes:

- Good coding ability, corresponding to at least a third or
  fourth-year undergraduate CS major. Assignments will be in Python.
- Background in basic probability, linear algebra, and calculus.
- Familiarity with machine learning is *helpful but not assumed*. Of
  particular relevance are linear classifiers: perceptron, naive
  Bayes, and logistic regression.

People sometimes want to take the course without having all of these
prerequisites. Frequent cases are:

- Junior CS students with strong programming skills but limited
  theoretical and mathematical background,
- Non-CS students with strong mathematical background but limited
  programming experience.

Students in the first group suffer in the exam and don't understand
the lectures, and students in the second group suffer in the problem sets. My advice is to get the background material first, and
then take this course.

## Collaboration policy

One of the goals of the assigned work is to assess your individual progress in meeting the learning objectives of the course. You may discuss the homework and projects with other students, but your work must be your own -- particularly all coding and writing. For example:

### Examples of acceptable collaboration

- Alice and Bob discuss alternatives for storing large, sparse vectors of feature counts, as required by a problem set.
- Bob is confused about how to implement the Viterbi algorithm, and asks Alice for a conceptual description of her strategy.
- Alice asks Bob if he encountered a failure condition at a "sanity check" in a coding assignment, and Bob explains at a conceptual level how he overcame that failure condition.
- Alice is having trouble getting adequate performance from her part-of-speech tagger. She finds a blog page or research paper that gives her some new ideas, which she implements.

### Examples of unacceptable collaboration

- Alice and Bob work together to write code for storing feature counts.
- Alice and Bob divide the assignment into parts, and each write the code for their part, and then share their solutions with each other to complete the assignment.
- Alice or Bob obtain a solution to a previous year's assignment or to a related assignment in another class, and use it as the starting point for their own solutions.
- Bob is having trouble getting adequate performance from his part-of-speech tagger. He finds source code online, and copies it into his own submission.
- Alice wants to win the Kaggle competition for a problem set. She finds the test set online, and customizes her submission to do well on it.

Some assignments will involve written responses. Using other people’s text or figures without attribution is plagiarism, and is never acceptable.

Suspected cases of academic misconduct will be (and have been!) referred to the Honor Advisory Council. For any questions involving these or any other Academic Honor Code issues, please consult me, my teaching assistants, or http://www.honor.gatech.edu.

## Acknowledgments

- Noah Smith's lecture slides
- Jacob Eisenstein's lecture notes and collaboration policy
- Michael Collins' lecture notes

**Last Updated**: May 21, 2018
